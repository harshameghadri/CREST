# Downstream analysis in CREST

Tools added in 0.3.0 to cover the usual steps after clustering: batch
integration, doublet detection, variance-stabilised HVGs, resolution
selection, reference mapping and likelihood-ratio DE. Each one follows a
reference implementation and is checked against it: in `tests/` (runs in CI)
and in a benchmark on real data (`bench/<module>/`, results in
`bench/<module>/results/`). All numbers below were measured on the 4-core
cloud machine used for development. Timings on your machine come from
`scripts/crest_paper_bench.sh`.

```python
import crest

crest.pp.highly_variable_genes(bf, flavor="seurat_v3", n_top_genes=2000, batch_key="sample")  # raw counts
crest.pp.scrublet(bf, batch_key="sample")                         # obs: doublet_score, predicted_doublet
crest.tl.harmony(bf, "sample")                                    # obsm['X_pca_harmony']
crest.pp.neighbors(bf, use_rep="X_pca_harmony")
table = crest.tl.leiden_sweep(bf, [0.2, 0.5, 1.0, 1.5, 2.0], n_seeds=3)   # n_clusters, stability
crest.tl.ingest(query, reference, obs="cell_type")                # label transfer + embedding
ds = crest.tl.DESeq2(pb, design="~ donor + condition", test="LRT", reduced="~ donor")
```

## Harmony (`crest.tl.harmony`, Rust)

A port of the harmony2 algorithm that R `harmony` ≥ 1.2 and `harmonypy` 2.x share:

* cosine-normalised soft k-means with the diversity penalty θ;
* block-wise assignment updates (5% of cells per block);
* the per-cluster ridge "mixture of experts" correction, with the automatic ridge λ = α·E;
* multiple batch covariates, the same objective and convergence tests, and the same defaults.

The two O(cells × clusters × PCs) steps run as GEMMs per batch level. Everything that works per cell runs in parallel.

Harmony's k-means initialisation is random, so the comparison with harmonypy is on outcomes, over 3 seeds each. Kang 2018 (24,679 cells, 50 PCs, batch = ctrl/stim), `bench/harmony/`:

| | time | iLISI (batch mixing, max 2) | cLISI (cell types, 1 = pure) | ARI Leiden vs annotation |
|---|---|---|---|---|
| uncorrected | | 1.05 | 1.01 | 0.755 |
| CREST | **2.1 s** | 1.81 | 1.01 | 0.820 |
| harmonypy 2.0.2 | 7.4 s | 1.81 | 1.01 | 0.822 |

Neighbourhoods agree: a cell's 15 nearest neighbours in CREST's corrected space and in harmonypy's share 98.0% on average. That is the same as the agreement between two seeds of the same tool (CREST 98.4%, harmonypy 98.2%), so the differences are at the level of seed noise.

Note: `scanpy.external.pp.harmony_integrate` (scanpy ≤ 1.11) transposes harmonypy 2.x's output and fails. Call `harmonypy.run_harmony` directly, or use CREST.

## Scrublet (`crest.pp.scrublet`)

Same pipeline and defaults as `scanpy.pp.scrublet`:

1. HVGs are chosen on normalised log data.
2. Doublets are simulated by summing the raw counts of random cell pairs (2× the number of cells).
3. Profiles are normalised to 1e6 and z-scored.
4. A 30-PC PCA is fitted on the observed cells.
5. A kNN classifier gives Scrublet's Bayesian doublet score.
6. The threshold is `skimage.filters.threshold_minimum` (ported; it matches skimage to 1e-9).

CREST keeps the matrices sparse and streams them through the native PCA kernels, where scanpy densifies 3 × cells × HVGs. It also uses exact GEMM kNN search.

Validated against the demuxlet doublet calls of Kang 2018 (27,848 droplets, 3,169 doublets between donors, per-condition batches), `bench/doublets/`, median of 3 seeds:

| | time | AUROC | AUPRC | cells called | precision |
|---|---|---|---|---|---|
| CREST | **16.9 s** | 0.863 | 0.537 | 994 | 0.69 |
| scanpy 1.11 | 210 s | 0.863 | 0.539 | 1,010 | 0.69 |

Scores correlate with scanpy's at Spearman 0.98. Simulated doublet pairs come from a different random generator, so the scores are equivalent statistically, not bit for bit.

## seurat_v3 highly variable genes (`flavor="seurat_v3"`)

Variance-stabilising HVG selection on raw counts, with an optional `batch_key`. The loess mean-variance trend is a port of netlib `dloess`, the code behind R `loess` and `skmisc.loess`:

* k-d tree cells;
* degree-2 local fits at the vertices;
* cubic Hermite interpolation between vertices.

It matches `skmisc.loess` to about 1e-14. The clipped sums stream through the chunk kernels, so the count matrix is never copied.

Against scanpy (tests): `variances_norm` agrees to 1e-14, the HVG sets are identical and `highly_variable_rank` is identical, with and without batches.

## Resolution sweep (`crest.tl.leiden_sweep`)

Leiden at several resolutions, and optionally several seeds, all on one neighbour graph. The runs go in parallel on all cores; a loop of `leiden()` calls would rebuild the graph each time and run on one core. Each resolution's labels are identical to `crest.tl.leiden`. The returned table has, per resolution:

* `n_clusters`;
* `ari_prev`: agreement with the next lower resolution, a clustree-style stability check;
* `ari_seeds`: with `n_seeds > 1`, the mean pairwise ARI across seeds, i.e. how reproducible the partition is. It drops sharply once a resolution starts splitting real clusters.

`crest.tl.adjusted_rand_index` matches `sklearn.metrics.adjusted_rand_score`.

## Reference mapping (`crest.tl.ingest`)

Like `scanpy.tl.ingest`:

1. Query cells are projected onto the reference PCA with the reference's genes, normalisation and scaling parameters. Missing genes count as unexpressed. Like scanpy, the query is centred on its own mean.
2. Each query cell gets its k nearest reference cells, using a new Rust reference → query kNN (exact GEMM, or an IVF index over the reference).
3. Labels are transferred by majority vote, with a per-cell `<label>_confidence`.
4. UMAP coordinates come from umap-learn's `transform` initialisation, a membership-weighted mean of the neighbours.

Kang 2018, reference = 12,315 control cells, query = 12,364 IFN-β stimulated cells, label = published cell type (`bench/ingest/`):

| | time | accuracy |
|---|---|---|
| CREST | **1.5 s** | 0.907 |
| scanpy 1.11 | 23.6 s | 0.898 |

The two tools assign the same label to 96.4% of cells. CREST's UMAP placement uses the transform initialisation only; scanpy additionally runs umap-learn's transform optimisation.

## DESeq2 likelihood-ratio test (`DESeq2(test="LRT", reduced=...)`)

This is `nbinomLRT`:

* the reduced model is fitted with the full model's dispersions;
* the statistic is deviance(reduced) − deviance(full), with a χ² p-value on p_full − p_reduced degrees of freedom;
* an intercept-only reduced model uses β = log(baseMean), as `fitNbinomGLMs` does;
* Cook's filtering, outlier refit and independent filtering are as in the Wald path;
* the contrast only chooses the reported fold change.

Checked against stored R DESeq2 1.42 output (`tests/data/*_LRT_R.csv`, regenerated by `tests/data/make_lrt_fixtures.R`): statistics, p-values and padj agree to 1e-6 relative. That holds for `~ batch + condition` vs `~ batch`, and for `~ condition` vs `~ 1` with outlier replacement.
