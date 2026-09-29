# Changelog

## Unreleased

### New
- **Pseudobulk DESeq2 in Rust.** `crest.tl.DESeq2` ports DESeq2 1.42 `DESeq()` +
  `results()`: median-of-ratios size factors, Cox-Reid gene-wise dispersions,
  parametric trend, MAP shrinkage, NB-GLM Wald tests, Cook's filtering and
  outlier replacement, and independent filtering. It matches R to ~1e-9 on
  simulated designs, is ~18× faster than R and ~28× faster than pydeseq2 on
  Kang 2018 pseudobulk. See `docs/deseq2.md`.
- `crest.tl.pseudobulk` (streamed raw-count sums per sample × group) and
  `crest.tl.pseudobulk_de` (per-cell-type DESeq2 in one call).
- `bench/deseq2/`: R comparison on simulated designs and on Kang et al. 2018.
- **Harmony** batch integration in Rust (`crest.tl.harmony`), the harmony2 algorithm of
  R harmony >= 1.2 / harmonypy 2.x; 3.5x faster than harmonypy at equal integration quality.
- **Scrublet** doublet detection (`crest.pp.scrublet`), sparse and streamed; 12x faster
  than `scanpy.pp.scrublet` with the same AUROC on demuxlet-labelled doublets.
- `highly_variable_genes(flavor="seurat_v3" | "seurat_v3_paper", batch_key=...)`, with a
  port of netlib loess (`crest._loess`); identical to scanpy.
- `crest.tl.leiden_sweep`: many resolutions/seeds on one graph in parallel, with
  ARI-based stability; `crest.tl.adjusted_rand_index`.
- `crest.tl.ingest`: project a query onto a reference PCA, transfer labels and UMAP.
  `uns['pca']['projection']` now stores what the projection needs.
- DESeq2 likelihood-ratio test: `DESeq2(test="LRT", reduced="~ ...")`, matching R.
- Native `knn_query` (reference -> query kNN, exact or IVF).
- `scripts/crest_paper_bench.sh` + `bench/paper/`: one-command publication benchmark
  (datasets, per-core CPU/clock/memory monitoring, statistics, figures).
- `scripts/crest_git_housekeeping.sh`: repository housekeeping (dev branch, stale branches).
- **Documentation site** (Sphinx + MyST, Read the Docs): installation, quickstart, concepts,
  full API reference, benchmarks, developer guide; `.readthedocs.yaml`; CI `docs` job.
- `work.md`: the development history; `CLAUDE.md` / `HANDOVER.md` rewritten for new sessions.

### Removed
- The `.bio` Polars expression namespace and its Rust plugin code (duplicated the
  validated API, was not validated itself, and tied the wheel to Polars' plugin ABI).
  The `polars`/`pyo3-polars` Rust dependencies go with it.
- SLAF support (`BioFrame.from_slaf`, `crest/slaf_io.py`, the `slaf` extra).
- Legacy scripts: `bench/*.py` from the biopolars era, `notebooks/`, the Docker setup,
  `docs/dev/`, and `bench/whitepaper/`'s scripts (superseded by `bench/paper/`; its
  0.2.0 results stay).

### Fixed
- `BioFrame.to_anndata()` / `from_anndata()` no longer need pyarrow.
- `crest_paper_bench.sh` stops with a clear message when the work directory is not writable,
  and records the work directory's device and filesystem.

## 0.2.0

A correctness and performance release. Every step of the standard scanpy
workflow has been re-implemented and validated against scanpy.

### Fixed (results from 0.1.0 should not be used)
- **kNN / Leiden / UMAP**: the HNSW index shuffled points and its id map was
  ignored, so neighbour indices pointed at random cells (recall@15 = 0.005).
- **Leiden** was a single-level local-move heuristic (thousands of singleton
  clusters) with a modularity gain off by a factor of 2; now Traag et al. 2019
  Leiden (local moving, refinement, aggregation). Modularity equals or exceeds
  `leidenalg`.
- **normalize_cpm / qc / filter_cells / scale / score_genes** Polars plugins
  were declared elementwise, so Polars computed per-cell sums on partial
  batches; `normalize_cpm` ignored `target_sum`.
- **rank_genes_groups** divided by non-zero counts instead of group sizes.
- **Randomized PCA** lost trailing components (no re-orthonormalisation).
- **HVG** "seurat" flavour was raw variance/mean (883/2000 overlap with scanpy).
- **Wilcoxon** p-values underflowed to 0.

### New
- `crest.pp` / `crest.tl` scanpy-style API on `BioFrame`, with raw counts held
  in compact CSR, a Polars triplet frame, or a Parquet dataset streamed from disk.
  Filters and `normalize_total` / `log1p` / `scale` are applied lazily inside
  fused native kernels: no normalised, scaled or dense copy is ever made.
- Exact PCA from a streamed Gram matrix with `sc.pp.scale(max_value)` applied
  implicitly (0.000 deg from scanpy's ARPACK result).
- kNN: tiled GEMM brute force (small n) and IVF + NN-descent (large n).
- UMAP: umap-learn semantics, parallelised by domain decomposition.
- Sparse Wilcoxon ranking only non-zeros; all-groups Welch t-test.
- `score_genes` reproducing scanpy's control-gene sampling.
- Readers: `read_10x_h5` (optionally streamed to Parquet), `read_h5ad`,
  `read_10x_mtx`; `write_parquet` / `read_parquet`; AnnData conversion.
- `bench/whitepaper/` benchmark harness.

### Changed
- `bio.deseq2` renamed `bio.nb_glm` (it fits an NB GLM with given dispersions;
  not the full DESeq2 procedure) and returns null when IRLS does not converge.
- In `bio.leiden` / `bio.umap`, `n_neighbors` now counts the cell itself (scanpy convention).
- `crest.tl.sparse_masked_pca` / `incremental_pca` removed (superseded by `crest.tl.pca`).
- Distribution renamed `crest-sc` (the import name stays `crest`); Python >= 3.10.
