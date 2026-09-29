# How CREST works

Read this page first if you want to understand what CREST does to your data, or to change the
code. Everything else follows from one design decision.

## The design decision: raw counts are never modified

scanpy normalises, log-transforms, subsets and scales **in place**. Each step writes a new
matrix (the scaled matrix is dense), so memory grows with every step.

CREST instead keeps **one read-only copy of the raw counts** and records everything else as
metadata:

| you call | what CREST stores |
|---|---|
| `pp.filter_cells`, `pp.filter_genes`, `bf.filter_cells(expr)` | which cells / genes are kept (`obs["cell_id"]`, `var["gene_id"]` point into the raw store) |
| `pp.normalize_total(bf, 1e4)` | `("normalize_total", 1e4)` appended to `bf.ops` |
| `pp.log1p(bf)` | `("log1p",)` appended to `bf.ops` |
| `pp.scale(bf, max_value=10)` | `bf.uns["scale"] = {"max_value": 10}` |

Every computing step (QC, HVG, PCA, DE, scores, pseudobulk) then streams the raw counts **one
chunk at a time** into a Rust kernel. The kernel applies the filters and transforms to each
cell *on the fly*, in a small per-thread buffer, and accumulates what it needs:

```text
raw store ──chunk──▶ Rust kernel: filter → normalise → log1p → (scale) → accumulate ──▶ small result
                ▲                                                                    │
                └───────────────────── next chunk ◀──────────────────────────────────┘
```

The kernel sees exactly the numbers scanpy would hold in memory; only *when* the arithmetic
happens changes, so the results match. Memory is one chunk (≤ 16.8 M non-zeros by default)
plus results that scale with cells *or* genes, never cells × genes.

## The BioFrame

{class}`crest.core.BioFrame` is CREST's AnnData. It is a dataclass with these fields:

| field | type | contents |
|---|---|---|
| `store` | `CSRStore` / `FrameStore` / `ParquetStore` | the raw counts (never modified) |
| `obs` | `polars.DataFrame` | one row per **kept** cell; `cell_id` = row in the store |
| `var` | `polars.DataFrame` | one row per **kept** gene; `gene_id` = column in the store |
| `obsm` | `dict[str, ndarray]` | per-cell embeddings: `X_pca`, `X_umap`, `X_pca_harmony` |
| `varm` | `dict[str, ndarray]` | per-gene matrices: `PCs` |
| `uns` | `dict` | everything else: parameters, the kNN graph, DE tables |
| `ops` | `list[tuple]` | the recorded transforms, in order |
| `chunk_nnz` | `int` | non-zeros per chunk (default 2²⁴) |

**Subsetting shares the store.** `filter_cells`, `filter_genes` and `copy` return a new
BioFrame with new `obs`/`var` tables that point into the same raw store. They are cheap and
safe, because nothing ever writes to the store.

**Functions modify `bf` in place and also return it**, like scanpy with `inplace=True`. The
filters are the exception: they return a **new** object. Always write
`bf = crest.pp.filter_cells(bf, ...)`.

### The three stores

| store | where the counts live | made by |
|---|---|---|
| `CSRStore` | in memory, compact CSR (`uint32` gene indices + `float32` counts) | `read_10x_h5`, `read_h5ad`, `from_scipy`, `from_csr`, `from_anndata` |
| `FrameStore` | in memory, a Polars frame of `(cell_id, gene_id, count)` triplets | `read_10x_mtx`, `from_triplets` |
| `ParquetStore` | **on disk**, a directory of `part-*.parquet` files, each holding whole cells | `read_10x_h5(backed=dir)`, `bf.write_parquet(dir)` → `read_parquet(dir)` |

All three expose the same `chunks()` iterator, so every analysis function works unchanged on
all of them. The Parquet store is what makes CREST *out-of-core*: only one part is in memory
at a time, so peak memory stays around 1 GB whatever the dataset size. The price is that each
pass re-reads the files, roughly 2× slower than in memory.

## Where results go

| step | writes |
|---|---|
| `pp.calculate_qc_metrics` | obs `n_genes_by_counts`, `total_counts`, `pct_counts_mt`; var `n_cells_by_counts`, `total_counts` |
| `pp.highly_variable_genes` | var `highly_variable`, `means`, `dispersions(_norm)`, cached `mean_log` / `std_log`; `uns["hvg"]` |
| `tl.pca` | `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]` (variance, variance_ratio, `projection` for `ingest`) |
| `pp.neighbors` | `uns["neighbors"]`: `indices`, `distances` (kNN), `rows`, `cols`, `weights` (fuzzy graph) |
| `tl.leiden` | obs `leiden` (strings, `"0"` = largest cluster) |
| `tl.leiden_sweep` | obs `leiden_<res>` per resolution; returns the stability table |
| `tl.umap` | `obsm["X_umap"]` |
| `tl.rank_genes_groups` | returns a Polars table; `uns["rank_genes_groups"]` |
| `tl.score_genes` | obs `<score_name>` |
| `tl.harmony` | `obsm["X_pca_harmony"]`, `uns["harmony"]` |
| `pp.scrublet` | obs `doublet_score`, `predicted_doublet`, `doublet_z`; `uns["scrublet"]` |
| `tl.ingest` (on the query) | `obsm["X_pca"]`, `obsm["X_umap"]`, obs `<label>` + `<label>_confidence` |
| `tl.pseudobulk_de` | returns a Polars table; `uns["pseudobulk_de"]` |

## Which values each step uses

| step | uses |
|---|---|
| QC, filters, `seurat_v3` HVG, Scrublet, pseudobulk / DESeq2 | **raw counts** (the recorded transforms are ignored) |
| `seurat` / `cell_ranger` HVG, DE, `score_genes` | **log-normalised** values (what `adata.raw` holds in scanpy) |
| PCA | log-normalised, then **scaled and clipped** if `pp.scale` was called |
| neighbors, Leiden, UMAP, Harmony, ingest | the embedding in `obsm` (`X_pca` by default) |

## Why PCA is exact without a dense matrix

`sc.pp.scale(max_value)` turns every zero of gene *j* into the same constant
$b_j = \mathrm{clip}(-\mu_j/\sigma_j)$. So the dense scaled matrix is
$Z = \mathbf{1} b^\top + S$, where $S$ is **sparse** and non-zero only where the data are
non-zero. Centring removes $\mathbf{1} b^\top$ exactly, so the principal components of $Z$
are those of $S$.

CREST accumulates the genes × genes Gram matrix of $S$ one chunk at a time, runs a symmetric
eigendecomposition, then projects the cells in a second pass. The result equals scanpy's
dense PCA (principal angles 0.000°); see {doc}`memory_model`.

## Performance model

* **Rust does the per-element work. Python only orchestrates.** Every loop over non-zeros or
  cells is in Rust (`src/`), releases the GIL, and uses all cores via rayon. Python code in
  `crest/` works on per-gene or per-group vectors (size genes or groups), never on
  cells × genes.
* **One pass per step.** Each kernel fuses filter + transform + accumulate, so a step reads
  the raw counts once (PCA twice).
* **Scale with cores, within limits.** The kernels scale well. Some steps are serial by
  nature: reading gzip-compressed HDF5, Leiden's local moving, and the Python glue. On
  64 threads the core pipeline is ~2.6× faster than on 1 (pbmc68k), not 64×. The benchmark
  reports this honestly; see {doc}`benchmarks`.

## Reproducibility

Every stochastic step takes `random_state` (default 0) and is deterministic for a given thread
count. UMAP's parallel optimisation is deterministic per thread count, not across thread
counts. Harmony's k-means initialisation is random, like harmonypy's; compare outcomes, not
coordinates.
