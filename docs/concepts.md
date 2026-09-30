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
happens changes, so the results match. Besides the raw counts themselves, a step needs one
chunk (for the in-memory stores a chunk is a view, not a copy), small per-thread buffers, and
its results, which scale with cells *or* genes, never with cells × genes.

## The BioFrame

{class}`crest.core.BioFrame` plays the role of AnnData. It is a Python dataclass with these
fields:

| field | type | contents |
|---|---|---|
| `store` | `CSRStore` / `FrameStore` / `ParquetStore` | the raw counts (never modified) |
| `obs` | `polars.DataFrame` | one row per **kept** cell; `cell_id` = row in the store |
| `var` | `polars.DataFrame` | one row per **kept** gene; `gene_id` = column in the store |
| `obsm` | `dict[str, ndarray]` | per-cell embeddings: `X_pca`, `X_umap`, `X_pca_harmony` |
| `varm` | `dict[str, ndarray]` | per-gene matrices: `PCs` |
| `uns` | `dict` | everything else: parameters, the kNN graph, DE tables |
| `ops` | `list[tuple]` | the recorded transforms, in order, e.g. `[("normalize_total", 10000.0), ("log1p",)]` |
| `chunk_nnz` | `int` | target non-zeros per chunk for the in-memory stores (default 2²⁴ ≈ 16.8 M); the Parquet store ignores it (one chunk = one file) |

There is no `X`: the counts are only reachable through `store`, and the current (filtered,
transformed) matrix is materialised only if you ask for it with `bf.to_scipy()` or
`bf.to_anndata()`.

**Subsetting shares the store.** `filter_cells`, `filter_genes` and `copy` return a new
BioFrame with its own (smaller) `obs`/`var` tables and copies of the `obsm`/`varm` rows it
keeps; the count matrix itself is shared, never copied. This is safe because nothing writes to
the store.

**What each function returns:**

* the filters (`pp.filter_cells`, `pp.filter_genes`, `bf.filter_cells`, `bf.filter_genes`,
  `pp.highly_variable_genes(..., subset=True)`) return a **new** BioFrame. Always write
  `bf = crest.pp.filter_cells(bf, ...)`;
* the transforms and tools (`normalize_total`, `log1p`, `scale`, `highly_variable_genes`,
  `pca`, `neighbors`, `leiden`, `umap`, `score_genes`, `harmony`, `scrublet`, `ingest`) modify
  `bf` in place and return the same object, like scanpy with `inplace=True`;
* the functions that produce a table (`rank_genes_groups`, `leiden_sweep`, `pseudobulk_de`)
  return it as a Polars DataFrame and also store it in `bf.uns`;
* `calculate_qc_metrics` modifies `bf` and returns `None` (or two tables with
  `inplace=False`).

### The three stores

| store | where the counts live | made by |
|---|---|---|
| `CSRStore` | in memory: CSR arrays `indptr` (`int64`), `indices` (`uint32` gene ids), `data` (`float32` counts), about 8 bytes per non-zero | `read_10x_h5`, `read_h5ad` (CSR or CSC `X`), `BioFrame.from_csr`, `from_scipy`, `from_anndata` |
| `FrameStore` | in memory: a Polars DataFrame of `(cell_id, gene_id, count)` triplets sorted by cell, about 12 bytes per non-zero | `read_10x_mtx`, `read_h5ad` (dense `X`), `BioFrame.from_triplets` |
| `ParquetStore` | **on disk**: a directory of `part-*.parquet` files of `(cell_id, gene_id, count)` triplets, each file holding whole cells (~8.4 M non-zeros by default), plus `obs.parquet` and `var.parquet` | `read_10x_h5(path, backed=dir)` streams the .h5 into it; `bf.write_parquet(dir)` then `crest.read_parquet(dir)` |

All three expose the same `chunks()` iterator, so every analysis function works unchanged on
all of them. The Parquet store is what makes CREST *out-of-core*: only one file is in memory
at a time, so the **count matrix** never has to fit in RAM. What still grows with the number
of cells are the per-cell results: `obs`, the 50-PC embedding (200 B per cell), the kNN graph
and UMAP coordinates, and at millions of cells these dominate. Measured peak memory for the
core workflow: 1.0–1.1 GB at 68k–200k cells (0.2.0 benchmark); 0.73 GB out-of-core vs 0.92 GB
in memory on Kang (29k cells); 3.7 vs 4.4 GB at 1M cells; **30 vs 37 GB at 9.7M cells**, where
the neighbour graph, Leiden and UMAP set the peak. The
price is re-reading the files on every pass: up to ~2× slower than in memory at 100k–200k
cells, barely slower on small data (where out-of-core also saves nothing).

## Where results go

| step | writes |
|---|---|
| `pp.calculate_qc_metrics` | obs `n_genes_by_counts`, `total_counts`, `pct_counts_mt`; var `n_cells_by_counts`, `total_counts` |
| `pp.highly_variable_genes` | var `highly_variable`, `means`, `dispersions(_norm)`, cached `mean_log` / `std_log`; `uns["hvg"]` |
| `tl.pca` | `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]` (variance, variance_ratio, `projection` for `ingest`) |
| `pp.neighbors` | `uns["neighbors"]`: `indices`, `distances` (kNN), `rows`, `cols`, `weights` (fuzzy graph) |
| `tl.leiden` | obs `leiden` (strings, `"0"` = largest cluster); `uns["leiden"]` (parameters) |
| `tl.leiden_sweep` | obs `leiden_<res>` per resolution; returns the stability table |
| `tl.umap` | `obsm["X_umap"]` |
| `tl.rank_genes_groups` | returns a Polars table; `uns["rank_genes_groups"]` |
| `tl.score_genes` | obs `<score_name>` |
| `tl.harmony` | `obsm["X_pca_harmony"]`, `uns["harmony"]` |
| `pp.scrublet` | obs `doublet_score`, `predicted_doublet`, `doublet_z`; `uns["scrublet"]` |
| `tl.ingest` (on the query) | `obsm["X_pca"]`, `obsm["X_umap"]`, obs `<label>` + `<label>_confidence`; `uns["ingest"]` (neighbours used) |
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

* **Rust does the per-element work; Python orchestrates.** The loops over non-zeros and cells
  of the analysis steps are in Rust (`src/`), release the GIL and run in parallel with rayon.
  The Python code in `crest/` works on per-gene, per-group or per-cell vectors, never on a
  dense cells × genes array. Two exceptions run in Python: reading files (`h5py` decompressing
  HDF5, or Polars reading Parquet) and the loess fit of `seurat_v3` HVGs.
* **Few passes per step.** Each kernel fuses filter + transform + accumulate. Most steps read
  the counts once; PCA and `seurat_v3` read them twice; the Wilcoxon test reads them once per
  memory-bounded block of genes.
* **Scales with cores, within limits.** The kernels scale well, but reading the file, Leiden
  (a sequential algorithm) and the Python glue run on one core. In the 0.3.0 benchmark
  (pbmc68k, core workflow, Threadripper 3975WX) CREST took 45.6 s on 1 thread and 10.6 s on
  64: 4.3× faster, not 64×, which corresponds to a serial fraction of about 24%. See
  {doc}`benchmarks`.

## Reproducibility

Every stochastic step takes `random_state` (default 0). With the same seed:

* the kNN graph and Leiden clusters are identical for any number of threads (checked on Kang,
  1 vs 4 threads);
* UMAP coordinates are identical for the same number of threads, but differ across thread
  counts, because the parallel optimisation splits the work by thread;
* Harmony's k-means initialisation is random like harmonypy's, so compare Harmony outcomes
  (mixing, clusters), not coordinates, with harmonypy.
