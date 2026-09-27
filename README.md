# CREST — Columnar Rust Engine for Single-cell Transcriptomics

CREST runs the standard single-cell RNA-seq workflow (QC → normalisation →
highly variable genes → scaling + PCA → neighbours → Leiden → UMAP →
differential expression) with native Rust kernels behind a scanpy-style Python
API. It reproduces scanpy's results while being faster at every step and never
materialising a normalised, scaled or dense copy of the count matrix — datasets
can be streamed from disk (Parquet) with memory bounded by one chunk.

```python
import crest

bf = crest.read_10x_h5("filtered_feature_bc_matrix.h5")           # or backed="dir/" for out-of-core
bf = crest.pp.filter_cells(bf, min_genes=200)
bf = crest.pp.filter_genes(bf, min_cells=3)
crest.pp.normalize_total(bf, target_sum=1e4)                       # lazy
crest.pp.log1p(bf)                                                 # lazy
crest.pp.highly_variable_genes(bf, n_top_genes=2000, flavor="seurat")
crest.pp.scale(bf, max_value=10)                                   # applied exactly inside PCA
crest.tl.pca(bf, n_comps=50)
crest.pp.neighbors(bf, n_neighbors=15)
crest.tl.leiden(bf, resolution=1.0)
crest.tl.umap(bf)
markers = crest.tl.rank_genes_groups(bf, "leiden", method="wilcoxon")   # Polars DataFrame
adata = bf.to_anndata()                                            # hand over to scanpy / scverse
```

## Installation

```bash
pip install crest-sc            # import name: crest
pip install "crest-sc[anndata]" # AnnData / scipy interop
```

Building from source needs a Rust toolchain and [maturin](https://www.maturin.rs):
`maturin develop --release`.

## What is validated against scanpy

Every step is checked against scanpy 1.11 (`tests/test_crest.py`, and
`bench/whitepaper/`). On PBMC3k with the scanpy tutorial parameters:

| step | agreement with scanpy |
|---|---|
| `filter_cells`, `filter_genes`, QC metrics | identical |
| `highly_variable_genes` (seurat) | identical gene set; `dispersions_norm` within 2e-6 |
| `scale` + `pca` | identical subspace (0.000° principal angles), variance ratio within 5e-9 |
| `neighbors` connectivities | identical given the same kNN (within 1e-5) |
| `leiden` | same number of clusters, equal or higher modularity |
| `umap` | equal kNN preservation / silhouette (stochastic layout) |
| `rank_genes_groups` t-test, Wilcoxon | scores within 1e-4 (1e-7 on identical input), same p-values and log fold-changes |
| `score_genes` | identical, including scanpy's control-gene sampling (within 3e-7) |

## Performance

<!-- BENCHMARK -->

Run it yourself: `bench/whitepaper/run_pipeline.py` (one tool per process,
per-step wall time and peak RSS, JIT warm-up excluded for scanpy),
`bench/whitepaper/make_dataset.py` (scale a real 10x matrix to any size), and
`bench/whitepaper/summarize.py` (tables + figures).

## How it works

* **Lazy, fused, streamed.** Raw counts stay in one compact store — CSR in
  memory, a Polars triplet frame, or a directory of Parquet parts read one at a
  time. Cell/gene filters and `normalize_total`/`log1p` are recorded and applied
  per cell inside each kernel, so every step is a single pass with no
  intermediate copies.
* **Exact PCA without densifying.** One pass accumulates the gene × gene Gram
  matrix from sparse rows (threads own disjoint row blocks), an eigen-solver
  gives the loadings, a second pass projects cells. `sc.pp.scale(max_value)` is
  applied exactly: zero entries map to a per-gene constant, so the scaled matrix
  is a sparse matrix plus a rank-one term that centring removes.
* **kNN.** Exact search with tiled GEMM (the FAISS "flat" trick) for small data;
  above 20k cells an inverted-file index — cells bucketed by k-means, each bucket
  searched against its 20 nearest buckets, like cell lists in molecular dynamics —
  polished by NN-descent. ≥99.5% recall.
* **UMAP.** umap-learn's objective and defaults, optimised in parallel by domain
  decomposition: each thread owns a range of cells, writes only those, and reads
  the rest from a snapshot refreshed 16× per epoch.
* **Leiden.** Traag et al. (2019): fast local moving, refinement, aggregation.
* **Sparse Wilcoxon.** All implicit zeros of a gene form one tie block with a
  closed-form rank, so only non-zero values are sorted.
* **Polars expressions.** Column-level operations are also available as a
  `.bio` Polars expression namespace (`pl.col("count").bio.log1p()`, …).

## Data formats

`read_10x_h5` (Cell Ranger v2/v3; `backed=` streams to Parquet), `read_10x_mtx`,
`read_h5ad` (no anndata needed), `BioFrame.from_scipy / from_triplets /
from_anndata`, `write_parquet` / `read_parquet`, `to_anndata`, `to_scipy`.

## Scope and limitations

* HVG flavours: `seurat`, `cell_ranger` (not yet `seurat_v3`).
* No batch integration yet (Harmony is planned).
* `bio.nb_glm` fits a negative-binomial GLM with given dispersions; it is the
  GLM core of a DESeq2-style pseudobulk test, not the full DESeq2 procedure.
* UMAP layouts are deterministic for a fixed thread count, not across thread counts.

## License

MIT
