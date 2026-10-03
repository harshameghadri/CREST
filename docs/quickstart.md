# Quickstart: PBMC 10k in five minutes

This walks through the standard workflow on 10x Genomics' public PBMC 10k dataset
(11,769 cells). If you know scanpy, every call will look familiar. The differences are
listed at the end.

## 1. Get the data

```bash
curl -LO https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5
```

## 2. Load and filter

```python
import crest

bf = crest.read_10x_h5("pbmc_10k_v3_filtered_feature_bc_matrix.h5")
print(bf)
```

```text
BioFrame 11769 cells × 33538 genes (CSRStore; raw counts)
    obs: ['barcode', 'cell_id']
    var: ['gene_ids', 'gene_name', 'feature_types', 'gene_id']
    obsm: []  uns: []
```

The object is a {class}`~crest.core.BioFrame`, CREST's counterpart of AnnData. `CSRStore`
means the counts are held in memory as a sparse matrix; `raw counts` means no transform has
been recorded yet. `cell_id` and `gene_id` are the row and column of each cell and gene in
that matrix (see {doc}`concepts`).

```python
crest.pp.calculate_qc_metrics(bf)     # adds obs n_genes_by_counts, total_counts, pct_counts_mt
bf = crest.pp.filter_cells(bf, min_genes=200, max_pct_mt=20)
bf = crest.pp.filter_genes(bf, min_cells=3)
```

Unlike scanpy, the filters **return** a new `BioFrame`, so assign the result. The new object
shares the count matrix with the old one (it is not copied); only the tables of kept cells
and genes are new.

### Why `polars`?

`bf.obs` (cells) and `bf.var` (genes) are [Polars](https://pola.rs) DataFrames, not pandas.
Polars is installed with CREST, and you only need to import it yourself to write a Polars
*expression*, for example a custom filter:

```python
import polars as pl

bf = bf.filter_cells(pl.col("total_counts") < 30_000)
```

A NumPy boolean mask does the same without Polars:

```python
bf = bf.filter_cells(bf.obs["total_counts"].to_numpy() < 30_000)
```

To work in pandas instead, convert a table with `bf.obs.to_pandas()` (needs `pyarrow`), or
hand the whole object over with `bf.to_anndata()`.

## 3. Normalise, select genes, reduce

```python
crest.pp.normalize_total(bf, target_sum=1e4)  # lazy: noted in bf.ops, nothing computed yet
crest.pp.log1p(bf)                            # lazy: noted in bf.ops, nothing computed yet
print(bf.ops)                                 # [('normalize_total', 10000.0), ('log1p',)]

crest.pp.highly_variable_genes(bf, n_top_genes=2000, flavor="seurat")
crest.pp.scale(bf, max_value=10)              # lazy: noted in bf.uns["scale"]; used by PCA only
crest.tl.pca(bf, n_comps=50)                  # obsm["X_pca"], varm["PCs"], uns["pca"]
```

**What "lazy" means here.** In scanpy, `sc.pp.normalize_total` and `sc.pp.log1p` rewrite
`adata.X` immediately. In CREST these two calls finish instantly: they only add an entry to
the list `bf.ops`, and the count matrix is left untouched. Every later step that needs
normalised values (`highly_variable_genes`, `pca`, `rank_genes_groups`, `score_genes`) reads the
raw counts chunk by chunk and applies the operations in `bf.ops` on the fly, inside its Rust
kernel, so it sees exactly the values scanpy would have in `adata.X`. Nothing normalised is
ever stored.

`scale` works the same way but only affects PCA, which applies the centring, scaling and
clipping exactly inside its own computation. Other steps keep using the log-normalised
values, as scanpy workflows do through `adata.raw`. See {doc}`concepts` for why this gives
identical results.

## 4. Graph, clusters, embedding

```python
crest.pp.neighbors(bf, n_neighbors=15)        # uns["neighbors"]: kNN + fuzzy connectivities
crest.tl.leiden(bf, resolution=1.0)           # obs["leiden"], "0" = largest cluster
crest.tl.umap(bf)                             # obsm["X_umap"]
```

Not sure which resolution to use? Run several in parallel on the same graph and look at how
stable each one is:

```python
table = crest.tl.leiden_sweep(bf, [0.2, 0.5, 1.0, 1.5, 2.0], n_seeds=3)
print(table)   # resolution, n_clusters, ari_prev, ari_seeds
```

## 5. Marker genes

```python
markers = crest.tl.rank_genes_groups(bf, "leiden", method="wilcoxon", n_genes=25)
print(markers.filter(pl.col("group") == "0").head(10))
```

The result is a long Polars DataFrame with columns `group`, `names`, `scores`,
`logfoldchanges`, `pvals` and `pvals_adj`. It is also stored in
`bf.uns["rank_genes_groups"]["table"]`.

## 6. Plot, or hand over to scanpy

CREST does not draw plots. Convert and use scanpy's plotting, or matplotlib directly:

```python
import matplotlib.pyplot as plt
xy = bf.obsm["X_umap"]
codes = bf.obs["leiden"].cast(pl.Int32).to_numpy()
plt.scatter(xy[:, 0], xy[:, 1], c=codes, s=1, cmap="tab20"); plt.gca().set_aspect("equal")

adata = bf.to_anndata()          # AnnData with raw counts in X, obs/var/obsm/uns copied
# import scanpy as sc; sc.pl.umap(adata, color="leiden")
```

## 7. Out of core: the same code for a dataset larger than RAM

```python
bf = crest.read_10x_h5("big.h5", backed="big_parquet/")    # converts to Parquet files once
# ... identical calls from step 2 on

bf = crest.read_parquet("big_parquet/")                    # reopen later without the .h5
```

With `backed=`, the counts stay on disk and are read one file at a time, so the count matrix
never has to fit in memory. Per-cell results (the table of cells, PCA, the neighbour graph,
UMAP) are still held in memory and grow with the number of cells. Each step re-reads the
files, which makes it up to about 2× slower than working in memory; on data that fits in RAM
comfortably, in-memory is the better choice.

## 8. Reading other formats

CREST reads 10x H5/MTX and `.h5ad` natively (no `anndata` needed). For anything else —
Loom, Visium, a Zarr-backed AnnData store, a CSV/Excel matrix, or any other format
`scanpy`/`anndata` can load — load it as an `AnnData` with the matching reader, then convert:

```python
import scanpy as sc
import crest

adata = sc.read_loom("sample.loom")       # or sc.read_visium(...), ad.read_zarr(...), ...
bf = crest.BioFrame.from_anndata(adata)   # carries over every obs/var column, and obsm
```

This loads the whole matrix into memory first (unlike `read_10x_h5(..., backed=...)`), so for
files too large for that, convert to `.h5ad` and use `crest.read_h5ad` instead.

## Differences from scanpy to know about

| scanpy | CREST |
|---|---|
| `adata = sc.read_10x_h5(...)` | `bf = crest.read_10x_h5(...)` (also `read_10x_mtx`, `read_h5ad`) |
| `sc.pp.filter_cells(adata, ...)` modifies in place | `bf = crest.pp.filter_cells(bf, ...)` **returns** the filtered object |
| `adata.obs` is pandas | `bf.obs` is **Polars** |
| `adata.raw = adata` before scaling | not needed: raw counts are never modified; DE uses the log-normalised values automatically |
| `sc.pp.scale` densifies | `crest.pp.scale` is recorded and applied exactly inside PCA |
| results in `adata.uns[...]` as recarrays | results returned as Polars DataFrames (and stored in `bf.uns`) |
| plotting in `sc.pl` | convert with `bf.to_anndata()` |
