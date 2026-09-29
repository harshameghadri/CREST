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
import polars as pl

bf = crest.read_10x_h5("pbmc_10k_v3_filtered_feature_bc_matrix.h5")
print(bf)
# BioFrame 11769 cells × 33538 genes (CSRStore; raw counts)

crest.pp.calculate_qc_metrics(bf)                       # adds obs/var QC columns
bf = crest.pp.filter_cells(bf, min_genes=200, max_pct_mt=20)
bf = crest.pp.filter_genes(bf, min_cells=3)
```

The object is a {class}`~crest.core.BioFrame`. `bf.obs` (cells) and `bf.var` (genes) are
**Polars** DataFrames, so any Polars expression works for custom filters:

```python
bf = bf.filter_cells(pl.col("total_counts") < 30_000)
```

The filters return a new `BioFrame` that **shares** the raw counts. Nothing is copied: the
kept cells and genes are recorded as index maps.

## 3. Normalise, select genes, reduce

```python
crest.pp.normalize_total(bf, target_sum=1e4)  # recorded, not applied yet
crest.pp.log1p(bf)                            # recorded, not applied yet
crest.pp.highly_variable_genes(bf, n_top_genes=2000, flavor="seurat")
crest.pp.scale(bf, max_value=10)              # recorded; PCA applies it exactly
crest.tl.pca(bf, n_comps=50)                  # obsm["X_pca"], varm["PCs"], uns["pca"]
```

`normalize_total`, `log1p` and `scale` return immediately: they only append to `bf.ops` /
`bf.uns`. Each later step applies them to one chunk of raw counts at a time, inside the Rust
kernel. See {doc}`concepts`.

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
bf = crest.read_10x_h5("big.h5", backed="big_parquet/")    # streams to Parquet once
# ... identical calls from step 2 on; memory stays ~1 GB whatever the size

bf = crest.read_parquet("big_parquet/")                    # reopen later without the .h5
```

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
