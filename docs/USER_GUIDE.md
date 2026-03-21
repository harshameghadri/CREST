# CREST User Guide

**Columnar Rust Engine for Single-cell Transcriptomics**

CREST is a high-performance scRNA-seq analysis library. Every compute-intensive operation runs as native Rust, exposed to Python as Polars expression plugins via the `.bio` namespace. No GPU required.

---

## Installation

```bash
pip install crest
```

Or build from source:

```bash
git clone https://github.com/crest-bio/crest
cd crest
pip install maturin
maturin develop --release
```

## Quick Start

```python
import polars as pl
import crest  # registers the .bio namespace on all Polars expressions

# Load your data as COO triplets (cell_id, gene_id, count)
df = pl.read_parquet("expression.parquet")

# Normalize + log-transform
df = df.with_columns(
    pl.col("count")
      .bio.normalize_cpm(pl.col("cell_id"))
      .bio.log1p()
      .alias("logcpm")
)
```

---

## Data Format

CREST operates on **sparse COO triplets** stored in Polars DataFrames:

| cell_id (UInt32) | gene_id (UInt32) | count (Float32) |
|---|---|---|
| 0 | 142 | 3.0 |
| 0 | 891 | 1.0 |
| 1 | 142 | 7.0 |
| ... | ... | ... |

This format is memory-efficient (only non-zero entries are stored) and plays to Polars' strengths with columnar operations and Parquet pushdown.

### Loading Data

**From 10x HDF5:**
```python
from crest.io import convert_h5_to_parquet_stream

convert_h5_to_parquet_stream(
    "raw_feature_bc_matrix.h5",
    "expression.parquet",
    chunk_size=50_000  # cells per chunk (controls peak memory)
)
df = pl.scan_parquet("expression.parquet")
```

**From SLAF (Lance-backed):**
```python
from crest.slaf_io import read_slaf_expression

agg_df, n_cells, n_genes = read_slaf_expression("dataset.slaf")
```

**From AnnData (via BioFrame):**
```python
from crest.core import BioFrame

bf = BioFrame.from_slaf("dataset.slaf")
# bf.X  -> LazyFrame of expression triplets
# bf.obs -> DataFrame of cell metadata
# bf.var -> DataFrame of gene metadata
```

---

## API Reference

All functions are accessed via `pl.col("column").bio.<function>()`.

### Preprocessing

#### `normalize_cpm(cell_id_col, target_sum=10_000.0)`
Normalize counts per cell to a target sum (default: CP10k, matching scanpy's `normalize_total`).

```python
df = df.with_columns(
    pl.col("count").bio.normalize_cpm(pl.col("cell_id")).alias("norm_count")
)
```

**How it works:** Two-pass algorithm. Pass 1 accumulates per-cell total counts. Pass 2 normalizes each entry: `count / cell_total * target_sum`.

#### `log1p()`
Natural log transform: `ln(1 + x)`. Standard variance-stabilizing transform for count data.

```python
df = df.with_columns(pl.col("norm_count").bio.log1p().alias("logcpm"))
```

#### `scale(gene_id_col, n_obs, max_value=10.0)`
Zero-center and scale to unit variance per gene, with clipping. Equivalent to scanpy's `pp.scale`.

```python
df = df.with_columns(
    pl.col("logcpm").bio.scale(pl.col("gene_id"), n_obs=20000, max_value=10.0).alias("scaled")
)
```

**Important:** `n_obs` must be the total number of cells (including those with zero expression for a given gene). This is needed because sparse format doesn't store zeros, but variance calculation must account for them.

### Quality Control

#### `qc_total_counts(cell_id_col)`
Per-cell total counts (sum of all gene counts). Maps each row to its cell's total.

```python
df = df.with_columns(
    pl.col("count").bio.qc_total_counts(pl.col("cell_id")).alias("total_counts")
)
```

#### `qc_n_genes(cell_id_col)`
Per-cell number of expressed genes (genes with count > 0).

```python
df = df.with_columns(
    pl.col("count").bio.qc_n_genes(pl.col("cell_id")).alias("n_genes")
)
```

#### `filter_cells(cell_id_col, min_genes=200, min_counts=0.0)`
Returns a Boolean mask: `True` for cells passing the quality thresholds.

```python
df = df.with_columns(
    pl.col("count").bio.filter_cells(pl.col("cell_id"), min_genes=200, min_counts=500.0).alias("keep")
).filter(pl.col("keep"))
```

### Dimensionality Reduction

#### `svd(gene_id_col, count_col, n_cells, n_genes, n_comps=50)`
Sparse randomized SVD (PCA). Operates directly on COO triplets without densifying. This is the Rust-native equivalent of scanpy's `pp.pca`.

```python
# Aggregate into lists first (required for aggregate plugins)
agg_df = df.group_by(pl.lit(1)).agg([
    pl.col("cell_id"), pl.col("gene_id"), pl.col("count")
])

result = agg_df.with_columns(
    pl.col("cell_id").bio.svd(
        pl.col("gene_id"), pl.col("count"),
        n_cells=20000, n_genes=28000, n_comps=50
    ).alias("pca_coords")
)
```

**Output:** `List(List(Float32))` — each inner list is one cell's PC coordinates.

#### `umap(n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0, n_epochs=200, spectral_n_iter=50)`
UMAP dimensionality reduction. Chain after `.bio.svd()`.

```python
result = result.with_columns(
    pl.col("pca_coords").bio.umap(
        n_components=2, n_neighbors=15, min_dist=0.1
    ).alias("umap_coords")
)
```

**Output:** `List(List(Float32))` — each inner list is one cell's UMAP coordinates.

**Implementation:** Hogwild! SGD with atomic floats + Rayon for lock-free parallel optimization. Spectral embedding initialization via subspace iteration.

### Clustering

#### `louvain(n_neighbors=15)`
Louvain community detection from PCA coordinates. Chain after `.bio.svd()`.

```python
result = result.with_columns(
    pl.col("pca_coords").bio.louvain(n_neighbors=15).alias("cluster")
)
```

**Output:** `List(UInt32)` — cluster ID per cell.

### Neighbors

#### `neighbors(n_neighbors=15)`
Build a K-nearest neighbor graph from PCA coordinates. Returns per-cell neighbor indices and distances.

```python
result = result.with_columns(
    pl.col("pca_coords").bio.neighbors(n_neighbors=15).alias("knn")
)
```

**Output:** `List(List(Float32))` — per cell, a flattened `[idx0, dist0, idx1, dist1, ...]`.

#### `connectivities(n_neighbors=15)`
Build UMAP-style fuzzy simplicial set connectivities. Returns sparse COO triplets with Gaussian kernel weights.

```python
result = result.with_columns(
    pl.col("pca_coords").bio.connectivities(n_neighbors=15).alias("conn")
)
```

**Output:** `List(Float32)` — flat COO triplets `[cell_i, cell_j, weight, ...]`. Every 3 consecutive values form one `(row, col, weight)` triplet. Parse with stride=3.

### Differential Expression

#### `rank_genes_groups(cell_id_col, gene_id_col, group_col, target_group=1)`
Welch's t-test with Benjamini-Hochberg FDR correction. Equivalent to scanpy's `tl.rank_genes_groups(method='t-test')`.

```python
# group_col: UInt32 column with cluster labels per row
result = agg_df.with_columns(
    pl.col("logcpm").bio.rank_genes_groups(
        pl.col("cell_id"), pl.col("gene_id"), pl.col("cluster"),
        target_group=0
    ).alias("de_results")
)
```

**Output:** `List(List(Float32))` — each inner list: `[gene_id, t_stat, p_value, adj_p_value, log2_fc]`, sorted by adjusted p-value (most significant first).

#### `score_genes(cell_id_col, gene_id_col, gene_set)`
Gene set signature scoring (mean_set - mean_background). Equivalent to scanpy's `tl.score_genes`.

```python
marker_genes = [142, 891, 2003, 5500]  # gene IDs as UInt32
df = df.with_columns(
    pl.col("logcpm").bio.score_genes(
        pl.col("cell_id"), pl.col("gene_id"), marker_genes
    ).alias("signature_score")
)
```

### Statistical Tests

#### `wilcoxon(other_expr)`
Tie-corrected Mann-Whitney U test (Wilcoxon rank-sum). Returns two-sided p-values.

```python
result = df.select(
    pl.col("group1_values").bio.wilcoxon(pl.col("group2_values")).alias("pvalue")
)
```

#### `deseq2(size_factors, design_matrix, num_covariates, dispersion)`
Negative Binomial GLM via IRLS (Iteratively Reweighted Least Squares). For bulk/pseudo-bulk differential expression.

```python
result = df.select(
    pl.col("counts").bio.deseq2(
        pl.col("size_factors"), pl.col("design"), pl.col("n_covs"), pl.col("disp")
    ).alias("betas")
)
```

---

## Complete Pipeline Example

```python
import polars as pl
import crest
import numpy as np

# 1. Load data
df = pl.scan_parquet("pbmc_68k.parquet").collect()
n_cells = df["cell_id"].n_unique()
n_genes = df["gene_id"].n_unique()

# 2. QC filtering
df = df.with_columns(
    pl.col("count").bio.filter_cells(pl.col("cell_id"), min_genes=200, min_counts=500.0).alias("keep")
).filter(pl.col("keep")).drop("keep")

# 3. Normalize and log-transform
df = df.with_columns(
    pl.col("count")
      .bio.normalize_cpm(pl.col("cell_id"))
      .bio.log1p()
      .alias("logcpm")
)

# 4. Aggregate for dimensionality reduction
agg = df.group_by(pl.lit(1)).agg([
    pl.col("cell_id"), pl.col("gene_id"), pl.col("logcpm").alias("count")
])

# 5. PCA (sparse SVD)
agg = agg.with_columns(
    pl.col("cell_id").bio.svd(
        pl.col("gene_id"), pl.col("count"),
        n_cells=n_cells, n_genes=n_genes, n_comps=50
    ).alias("pca")
)

# 6. UMAP
agg = agg.with_columns(
    pl.col("pca").bio.umap(n_components=2, n_neighbors=15).alias("umap")
)

# 7. Clustering
agg = agg.with_columns(
    pl.col("pca").bio.louvain(n_neighbors=15).alias("clusters")
)

# 8. Extract results
umap_coords = np.array(agg["umap"][0].to_list())
cluster_ids = np.array(agg["clusters"][0].to_list())

print(f"UMAP: {umap_coords.shape}, Clusters: {len(np.unique(cluster_ids))}")
```

---

## The `.bio` Namespace — How It Works

CREST uses Polars' plugin system. When you `import crest`, it registers a custom expression namespace called `.bio` on every `pl.Expr`. Under the hood:

1. `import crest` triggers `__init__.py` which calls `@pl.api.register_expr_namespace("bio")`
2. Each `.bio.xxx()` call returns a Polars expression wrapping `register_plugin_function()`
3. When the expression is evaluated (`.collect()`, `.with_columns()`, etc.), Polars loads the compiled Rust shared library (`crest.abi3.so`) and calls the Rust function directly
4. **No Python GIL** — the Rust code runs outside the GIL, with full Rayon parallelism

This means:
- You can chain `.bio` calls with any Polars expression
- Lazy evaluation and query optimization still apply
- The Rust functions operate on Arrow arrays with zero-copy

### Elementwise vs Aggregate Operations

| Type | Functions | Pattern |
|---|---|---|
| **Elementwise** | `log1p`, `normalize_cpm`, `qc_*`, `filter_cells`, `scale`, `score_genes` | Operates row-by-row, can use in `.with_columns()` directly |
| **Aggregate** | `svd`, `umap`, `louvain`, `neighbors`, `connectivities`, `rank_genes_groups` | Consumes entire columns, requires data aggregated into `List` columns first |

For aggregate operations, you must first group your data into List columns:

```python
agg = df.group_by(pl.lit(1)).agg([
    pl.col("cell_id"), pl.col("gene_id"), pl.col("count")
])
# Now each column is a single List containing all values
```

---

## Performance Characteristics

Benchmarked on Docker/aarch64 with 10M rows:

| Function | Time | Throughput |
|---|---|---|
| `log1p` | 0.10s | ~100M rows/s |
| `normalize_cpm` | 0.12s | ~83M rows/s |
| `qc_total_counts` | 0.09s | ~111M rows/s |
| `qc_n_genes` | 0.09s | ~111M rows/s |
| `filter_cells` | 0.09s | ~111M rows/s |
| `scale` | 0.12s | ~83M rows/s |

All preprocessing functions use single-pass or two-pass streaming algorithms that scale linearly with input size. Memory usage is O(n_cells) or O(n_genes), not O(n_cells * n_genes).

### Scaling to 50M+ Cells

CREST is designed for Apple Silicon with 32-64GB unified memory:

- **Sparse format**: Only non-zero entries are stored (scRNA-seq is ~95% sparse)
- **Float32 everywhere**: Half the memory of Float64 (sufficient for count data)
- **Streaming I/O**: `convert_h5_to_parquet_stream` never loads the full matrix
- **Parquet pushdown**: LazyFrame operations push filters down to the file level
- **No densification**: SVD operates directly on sparse COO/CSR
- **BioFrame semi-joins**: Cell/gene filtering uses semi-joins (no column duplication)

---

## Python Helpers (Non-Rust)

These orchestrate Rust plugins or provide data loading:

### `crest.io`
- `convert_h5_to_parquet_stream(file_path, output_path, chunk_size=50000)` — Stream 10x HDF5 to Parquet

### `crest.pp`
- `highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")` — HVG selection via Polars group-by

### `crest.tl`
- `sparse_masked_pca(df, n_cells, n_genes, n_comps=50)` — SciPy-backed PCA with implicit centering
- `incremental_pca(df, n_cells, n_genes, n_comps=50, chunk_size=50000)` — Out-of-core PCA via DuckDB streaming

### `crest.core`
- `BioFrame(X, obs, var)` — AnnData-like wrapper with `filter_cells()` / `filter_genes()` semi-join methods

### `crest.slaf_io`
- `read_slaf_expression(slaf_path)` — Load SLAF dataset into SVD-ready format
- `run_svd_umap(slaf_path, ...)` — Full SVD + UMAP pipeline on SLAF data
- `run_svd_louvain(slaf_path, ...)` — Full SVD + Louvain pipeline
- `write_embeddings_to_slaf(slaf_path, embeddings)` — Write results back to SLAF

---

## Troubleshooting

**"attribute 'bio' not found"**
You forgot to import crest: `import crest`

**"undefined symbol: _polars_plugin_field_xxx"**
The Rust shared library is outdated. Rebuild: `maturin develop --release`

**"PCA has N dimensions but KD-tree supports max 50. Truncating."**
This is a warning, not an error. The KD-tree for neighbors/Louvain uses the first 50 PCA components. Use `n_comps=50` in SVD to avoid truncation.

**Large datasets running out of memory**
- Use `pl.scan_parquet()` (lazy) instead of `pl.read_parquet()` (eager)
- Use `convert_h5_to_parquet_stream()` with smaller `chunk_size`
- Use `BioFrame.filter_cells()` / `filter_genes()` early to reduce data
- Use `incremental_pca()` for >1M cells

---

## Known Limitations & Scaling Notes

### UMAP KNN is O(n²)
The internal UMAP graph construction (`graph.rs`) uses **brute-force KNN** — it computes pairwise distances between all points. This means `.bio.umap()` scales as O(n²) and is practical up to ~50k cells. For larger datasets:
- Run SVD/PCA first to reduce to 50 components
- Use `incremental_pca()` + an external UMAP library for >100k cells
- The `.bio.neighbors()` and `.bio.connectivities()` functions use a KD-tree and scale much better

**Roadmap:** Replace brute-force with KD-tree or HNSW in `graph.rs` to enable UMAP at 50M+ scale.

### KD-tree dimension limit
The KD-tree used in `neighbors`, `connectivities`, and `louvain` supports a maximum of **50 dimensions** (compile-time constant). PCA components beyond 50 are silently truncated with a warning. Since standard scRNA-seq pipelines use 30-50 PCs, this is rarely a problem.

### Hogwild! SGD (UMAP optimizer)
The UMAP SGD optimizer uses lock-free Hogwild! updates via atomic floats. Individual `load + add + store` operations are NOT atomic — concurrent writes can lose updates. This is intentional and mathematically justified for SGD convergence (see Niu et al., 2011). In practice, it produces correct embeddings with significant speedup from Rayon parallelism.

### t-test p-value approximation
`rank_genes_groups` uses a normal approximation for p-values when degrees of freedom > 30, and a corrected approximation for smaller df. This is accurate to ~1% for typical scRNA-seq group sizes (hundreds to thousands of cells). For very small groups (< 10 cells), consider using the Wilcoxon test instead.

### Float32 precision
All expression data uses Float32 throughout the pipeline. This is intentional — it halves memory usage compared to Float64 and is more than sufficient for count data (which is integer-valued). Statistical computations in `deseq2`, `rank_genes_groups`, and `scale` promote to Float64 internally for intermediate calculations.
