# CREST

**Columnar Rust Engine for Single-cell Transcriptomics**

CREST is a high-performance single-cell RNA-seq analysis library. Every compute-intensive operation is implemented in native Rust and exposed to Python as a Polars expression plugin. No GPU required.

Designed to scale to **50M+ cells** on Apple Silicon hardware with 32-64GB unified memory.

---

## Why CREST

Standard scRNA-seq tools (scanpy, Seurat) hit fundamental bottlenecks at scale:

- **Dense matrix explosion**: PCA mean-centering a sparse matrix creates a dense copy (1M cells x 20k genes = 160GB).
- **Python GIL**: NumPy/SciPy release the GIL for BLAS, but orchestration, I/O, and custom kernels remain single-threaded.
- **Memory materialization**: AnnData loads entire datasets into RAM before any operation begins.

CREST avoids all three by operating directly on **sparse COO triplets** in **columnar Polars DataFrames**, with every hot path running as compiled Rust outside the GIL.

---

## Architecture

```
                         Python API
                    import crest
                    pl.col("count").bio.normalize_cpm(...)
                              |
                    +---------v-----------+
                    |  .bio namespace     |
                    |  (crest/__init__.py) |
                    |  register_plugin()  |
                    +---------+-----------+
                              |
                   Polars Plugin System (pyo3-polars)
                              |
          +-------------------+-------------------+
          |                   |                   |
    +-----v------+    +------v------+    +-------v-------+
    | preprocess |    | dim. reduc. |    | diff. expr.   |
    |            |    |             |    |               |
    | normalize  |    | SVD (PCA)   |    | rank_genes    |
    | log1p      |    | UMAP        |    | wilcoxon      |
    | scale      |    | neighbors   |    | DESeq2 IRLS   |
    | qc_metrics |    | connect.    |    |               |
    | filter     |    | louvain     |    |               |
    | score_genes|    |             |    |               |
    +-----+------+    +------+------+    +-------+-------+
          |                  |                   |
          +------------------+-------------------+
                             |
                  +----------v-----------+
                  |  Rust Core Libraries |
                  |                      |
                  |  Rayon (parallelism)  |
                  |  kiddo (KD-tree)     |
                  |  single-svdlib       |
                  |  nalgebra            |
                  |  graphrs (Louvain)   |
                  +----------+-----------+
                             |
                  +----------v-----------+
                  |  Arrow Columnar      |
                  |  Memory (zero-copy)  |
                  +----------------------+


    Data Flow (COO Triplet Format)

    +-------------+      +-------------+      +-------------+
    | 10x HDF5    | ---> | Parquet     | ---> | Polars       |
    | (sparse)    |      | (columnar)  |      | LazyFrame    |
    +-------------+      +-------------+      +-------------+
                                                    |
    +-----+-----+-----+          +------------------v----------+
    |cell | gene|count|          |  cell_id | gene_id | count  |
    |  0  | 142 | 3.0 |          |    0     |   142   |  3.0   |
    |  0  | 891 | 1.0 |   ===>  |    0     |   891   |  1.0   |
    |  1  | 142 | 7.0 |          |    1     |   142   |  7.0   |
    +-----+-----+-----+          +-----------------------------+
     Only non-zero              Parquet pushdown filtering
     entries stored             Predicate/projection pushdown
```

### Source Layout

```
crest/
├── src/                        Rust source (pyo3-polars plugin)
│   ├── lib.rs                  Plugin entry, log1p, wilcoxon
│   ├── preprocessing.rs        normalize_cpm, qc_metrics, filter, scale, score_genes
│   ├── svd.rs                  Sparse randomized SVD via single-svdlib
│   ├── umap_binding.rs         UMAP Polars plugin entry
│   ├── umap/                   Native UMAP implementation
│   │   ├── core.rs             Orchestrator (graph -> spectral -> SGD)
│   │   ├── graph.rs            Fuzzy simplicial set construction
│   │   ├── sgd.rs              Hogwild! SGD with atomic floats + Rayon
│   │   ├── spectral.rs         Spectral embedding via subspace iteration
│   │   └── utils.rs            Distance functions, a/b parameter fitting
│   ├── neighbors.rs            KNN graph + connectivities (KD-tree)
│   ├── leiden.rs               Louvain community detection
│   ├── rank_genes.rs           Welch t-test + Benjamini-Hochberg FDR
│   ├── deseq2.rs               Negative Binomial GLM (IRLS solver)
│   └── stats.rs                Reserved for future statistical functions
├── crest/                      Python package
│   ├── __init__.py             CrestExpr class (.bio namespace registration)
│   ├── core.py                 BioFrame dataclass (AnnData-like wrapper)
│   ├── io.py                   HDF5 to Parquet streaming converter
│   ├── pp.py                   HVG selection (Polars-native)
│   ├── tl.py                   Sparse masked PCA, incremental PCA
│   └── slaf_io.py              SLAF format I/O bridge
├── bench/                      Benchmarks and integration tests
├── docs/                       User guide
├── Cargo.toml                  Rust dependencies
└── pyproject.toml              Python build (maturin)
```

---

## Installation

### From PyPI (when published)

```bash
pip install crest
```

### From source

```bash
git clone https://github.com/harshameghadri/CREST.git
cd CREST
pip install maturin
maturin develop --release
```

Requires:
- Python >= 3.10
- Rust toolchain (stable)
- polars >= 1.0.0
- numpy >= 1.24.0

---

## Quick Start

```python
import polars as pl
import crest  # registers the .bio namespace

# Load sparse triplet data
df = pl.read_parquet("expression.parquet")

# Preprocessing pipeline
df = df.with_columns(
    pl.col("count")
      .bio.normalize_cpm(pl.col("cell_id"))
      .bio.log1p()
      .alias("logcpm")
)

# Dimensionality reduction (aggregate into lists first)
agg = df.group_by(pl.lit(1)).agg([
    pl.col("cell_id"), pl.col("gene_id"), pl.col("logcpm").alias("count")
])

agg = agg.with_columns(
    pl.col("cell_id").bio.svd(
        pl.col("gene_id"), pl.col("count"),
        n_cells=20000, n_genes=28000, n_comps=50
    ).alias("pca")
)

# UMAP + clustering
agg = agg.with_columns(
    pl.col("pca").bio.umap(n_components=2, n_neighbors=15).alias("umap")
)
agg = agg.with_columns(
    pl.col("pca").bio.louvain(n_neighbors=15).alias("clusters")
)
```

---

## API Reference

All functions are accessed via `pl.col("column").bio.<function>()`.

### Preprocessing

| Function | Description | Output Type |
|---|---|---|
| `normalize_cpm(cell_id_col)` | CP10k normalization per cell | Float32 |
| `log1p()` | Natural log transform ln(1+x) | Float32 |
| `scale(gene_id_col, n_obs, max_value=10.0)` | Zero-center, unit variance, clip | Float32 |
| `qc_total_counts(cell_id_col)` | Per-cell total counts | Float32 |
| `qc_n_genes(cell_id_col)` | Per-cell expressed gene count | UInt32 |
| `filter_cells(cell_id_col, min_genes=200, min_counts=0.0)` | QC boolean mask | Boolean |
| `score_genes(cell_id_col, gene_id_col, gene_set)` | Gene signature score | Float32 |

### Dimensionality Reduction

| Function | Description | Output Type |
|---|---|---|
| `svd(gene_id_col, count_col, n_cells, n_genes, n_comps=50)` | Sparse randomized SVD (PCA) | List(List(Float32)) |
| `umap(n_components=2, n_neighbors=15, min_dist=0.1, ...)` | UMAP embedding | List(List(Float32)) |

### Clustering and Neighbors

| Function | Description | Output Type |
|---|---|---|
| `louvain(n_neighbors=15)` | Louvain community detection | UInt32 |
| `neighbors(n_neighbors=15)` | KNN graph (indices + distances) | List(List(Float32)) |
| `connectivities(n_neighbors=15)` | Fuzzy simplicial set (COO triplets) | List(Float32) |

### Differential Expression and Statistics

| Function | Description | Output Type |
|---|---|---|
| `rank_genes_groups(cell_id, gene_id, group, target=1)` | Welch t-test + BH FDR | List(List(Float32)) |
| `wilcoxon(other_expr)` | Mann-Whitney U test | Float64 |
| `deseq2(size_factors, design, n_covs, dispersion)` | Negative Binomial GLM (IRLS) | List(Float32) |

For detailed usage examples, parameters, and output formats, see the [User Guide](docs/USER_GUIDE.md).

---

## Data Format

CREST operates on sparse COO triplets stored as three columns in a Polars DataFrame:

| Column | Type | Description |
|---|---|---|
| `cell_id` | UInt32 | Cell identifier (0-indexed) |
| `gene_id` | UInt32 | Gene identifier (0-indexed) |
| `count` | Float32 | Expression value |

Only non-zero entries are stored. This format is memory-efficient for scRNA-seq data, which is typically 90-95% sparse.

### Loading Data

```python
# From 10x Genomics HDF5 (streaming, constant memory)
from crest.io import convert_h5_to_parquet_stream
convert_h5_to_parquet_stream("raw_feature_bc_matrix.h5", "expression.parquet")

# From Parquet (lazy evaluation with predicate pushdown)
df = pl.scan_parquet("expression.parquet")

# From SLAF (Lance-backed columnar storage)
from crest.core import BioFrame
bf = BioFrame.from_slaf("dataset.slaf")
```

---

## Performance

Benchmarked on aarch64 Linux (Docker) with 10 million rows:

| Function | Time | Throughput |
|---|---|---|
| log1p | 0.10s | 100M rows/s |
| normalize_cpm | 0.12s | 83M rows/s |
| qc_total_counts | 0.10s | 100M rows/s |
| qc_n_genes | 0.10s | 100M rows/s |
| filter_cells | 0.09s | 111M rows/s |
| scale | 0.12s | 83M rows/s |

All preprocessing functions use single-pass or two-pass streaming algorithms that scale linearly with input size. Memory usage is O(n_cells) or O(n_genes), not O(n_cells x n_genes).

### Design decisions for scale

- **Sparse throughout**: Only non-zero entries are stored and processed. No dense materialization.
- **Float32**: Half the memory of Float64. Sufficient for integer-valued count data.
- **Streaming I/O**: HDF5-to-Parquet conversion processes fixed-size chunks. Peak memory is independent of dataset size.
- **Parquet pushdown**: LazyFrame operations push filters and projections down to the file level.
- **No densification in SVD**: The Rust SVD operates directly on COO/CSR sparse matrices.
- **Semi-join filtering**: BioFrame cell/gene filtering uses semi-joins to avoid column duplication.

---

## How the Plugin System Works

CREST uses the Polars expression plugin architecture. The integration path:

1. `import crest` registers the `.bio` namespace on `pl.Expr` via `@pl.api.register_expr_namespace`.
2. Each `.bio.xxx()` method returns a Polars expression wrapping `register_plugin_function()`.
3. When the expression is evaluated (`.collect()`, `.with_columns()`), Polars dynamically loads the compiled Rust shared library (`crest.abi3.so`).
4. The Rust function receives Arrow-backed `Series` objects with zero-copy access.
5. All Rust code runs outside the Python GIL with full Rayon thread-pool parallelism.

This means `.bio` calls compose naturally with any Polars expression, and lazy evaluation and query optimization still apply.

### Operation types

**Elementwise** (output length = input length): `log1p`, `normalize_cpm`, `qc_total_counts`, `qc_n_genes`, `filter_cells`, `scale`, `score_genes`. Use directly in `.with_columns()`.

**Aggregate** (output length differs from input): `svd`, `umap`, `louvain`, `neighbors`, `connectivities`, `rank_genes_groups`. Require data pre-aggregated into `List` columns via `group_by(...).agg(...)`.

---

## Known Limitations

- **UMAP KNN is O(n^2)**: The internal UMAP graph construction uses brute-force nearest neighbor search. Practical limit is approximately 50k cells. The `neighbors()` and `connectivities()` functions use a KD-tree and scale better. Replacing brute-force with KD-tree in the UMAP path is on the roadmap.

- **KD-tree dimension cap**: The KD-tree (kiddo) supports a maximum of 50 dimensions at compile time. PCA components beyond 50 are truncated with a warning. Standard scRNA-seq pipelines use 30-50 PCs, so this rarely matters in practice.

- **t-test p-value approximation**: `rank_genes_groups` uses a normal approximation for p-values when degrees of freedom exceed 30. For very small groups (fewer than 10 cells per group), consider using the Wilcoxon test instead.

---

## Testing

```bash
# Rust unit tests (12 tests)
cargo test --release

# Python integration tests + benchmarks (requires maturin develop --release first)
python bench/test_all.py
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Write tests for any new Rust functions (unit tests in `#[cfg(test)]` modules).
3. Run `cargo test --release` and `cargo check` (must pass with zero warnings).
4. Run `maturin develop --release && python bench/test_all.py` to verify integration.
5. Submit a pull request with a clear description of what changed and why.

---

## License

MIT. See [LICENSE](LICENSE) for details.
