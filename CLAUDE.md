# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Bio-Polars Project

## Mission Statement
BUILD bio-polars: High-performance sparse bioinformatics wrapper for Polars. DELIVER incremental, tested, production-ready code.

## System Constraints (M1 Mac)

### Memory Management
```python
# CRITICAL: Monitor memory before operations
import psutil
import os

def check_memory():
    """Check available memory before operations."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    if available_gb < 2.0:
        raise MemoryError(f"Only {available_gb:.1f}GB available. Abort operation.")
    
    if available_gb < 4.0:
        print(f"⚠️ Low memory: {available_gb:.1f}GB. Using chunked processing.")
        return 'chunked'
    return 'normal'

# ALWAYS call before large operations
mode = check_memory()
```

### M1 Mac Limits
- **MAX_CELLS_IN_MEMORY**: 500,000 (for 8GB RAM) / 1,000,000 (for 16GB RAM)
- **CHUNK_SIZE**: 50,000 cells when streaming
- **MONITOR**: Memory every 100k cells processed
- **ABORT**: If available RAM < 2GB
- **SWAP PREVENTION**: Kill process if swap usage > 1GB

## Package Management (uv)

### Initial Setup
```bash
# Install uv (fastest modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project with uv
uv init bio-polars
cd bio-polars

# Create virtual environment with Python 3.11 (optimal for M1)
uv venv --python 3.11

# Activate environment
source .venv/bin/activate

# Install dependencies via uv (10x faster than pip)
uv pip install polars pyarrow scipy numpy pandas anndata h5py zarr
uv pip install pytest pytest-benchmark memory-profiler psutil
uv pip install --no-deps scanpy  # For benchmarking only
```

### Package Configuration (pyproject.toml)
```toml
[project]
name = "bio-polars"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "polars>=0.20.0",
    "pyarrow>=14.0.0",
    "scipy>=1.11.0",
    "numpy<2.0",  # Compatibility
    "pandas>=2.0.0",
    "anndata>=0.10.0",
    "h5py>=3.9.0",
    "zarr>=2.16.0",
    "psutil>=5.9.0",  # Memory monitoring
    "tqdm>=4.65.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-benchmark>=4.0.0",
    "memory-profiler>=0.61.0",
    "ruff>=0.1.0",
    "ipython>=8.10.0"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "ipykernel>=6.25.0",
]
```

### Environment Management Protocol
```bash
# ALWAYS use uv for package operations
uv pip install <package>  # Not pip install
uv pip sync              # Sync with pyproject.toml
uv pip compile pyproject.toml -o requirements.txt  # Lock deps

# For reproducibility
uv pip freeze > requirements.lock
```

## Operating Instructions

### Core Behaviors
- **MONITOR** memory before EVERY operation using psutil
- **WRITE** code immediately - explanations only when asked
- **TEST** every function before moving forward
- **COMMIT** atomically: `feat(module): action [TICKET-ID]`
- **TRACK** progress in `TASK_TRACKER.md` after each completion
- **FAIL FAST** - report blockers immediately, pivot quickly
- **BENCHMARK** against baseline continuously

### Task Tracking Protocol
MAINTAIN `TASK_TRACKER.md` with:
```markdown
# Task Tracker - Bio-Polars

## System Status
- Platform: macOS M1
- RAM Available: 8.2GB / 16GB
- Python: 3.11.7
- Package Manager: uv 0.4.0

## Completed Tasks
- [x] TICKET-001: Repository setup (2024-XX-XX HH:MM) - 5 min - Peak RAM: 0.2GB
- [x] TICKET-002: MTX reader (2024-XX-XX HH:MM) - 15 min - Peak RAM: 1.1GB

## In Progress
- [ ] TICKET-003: Filter cells operation

## Performance Metrics
- MTX read: 0.3s (vs scanpy: 0.8s) ✅ - RAM: 1.1GB
- Memory peak: 1.2GB (vs scanpy: 3.1GB) ✅
```

UPDATE after EVERY ticket completion including memory metrics.

## Project Architecture

### Directory Structure
```
bio-polars/
├── src/bio_polars/
│   ├── __init__.py
│   ├── core/
│   │   ├── dataset.py      # BioPolarsDataset class
│   │   └── sparse_ops.py   # Sparse operations
│   ├── io/
│   │   ├── mtx.py         # 10x MTX reader
│   │   ├── h5.py          # H5AD/HDF5 readers
│   │   └── converters.py  # Format conversions
│   └── benchmarks/
│       └── utils.py       # Timing utilities
├── tests/
│   ├── conftest.py
│   ├── test_io.py
│   └── test_core.py
├── bench/
│   ├── run_bench.py
│   └── datasets/
├── pyproject.toml
├── TASK_TRACKER.md
└── README.md
```

### Core Data Model
```python
@dataclass
class BioPolarsDataset:
    """Memory-aware sparse dataset. NEVER densify X unless explicit."""
    X: scipy.sparse.csr_matrix      # Shape: (n_cells, n_genes)
    obs: pl.DataFrame                # Cell metadata
    var: pl.DataFrame                # Gene metadata
    layers: dict[str, csr_matrix]   # Optional layers
    uns: dict                        # Unstructured metadata
    
    # Memory safety
    MAX_DENSE_SIZE = 1e8  # 100M elements max for dense ops
    
    def __post_init__(self):
        assert self.X.shape[0] == len(self.obs)
        assert self.X.shape[1] == len(self.var)
        self._check_memory_safety()
    
    def _check_memory_safety(self):
        """Prevent accidental OOM on M1 Mac."""
        if self.X.shape[0] * self.X.shape[1] > self.MAX_DENSE_SIZE:
            self._dense_blocked = True
            print(f"⚠️ Dense operations blocked: {self.X.shape}")
    
    @property
    def memory_usage(self) -> dict:
        """Report memory usage in GB."""
        x_mem = (self.X.data.nbytes + self.X.indices.nbytes + 
                self.X.indptr.nbytes) / 1e9
        obs_mem = self.obs.estimated_size('gb')
        var_mem = self.var.estimated_size('gb')
        return {'X': x_mem, 'obs': obs_mem, 'var': var_mem, 
                'total': x_mem + obs_mem + var_mem}
```

## Memory Monitoring Utilities

### Required Utils (src/bio_polars/utils/memory.py)
```python
import psutil
import os
import functools
import warnings

def memory_guard(max_gb: float = None):
    """Decorator to guard against OOM."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mem_before = psutil.Process().memory_info().rss / 1e9
            available = psutil.virtual_memory().available / 1e9
            
            if available < 2.0:
                raise MemoryError(f"Insufficient RAM: {available:.1f}GB")
            
            result = func(*args, **kwargs)
            
            mem_after = psutil.Process().memory_info().rss / 1e9
            delta = mem_after - mem_before
            
            if delta > 1.0:
                warnings.warn(f"Large memory increase: {delta:.1f}GB")
            
            return result
        return wrapper
    return decorator

def estimate_sparse_size(n_rows: int, n_cols: int, nnz: int) -> float:
    """Estimate CSR memory in GB."""
    # CSR: data (float32) + indices (int32) + indptr (int64)
    data_size = nnz * 4  # float32
    indices_size = nnz * 4  # int32
    indptr_size = (n_rows + 1) * 8  # int64
    return (data_size + indices_size + indptr_size) / 1e9

def get_memory_status() -> dict:
    """Get current memory status for M1 Mac."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process = psutil.Process()
    
    return {
        'available_gb': vm.available / 1e9,
        'percent_used': vm.percent,
        'swap_used_gb': swap.used / 1e9,
        'process_rss_gb': process.memory_info().rss / 1e9,
        'status': 'OK' if vm.available > 4e9 else 'LOW'
    }

# Auto-abort if swap > 1GB (M1 Mac performance cliff)
def check_swap_death():
    if psutil.swap_memory().used > 1e9:
        print("💀 Swap death detected! Aborting...")
        os._exit(1)
```

### Memory-Safe Operations Pattern
```python
@memory_guard(max_gb=4.0)
def filter_cells(self, mask: np.ndarray) -> BioPolarsDataset:
    """Filter with memory monitoring."""
    # Check if operation is safe
    estimated_size = self.X.nnz * mask.sum() / len(mask) * 12 / 1e9
    if estimated_size > psutil.virtual_memory().available / 1e9:
        # Fall back to chunked processing
        return self._filter_cells_chunked(mask)
    
    # Normal processing
    ...
```

## CORRECTED Implementation Roadmap (Post-Polars Analysis)

⚠️ **CRITICAL**: Previous roadmap was architecturally flawed. Following corrected approach based on Polars patterns.

### PHASE 0: Polars-Compatible Foundation [Week 1]

#### TICKET-001-v2: Polars-Native Repository Setup
```bash
ACTIONS:
1. curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv (correct)
2. uv init bio-polars && cd bio-polars (correct)
3. uv venv --python 3.11 && source .venv/bin/activate (correct)
4. CREATE pyproject.toml with CORRECTED deps (see below)
5. uv pip sync  # Install Arrow/Polars-focused dependencies
6. SETUP pre-commit hooks (correct)
7. CREATE TASK_TRACKER.md with system info (correct)
8. COMMIT: "feat: initialize Polars-compatible architecture [TICKET-001-v2]"

CORRECTED pyproject.toml:
[project]
name = "bio-polars"
version = "0.2.0"  # Breaking change - new architecture
dependencies = [
    "polars>=0.20.0",        # Core DataFrame engine
    "pyarrow>=14.0.0",       # Arrow memory management
    "numpy>=1.24.0",         # Minimal numpy for compatibility
    "h5py>=3.9.0",          # H5 I/O
    "zarr>=2.16.0",         # Zarr I/O
    "psutil>=5.9.0",        # System monitoring only
]

[project.optional-dependencies]
compat = ["scipy>=1.11.0", "anndata>=0.10.0"]  # Legacy compatibility only
dev = ["pytest>=7.4.0", "pytest-benchmark>=4.0.0", "memory-profiler>=0.61.0"]

DELIVERABLE: Polars-compatible repo with Arrow backend
```

#### TICKET-002-v2: Polars-Native Data Model
```python
IMPLEMENT in src/bio_polars/core/dataset.py:

import polars as pl
import pyarrow as pa
from contextlib import contextmanager

class BioDataset:
    """Polars-native bioinformatics dataset following Arrow principles."""

    def __init__(self, data: pl.DataFrame):
        """Initialize with Polars DataFrame.

        Required schema:
        - cell_id: UInt32
        - gene_counts: List[Struct[gene_id: UInt32, count: Float32]]
        - Additional cell metadata columns
        """
        self.data = data
        self._validate_schema()
        self.memory_pool = pa.default_memory_pool()

    @classmethod
    def from_scipy_sparse(cls, X, obs: pl.DataFrame, var: pl.DataFrame):
        """Migration bridge from legacy scipy.sparse format."""
        from scipy.sparse import csr_matrix

        # Convert CSR to Polars List[Struct] format
        sparse_data = []
        for cell_idx in range(X.shape[0]):
            row = X.getrow(cell_idx)
            gene_counts = [
                {"gene_id": int(gene_idx), "count": float(count)}
                for gene_idx, count in zip(row.indices, row.data)
                if count > 0
            ]
            sparse_data.append({
                "cell_id": cell_idx,
                "gene_counts": gene_counts
            })

        # Create base DataFrame
        df = pl.DataFrame(sparse_data)

        # Join with cell metadata using Polars native operations
        df = df.join(obs.with_row_index("cell_id"), on="cell_id", how="left")

        return cls(df)

    def filter_cells(self, predicate: pl.Expr) -> "BioDataset":
        """Filter cells using native Polars operations - FAST!"""
        filtered_data = self.data.filter(predicate)
        return BioDataset(filtered_data)

    def select_genes(self, gene_ids: list[int]) -> "BioDataset":
        """Select genes using Polars list operations."""
        filtered_data = self.data.with_columns([
            pl.col("gene_counts").list.eval(
                pl.element().filter(
                    pl.element().struct.field("gene_id").is_in(gene_ids)
                )
            )
        ])
        return BioDataset(filtered_data)

    def to_scipy_sparse(self):
        """Export to legacy format for compatibility."""
        # Convert back to (X, obs, var) for scanpy/anndata
        pass

    @property
    def n_cells(self) -> int:
        return len(self.data)

    @property
    def memory_usage_gb(self) -> float:
        return self.data.estimated_size('gb')

TEST in tests/test_core.py:
- Test Polars DataFrame operations
- Validate Arrow memory usage
- Test conversion roundtrips
```

#### TICKET-003-v2: Polars-Style I/O with Lazy Evaluation
```python
IMPLEMENT in src/bio_polars/io/mtx.py:

import polars as pl
import pyarrow as pa
from pathlib import Path

def scan_10x_mtx(path: str | Path) -> "LazyBioDataset":
    """Lazy scan following pl.scan_csv() pattern."""
    path = Path(path)

    # Parse metadata files immediately (small files)
    barcodes_lf = pl.scan_csv(
        path / "barcodes.tsv.gz",
        has_header=False,
        new_columns=["barcode"]
    )

    features_lf = pl.scan_csv(
        path / "features.tsv.gz",
        separator="\t",
        has_header=False,
        new_columns=["gene_id", "gene_name", "feature_type"]
    )

    # Return lazy dataset (matrix not loaded yet)
    return LazyBioDataset(
        matrix_path=path / "matrix.mtx.gz",
        barcodes_lf=barcodes_lf,
        features_lf=features_lf
    )

class LazyBioDataset:
    """Lazy evaluation like pl.LazyFrame - build query plan without execution."""

    def __init__(self, matrix_path: Path, barcodes_lf: pl.LazyFrame, features_lf: pl.LazyFrame):
        self.matrix_path = matrix_path
        self.barcodes_lf = barcodes_lf
        self.features_lf = features_lf
        self._filters = []
        self._gene_selection = None

    def filter_cells(self, predicate: pl.Expr) -> "LazyBioDataset":
        """Add cell filter to query plan - no execution."""
        new_dataset = LazyBioDataset(self.matrix_path, self.barcodes_lf, self.features_lf)
        new_dataset._filters = self._filters + [predicate]
        return new_dataset

    def select_genes(self, gene_ids: list[int]) -> "LazyBioDataset":
        """Add gene selection to query plan."""
        new_dataset = LazyBioDataset(self.matrix_path, self.barcodes_lf, self.features_lf)
        new_dataset._filters = self._filters.copy()
        new_dataset._gene_selection = gene_ids
        return new_dataset

    def collect(self, streaming: bool = True) -> "BioDataset":
        """Execute query plan with optional streaming."""
        if streaming:
            return self._collect_streaming()
        else:
            return self._collect_eager()

    def _collect_streaming(self) -> "BioDataset":
        """Stream large MTX file in chunks following Polars pattern."""
        # 1. Parse matrix headers to get dimensions
        # 2. Apply gene selection to reduce I/O
        # 3. Stream matrix data in row chunks (50k cells)
        # 4. Convert to Polars List[Struct] format per chunk
        # 5. Apply cell filters using Polars operations
        # 6. Combine chunks with metadata joins
        pass

def read_10x_mtx(path: str | Path) -> "BioDataset":
    """Eager read - convenience function."""
    return scan_10x_mtx(path).collect(streaming=True)

TEST in tests/test_io.py:
- Test lazy evaluation (no file I/O until .collect())
- Test streaming with large synthetic datasets
- Validate query optimization works
```

#### TICKET-004-v2: Legacy Compatibility Bridge
```python
IMPLEMENT in src/bio_polars/compat/anndata.py:

from typing import Optional
import anndata
from scipy.sparse import csr_matrix

def to_anndata(bio_dataset: "BioDataset") -> anndata.AnnData:
    """Convert to AnnData for scanpy compatibility."""
    X, obs, var = bio_dataset.to_scipy_sparse()
    return anndata.AnnData(X=X, obs=obs.to_pandas(), var=var.to_pandas())

def from_anndata(adata: anndata.AnnData) -> "BioDataset":
    """Convert from AnnData to BioDataset."""
    obs = pl.from_pandas(adata.obs.reset_index())
    var = pl.from_pandas(adata.var.reset_index())
    return BioDataset.from_scipy_sparse(adata.X, obs, var)

def write_h5ad(bio_dataset: "BioDataset", path: str, **kwargs):
    """Write H5AD via AnnData conversion."""
    adata = to_anndata(bio_dataset)
    adata.write_h5ad(path, **kwargs)

def read_h5ad(path: str) -> "BioDataset":
    """Read H5AD via AnnData."""
    adata = anndata.read_h5ad(path)
    return from_anndata(adata)

TEST: Full roundtrip compatibility with scanpy workflows
```

#### TICKET-005-v2: Performance Validation Framework
```python
CREATE bench/polars_comparison.py:

class BenchmarkRunner:
    def time_operation(self, func, *args, **kwargs):
        """Measure time and memory."""
        
    def compare_to_baseline(self, bio_polars_func, scanpy_func):
        """Generate comparison report."""

MEASURE:
- Read time
- Filter time  
- Memory peak
- Output to JSON
```

### PHASE 1: Core Operations [Week 2-3]

#### TICKET-006: Normalization Methods
```python
def normalize_counts(self, method='cpm', target_sum=1e4):
    """Per-cell normalization preserving sparsity."""
    if method == 'cpm':
        # Counts per million
        # Work on .data array only
    elif method == 'log1p':
        # Natural log(1+x)
        # Element-wise on .data
```

#### TICKET-007: Aggregation Operations
```python
def aggregate_by_obs(self, by: str, func='sum') -> BioPolarsDataset:
    """Group cells and aggregate counts."""
    # Build group mapping
    # Create new CSR for groups
    # Aggregate obs metadata
```

#### TICKET-008: Quality Control Metrics
```python
def calculate_qc_metrics(self) -> pl.DataFrame:
    """Compute standard QC metrics."""
    # n_genes_per_cell (row nnz)
    # n_counts_per_cell (row sum)
    # pct_mitochondrial
    # Return as pl.DataFrame
```

#### TICKET-009: H5/HDF5 Reader
```python
def read_10x_h5(path: str) -> BioPolarsDataset:
    """Read Cell Ranger H5 format."""
    # Use h5py
    # Direct CSR construction from /matrix group
    # Handle multiple genomes
```

#### TICKET-010: Lazy/Streaming Interface
```python
class LazyBioPolarsDataset:
    """Lazy evaluation for large datasets."""
    
    def scan_10x_mtx(path: str) -> LazyBioPolarsDataset:
        """Scan without loading."""
        
    def collect(self, streaming=True) -> BioPolarsDataset:
        """Materialize with optional streaming."""
```

### PHASE 2: Advanced Features [Week 4-5]

#### TICKET-011: Chunked Processing
```python
def process_chunked(self, func, chunk_size=50000):
    """Process dataset in chunks for memory efficiency."""
    # Split CSR by rows
    # Apply func per chunk
    # Combine results
```

#### TICKET-012: Parallel Operations
```python
def parallel_apply(self, func, n_jobs=-1):
    """Parallel execution for row/col operations."""
    # Use joblib/multiprocessing
    # Thread-safe CSR access
```

#### TICKET-013: Arrow Sparse Integration
```python
def to_arrow_sparse(self) -> pyarrow.SparseTensor:
    """Convert to Arrow sparse tensor."""
    
def from_arrow_sparse(tensor: pyarrow.SparseTensor) -> BioPolarsDataset:
    """Create from Arrow sparse tensor."""
```

#### TICKET-014: Zarr Support
```python
def read_zarr(path: str) -> BioPolarsDataset:
    """Read Zarr chunked arrays."""
    
def write_zarr(self, path: str, chunks=(1000, 1000)):
    """Write to Zarr format."""
```

#### TICKET-015: CLI Interface
```bash
CREATE bio_polars/cli.py:

Commands:
- bio-polars inspect <path>
- bio-polars filter --min-genes 200 --max-mito 0.2
- bio-polars benchmark <dataset>
- bio-polars convert <input> <output>
```

### PHASE 3: Optimization [Week 6-8]

#### TICKET-016: Rust Sparse Kernels
```rust
// src/rust/sparse_ops.rs
fn filter_csr_rows(data: &[f32], indices: &[i32], 
                   indptr: &[i64], mask: &[bool]) -> CsrMatrix
```

#### TICKET-017: GPU Support (Optional)
```python
def to_cupy_sparse(self):
    """Convert to CuPy sparse for GPU ops."""
```

#### TICKET-018: Memory Profiling
```python
def profile_memory(self, operation: str):
    """Profile memory usage per operation."""
```

#### TICKET-019: Performance Tuning
- Profile hotspots with cProfile
- Optimize inner loops
- Add numba JIT where beneficial

#### TICKET-020: Documentation Site
- Setup MkDocs
- API reference
- Tutorials
- Benchmarks

## Success Metrics

### Performance Targets
```yaml
MTX Read (100k cells):
  bio-polars: < 0.5s
  scanpy: ~ 1.2s
  speedup: > 2x ✓

Filter Cells (100k -> 50k):
  bio-polars: < 0.1s
  scanpy: ~ 0.3s
  speedup: > 3x ✓

Memory Peak (1M cells):
  bio-polars: < 2GB
  scanpy: ~ 5GB
  reduction: > 60% ✓

Correctness:
  numerical_tolerance: 1e-6
  metadata_integrity: 100%
```

### Validation Datasets
1. **PBMC 3k** (small): Functional correctness
2. **PBMC 68k** (medium): Performance baseline  
3. **Mouse Brain 1.3M** (large): Scalability test

## Testing Strategy

### Unit Test Pattern
```python
def test_filter_preserves_sparsity():
    # GIVEN sparse matrix
    X = scipy.sparse.random(100, 200, density=0.1, format='csr')
    dataset = BioPolarsDataset(X, mock_obs, mock_var)
    
    # WHEN filtering
    mask = np.random.choice([True, False], 100)
    filtered = dataset.filter_cells(mask)
    
    # THEN sparsity preserved
    assert scipy.sparse.issparse(filtered.X)
    assert filtered.X.nnz / filtered.X.size < 0.2
```

### Integration Test Pattern
```python
def test_end_to_end_pipeline():
    # Full pipeline on real dataset
    dataset = read_10x_mtx("data/pbmc3k")
    dataset = dataset.filter_cells(dataset.obs['n_genes'] > 200)
    dataset = dataset.filter_genes(min_cells=3)
    dataset.normalize_counts(method='cpm')
    dataset.write_h5ad("output.h5ad")
    
    # Validate against reference
    reference = anndata.read_h5ad("reference.h5ad")
    np.testing.assert_allclose(dataset.X.data, reference.X.data, rtol=1e-6)
```

## Development Workflow (uv-based)

### Daily Development Pattern
```bash
# Morning setup
cd bio-polars
source .venv/bin/activate
uv pip sync  # Ensure deps are current
git pull origin develop

# Before implementing new feature
uv pip install ipython  # For interactive testing
python -c "from bio_polars.utils.memory import get_memory_status; print(get_memory_status())"

# After implementing feature
uv pip install pytest-xdist  # For parallel tests
pytest tests/ -n auto --maxfail=1  # Stop on first failure

# Adding new dependency
uv pip install new-package
uv pip freeze > requirements.lock  # Lock state

# Before committing
ruff check src/  # Linting
ruff format src/  # Formatting
```

### Rapid Iteration Commands
```bash
# Quick test single module
python -m pytest tests/test_io.py::test_read_10x_mtx -xvs

# Memory profiling
mprof run python bench/profile_operation.py
mprof plot  # Visualize memory usage

# Interactive development
ipython
> %load_ext autoreload
> %autoreload 2
> from bio_polars import *
> ds = read_10x_mtx("data/pbmc3k")
> ds.memory_usage
```

### Package Building
```bash
# Build wheel with uv
uv pip install build
python -m build

# Test in fresh environment
uv venv test-env --python 3.11
uv pip install dist/bio_polars-*.whl
python -c "import bio_polars; print(bio_polars.__version__)"
```

## Git Workflow

### Branch Strategy
```bash
main           # Stable releases
├── develop    # Integration branch
├── feat/*     # Feature branches
├── fix/*      # Bug fixes
└── perf/*     # Performance improvements
```

### Commit Convention
```
feat(io): add Zarr reader [TICKET-014]
fix(core): prevent densification in filter [TICKET-003]
perf(ops): optimize CSR row filtering [TICKET-016]
docs(api): update BioPolarsDataset docstring
test(io): add MTX edge cases
chore(deps): update polars to 0.20.0
```

## Continuous Integration

### CI Pipeline (.github/workflows/ci.yml)
```yaml
on: [push, pull_request]

jobs:
  test:
    matrix:
      python: [3.10, 3.11, 3.12]
      os: [ubuntu-latest, macos-latest]
    steps:
      - Test suite
      - Coverage report
      - Memory profiling
      - Benchmark vs baseline

  benchmark:
    if: github.ref == 'refs/heads/main'
    steps:
      - Run full benchmark suite
      - Compare to previous results
      - Fail if regression > 10%
```

## Error Handling Protocol

### On Test Failure
1. CAPTURE full traceback
2. IDENTIFY root cause
3. FIX in minimal diff
4. ADD regression test
5. UPDATE TASK_TRACKER.md

### On Performance Regression
1. PROFILE with cProfile/memory_profiler
2. IDENTIFY hotspot
3. OPTIMIZE algorithm/implementation
4. DOCUMENT tradeoffs

### On Blocked Task
1. LOG blocker in TASK_TRACKER.md
2. IMPLEMENT workaround if possible
3. CREATE follow-up ticket
4. PROCEED to next task

## Resource Optimization

### Memory Management
- NEVER call `.toarray()` on large matrices
- USE generators for large iterations
- IMPLEMENT chunked processing for > 500k cells
- CLEAR intermediate results explicitly

### Computation Optimization
- PREFER CSR for row operations
- USE CSC views for column operations  
- IMPLEMENT parallel ops for independent computations
- CACHE expensive computations (gene stats)

## M1 Mac Troubleshooting

### Common Memory Issues & Solutions
```python
# ISSUE: Kernel died during operation
# SOLUTION: Add memory limit
import resource
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, -1))  # 8GB limit

# ISSUE: Slow operations when RAM < 4GB
# SOLUTION: Use chunked processing
if psutil.virtual_memory().available < 4e9:
    return process_chunked(data, chunk_size=10000)

# ISSUE: Memory leak in loops
# SOLUTION: Explicit garbage collection
import gc
for chunk in chunks:
    process(chunk)
    del chunk
    gc.collect()  # Force cleanup

# ISSUE: Swap usage degrades performance
# SOLUTION: Monitor and abort
@contextmanager
def no_swap():
    swap_before = psutil.swap_memory().used
    yield
    swap_after = psutil.swap_memory().used
    if swap_after - swap_before > 500e6:  # 500MB
        raise MemoryError("Excessive swap usage")
```

### M1-Optimized Settings
```python
# Use native ARM64 libraries
os.environ['OPENBLAS_NUM_THREADS'] = '8'  # M1 has 8 cores
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'

# Optimal chunk sizes for M1
CHUNK_SIZES = {
    '8GB': 50_000,   # M1 base model
    '16GB': 100_000,  # M1 Pro
    '32GB': 200_000,  # M1 Max
}
```

## Quick Reference

### Common Operations
```python
# Read data
ds = read_10x_mtx("path/to/10x")

# Quality control
ds = ds.filter_cells(ds.obs['n_genes'] > 200)
ds = ds.filter_cells(ds.obs['pct_mito'] < 0.2)
ds = ds.filter_genes(min_cells=3)

# Normalize
ds.normalize_counts(method='cpm')
ds.log1p()  # In-place

# Aggregate by cluster
clustered = ds.aggregate_by_obs('leiden_cluster')

# Export
ds.write_h5ad("processed.h5ad")
ds.to_anndata()  # For Scanpy
```

## POLARS ARCHITECTURE ANALYSIS & STRATEGIC INSIGHTS

### Critical Discovery: Current Architecture is Fundamentally Flawed

**DEEP REPOSITORY ANALYSIS COMPLETED** - Comprehensive study of Polars repository (20+ Rust crates, PyO3 bindings) reveals bio-polars requires architectural redesign for production success.

### Core Polars Design Patterns (Production-Grade Reference)

```rust
// POLARS CORE ARCHITECTURE - What we must emulate
pub struct DataFrame {
    height: usize,
    columns: Vec<Column>,           // Apache Arrow columnar format
    cached_schema: OnceLock<SchemaRef>, // Lazy schema computation
}

pub struct ChunkedArray<T: PolarsDataType> {
    field: Arc<Field>,
    chunks: Vec<ArrayRef>,          // Arrow arrays for SIMD
    flags: StatisticsFlagsIM,       // Cached min/max/null_count
    length: usize,
    null_count: usize,
}

// PyO3 BINDING PATTERN - Thread-safe Python wrapper
#[pyclass(frozen)]
pub struct PyDataFrame {
    df: RwLock<DataFrame>,          // Concurrent reads, exclusive writes
}
```

### Design System Thinking: Bio-Polars Must Follow Polars Patterns

**ARCHITECTURAL PRINCIPLE**: Bio-polars is NOT a custom implementation - it's a specialized **extension** of Polars for bioinformatics. We must leverage Polars' proven architecture, not reinvent it.

### Critical Flaws in Current Implementation

| Component | Current Bio-Polars | Polars Standard | Impact | Fix Strategy |
|-----------|-------------------|-----------------|---------|--------------|
| **Data Model** | `scipy.sparse.csr_matrix` | Apache Arrow columnar | 🔴 5-10x slower, no SIMD | Redesign around `pl.DataFrame` |
| **Memory Layout** | Row-based sparse | Column-based compressed | 🔴 Poor cache locality | Use Arrow memory pools |
| **Streaming** | Load all into RAM | Lazy evaluation + chunking | 🔴 OOM on large datasets | Implement `LazyBioDataset` |
| **Parallelism** | Single-threaded | Rayon work-stealing | 🔴 Doesn't use multiple cores | Use Polars parallel engine |
| **Type System** | Loose Python types | Strict Arrow schema | 🟡 Runtime errors | Adopt Arrow type system |

### Strategic Commands for Future Development

**BEFORE implementing ANY ticket, run these analysis commands:**

```bash
# 1. ANALYZE existing Polars patterns for the feature
rg "impl.*filter" /path/to/polars/crates/polars-core/src --type rust -A 10
rg "scan_.*" /path/to/polars/crates/polars-io/src --type rust -A 5

# 2. STUDY Polars I/O implementations
find /path/to/polars/crates/polars-io/src -name "*.rs" | xargs grep -l "streaming"
read /path/to/polars/crates/polars-io/src/csv/read_impl.rs  # Study chunked reading

# 3. EXAMINE PyO3 binding patterns
rg "PyDataFrame" /path/to/polars/crates/polars-python/src --type rust -A 5
```

### Design System: Data Format Strategy

**PHASE 1: Compatibility Bridge (Immediate)**
```python
# DON'T rebuild from scratch - create compatibility layer
class BioDataset:
    def __init__(self, data: pl.DataFrame = None, X=None, obs=None, var=None):
        if data is not None:
            # NEW: Arrow-based path (future)
            self._data = data
            self._mode = "arrow"
        else:
            # LEGACY: scipy.sparse path (current compatibility)
            self._X, self._obs, self._var = X, obs, var
            self._mode = "legacy"

    @property
    def X(self):
        if self._mode == "arrow":
            return self._data.to_sparse_matrix()  # Convert on-demand
        return self._X
```

**PHASE 2: Native Polars Integration**
```python
# STRATEGIC: Store sparse data as Polars List[Struct] columns
class BioDatasetV2:
    def __init__(self, data: pl.DataFrame):
        # Schema: [cell_id: UInt32, gene_data: List[Struct[gene_id: UInt32, count: Float32]]]
        self.data = data

    @classmethod
    def from_csr(cls, csr_matrix, obs, var):
        # Convert CSR to columnar format using Polars efficiency
        sparse_data = []
        for row_idx in range(csr_matrix.shape[0]):
            row = csr_matrix.getrow(row_idx)
            gene_counts = [
                {"gene_id": col_idx, "count": value}
                for col_idx, value in zip(row.indices, row.data)
            ]
            sparse_data.append({"cell_id": row_idx, "gene_data": gene_counts})

        df = pl.DataFrame(sparse_data)
        # Merge with obs metadata using Polars joins (fast!)
        return cls(df.join(obs, left_on="cell_id", right_on="barcode"))
```

### Streaming Implementation Strategy

**DON'T invent custom streaming - use Polars patterns:**

```python
# STUDY these Polars functions and replicate the pattern:
# - pl.scan_csv() -> bio.scan_10x_mtx()
# - pl.scan_parquet() -> bio.scan_10x_h5()
# - LazyFrame.collect(streaming=True) -> LazyBioDataset.collect()

def scan_10x_mtx(path: str) -> LazyBioDataset:
    """Lazy scan following pl.scan_csv pattern."""
    # 1. Parse headers without loading data
    # 2. Return lazy representation
    # 3. Enable query optimization before materialization
    return LazyBioDataset(scan_plan=MTXScanPlan(path))

class LazyBioDataset:
    def filter_cells(self, predicate: pl.Expr) -> LazyBioDataset:
        # Build query plan like Polars LazyFrame
        return LazyBioDataset(self.plan.filter(predicate))

    def collect(self, streaming: bool = True) -> BioDataset:
        # Execute with streaming engine for memory efficiency
        if streaming:
            return self.plan.collect_streaming()
        return self.plan.collect()
```

### Memory Management: Follow Polars Memory Pool Pattern

```python
# DON'T use psutil monitoring - use Arrow memory pools like Polars
import pyarrow as pa

class BioMemoryManager:
    def __init__(self):
        # Use Arrow memory pool like Polars does
        self.pool = pa.default_memory_pool()
        self.initial_bytes = self.pool.bytes_allocated()

    @contextmanager
    def track_allocation(self, operation_name: str):
        before = self.pool.bytes_allocated()
        yield
        after = self.pool.bytes_allocated()
        delta_gb = (after - before) / 1e9
        if delta_gb > 1.0:
            warnings.warn(f"{operation_name} allocated {delta_gb:.1f}GB")
```

### Performance Optimization Commands

**ALWAYS benchmark against Polars equivalents:**

```bash
# Measure Polars baseline performance
python -c "
import polars as pl
import time
df = pl.read_csv('large_dataset.csv')
start = time.time()
filtered = df.filter(pl.col('value') > 100)
print(f'Polars filter: {time.time() - start:.3f}s')
"

# Compare bio-polars implementation
python -c "
import bio_polars as bp
start = time.time()
dataset = bp.read_10x_mtx('data/')
filtered = dataset.filter_cells(dataset.obs['n_genes'] > 200)
print(f'Bio-polars filter: {time.time() - start:.3f}s')
"
```

### Production Readiness Commands

**Execute these commands before claiming production readiness:**

```bash
# 1. STREAMING TEST: Must handle 10M+ cells without OOM
python -c "
dataset = bio_polars.scan_10x_h5('huge_dataset.h5')
filtered = dataset.filter_cells(pl.col('n_genes') > 200)
result = filtered.collect(streaming=True)  # Must not OOM
"

# 2. PARALLEL TEST: Must use all CPU cores
python -c "
import time, psutil
start = time.time()
dataset.normalize_counts(method='cpm')  # Should max out CPU
duration = time.time() - start
cpu_percent = psutil.cpu_percent()
print(f'Used {cpu_percent}% CPU for {duration:.1f}s')
"

# 3. MEMORY EFFICIENCY: Must use <50% of Scanpy memory
mprof run python benchmark_memory.py
# Compare peak memory usage
```

### Strategic Development Approach

**NEVER implement from scratch - always follow this pattern:**

1. **STUDY** equivalent Polars implementation first
2. **ADAPT** Polars pattern to bioinformatics data
3. **BENCHMARK** against Polars baseline performance
4. **ITERATE** until matching Polars efficiency

**Example Development Sequence:**
```bash
# Step 1: Study Polars
rg "filter" /path/to/polars/crates/polars-core/src/frame/mod.rs -A 10

# Step 2: Understand the pattern
# Polars filter creates new DataFrame with filtered columns
# Uses Arrow compute kernels for vectorized operations
# Preserves schema and memory layout

# Step 3: Adapt for bio-polars
# Don't reinvent filtering - use Polars DataFrame.filter()
# Convert sparse data to Polars format, filter, convert back

# Step 4: Benchmark
python bench/compare_filter_performance.py
```

### Critical Production Blockers to Address

1. **🔴 IMMEDIATE: Replace scipy.sparse with Arrow format**
   - Command: `bio.to_arrow()` → `pl.DataFrame.from_arrow()`
   - Timeline: Week 1-2 (breaking change)

2. **🔴 URGENT: Implement streaming for >1M cells**
   - Command: `bio.scan_10x_h5()` → `LazyBioDataset.collect(streaming=True)`
   - Timeline: Week 2-3

3. **🔴 CRITICAL: Add parallel execution**
   - Command: Use `pl.DataFrame.with_columns()` parallel engine
   - Timeline: Week 3-4

### Success Validation Commands

**Project is production-ready when these commands pass:**

```bash
# Scalability test
python -c "bio_polars.scan_10x_h5('10M_cells.h5').collect(streaming=True)"

# Performance test
python -c "assert bio_polars.read_time < polars.read_csv_time * 1.5"

# Memory test
python -c "assert bio_polars.peak_memory < scanpy.peak_memory * 0.5"

# Compatibility test
python -c "bio_polars.to_anndata().write_h5ad('test.h5ad')"
```

## Project Status

**CURRENT STATE**: Basic prototype with architectural flaws identified
**CRITICAL INSIGHT**: Must redesign around Polars patterns, not scipy.sparse
**NEXT STEPS**: Begin architectural migration following Polars design system

## Essential Commands (Once Initialized)

### Setup Commands
```bash
# Install uv package manager (required first step)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize project structure
uv init bio-polars && cd bio-polars
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip sync  # After creating pyproject.toml
```

### Development Commands
```bash
# Test commands
pytest tests/ -xvs                    # Run all tests
pytest tests/test_io.py -xvs          # Run specific test file
pytest -k "test_filter" -xvs          # Run tests matching pattern

# Linting and formatting
ruff check src/                       # Check code quality
ruff format src/                      # Format code

# Memory monitoring (critical for M1 Mac)
python -c "import psutil; print(f'Available: {psutil.virtual_memory().available/1e9:.1f}GB')"

# Package management
uv pip install <package>              # Add new dependency
uv pip freeze > requirements.lock     # Lock dependencies
```

### Benchmarking Commands
```bash
# Memory profiling
mprof run python bench/profile_operation.py
mprof plot

# Performance testing
python bench/run_bench.py
```

## Immediate Actions for New Development

1. **INSTALL uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **CHECK MEMORY**: Run memory check command above
3. **START**: Execute TICKET-001 with uv setup
4. **TRACK**: Update TASK_TRACKER.md after EACH ticket (include memory usage)
5. **MONITOR**: Check memory status before each operation
6. **TEST**: Run pytest after EVERY implementation
7. **BENCHMARK**: Compare to baseline after core features
8. **ITERATE**: Continue through tickets sequentially

## Success Criteria

Project is COMPLETE when:
- [ ] All 20 tickets implemented and tested
- [ ] Benchmarks show >2x speedup on metadata ops
- [ ] Zero densification of sparse matrices
- [ ] Memory usage < 50% of Scanpy for same operations
- [ ] No swap usage on M1 Mac during standard workflows
- [ ] Full AnnData/Seurat interoperability
- [ ] Documentation website live
- [ ] PyPI package published via uv

---

✅ **NEXT ACTION**: 
1. Install uv package manager
2. CREATE repository with uv init
3. Implement TICKET-001 with memory monitoring
4. Update TASK_TRACKER.md with system specs and first task completion
