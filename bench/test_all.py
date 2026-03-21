"""Integration tests and benchmarks for all biopolars functions."""
import polars as pl
import biopolars
import numpy as np
import time

print("=== BioPolars Integration Test Suite ===")
print()

# Test 1: log1p
print("[1] Testing log1p...")
df = pl.DataFrame({"count": [1.0, 2.0, 0.0, 10.0]}, schema={"count": pl.Float32})
result = df.with_columns(pl.col("count").bio.log1p().alias("log1p"))
assert abs(result["log1p"][0] - 0.6931) < 0.01
print("    PASS")

# Test 2: normalize_cpm
print("[2] Testing normalize_cpm...")
df = pl.DataFrame({
    "count": [2.0, 3.0, 5.0, 1.0, 4.0],
    "cell_id": [0, 0, 0, 1, 1],
}, schema={"count": pl.Float32, "cell_id": pl.UInt32})
result = df.with_columns(pl.col("count").bio.normalize_cpm(pl.col("cell_id")).alias("normalized"))
assert abs(result["normalized"][0] - 2000.0) < 1.0, f"Expected 2000, got {result['normalized'][0]}"
assert abs(result["normalized"][4] - 8000.0) < 1.0, f"Expected 8000, got {result['normalized'][4]}"
print("    PASS")

# Test 3: qc_total_counts
print("[3] Testing qc_total_counts...")
result = df.with_columns(pl.col("count").bio.qc_total_counts(pl.col("cell_id")).alias("total"))
assert abs(result["total"][0] - 10.0) < 0.01
assert abs(result["total"][3] - 5.0) < 0.01
print("    PASS")

# Test 4: qc_n_genes
print("[4] Testing qc_n_genes...")
df2 = pl.DataFrame({
    "count": [2.0, 0.0, 5.0, 0.0, 4.0],
    "cell_id": [0, 0, 0, 1, 1],
}, schema={"count": pl.Float32, "cell_id": pl.UInt32})
result = df2.with_columns(pl.col("count").bio.qc_n_genes(pl.col("cell_id")).alias("n_genes"))
assert result["n_genes"][0] == 2
assert result["n_genes"][3] == 1
print("    PASS")

# Test 5: filter_cells
print("[5] Testing filter_cells...")
result = df.with_columns(pl.col("count").bio.filter_cells(pl.col("cell_id"), min_genes=2).alias("keep"))
assert result["keep"][0] == True   # cell 0: 3 genes
assert result["keep"][3] == True   # cell 1: 2 genes
print("    PASS")

# Test 6: scale
print("[6] Testing scale...")
df3 = pl.DataFrame({
    "value": [1.0, 2.0, 3.0],
    "gene_id": [0, 0, 0],
}, schema={"value": pl.Float32, "gene_id": pl.UInt32})
result = df3.with_columns(pl.col("value").bio.scale(pl.col("gene_id"), n_obs=3).alias("scaled"))
assert abs(result["scaled"][0] - (-1.0)) < 0.01
assert abs(result["scaled"][1] - 0.0) < 0.01
assert abs(result["scaled"][2] - 1.0) < 0.01
print("    PASS")

# Test 7: Large-scale benchmark
print()
print("=== Performance Benchmark (10M rows) ===")
np.random.seed(42)
N = 10_000_000
cell_ids = np.random.randint(0, 100_000, size=N, dtype=np.uint32)
counts = np.random.exponential(scale=2.0, size=N).astype(np.float32)
big_df = pl.DataFrame({"cell_id": cell_ids, "count": counts})
mb = big_df.estimated_size("mb")
print(f"Dataset: {N:,} rows, {mb:.1f} MB")

t0 = time.time()
result = big_df.with_columns(pl.col("count").bio.log1p().alias("log1p"))
t1 = time.time()
print(f"[Rust] log1p on {N:,} rows: {t1-t0:.4f}s")

t0 = time.time()
result = big_df.with_columns(pl.col("count").bio.normalize_cpm(pl.col("cell_id")).alias("norm"))
t1 = time.time()
print(f"[Rust] normalize_cpm on {N:,} rows: {t1-t0:.4f}s")

t0 = time.time()
result = big_df.with_columns(pl.col("count").bio.qc_total_counts(pl.col("cell_id")).alias("total"))
t1 = time.time()
print(f"[Rust] qc_total_counts on {N:,} rows: {t1-t0:.4f}s")

t0 = time.time()
result = big_df.with_columns(pl.col("count").bio.qc_n_genes(pl.col("cell_id")).alias("n_genes"))
t1 = time.time()
print(f"[Rust] qc_n_genes on {N:,} rows: {t1-t0:.4f}s")

t0 = time.time()
result = big_df.with_columns(pl.col("count").bio.filter_cells(pl.col("cell_id"), min_genes=200).alias("keep"))
t1 = time.time()
print(f"[Rust] filter_cells on {N:,} rows: {t1-t0:.4f}s")

# Scale benchmark (with gene_ids)
gene_ids = np.random.randint(0, 30_000, size=N, dtype=np.uint32)
big_df2 = pl.DataFrame({"count": counts, "gene_id": gene_ids})

t0 = time.time()
result = big_df2.with_columns(pl.col("count").bio.scale(pl.col("gene_id"), n_obs=100_000).alias("scaled"))
t1 = time.time()
print(f"[Rust] scale on {N:,} rows: {t1-t0:.4f}s")

print()
print("All tests PASSED!")
