import polars as pl
import time
import numpy as np
import crest  # Our custom Rust-Polars plugin

# Simulate a massive sparse single-cell dataset
# Let's say 1,000,000 cells (for speed of generation right now)
# Real boss mode would be 10M, but generating that in python takes a minute. Let's do 10M cells * 10 nonzero genes = 100,000,000 rows.
print("Generating 100M row COO triplet DataFrame representing 10 Million cells...")

# Random data generation (simulating the COO format output of an MTX parser)
np.random.seed(42)
num_nonzero = 100_000_000
cell_ids = np.random.randint(0, 10_000_000, size=num_nonzero, dtype=np.uint32)
gene_ids = np.random.randint(0, 30_000, size=num_nonzero, dtype=np.uint32)
counts = np.random.exponential(scale=2.0, size=num_nonzero).astype(np.float32)

print("Constructing Polars DataFrame...")
df = pl.DataFrame({
    "cell_id": cell_ids,
    "gene_id": gene_ids,
    "count": counts
})

print(f"Data Schema: {df.schema}")
print(f"Memory Usage: {df.estimated_size('gb'):.2f} GB")

# Test 1: Native Polars Filter (which should be instantaneous)
t0 = time.time()
filtered = df.filter(pl.col("count") > 0.5)
t1 = time.time()
print(f"[Polars Core] Filtered {len(filtered)} rows in {(t1-t0):.4f} seconds.")

# Test 2: Custom Rust Plugin `.bio.log1p()`
print("\nInvoking hardcore Rust PyO3-Polars Engine: .bio.log1p()...")
t0 = time.time()
# The heavy lifting natively bypasses Python entirely
result = df.with_columns(
    pl.col("count").bio.log1p().alias("count_log1p")
)
t1 = time.time()
print(f"[Rust Engine] Completed 100 Million log1p calculations in {(t1-t0):.4f} seconds.")
print(result.head())

# Bonus: Lazy Evaluation
print("\nTesting Lazy streaming query optimization...")
lazy_df = df.lazy()
lazy_query = (
    lazy_df
    .filter(pl.col("count") > 0.1)
    .with_columns(pl.col("count").bio.log1p().alias("count_log1p"))
)
print("Query Plan built. Executing...")
t0 = time.time()
final_result = lazy_query.collect()
t1 = time.time()
print(f"[Lazy Execution] Filter + Rust log1p on 100M rows took {(t1-t0):.4f} seconds.")
