import time
import os
import psutil
import polars as pl
import scipy.stats as stats
import numpy as np

print("==================================================")
print("  BIOPOLARS VS SCIPY: MASSIVE WILCOXON BENCHMARK  ")
print("==================================================")

np.random.seed(42)

# Gene differential expression typically tests ~2000 HVGs across clusters
n_genes = 2000
n1, n2 = 1000, 1000  # cells per cluster

print(f"Generating Count Matrices for {n_genes} Genes across 2 clusters ({n1} and {n2} cells)...")
group1_matrix = np.floor(np.random.exponential(scale=2.0, size=(n_genes, n1))).astype(np.float32)
group2_matrix = np.floor(np.random.exponential(scale=2.5, size=(n_genes, n2))).astype(np.float32)


print("\n--- PHASE 1: SciPy Baseline (For Loop) ---")
scipy_pvals = np.zeros(n_genes)

t0 = time.time()
# SciPy MUST be for-looped across genes because mannwhitneyu is not vectorized for 2D matrices across axis=1
for i in range(n_genes):
    # Only calculate if data exists to match real-world
    stat, pval = stats.mannwhitneyu(group1_matrix[i], group2_matrix[i], alternative="two-sided")
    scipy_pvals[i] = pval
t1 = time.time()
scipy_time = t1 - t0

print(f"SciPy Time (2000 genes): {scipy_time:.2f} seconds")

print("\n--- PHASE 2: BioPolars Native Parallel Formulation ---")
import biopolars

# Pack matrix into Polars Lists (simulating the streaming groupby output)
df = pl.DataFrame({
    "gene_id": np.arange(n_genes),
    "g1": group1_matrix.tolist(),
    "g2": group2_matrix.tolist()
}, schema={"gene_id": pl.UInt32, "g1": pl.List(pl.Float32), "g2": pl.List(pl.Float32)}).lazy()

t0 = time.time()
# The native rust plugin executes implicitly in parallel over all rows (genes) using Polars' thread pool
res = df.with_columns(
    pl.col("g1").bio.wilcoxon(pl.col("g2")).alias("p_value")
).collect()
t1 = time.time()
bp_time = t1 - t0

bp_pvals = res["p_value"].to_numpy()

print(f"BioPolars Time (2000 genes): {bp_time:.2f} seconds")

print("\n==================================================")
print("               SCIENTIFIC VERIFICATION            ")
print("==================================================")
# Use isclose with ATOL to tolerate SciPy using Survival Function vs Rust using 1-CDF at tails
matching = np.isclose(scipy_pvals, bp_pvals, atol=1e-8).sum()
print(f"Exact Matches (up to atol=1e-8): {matching}/{n_genes}")

if bp_time > 0:
    print(f"\n🚀 Time Speedup: {scipy_time / bp_time:.2f}x faster globally!")
