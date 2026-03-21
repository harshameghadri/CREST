import time
import os
import scanpy as sc
import anndata
import scipy.sparse as sp
import numpy as np
import polars as pl
import crest
import urllib.request
import psutil

print("==================================================")
print("  BIOPOLARS VS SCANPY: FINAL BOSS BENCHMARK       ")
print("==================================================")

import subprocess

# 1. Download Real Data
url = "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5"
h5_path = "pbmc_10k_v3.h5"
if not os.path.exists(h5_path):
    print(f"Downloading real 10x dataset (10k PBMCs) to {h5_path}...")
    subprocess.run(["curl", "-A", "Mozilla/5.0", "-o", h5_path, url], check=True)

print("\n--- PHASE 1: I/O PARSING SPEED (Real 10k PBMC) ---")
# Scanpy I/O
t0 = time.time()
adata = sc.read_10x_h5(h5_path)
t1 = time.time()
scanpy_io_time = t1 - t0
print(f"Scanpy read_10x_h5: {scanpy_io_time:.4f} sec")

# BioPolars I/O
from crest.io import read_10x_h5
t0 = time.time()
bp_data = read_10x_h5(h5_path)
t1 = time.time()
bp_io_time = t1 - t0
print(f"BioPolars read_10x_h5: {bp_io_time:.4f} sec")
if bp_io_time > 0:
    print(f"Speedup: {scanpy_io_time / bp_io_time:.2f}x faster")


print("\n--- PHASE 2: 1.3 MILLION CELL SCALABILITY (Final Boss) ---")
print("Generating exact sparse matrix footprint for 1.3 Million cells...")
num_cells = 1_300_000
num_genes = 20_000
density = 50 / num_genes # ~50 genes per cell = 65,000,000 expressions

nnz = int(num_cells * num_genes * density)
np.random.seed(42)
row_ind = np.random.randint(0, num_cells, nnz, dtype=np.uint32)
col_ind = np.random.randint(0, num_genes, nnz, dtype=np.uint32)
data_vals = np.random.exponential(scale=2.0, size=nnz).astype(np.float32)

print(f"Total non-zero elements (nnz): {nnz:,}")

t0 = time.time()
sparse_matrix = sp.csr_matrix((data_vals, (row_ind, col_ind)), shape=(num_cells, num_genes))
adata_large = anndata.AnnData(X=sparse_matrix)
print(f"Constructed Scanpy AnnData in {time.time() - t0:.2f} sec")


t0 = time.time()
bp_large = pl.DataFrame({
    "cell_id": row_ind,
    "gene_id": col_ind,
    "count": data_vals
})
print(f"Constructed BioPolars DataFrame in {time.time() - t0:.2f} sec")

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Memory profiling Scanpy
print("\nRunning Scanpy log1p...")
start_mem = get_process_memory()
t0 = time.time()
sc.pp.log1p(adata_large)
t1 = time.time()
end_mem = get_process_memory()
scanpy_time = t1 - t0
print(f"Scanpy memory footprint moved from {start_mem:.1f}MB to {end_mem:.1f}MB")

# Memory profiling BioPolars
print("\nRunning BioPolars log1p...")
start_mem = get_process_memory()
t0 = time.time()
result = bp_large.with_columns(pl.col("count").bio.log1p().alias("count_log1p"))
t1 = time.time()
end_mem = get_process_memory()
bp_time = t1 - t0
print(f"BioPolars memory footprint moved from {start_mem:.1f}MB to {end_mem:.1f}MB")

print("\n==================================================")
print("               RESULTS SUMMARY                    ")
print("==================================================")
print(f"Scanpy Execution:    {scanpy_time:.2f} sec")
print(f"BioPolars Execution: {bp_time:.2f} sec")
print("--------------------------------------------------")
if bp_time > 0:
    print(f"Time Speedup: {scanpy_time / bp_time:.2f}x faster")
