import time
import os
import psutil
import polars as pl
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata
from biopolars.tl import sparse_masked_pca

print("==================================================")
print("  BIOPOLARS VS SCANPY: IMPLICIT SPARSE PCA        ")
print("==================================================")

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# We construct a dense matrix simulation that might crash scanpy
num_cells = 500_000
num_genes = 2000 # Assume already filtered to HVG
density = 0.05
nnz = int(num_cells * num_genes * density)

print(f"Generating {num_cells:,} Cells x {num_genes} HVGs ({nnz:,} non-zeros)...")
np.random.seed(42)
row_ind = np.random.randint(0, num_cells, nnz, dtype=np.uint32)
col_ind = np.random.randint(0, num_genes, nnz, dtype=np.uint32)
data_vals = np.random.exponential(scale=2.0, size=nnz).astype(np.float32)

print("\n--- PHASE 1: BioPolars Implicitly Centered sparse_masked_pca ---")
df = pl.DataFrame({
    "cell_id": row_ind,
    "gene_id": col_ind,
    "count": data_vals
})

start_mem = get_process_memory()
t0 = time.time()
results = sparse_masked_pca(df, n_cells=num_cells, n_genes=num_genes, n_comps=50)
t1 = time.time()
end_mem = get_process_memory()

biopolars_time = t1 - t0
print(f"BioPolars PCA execution: {biopolars_time:.2f} seconds")
print(f"Memory moved from {start_mem:.1f}MB to {end_mem:.1f}MB")

print("\n--- PHASE 2: Scanpy PCA Benchmark ---")
sparse_matrix = sp.csr_matrix((data_vals, (row_ind, col_ind)), shape=(num_cells, num_genes))
adata = anndata.AnnData(X=sparse_matrix)

start_mem = get_process_memory()
t0 = time.time()
sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
t1 = time.time()
end_mem = get_process_memory()

scanpy_time = t1 - t0
print(f"Scanpy PCA execution: {scanpy_time:.2f} seconds")
print(f"Memory moved from {start_mem:.1f}MB to {end_mem:.1f}MB")

print("\n==================================================")
print("               RESULTS SUMMARY                    ")
print("==================================================")
if biopolars_time > 0:
    print(f"Time Speedup: {scanpy_time / biopolars_time:.2f}x faster")
