import time
import os
import psutil
import polars as pl
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata
from biopolars.pp import highly_variable_genes

print("==================================================")
print("  BIOPOLARS VS SCANPY: HIGHLY VARIABLE GENES      ")
print("==================================================")

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

num_cells = 1_000_000
num_genes = 20_000
density = 50 / num_genes
nnz = int(num_cells * num_genes * density)

print(f"Generating 1 Million Cells x 20,000 Genes ({nnz:,} non-zeros)...")
np.random.seed(42)
row_ind = np.random.randint(0, num_cells, nnz, dtype=np.uint32)
col_ind = np.random.randint(0, num_genes, nnz, dtype=np.uint32)
data_vals = np.random.exponential(scale=2.0, size=nnz).astype(np.float32)

print("\n--- PHASE 1: Scanpy HVG Benchmark ---")
sparse_matrix = sp.csr_matrix((data_vals, (row_ind, col_ind)), shape=(num_cells, num_genes))
adata = anndata.AnnData(X=sparse_matrix)

start_mem = get_process_memory()
t0 = time.time()
# Calculate Seurat highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
t1 = time.time()
end_mem = get_process_memory()

scanpy_time = t1 - t0
print(f"Scanpy execution: {scanpy_time:.2f} seconds")
print(f"Memory moved from {start_mem:.1f}MB to {end_mem:.1f}MB")

print("\n--- PHASE 2: BioPolars Streaming HVG Benchmark ---")
# To simulate streaming, we create the dataframe and evaluate it lazily
bp_df = pl.LazyFrame(pl.DataFrame({
    "cell_id": row_ind,
    "gene_id": col_ind,
    "count": data_vals
}))

start_mem = get_process_memory()
t0 = time.time()

# 1. Define lazy computation
lazy_hvg = highly_variable_genes(bp_df, n_top_genes=2000, flavor="seurat", total_cells=num_cells)

# 2. Execute natively in Rust via Polars Engine
result = lazy_hvg.collect()

t1 = time.time()
end_mem = get_process_memory()

biopolars_time = t1 - t0
print(f"BioPolars execution: {biopolars_time:.2f} seconds")
print(f"Memory moved from {start_mem:.1f}MB to {end_mem:.1f}MB")
print(f"Result shape (only top 2000 HVG data retained): {result.shape}")

print("\n==================================================")
print("               RESULTS SUMMARY                    ")
print("==================================================")
if biopolars_time > 0:
    print(f"Time Speedup: {scanpy_time / biopolars_time:.2f}x faster")
