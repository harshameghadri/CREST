import time
import os
import psutil
import numpy as np
import pynndescent
import scanpy as sc
import anndata

print("==================================================")
print("  BIOPOLARS VS SCANPY: K-NEAREST NEIGHBORS (HNSW) ")
print("==================================================")

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# We will generate synthetic PCA coordinates for 1M cells (e.g., 50 dimensions)
n_cells = 1_000_000
n_comps = 50

print(f"Generating PCA coordinates for {n_cells:,} cells x {n_comps} dimensions...")
np.random.seed(42)
X_pca = np.random.randn(n_cells, n_comps).astype(np.float32)

print("\n--- PHASE 1: Scanpy Exact/Approximate KNN Benchmark ---")
adata = anndata.AnnData(X=np.empty((n_cells, 1)))
adata.obsm['X_pca'] = X_pca

start_mem = get_process_memory()
t0 = time.time()
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, method='umap', use_rep='X_pca')
t1 = time.time()
end_mem = get_process_memory()

scanpy_time = t1 - t0
print(f"Scanpy execution: {scanpy_time:.2f} seconds")
print(f"Memory moved from {start_mem:.1f}MB to {end_mem:.1f}MB")


print("\n--- PHASE 2: BioPolars PyNNDescent (HNSW/ANN) Benchmark ---")
start_mem = get_process_memory()
t0 = time.time()

# Drop down directly into heavily parallelized Approximate Nearest Neighbors
index = pynndescent.NNDescent(
    X_pca, 
    n_neighbors=15, 
    metric="euclidean", 
    n_jobs=-1,  # use all cores
    random_state=42
)
indices, distances = index.neighbor_graph

t1 = time.time()
end_mem = get_process_memory()

biopolars_time = t1 - t0
print(f"BioPolars (PyNNDescent) execution: {biopolars_time:.2f} seconds")
print(f"Memory moved from {start_mem:.1f}MB to {end_mem:.1f}MB")

print("\n==================================================")
print("               RESULTS SUMMARY                    ")
print("==================================================")
if biopolars_time > 0:
    print(f"Time Speedup: {scanpy_time / biopolars_time:.2f}x faster")
