import polars as pl
import numpy as np
import biopolars
import time

# Create a synthetic 10k x 50 dense feature matrix mimicking PCA output
n_cells = 10_000
n_pcs = 50

print(f"Generating synthetic PCA coordinates: {n_cells} cells x {n_pcs} PCs")
np.random.seed(42)
# Create a dataset with 3 clear cluster centers for the graph
centers = [
    np.random.randn(n_pcs) * 5 + 10,
    np.random.randn(n_pcs) * 5 - 10,
    np.random.randn(n_pcs) * 5
]

pca_data = np.vstack([
    np.random.randn(3334, n_pcs) + centers[0],
    np.random.randn(3333, n_pcs) + centers[1],
    np.random.randn(3333, n_pcs) + centers[2]
]).astype(np.float32)

# Pack into a Polars List(Float32) schema
df = pl.DataFrame({
    "cell_id": np.arange(n_cells, dtype=np.uint32),
    "pca_coords": [list(row) for row in pca_data]
})

print("Testing Native Rust Louvain Clustering (n_neighbors=15)...")
start = time.time()

# Run Louvain through our pyo3-polars binding
res = df.with_columns(
    pl.col("pca_coords").bio.louvain(n_neighbors=15).alias("cluster_id")
)

# Collect natively executes the Rust lazy expression
res_df = res.collect() if isinstance(res, pl.LazyFrame) else res

end = time.time()
print(f"Native Louvain completed in {end - start:.4f} seconds!")

print(res_df.group_by("cluster_id").count())
