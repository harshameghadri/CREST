import polars as pl
import numpy as np
import biopolars
import time

# Create a synthetic 10k x 50 dense feature matrix mimicking PCA output
n_cells = 10_000
n_pcs = 50

print(f"Generating synthetic PCA coordinates: {n_cells} cells x {n_pcs} PCs")
np.random.seed(42)
pca_data = np.random.randn(n_cells, n_pcs).astype(np.float32)

# Pack into a Polars List(Float32) schema
df = pl.DataFrame({
    "cell_id": np.arange(n_cells, dtype=np.uint32),
    "pca_coords": [list(row) for row in pca_data]
})

print(df.head())

print("Testing Native Rust UMAP (n_neighbors=15, n_components=2)...")
start = time.time()

# Run UMAP through our pyo3-polars binding
res = df.with_columns(
    pl.col("pca_coords").bio.umap(n_components=2, n_neighbors=15).alias("umap_coords")
)
# Collect natively executes the Rust lazy expression
res_df = res.collect() if isinstance(res, pl.LazyFrame) else res

end = time.time()
print(f"Native UMAP completed in {end - start:.4f} seconds!")
print(res_df.head(5))
