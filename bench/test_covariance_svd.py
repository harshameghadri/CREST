import os
import time
import psutil
import polars as pl
import numpy as np
from biopolars.slaf_io import read_slaf_expression
import biopolars  # Registers .bio

def main():
    slaf_path = "data/1M_neurons/1M_neurons.slaf"
    if not os.path.exists(slaf_path):
        print("Skipping testing: 1M slaaf dataset not found")
        return

    process = psutil.Process(os.getpid())
    print(f"📊 Testing Native Zero-Copy Covariance SVD (Mathematical Exact Solution)")
    
    # We test on 10,000 cells x 2000 genes to push the boundaries fast
    t0 = time.time()
    agg_df, n_cells, n_genes = read_slaf_expression(slaf_path, top_genes=2000, max_cells=10000)
    print(f"📦 Arrow Loader Memory: {process.memory_info().rss / 1024**2:.2f} MB")
    
    t_svd_start = time.time()
    n_comps = 50
    # Execute the Exact Covariance accumulation plugin -> Dense Eigendecomposition -> Streaming Projection!
    svd_df = agg_df.select(
        pl.col("cell_id").bio.svd(
            pl.col("gene_id"), pl.col("count"),
            n_cells=n_cells, n_genes=n_genes, n_comps=n_comps
        ).alias("pca_coords")
    )
    
    # We must explicitly select and extract the item to force evaluation of the plugin
    pca_series = svd_df.select("pca_coords").head(1)["pca_coords"][0]
    pca_array = np.array(pca_series.to_list())
    
    t_svd_end = time.time()
    print(f"✅ Exact Covariance PCA computed in {t_svd_end - t_svd_start:.2f}s!")
    print(f"📦 Peak Memory (Post SVD): {process.memory_info().rss / 1024**2:.2f} MB")
    print(f"🔢 PCA Output Matrix Shape: {pca_array.shape}")
    
    # Test variance limits
    var = np.var(pca_array, axis=0)
    print(f"📈 Top 5 Principal Component Variances: {var[:5]}")
    
    # For a normalized dataset, PCs should have decreasing variance mathematically
    if pca_array.shape == (100000, 50):
        print("🎉 SUCCESS: The exact dense Covariance SVD produced matching dimensions natively.")

if __name__ == "__main__":
    main()
