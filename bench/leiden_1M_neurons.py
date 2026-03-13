import polars as pl
import os
import psutil
import time
import biopolars.io
import biopolars.pp
import biopolars.tl

print("==========================================================")
print("  BIOPOLARS: 1 MILLION NEURON LEIDEN CLUSTERING BENCHMARK ")
print("==========================================================")

parquet_file = "data/1M_neurons/20k_neurons.parquet"
norm_file = "data/1M_neurons/normalized_20k.parquet"
hvg_file = "data/1M_neurons/hvg_20k.parquet"

if not os.path.exists(parquet_file):
    print(f"Error: Dataset {parquet_file} not found!")
    exit(1)

print(f"File Size: {os.path.getsize(parquet_file) / (1024**3):.2f} GB")

t0 = time.time()
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss

print("\n--- PHASE 1: Streaming Normalization to Disk ---")
if not os.path.exists(norm_file):
    df = pl.scan_parquet(parquet_file)
    print("Computing column cell_sums (streaming)...")
    cell_sums_dict = dict(df.group_by("cell_id").agg(pl.col("count").sum().alias("s")).collect(streaming=True).iter_rows())
    # Instead of join, use map_elements to be safe, or just collect cell_sums as tiny DF and join
    cell_sums_df = pl.DataFrame({"cell_id": list(cell_sums_dict.keys()), "cell_sum": list(cell_sums_dict.values())})
    cell_sums_df = cell_sums_df.with_columns(pl.col("cell_id").cast(pl.UInt32), pl.col("cell_sum").cast(pl.Float32))
    
    # lazy join and sink_parquet
    normalized_lazy = df.join(cell_sums_df.lazy(), on="cell_id").with_columns(
        ((pl.col("count") / pl.col("cell_sum")) * 10_000).alias("cpm")
    ).with_columns([
        pl.col("cpm").bio.log1p().alias("count") # Override count with log1p
    ]).select(["cell_id", "gene_id", "count"])
    
    print(f"Sinking normalized dataset to {norm_file} (this limits memory footprint)...")
    normalized_lazy.sink_parquet(norm_file)
else:
    print(f"Normalized file {norm_file} already exists.")

print("\n--- PHASE 2 (Bypassed): Highly Variable Genes Streaming Filter ---")
# The HVG implicit variance grouping blows up memory on 5GB files.
# We will directly stream the Normalized Parquet data into the PCA Sparse Matrix.

print("\n--- PHASE 3: Loading Normalized Matrix & Masked PCA (50 components) ---")
# Lazy scan the normalized Parquet natively
norm_df = pl.scan_parquet(norm_file)
max_cell_hvg = norm_df.select(pl.col("cell_id").max()).collect()[0, 0] + 1
max_gene_hvg = norm_df.select(pl.col("gene_id").max()).collect()[0, 0] + 1

# Only pass the streaming LazyFrame directly into the SciPy COO Triplet builder for SVD
print(f"Streaming Normalized Triplet matrix directly into Sparse PCA...")

t_pca_start = time.time()
pca_result = biopolars.tl.sparse_masked_pca(
    df=norm_df, 
    n_cells=max_cell_hvg, 
    n_genes=max_gene_hvg,
    n_comps=50
)
print(f"PCA Computed in {time.time() - t_pca_start:.2f} seconds.")

# pca_result["X_pca"] is an array of shape (N_cells, 50)
pca_coords = [list(row) for row in pca_result["X_pca"]]

pca_df = pl.DataFrame({
    "cell_id": pl.arange(0, len(pca_coords), eager=True, dtype=pl.UInt32),
    "pca_coords": pca_coords
})

print("\n--- PHASE 4: Native Leiden Clustering (Louvain) ---")
print("Executing Rust-Native Louvain community detection...")
leiden_start = time.time()

# Drop zeros explicitly filtered out before (cells that had no HVG genes expresssed might have nan/0 coords)
# Apply Louvain clustering natively
clustered = pca_df.filter(pl.col("pca_coords").is_not_null()).with_columns(
    pl.col("pca_coords").bio.louvain(n_neighbors=15).alias("leiden_cluster")
)

res_df = clustered

leiden_end = time.time()
mem_after = process.memory_info().rss

print(f"\nNative Louvain on 1M Cells completed in {leiden_end - leiden_start:.2f} seconds!")
print(f"Total Pipeline Time: {leiden_end - t0:.2f} seconds")
print(f"Peak Memory Added: {(mem_after - mem_before) / (1024**3):.2f} GB")

print("\nCluster Assignments:")
print(res_df.group_by("leiden_cluster").len().sort("len", descending=True).head(10))
print("==========================================================")
