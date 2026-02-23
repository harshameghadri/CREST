import polars as pl
import os
import psutil
import time

import biopolars.io
import biopolars.pp

print("==========================================================")
print("  BIOPOLARS: 20K NEURON REAL DATASET BENCHMARK            ")
print("==========================================================")

h5_file = "data/1M_neurons/1M_neurons_neuron20k.h5"
parquet_file = "data/1M_neurons/20k_neurons.parquet"

if not os.path.exists(h5_file):
    print(f"Error: Dataset {h5_file} not found!")
    exit(1)

# Step 1: Parse the 20k cell H5 file into Parquet stream
print("\n--- PHASE 1: Streaming HDF5 to Parquet ---")
t0 = time.time()
if not os.path.exists(parquet_file):
    # Using our chunked parser
    biopolars.io.convert_h5_to_parquet_stream(
        file_path=h5_file,
        output_path=parquet_file,
        chunk_size=50_000
    )
    t1 = time.time()
    print(f"Conversion complete! Took {t1 - t0:.2f} seconds.")
else:
    print("Parquet file already exists. Skipping conversion.")

# Step 2: Lazy loading and full single-cell lifecycle
print("\n--- PHASE 2: End-to-End Log1p, HVG, PCA on 20k Cells ---")
print(f"File Size: {os.path.getsize(parquet_file) / (1024**3):.2f} GB")

# Spin up our LazyFrame
df = pl.scan_parquet(parquet_file)

t0 = time.time()
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss

# 1. Calculate CPM using explicit Join instead of Window (.over) to maintain streaming
cell_sums = df.group_by("cell_id").agg(pl.col("count").sum().alias("cell_sum"))
normalized = df.join(cell_sums, on="cell_id").with_columns(
    ((pl.col("count") / pl.col("cell_sum")) * 10_000).alias("cpm")
).with_columns([
    pl.col("cpm").bio.log1p().alias("log1p_count")
])

# 2. Build explicit metadata (obs) to formally encode the graph shape
# In production, this would be loaded from a `.obs` CSV or AnnData
unique_cells = df.select("cell_id").unique().collect()
obs = unique_cells.with_columns(pl.lit("Unknown").alias("cluster"))
print(f"BioFrame Initialization: Detected {obs.height} total cells.")

# Instantiate BioFrame V2
adata = biopolars.BioFrame(X=normalized, obs=obs)

# 3. Compute Highly Variable Genes natively with zero-inflation internally tracked
hvgs = biopolars.pp.highly_variable_genes(adata)

result = hvgs.X.collect(engine="streaming")

mem_after = process.memory_info().rss
t1 = time.time()

print(f"\nDiscovered {len(result)} Highly Variable Genes across 20k cells!")
print(result.head())
print("----------------------------------------------------------")
print(f"Time Taken (Normalization + HVGs): {t1 - t0:.2f} seconds")
print(f"Peak Memory Added: {(mem_after - mem_before) / (1024**3):.2f} GB")
print("==========================================================")
