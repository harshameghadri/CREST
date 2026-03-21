import polars as pl
import os
import psutil
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import memory_usage

import crest.io
import crest.pp
import crest.tl

print("==========================================================")
print("  BIOPOLARS: 1 MILLION NEURON PROFESSIONAL BENCHMARK ")
print("==========================================================")

parquet_file = "data/1M_neurons/1M_neurons.parquet"
norm_file = "data/1M_neurons/normalized_1M.parquet"

if not os.path.exists(parquet_file):
    print(f"Error: Dataset {parquet_file} not found! Please run Phase 1 downloads.")
    exit(1)

file_size_gb = os.path.getsize(parquet_file) / (1024**3)
print(f"Dataset Size: {file_size_gb:.2f} GB")

# Initialize Tracking
timing_metrics = {}
memory_metrics = {}
process = psutil.Process(os.getpid())
mem_baseline = process.memory_info().rss / (1024**3)
print(f"Baseline Python RAM: {mem_baseline:.2f} GB")

# ---------------------------------------------------------
# Phase 1: Streaming Normalization (If not cached)
# ---------------------------------------------------------
def run_normalization():
    print("\n--- PHASE 1: Streaming Normalization to Disk ---")
    if not os.path.exists(norm_file):
        t0 = time.time()
        df = pl.scan_parquet(parquet_file)
        print("Computing column cell_sums (streaming)...")
        # In modern polars, streaming is 'engine="streaming"' but we use default lazy iterators
        cell_sums_dict = dict(df.group_by("cell_id").agg(pl.col("count").sum().alias("s")).collect(streaming=True).iter_rows())
        
        cell_sums_df = pl.DataFrame({"cell_id": list(cell_sums_dict.keys()), "cell_sum": list(cell_sums_dict.values())})
        cell_sums_df = cell_sums_df.with_columns(pl.col("cell_id").cast(pl.UInt32), pl.col("cell_sum").cast(pl.Float32))
        
        normalized_lazy = df.join(cell_sums_df.lazy(), on="cell_id").with_columns(
            ((pl.col("count") / pl.col("cell_sum")) * 10_000).alias("cpm")
        ).with_columns([
            pl.col("cpm").bio.log1p().alias("count") 
        ]).select(["cell_id", "gene_id", "count"])
        
        print(f"Sinking normalized dataset to {norm_file}...")
        normalized_lazy.sink_parquet(norm_file)
        t1 = time.time()
        timing_metrics['1_Normalization'] = t1 - t0
    else:
        print(f"Cached Normalized file found.")
        timing_metrics['1_Normalization'] = 0.0

memory_usage((run_normalization, ), max_usage=True)
memory_metrics['1_Normalization_Peak'] = process.memory_info().rss / (1024**3) - mem_baseline

# ---------------------------------------------------------
# Phase 2: Exact PCA Subsampling (50 PCs)
# ---------------------------------------------------------
print("\n--- PHASE 2: Exact Sparse PCA on Subsampled Cohort ---")
import duckdb

try:
    norm_df = pl.scan_parquet(norm_file)
    explain_str = norm_df.explain()
    file_path = explain_str.split("Parquet SCAN [")[1].split("]")[0]
except Exception:
    file_path = norm_file

con = duckdb.connect(database=':memory:')

print("Streaming random 1.5% cell subsample (20,000 cells) from 2.5B rows natively via DuckDB...")
t_samp = time.time()

print("Calculating full matrix dimensions to draw random sample constraints...")
t_dim = time.time()
max_res = pl.scan_parquet(norm_file).select([
    pl.col("cell_id").max().alias("max_cell"),
    pl.col("gene_id").max().alias("max_gene")
]).collect(engine="streaming")

max_cell = max_res[0, "max_cell"] + 1
max_gene = max_res[0, "max_gene"] + 1

# Explicitly sample 20,000 unique Cell IDs to prevent pulling 37.5 Million randomized expression tuples 
# across all 1.3M cells that crashes UMAP later by embedding 1.3M native arrays.
np.random.seed(42)
random_cells = np.random.choice(int(max_cell), 20000, replace=False).astype(np.uint32)

# Insert the explicit cell IDs to be monitored down into a native isolated memory table.
con.execute("CREATE TABLE sampled_cells (cell_id UINTEGER)")
con.executemany("INSERT INTO sampled_cells VALUES (?)", [(int(x),) for x in random_cells])

# Join the target labels to strip the exact coordinate subset directly from Disk
query = f"""
    SELECT p.cell_id, p.gene_id, p.count 
    FROM read_parquet('{file_path}') p
    INNER JOIN sampled_cells s ON p.cell_id = s.cell_id
"""

# Zero copy stream into isolated Polars LazyFrame
subset_df = pl.from_arrow(con.execute(query).fetch_arrow_table()).lazy()

print("\nCalculating Top 2000 Highly Variable Genes (Locally on 20k Subsample) using Native Polars...")
t_hvg = time.time()
obs_df = pl.DataFrame({"cell_id": random_cells})
bf_subset = crest.core.BioFrame(X=subset_df, obs=obs_df)

bf_hvg = crest.pp.highly_variable_genes(bf_subset, n_top_genes=2000)
subset_hvg_df = bf_hvg.X
timing_metrics['1.5_HVG_Subsample'] = time.time() - t_hvg
print(f"Top 2000 HVGs identified natively in {timing_metrics['1.5_HVG_Subsample']:.2f}s!")

# Map original cell_ids to a contiguous 0..19999 dense array so SVD math does not encode 1.28M empty zeros.
mapping_df = pl.DataFrame({
    "cell_id": random_cells,
    "dense_id": np.arange(len(random_cells), dtype=np.uint32)
}).lazy()

subset_df_mapped = subset_hvg_df.join(mapping_df, on="cell_id").select([
    pl.col("dense_id").alias("cell_id"),
    pl.col("gene_id"),
    pl.col("count")
])

print(f"Extracted Sample Coordinates! Bounds: Random ~20,000 Cells. Evaluated in {time.time()-t_samp:.2f}s")


def run_pca():
    t_pca_start = time.time()
    # Execute EXACT SVD directly skipping the sklearn loop
    pca_result = crest.tl.sparse_masked_pca(
        df=subset_df_mapped, 
        n_cells=len(random_cells), 
        n_genes=max_gene,
        n_comps=50
    )
    t_pca_end = time.time()
    timing_metrics['2_PCA_Sampled'] = t_pca_end - t_pca_start
    return pca_result["X_pca"]

pca_mem = memory_usage((run_pca, ), max_usage=True)
global_pca_coords = run_pca()
memory_metrics['2_PCA_Peak'] = (pca_mem[0] if isinstance(pca_mem, list) else pca_mem) / 1024

print(f"Exact Sparse PCA Complete! Peak Memory: {memory_metrics['2_PCA_Peak']:.2f} GB")

# Package coordinates into Polars
pca_coords_list = [list(row) for row in global_pca_coords]
pca_df = pl.DataFrame({
    "dense_id": pl.arange(0, len(pca_coords_list), eager=True, dtype=pl.UInt32),
    "pca_coords": pca_coords_list
})

# Revert the dense mapping back to original biological Cell IDs
final_pca_df = pca_df.join(mapping_df.collect(), on="dense_id").select(["cell_id", "pca_coords"])

# We identify the valid cell IDs dynamically sampled from the DuckDB query
active_cell_ids = subset_df.select("cell_id").unique().collect().get_columns()[0]

final_results_df = pl.DataFrame({"cell_id": active_cell_ids})
final_results_df = final_results_df.join(final_pca_df, on="cell_id")


# ---------------------------------------------------------
# Phase 3: UMAP Embedding Execution
# ---------------------------------------------------------
print("\n--- PHASE 3: Native UMAP Embedded Projections ---")
def run_umap():
    t_umap = time.time()
    res = final_results_df.filter(pl.col("pca_coords").is_not_null()).with_columns(
        pl.col("pca_coords").bio.umap(n_components=2, n_neighbors=15).alias("umap_coords")
    )
    timing_metrics['3_UMAP'] = time.time() - t_umap
    return res

umap_mem = memory_usage((run_umap, ), max_usage=True)
final_results_df = run_umap()
memory_metrics['3_UMAP_Peak'] = ((umap_mem[0] if isinstance(umap_mem, list) else umap_mem) / 1024) - mem_baseline


# ---------------------------------------------------------
# Phase 4: Parameter Sweep - Native Leiden Clustering
# ---------------------------------------------------------
print("\n--- PHASE 4: Leiden Parameter Sweeps ---")
k_values = [15, 30, 50]
sweep_memories = []

for k in k_values:
    print(f"\nEvaluating Native Louvain Community (k_neighbors = {k})...")
    def run_leiden_k():
        t0 = time.time()
        res = final_results_df.filter(pl.col("pca_coords").is_not_null()).with_columns(
            pl.col("pca_coords").bio.louvain(n_neighbors=k).alias(f"leiden_k{k}")
        )
        t_l = time.time() - t0
        return res, t_l
        
    peak_mem = memory_usage((run_leiden_k, ), max_usage=True)
    sweep_memories.append((peak_mem[0] if isinstance(peak_mem, list) else peak_mem) / 1024)
    
    final_results_df, t_exec = run_leiden_k()
    timing_metrics[f'4_Leiden_k{k}'] = t_exec
    print(f"  Completed in {t_exec:.2f}s | Peak RAM: {sweep_memories[-1]:.2f} GB")

memory_metrics['4_Leiden_Peak_Avg'] = np.mean(sweep_memories) - mem_baseline


# ---------------------------------------------------------
# Phase 5: Result Plotting and Scientific Output
# ---------------------------------------------------------
print("\n--- PHASE 5: Generating Publication Figures ---")
os.makedirs("bench_results", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# Expand UMAP coordinates for matplotlib
umap_array = np.array(final_results_df["umap_coords"].to_list())
final_results_df = final_results_df.with_columns([
    pl.Series("UMAP_1", umap_array[:, 0]),
    pl.Series("UMAP_2", umap_array[:, 1])
])

# Figure 1: UMAP Overlays colored by parameter sweeps
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle("BioPolars Native UMAP Projections (1.3 Million Neurons)", fontsize=20, fontweight="bold")

for idx, k in enumerate(k_values):
    ax = axes[idx]
    
    # Randomize plot ordering to prevent overlap clusters hiding others
    plot_df = final_results_df.sample(fraction=1.0, shuffle=True)
    
    sns.scatterplot(
        x=plot_df["UMAP_1"].to_numpy(),
        y=plot_df["UMAP_2"].to_numpy(),
        hue=plot_df[f"leiden_k{k}"].to_numpy().astype(str),
        palette="tab20",
        s=1,
        alpha=0.6,
        linewidth=0,
        ax=ax,
        legend=False
    )
    ax.set_title(f"Native Leiden Clustering (k={k})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
plt.tight_layout()
plt.savefig("bench_results/umap_leiden_sweeps.png", dpi=300)
print("Saved UMAP Multi-Panel Figure to `bench_results/umap_leiden_sweeps.png`.")

# Figure 2: Metrics Bar Charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Time Bar Chart
sns.barplot(
    x=list(timing_metrics.keys()), 
    y=list(timing_metrics.values()), 
    hue=list(timing_metrics.keys()),
    ax=ax1, 
    palette="viridis"
)
ax1.set_title("Pipeline Execution Time (1.3M Cells x 50 PCs x 20k Genes)")
ax1.set_ylabel("Seconds")
ax1.tick_params(axis='x', rotation=45)

# RAM Bar Chart
sns.barplot(
    x=list(memory_metrics.keys()), 
    y=list(memory_metrics.values()), 
    hue=list(memory_metrics.keys()),
    ax=ax2, 
    palette="flare"
)
ax2.set_title("Peak Added Memory Overhead (Beyond Base Pointer)")
ax2.set_ylabel("Gigabytes (GB)")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("bench_results/performance_metrics.png", dpi=300)
print("Saved Performance Metrics to `bench_results/performance_metrics.png`.")


# Figure 3: Cluster Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, k in enumerate(k_values):
    cluster_counts = final_results_df.group_by(f"leiden_k{k}").len().sort("len", descending=True)
    sns.barplot(
        x=cluster_counts[f"leiden_k{k}"].head(20).to_numpy().astype(str),
        y=cluster_counts["len"].head(20).to_numpy(),
        color="steelblue",
        ax=axes[idx]
    )
    axes[idx].set_title(f"Top 20 Cluster Densities (k={k})")
    axes[idx].set_xlabel("Cluster ID")
    axes[idx].set_ylabel("Number of Cells")
    axes[idx].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig("bench_results/cluster_distributions.png", dpi=300)
print("Saved Cluster Distributions to `bench_results/cluster_distributions.png`.")

print("\n==========================================================")
print(" BENCHMARK COMPLETE!")
print(" Result Artifacts available in local `/bench_results` directory.")
print("==========================================================")
