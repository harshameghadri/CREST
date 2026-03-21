import polars as pl
import os
import crest
import time
import numpy as np
import matplotlib.pyplot as plt

norm_file = "/tmp/normalized_1M.parquet"

if not os.path.exists(norm_file):
    print(f"Error: Dataset {norm_file} not found! Cannot run 1M benchmark.")
    exit(1)

print("Scanning normalized dataset...")
df = pl.scan_parquet(norm_file)

# Determine global max bounds for sparse matrix instantiation
max_res = pl.scan_parquet(norm_file).select([
    pl.col("cell_id").max().alias("max_cell"),
    pl.col("gene_id").max().alias("max_gene")
]).collect()

n_cells = max_res[0, "max_cell"] + 1
n_genes = max_res[0, "max_gene"] + 1

print(f"Dataset bounds: {n_cells:,} cells, {n_genes:,} genes.")
print("Aggregating into single contiguous COO payload for Rust SVD engine...")

df_grouped = df.group_by(pl.lit(1).alias("dataset_id")).agg([
    pl.col("cell_id"),
    pl.col("gene_id"),
    pl.col("count")
])

print("Executing Native SVD -> Native UMAP Pipeline (No GIL)...")

res = df_grouped.with_columns(
    pl.col("cell_id").bio.svd(pl.col("gene_id"), pl.col("count"), n_cells=n_cells, n_genes=n_genes, n_comps=50).alias("pca_coords")
).with_columns(
    pl.col("pca_coords").bio.umap(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=200).alias("umap_coords")
)

t0 = time.time()
final_df = res.collect()
t1 = time.time()

print(f"Full 1M Cell Native SVD -> UMAP completed in {t1 - t0:.2f} seconds!")

umap_matrix = final_df["umap_coords"][0]
print(f"Resulting UMAP shape: {len(umap_matrix)} rows x {len(umap_matrix[0]) if len(umap_matrix)>0 else 0} cols")

umap_array = np.array(umap_matrix.to_list())

# Filter out empty cells (rows with perfectly [0,0] might just be unrecorded cells, so we can ignore them in plotting to avoid a massive dot at 0,0)
valid_mask = ~((umap_array[:,0] == 0) & (umap_array[:,1] == 0))
valid_umap = umap_array[valid_mask]

plt.figure(figsize=(12, 12))
plt.scatter(valid_umap[:, 0], valid_umap[:, 1], s=0.01, alpha=0.3, c='darkblue', edgecolors='none')
plt.title("BioPolars: Native Rust SVD + Exact UMAP on 1.3M Neurons", fontsize=16, fontweight='bold')
plt.axis('off')

os.makedirs("/tmp/artifacts", exist_ok=True)
plt.savefig("/tmp/artifacts/umap_1m_biological.png", dpi=400, bbox_inches='tight')
print("Saved ultra-dense biological scatter plot to /tmp/artifacts/umap_1m_biological.png")
