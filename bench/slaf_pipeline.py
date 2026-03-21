"""
SLAF × BioPolars End-to-End Pipeline Benchmark.

Demonstrates the full pipeline:
  1. Load expression data from a SLAF dataset
  2. Run native Rust SVD (50 PCs)
  3. Run native Rust UMAP (2D embedding)
  4. Write UMAP coordinates back to SLAF
  5. Optionally run Louvain clustering

Usage:
    python bench/slaf_pipeline.py <path_to_slaf_dataset>

Example:
    python bench/slaf_pipeline.py data/pbmc3k.slaf
"""

import sys
import time
import os

import numpy as np
import polars as pl


def main():
    if len(sys.argv) < 2:
        print("Usage: python bench/slaf_pipeline.py <slaf_path>")
        print("Example: python bench/slaf_pipeline.py data/pbmc3k.slaf")
        sys.exit(1)

    slaf_path = sys.argv[1]

    if not os.path.exists(slaf_path):
        print(f"Error: SLAF dataset not found: {slaf_path}")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Step 1: Load SLAF expression data
    # -------------------------------------------------------------------
    print("=" * 60)
    print("SLAF × BioPolars Pipeline")
    print("=" * 60)

    from biopolars.slaf_io import read_slaf_expression
    import biopolars  # noqa: F401 — registers .bio namespace

    t0 = time.time()
    agg_df, n_cells, n_genes = read_slaf_expression(slaf_path)
    t_load = time.time() - t0
    print(f"\n📦 Loaded SLAF: {n_cells:,} cells × {n_genes:,} genes [{t_load:.2f}s]")

    # -------------------------------------------------------------------
    # Step 2: Run SVD (native Rust, no GIL)
    # -------------------------------------------------------------------
    print("\n🔬 Running native Rust SVD (50 PCs)...")
    t0 = time.time()
    svd_df = agg_df.with_columns(
        pl.col("cell_id").bio.svd(
            pl.col("gene_id"), pl.col("count"),
            n_cells=n_cells, n_genes=n_genes, n_comps=50
        ).alias("pca_coords")
    )
    t_svd = time.time() - t0
    print(f"   ✅ SVD completed [{t_svd:.2f}s]")

    # -------------------------------------------------------------------
    # Step 3: Run UMAP (native Rust, no GIL)
    # -------------------------------------------------------------------
    print("\n🗺️  Running native Rust UMAP (2D, 15 neighbors, 200 epochs)...")
    t0 = time.time()
    result_df = svd_df.with_columns(
        pl.col("pca_coords").bio.umap(
            n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=200
        ).alias("umap_coords")
    )
    t_umap = time.time() - t0
    print(f"   ✅ UMAP completed [{t_umap:.2f}s]")

    # Extract UMAP coordinates
    umap_nested = result_df["umap_coords"][0]
    umap_array = np.array(umap_nested.to_list())
    print(f"   📊 Embedding shape: {umap_array.shape}")

    # -------------------------------------------------------------------
    # Step 4: Write UMAP back to SLAF (optional)
    # -------------------------------------------------------------------
    write_back = "--write-back" in sys.argv
    if write_back:
        from biopolars.slaf_io import write_embeddings_to_slaf

        print("\n💾 Writing UMAP embeddings back to SLAF...")
        t0 = time.time()
        write_embeddings_to_slaf(slaf_path, umap_array, column_name="X_umap")
        t_write = time.time() - t0
        print(f"   ✅ Write-back completed [{t_write:.2f}s]")

    # -------------------------------------------------------------------
    # Step 5: Generate scatter plot
    # -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid_mask = ~((umap_array[:, 0] == 0) & (umap_array[:, 1] == 0))
        valid_umap = umap_array[valid_mask]

        plt.figure(figsize=(12, 12))
        plt.scatter(valid_umap[:, 0], valid_umap[:, 1],
                    s=0.5, alpha=0.5, c="darkblue", edgecolors="none")
        plt.title(
            f"SLAF × BioPolars: Rust SVD→UMAP ({n_cells:,} cells)",
            fontsize=16, fontweight="bold",
        )
        plt.axis("off")

        out_dir = "/tmp/artifacts"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "slaf_umap_result.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n🖼️  Saved scatter plot to {out_path}")

    except ImportError:
        print("\n⚠️  matplotlib not available, skipping plot.")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    total = t_load + t_svd + t_umap
    print("\n" + "=" * 60)
    print(f"⏱️  Total pipeline time: {total:.2f}s")
    print(f"   Load:  {t_load:.2f}s")
    print(f"   SVD:   {t_svd:.2f}s")
    print(f"   UMAP:  {t_umap:.2f}s")
    print(f"   Cells: {n_cells:,}, Genes: {n_genes:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
