"""
CREST Validation Benchmark: 10x GEM-X FLEX 320K FFPE Dataset
============================================================

Three-way comparison:
  1. 10x Cell Ranger (vendor reference — north star)
  2. Scanpy (community standard)
  3. CREST (our pipeline)

Dataset: 320K scFFPE 16-plex GEM-X FLEX
- 8 tissues: Breast, Colorectal, Endo, Glioblastoma, Kidney, LymphNode, Lung, SkinMelanoma
- This script starts with a single tissue for validation, then optionally scales to multi-tissue

Usage:
    # Single tissue (default: Kidney, ~33K cells)
    python bench/validate_flex_320k.py

    # Pick a different tissue
    python bench/validate_flex_320k.py --tissue glioblastoma

    # Multi-tissue integration benchmark (requires Harmony)
    python bench/validate_flex_320k.py --multi
"""

import argparse
import csv
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dataset registry: tissue name → (URL prefix, sample name)
# ---------------------------------------------------------------------------
BASE = "https://cf.10xgenomics.com/samples/cell-exp/9.0.0"
TISSUES = {
    "breast":       ("320k_scFFPE_16-plex_GEM-X_FLEX_BreastCancer1_BC7-8",),
    "colorectal":   ("320k_scFFPE_16-plex_GEM-X_FLEX_Colorectal_BC3-4",),
    "endo":         ("320k_scFFPE_16-plex_GEM-X_FLEX_Endo_BC15-16",),
    "glioblastoma": ("320k_scFFPE_16-plex_GEM-X_FLEX_Glioblastoma_BC1-2",),
    "kidney":       ("320k_scFFPE_16-plex_GEM-X_FLEX_Kidney_BC11-12",),
    "lymphnode":    ("320k_scFFPE_16-plex_GEM-X_FLEX_LNReactive_BC9-10",),
    "lung":         ("320k_scFFPE_16-plex_GEM-X_FLEX_LungCancer2_BC5-6",),
    "skin":         ("320k_scFFPE_16-plex_GEM-X_FLEX_SkinMelanoma_BC13-14",),
}


def download_file(url: str, dest: str):
    """Download a file if it doesn't exist yet."""
    if os.path.exists(dest):
        print(f"  [skip] {os.path.basename(dest)} already exists")
        return
    print(f"  [download] {os.path.basename(dest)} ...")
    subprocess.run(
        ["curl", "-sL", "-o", dest, url],
        check=True,
    )
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  [done] {size_mb:.1f} MB")


def download_tissue(tissue: str, data_dir: str):
    """Download filtered matrix H5 and analysis tar.gz for a tissue."""
    sample = TISSUES[tissue][0]
    prefix = f"{BASE}/{sample}/{sample}"

    os.makedirs(data_dir, exist_ok=True)

    h5_path = os.path.join(data_dir, f"{tissue}_filtered.h5")
    analysis_path = os.path.join(data_dir, f"{tissue}_analysis.tar.gz")

    download_file(f"{prefix}_count_sample_filtered_feature_bc_matrix.h5", h5_path)
    download_file(f"{prefix}_count_analysis.tar.gz", analysis_path)

    return h5_path, analysis_path


def extract_cellranger_results(analysis_tar_path: str, data_dir: str, tissue: str):
    """Extract 10x Cell Ranger clustering, PCA, and UMAP from analysis tar.gz."""
    out_dir = os.path.join(data_dir, f"{tissue}_analysis")
    if os.path.exists(out_dir):
        print(f"  [skip] {tissue}_analysis/ already extracted")
        return out_dir

    print(f"  [extract] {os.path.basename(analysis_tar_path)} ...")
    with tarfile.open(analysis_tar_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    return out_dir


def load_cellranger_clusters(analysis_dir: str):
    """Load Cell Ranger graph-based clustering results."""
    # Try graphclust first, then kmeans
    cluster_path = None
    for root, dirs, files in os.walk(analysis_dir):
        if "clusters.csv" in files and "graphclust" in root:
            cluster_path = os.path.join(root, "clusters.csv")
            break

    if cluster_path is None:
        # Fallback: any clusters.csv
        for root, dirs, files in os.walk(analysis_dir):
            if "clusters.csv" in files:
                cluster_path = os.path.join(root, "clusters.csv")
                break

    if cluster_path is None:
        print("  [warn] No Cell Ranger clusters found")
        return None

    clusters = {}
    with open(cluster_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            barcode = row["Barcode"]
            cluster = int(row["Cluster"])
            clusters[barcode] = cluster

    labels = np.array([clusters[bc] for bc in sorted(clusters.keys())])
    print(f"  [cellranger] {len(labels)} cells, {len(set(labels))} clusters")
    return labels


def load_cellranger_pca(analysis_dir: str):
    """Load Cell Ranger PCA projection."""
    pca_path = None
    for root, dirs, files in os.walk(analysis_dir):
        if "projection.csv" in files and "pca" in root:
            pca_path = os.path.join(root, "projection.csv")
            break

    if pca_path is None:
        print("  [warn] No Cell Ranger PCA found")
        return None

    rows = []
    with open(pca_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [float(v) for k, v in row.items() if k != "Barcode"]
            rows.append(vals)

    pca = np.array(rows, dtype=np.float32)
    print(f"  [cellranger] PCA: {pca.shape}")
    return pca


def load_cellranger_umap(analysis_dir: str):
    """Load Cell Ranger UMAP projection."""
    umap_path = None
    for root, dirs, files in os.walk(analysis_dir):
        if "projection.csv" in files and "umap" in root:
            umap_path = os.path.join(root, "projection.csv")
            break

    if umap_path is None:
        print("  [warn] No Cell Ranger UMAP found")
        return None

    rows = []
    with open(umap_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [float(v) for k, v in row.items() if k != "Barcode"]
            rows.append(vals)

    umap = np.array(rows, dtype=np.float32)
    print(f"  [cellranger] UMAP: {umap.shape}")
    return umap


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def adjusted_rand_index(labels_true, labels_pred):
    """Compute ARI between two clusterings."""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels_true, labels_pred)


def normalized_mutual_info(labels_true, labels_pred):
    """Compute NMI between two clusterings."""
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(labels_true, labels_pred)


def pca_cosine_similarity(pca_a, pca_b, n_comps=10):
    """Compute mean absolute cosine similarity for top n_comps PCs (sign-invariant)."""
    n = min(pca_a.shape[1], pca_b.shape[1], n_comps)
    sims = []
    for i in range(n):
        a = pca_a[:, i]
        b = pca_b[:, i]
        cos = np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        sims.append(cos)
    return np.mean(sims), sims


# ---------------------------------------------------------------------------
# Scanpy pipeline
# ---------------------------------------------------------------------------

def run_scanpy_pipeline(h5_path: str, n_comps=50, n_neighbors=15, resolution=1.0):
    """Run standard scanpy pipeline: load → preprocess → PCA → neighbors → leiden → UMAP."""
    import scanpy as sc

    timings = {}

    t0 = time.perf_counter()
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    timings["io"] = time.perf_counter() - t0

    print(f"  [scanpy] Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # QC + filter
    t0 = time.perf_counter()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    timings["filter"] = time.perf_counter() - t0

    # Normalize
    t0 = time.perf_counter()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    timings["normalize"] = time.perf_counter() - t0

    # HVG
    t0 = time.perf_counter()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    timings["hvg"] = time.perf_counter() - t0

    print(f"  [scanpy] After filter/HVG: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # PCA
    t0 = time.perf_counter()
    sc.tl.pca(adata, n_comps=n_comps)
    timings["pca"] = time.perf_counter() - t0

    # Neighbors
    t0 = time.perf_counter()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    timings["neighbors"] = time.perf_counter() - t0

    # Leiden
    t0 = time.perf_counter()
    sc.tl.leiden(adata, resolution=resolution)
    timings["leiden"] = time.perf_counter() - t0

    # UMAP
    t0 = time.perf_counter()
    sc.tl.umap(adata)
    timings["umap"] = time.perf_counter() - t0

    n_clusters = len(adata.obs["leiden"].unique())
    print(f"  [scanpy] {n_clusters} Leiden clusters")

    return {
        "adata": adata,
        "pca": adata.obsm["X_pca"],
        "leiden": adata.obs["leiden"].astype(int).values,
        "umap": adata.obsm["X_umap"],
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# CREST pipeline
# ---------------------------------------------------------------------------

def run_crest_pipeline(h5_path: str, n_comps=50, n_neighbors=15, resolution=1.0):
    """Run CREST pipeline: load H5 → COO triplets → SVD → Leiden → UMAP."""
    import polars as pl
    import crest

    timings = {}

    # Load H5 to COO triplets
    t0 = time.perf_counter()
    from crest.io import read_10x_h5
    triplets = read_10x_h5(h5_path)
    timings["io"] = time.perf_counter() - t0

    n_cells_raw = triplets["cell_id"].n_unique()
    n_genes_raw = triplets["gene_id"].n_unique()
    print(f"  [crest] Loaded: {n_cells_raw} cells x {n_genes_raw} genes ({len(triplets)} triplets)")

    # QC: filter cells with <200 genes
    t0 = time.perf_counter()
    genes_per_cell = triplets.group_by("cell_id").agg(pl.col("gene_id").n_unique().alias("n_genes"))
    valid_cells = genes_per_cell.filter(pl.col("n_genes") >= 200)["cell_id"]
    triplets = triplets.filter(pl.col("cell_id").is_in(valid_cells))

    # Filter genes expressed in <3 cells
    cells_per_gene = triplets.group_by("gene_id").agg(pl.col("cell_id").n_unique().alias("n_cells"))
    valid_genes = cells_per_gene.filter(pl.col("n_cells") >= 3)["gene_id"]
    triplets = triplets.filter(pl.col("gene_id").is_in(valid_genes))
    timings["filter"] = time.perf_counter() - t0

    # Normalize (CPM to 10k + log1p)
    t0 = time.perf_counter()
    result = triplets.with_columns(
        pl.col("count").bio.normalize_cpm(pl.col("cell_id"), target_sum=10000).alias("count")
    )
    result = result.with_columns(
        pl.col("count").bio.log1p().alias("count")
    )
    timings["normalize"] = time.perf_counter() - t0

    # HVG selection (top 2000 by variance)
    t0 = time.perf_counter()
    gene_stats = result.group_by("gene_id").agg([
        pl.col("count").var().alias("variance"),
        pl.col("count").mean().alias("mean"),
    ])
    top_genes = gene_stats.sort("variance", descending=True).head(2000)["gene_id"]
    result = result.filter(pl.col("gene_id").is_in(top_genes))
    timings["hvg"] = time.perf_counter() - t0

    # Remap gene_id and cell_id to dense contiguous indices
    unique_cells = result["cell_id"].unique().sort()
    unique_genes = result["gene_id"].unique().sort()
    cell_map = pl.DataFrame({
        "cell_id": unique_cells,
        "cell_idx": pl.Series(range(len(unique_cells)), dtype=pl.UInt32),
    })
    gene_map = pl.DataFrame({
        "gene_id": unique_genes,
        "gene_idx": pl.Series(range(len(unique_genes)), dtype=pl.UInt32),
    })
    result = result.join(cell_map, on="cell_id").join(gene_map, on="gene_id")

    n_cells = len(unique_cells)
    n_genes = len(unique_genes)
    print(f"  [crest] After filter/HVG: {n_cells} cells x {n_genes} genes")

    # SVD (PCA)
    t0 = time.perf_counter()
    agg_pca = result.select(
        pl.col("cell_idx").bio.svd(
            pl.col("gene_idx"), pl.col("count"),
            n_cells=n_cells, n_genes=n_genes, n_comps=n_comps
        ).alias("pca")
    )
    # Unpack 1-row aggregate to multi-row DataFrame
    pca_inner = agg_pca["pca"][0]
    pca_df = pl.DataFrame({"pca": pca_inner})
    timings["pca"] = time.perf_counter() - t0

    # Extract PCA as numpy for comparison
    pca_np = np.array(pca_df["pca"].to_list(), dtype=np.float32)
    print(f"  [crest] PCA: {pca_np.shape}")

    # Leiden clustering
    t0 = time.perf_counter()
    leiden_result = pca_df.select(
        pl.col("pca").bio.leiden(n_neighbors=n_neighbors, resolution=resolution).alias("leiden")
    )
    leiden_labels = np.array(leiden_result["leiden"][0].to_list(), dtype=np.uint32)
    timings["leiden"] = time.perf_counter() - t0

    n_clusters = len(set(leiden_labels))
    print(f"  [crest] {n_clusters} Leiden clusters")

    # UMAP
    t0 = time.perf_counter()
    umap_result = pca_df.select(
        pl.col("pca").bio.umap(n_components=2, n_neighbors=n_neighbors).alias("umap")
    )
    umap_inner = umap_result["umap"][0]
    umap_np = np.array(pl.DataFrame({"umap": umap_inner})["umap"].to_list(), dtype=np.float32)
    timings["umap"] = time.perf_counter() - t0

    return {
        "pca": pca_np,
        "leiden": leiden_labels,
        "umap": umap_np,
        "timings": timings,
        "n_cells": n_cells,
        "cell_map": cell_map,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(scanpy_res, crest_res, cellranger_clusters, cellranger_pca):
    """Compare all three pipelines and print results."""
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)

    # --- Timing comparison ---
    print("\n--- Timing (seconds) ---")
    print(f"{'Stage':<15} {'Scanpy':>10} {'CREST':>10} {'Speedup':>10}")
    print("-" * 50)
    total_scanpy = 0
    total_crest = 0
    for stage in ["io", "filter", "normalize", "hvg", "pca", "neighbors", "leiden", "umap"]:
        s_time = scanpy_res["timings"].get(stage, 0)
        c_time = crest_res["timings"].get(stage, 0)
        total_scanpy += s_time
        total_crest += c_time
        speedup = s_time / c_time if c_time > 0 else float("inf")
        crest_str = f"{c_time:.3f}" if c_time > 0 else "N/A"
        print(f"{stage:<15} {s_time:>10.3f} {crest_str:>10} {speedup:>9.1f}x")
    speedup_total = total_scanpy / total_crest if total_crest > 0 else float("inf")
    print(f"{'TOTAL':<15} {total_scanpy:>10.3f} {total_crest:>10.3f} {speedup_total:>9.1f}x")

    # --- PCA comparison ---
    print("\n--- PCA Cosine Similarity (sign-invariant) ---")
    # Compare CREST vs Scanpy
    n_shared = min(scanpy_res["pca"].shape[0], crest_res["pca"].shape[0])
    if n_shared > 0:
        mean_cos, per_pc = pca_cosine_similarity(
            scanpy_res["pca"][:n_shared],
            crest_res["pca"][:n_shared],
            n_comps=10,
        )
        print(f"CREST vs Scanpy (top 10 PCs): mean |cos| = {mean_cos:.4f}")
        for i, c in enumerate(per_pc[:5]):
            print(f"  PC{i+1}: {c:.4f}")

    # Compare Scanpy vs CellRanger PCA
    if cellranger_pca is not None:
        n_cr = min(scanpy_res["pca"].shape[0], cellranger_pca.shape[0])
        n_comps_cr = min(cellranger_pca.shape[1], 10)
        mean_cos_cr, _ = pca_cosine_similarity(
            scanpy_res["pca"][:n_cr, :n_comps_cr],
            cellranger_pca[:n_cr, :n_comps_cr],
            n_comps=n_comps_cr,
        )
        print(f"Scanpy vs CellRanger (top {n_comps_cr} PCs): mean |cos| = {mean_cos_cr:.4f}")

    # --- Clustering comparison ---
    print("\n--- Clustering (Leiden) ---")
    scanpy_leiden = scanpy_res["leiden"]
    crest_leiden = crest_res["leiden"]

    # CREST may have different cell ordering — use the min length
    n_compare = min(len(scanpy_leiden), len(crest_leiden))

    print(f"Scanpy clusters: {len(set(scanpy_leiden))}")
    print(f"CREST clusters:  {len(set(crest_leiden))}")

    if n_compare > 0:
        ari = adjusted_rand_index(scanpy_leiden[:n_compare], crest_leiden[:n_compare])
        nmi = normalized_mutual_info(scanpy_leiden[:n_compare], crest_leiden[:n_compare])
        print(f"CREST vs Scanpy:  ARI = {ari:.4f}, NMI = {nmi:.4f}")

    # Compare vs Cell Ranger
    if cellranger_clusters is not None:
        n_cr = min(len(scanpy_leiden), len(cellranger_clusters))
        if n_cr > 0:
            ari_cr = adjusted_rand_index(cellranger_clusters[:n_cr], scanpy_leiden[:n_cr])
            nmi_cr = normalized_mutual_info(cellranger_clusters[:n_cr], scanpy_leiden[:n_cr])
            print(f"Scanpy vs CellRanger: ARI = {ari_cr:.4f}, NMI = {nmi_cr:.4f}")

        n_crest_cr = min(len(crest_leiden), len(cellranger_clusters))
        if n_crest_cr > 0:
            ari_crest_cr = adjusted_rand_index(cellranger_clusters[:n_crest_cr], crest_leiden[:n_crest_cr])
            nmi_crest_cr = normalized_mutual_info(cellranger_clusters[:n_crest_cr], crest_leiden[:n_crest_cr])
            print(f"CREST vs CellRanger:  ARI = {ari_crest_cr:.4f}, NMI = {nmi_crest_cr:.4f}")

    print(f"\nCellRanger clusters: {len(set(cellranger_clusters)) if cellranger_clusters is not None else 'N/A'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    thresholds = {
        "PCA cosine similarity ≥ 0.95": mean_cos >= 0.95 if n_shared > 0 else None,
        "ARI ≥ 0.5 (vs Scanpy)": ari >= 0.5 if n_compare > 0 else None,
        "Total speedup > 1x": speedup_total > 1.0,
    }
    for desc, passed in thresholds.items():
        status = "PASS" if passed else ("FAIL" if passed is False else "SKIP")
        print(f"  [{status}] {desc}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CREST Validation Benchmark")
    parser.add_argument("--tissue", default="kidney", choices=list(TISSUES.keys()),
                        help="Tissue to benchmark (default: kidney)")
    parser.add_argument("--data-dir", default="bench/data/flex_320k",
                        help="Directory to store downloaded data")
    parser.add_argument("--n-comps", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--n-neighbors", type=int, default=15, help="Number of neighbors")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution")
    parser.add_argument("--multi", action="store_true",
                        help="Run multi-tissue integration benchmark")
    args = parser.parse_args()

    print("=" * 70)
    print("  CREST Validation Benchmark: 10x GEM-X FLEX 320K FFPE")
    print("=" * 70)

    if args.multi:
        # Multi-tissue: Glioblastoma + Kidney + LymphNode ≈ 98K cells
        tissues = ["glioblastoma", "kidney", "lymphnode"]
        print(f"\nMulti-tissue mode: {', '.join(tissues)}")
        print("NOTE: Integration benchmark requires Harmony (not yet implemented)")
        sys.exit(1)

    tissue = args.tissue
    print(f"\nTissue: {tissue}")
    print(f"Parameters: n_comps={args.n_comps}, n_neighbors={args.n_neighbors}, resolution={args.resolution}")

    # 1. Download data
    print(f"\n--- Step 1: Download {tissue} data ---")
    h5_path, analysis_path = download_tissue(tissue, args.data_dir)

    # 2. Extract Cell Ranger reference results
    print(f"\n--- Step 2: Extract Cell Ranger analysis ---")
    analysis_dir = extract_cellranger_results(analysis_path, args.data_dir, tissue)
    cr_clusters = load_cellranger_clusters(analysis_dir)
    cr_pca = load_cellranger_pca(analysis_dir)
    cr_umap = load_cellranger_umap(analysis_dir)

    # 3. Run Scanpy pipeline
    print(f"\n--- Step 3: Run Scanpy pipeline ---")
    scanpy_res = run_scanpy_pipeline(
        h5_path,
        n_comps=args.n_comps,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
    )

    # 4. Run CREST pipeline
    print(f"\n--- Step 4: Run CREST pipeline ---")
    crest_res = run_crest_pipeline(
        h5_path,
        n_comps=args.n_comps,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
    )

    # 5. Compare
    compare_results(scanpy_res, crest_res, cr_clusters, cr_pca)

    # 6. Save results
    results_dir = os.path.join(args.data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{tissue}_results.csv")

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for stage, t in scanpy_res["timings"].items():
            writer.writerow([f"scanpy_{stage}_sec", f"{t:.4f}"])
        for stage, t in crest_res["timings"].items():
            writer.writerow([f"crest_{stage}_sec", f"{t:.4f}"])
        writer.writerow(["scanpy_n_clusters", len(set(scanpy_res["leiden"]))])
        writer.writerow(["crest_n_clusters", len(set(crest_res["leiden"]))])
        if cr_clusters is not None:
            writer.writerow(["cellranger_n_clusters", len(set(cr_clusters))])

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
