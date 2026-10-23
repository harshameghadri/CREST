"""Leiden modularity: CREST vs leidenalg over many seeds on the same graph.

    python bench/paper/modularity_sweep.py --data bench_data/pbmc68k.h5 --out RESULTS/modules/modularity \
        [--seeds 20] [--resolution 1.0] [--n-iterations 2] [--graph crest|scanpy]

This is the check behind the docs' "mean +0.0015 over 20 seeds at resolution 1 on PBMC 68k".
The graph is built once (default: CREST's pipeline, the same parameters as the paper
benchmark: QC filter, normalize_total + log1p, 2,000 `seurat` HVGs, scale, 50 PCs, 15-NN UMAP
connectivities; ``--graph scanpy`` uses scanpy's connectivities instead). On that one graph,
for each seed 0..S-1:

* CREST ``leiden`` (RB configuration / modularity, ``--n-iterations``),
* leidenalg ``find_partition(RBConfigurationVertexPartition, n_iterations=..., seed=...)``,

and both partitions are scored by igraph's ``Graph.modularity`` (weighted, same
resolution), an implementation independent of both. Reports per seed and overall: mean
and median modularity gap (CREST - leidenalg), how many seeds CREST is below, cluster
counts, ARI between the two, and wall times. Writes ``modularity_sweep.{csv,json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_one  # noqa: E402  (shared pipeline parameters)


def crest_graph(path: Path, min_genes: int):
    import crest
    b = crest.read_10x_h5(path)
    b = crest.pp.filter_cells(b, min_genes=min_genes)
    b = crest.pp.filter_genes(b, min_cells=3)
    crest.pp.normalize_total(b, run_one.NORMALIZE_TARGET_SUM)
    crest.pp.log1p(b)
    crest.pp.highly_variable_genes(b, n_top_genes=run_one.N_HVG, flavor="seurat")
    crest.pp.scale(b, max_value=run_one.SCALE_MAX_VALUE)
    crest.tl.pca(b, n_comps=run_one.N_PCS)
    crest.pp.neighbors(b, n_neighbors=run_one.N_NEIGHBORS)
    return crest.pp.connectivities_matrix(b)


def scanpy_graph(path: Path, min_genes: int):
    import scanpy as sc
    a = sc.read_10x_h5(path)
    a.var_names_make_unique()
    sc.pp.filter_cells(a, min_genes=min_genes)
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=run_one.NORMALIZE_TARGET_SUM)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=run_one.N_HVG, flavor="seurat")
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=run_one.SCALE_MAX_VALUE)
    sc.tl.pca(a, n_comps=run_one.N_PCS, svd_solver="arpack")
    sc.pp.neighbors(a, n_neighbors=run_one.N_NEIGHBORS, n_pcs=run_one.N_PCS, random_state=0)
    return a.obsp["connectivities"].tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="10x .h5 (as written by datasets.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--n-iterations", type=int, default=2,
                    help="Leiden iterations for both (CREST's default 2; leidenalg's default is 2 as well)")
    ap.add_argument("--graph", choices=["crest", "scanpy"], default="crest")
    a = ap.parse_args()
    import igraph as ig
    import leidenalg
    import scipy.sparse as sp
    from sklearn.metrics import adjusted_rand_score
    from crest import crest as native

    path, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = run_one._meta(path.stem, path)
    t0 = time.perf_counter()
    C = (crest_graph if a.graph == "crest" else scanpy_graph)(path, meta.get("min_genes", 200))
    t_graph = time.perf_counter() - t0
    U = sp.triu(C, 1).tocoo()
    rows, cols, w = U.row.astype(np.uint32), U.col.astype(np.uint32), U.data.astype(np.float32)
    g = ig.Graph.Weighted_Adjacency(sp.triu(C, 1).tocsr(), mode="undirected")
    weights = g.es["weight"]
    n = C.shape[0]
    print(f"{path.stem}: {n} cells, {g.ecount()} edges ({a.graph} graph, {t_graph:.1f} s)")

    recs = []
    for s in range(a.seeds):
        t0 = time.perf_counter()
        lc = np.asarray(native.leiden(n, rows, cols, w, a.resolution, a.n_iterations, s)).astype(np.int64)
        tc = time.perf_counter() - t0
        t0 = time.perf_counter()
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=weights,
                                        resolution_parameter=a.resolution, n_iterations=a.n_iterations, seed=s)
        tl = time.perf_counter() - t0
        ll = np.asarray(part.membership, np.int64)
        qc = g.modularity(lc.tolist(), weights=weights, resolution=a.resolution)
        ql = g.modularity(ll.tolist(), weights=weights, resolution=a.resolution)
        recs.append({"seed": s, "modularity_crest": qc, "modularity_leidenalg": ql, "gap": qc - ql,
                     "n_clusters_crest": int(len(np.unique(lc))), "n_clusters_leidenalg": int(len(np.unique(ll))),
                     "ari": float(adjusted_rand_score(lc, ll)), "time_crest_s": tc, "time_leidenalg_s": tl})
        print(recs[-1], flush=True)
    df = pl.DataFrame(recs)
    gap = df["gap"].to_numpy()
    summary = {"dataset": path.stem, "graph": a.graph, "n_cells": int(n), "n_edges": int(g.ecount()),
               "resolution": a.resolution, "n_iterations": a.n_iterations, "seeds": a.seeds,
               "gap_mean": float(gap.mean()), "gap_median": float(np.median(gap)), "gap_min": float(gap.min()),
               "gap_max": float(gap.max()), "seeds_crest_below": int((gap < 0).sum()),
               "modularity_crest_mean": float(df["modularity_crest"].mean()),
               "modularity_leidenalg_mean": float(df["modularity_leidenalg"].mean()),
               "sd_crest": float(df["modularity_crest"].std()), "sd_leidenalg": float(df["modularity_leidenalg"].std()),
               "ari_median": float(df["ari"].median()), "time_crest_median_s": float(df["time_crest_s"].median()),
               "time_leidenalg_median_s": float(df["time_leidenalg_s"].median())}
    df.write_csv(out / "modularity_sweep.csv")
    (out / "modularity_sweep.json").write_text(json.dumps({"summary": summary, "per_seed": recs}, indent=1))
    with open(out / "modularity_sweep.md", "w") as fh:
        fh.write(f"Leiden modularity, CREST vs leidenalg (RBConfigurationVertexPartition), {path.stem}: {n:,} cells, "
                 f"{a.graph}'s 15-NN connectivities, resolution {a.resolution:g}, n_iterations {a.n_iterations}, "
                 f"seeds 0-{a.seeds - 1}; modularity scored by igraph `Graph.modularity`.\n\n")
        fh.write(f"Mean gap (CREST - leidenalg) **{summary['gap_mean']:+.5f}** (median {summary['gap_median']:+.5f}, "
                 f"range {summary['gap_min']:+.5f} to {summary['gap_max']:+.5f}); CREST below on "
                 f"**{summary['seeds_crest_below']}/{a.seeds}** seeds. Seed-to-seed SD: CREST "
                 f"{summary['sd_crest']:.5f}, leidenalg {summary['sd_leidenalg']:.5f}. Median ARI between the two "
                 f"{summary['ari_median']:.3f}. Median time: CREST {summary['time_crest_median_s']:.2f} s, "
                 f"leidenalg {summary['time_leidenalg_median_s']:.2f} s.\n\n")
        fh.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for row in df.iter_rows():
            fh.write("| " + " | ".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in row) + " |\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
