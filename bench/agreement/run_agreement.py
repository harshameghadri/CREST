"""Agreement run: CREST against scanpy on the same data, untimed.

The metrics, thresholds and the action on failure of each metric are fixed in
``bench/agreement/THRESHOLDS.md``; this script implements them and changes none of them.

    python bench/agreement/run_agreement.py scanpy  --dataset kang --data-dir DATA --out OUT
    python bench/agreement/run_agreement.py crest   --dataset kang --data-dir DATA --out OUT
    python bench/agreement/run_agreement.py compare --dataset kang --data-dir DATA --out OUT
    python bench/agreement/run_agreement.py exact   --dataset kang --data-dir DATA --out OUT   # optional diagnostic

``scanpy`` and ``crest`` write the per-tool arrays of one dataset under ``OUT/<dataset>/`` (every
sub-stage is skipped when its file exists, so an interrupted run resumes); ``compare`` reads them and
writes ``results.json`` and ``results.md`` with, for every metric, value, threshold and PASS / FAIL.
``--mode knn`` stops after the neighbour graphs (datasets that only enter the kNN recall metric).
The ``crest`` stage must run after the ``scanpy`` stage (it reads scanpy's PCA).

Arms (see THRESHOLDS.md, section 2). A: scanpy PCA, kNN, Leiden. D: CREST PCA, kNN, Leiden.
B: scanpy's connectivities clustered by CREST's Leiden. C: scanpy's PCA, CREST's kNN and Leiden.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# bench/paper only: the repository root must not be put in front of the installed package
sys.path.insert(0, str(HERE.parents[0] / "paper"))

# --------------------------------------------------------------------------- registered constants
# Every value below is fixed in THRESHOLDS.md; changing one needs an entry in its Deviations section.
N_SEEDS = 10                   # seeds per tool (0..N_SEEDS-1)
MOD_SEEDS_SMALL, MOD_SEEDS_LARGE = 20, 10
SMALL_MAX_CELLS = 100_000      # "small" datasets: more modularity / UMAP / reference-DE seeds
UMAP_SEEDS_SMALL, UMAP_SEEDS_LARGE = 3, 1
DE_REF_SEEDS_SMALL = 2         # scanpy's own seeds 1, 2 against seed 0 (context only)
RESOLUTION, N_ITER = 1.0, 2
N_NEIGHBORS = 15               # 14 neighbours per cell, self excluded
K = N_NEIGHBORS - 1

ARI_FLOOR_ANNOT = 0.01         # metric 1
ARI_FLOOR_SEEDS = 0.02         # metric 2
MOD_TOL = 0.003                # metric 3
DE_MIN_JACCARD_PAIR = 0.6      # metric 4: cluster pairing
DE_MIN_PAIRS = 3
DE_TOP = 50
DE_SIG_PADJ, DE_SIG_LFC = 0.05, 0.5
DE_OWN_MIN = {"spearman_scores": 0.90, "spearman_lfc": 0.90, "top50_jaccard": 0.50, "sig_jaccard": 0.70}
DE_SAME_MIN = {"spearman_scores": 0.99, "spearman_lfc": 0.99, "top50_jaccard": 0.80, "sig_jaccard": 0.90}
RECALL_QUERIES = 10_000
RECALL_MIN_QUERIES = 5_000
RECALL_MEAN_MIN = 0.99
RECALL_FRAC_BELOW_05_MAX = 0.005
RECALL_FRAC_BELOW_08_MAX = 0.010
TRUST_K = 15
TRUST_QUERIES = 5_000
TRUST_TOL = 0.01
GLOBAL_TOL = 0.10
GLOBAL_MIN_CLUSTER = 100


# =========================================================================== metric functions (pure)
def pair_aris(runs_a, runs_b=None) -> np.ndarray:
    """ARI over all pairs of partitions: the pairs i < j of ``runs_a`` (within a set of runs) or
    every (i, j) of ``runs_a`` x ``runs_b`` (across two sets). Returns a 1-d array."""
    from sklearn.metrics import adjusted_rand_score
    if runs_b is None:
        return np.array([adjusted_rand_score(runs_a[i], runs_a[j])
                         for i in range(len(runs_a)) for j in range(i + 1, len(runs_a))])
    return np.array([adjusted_rand_score(x, y) for x in runs_a for y in runs_b])


def seed_floor(within: np.ndarray, floor: float = ARI_FLOOR_SEEDS) -> float:
    """Lower bound ``L = min(min(W), mean(W) - floor)`` of an acceptable cross-tool mean ARI."""
    w = np.asarray(within, float)
    return float(min(w.min(), w.mean() - floor))


def eval_cross_ari(cross: np.ndarray, within: np.ndarray) -> dict:
    lb = seed_floor(within)
    m = float(np.mean(cross))
    return {"cross_mean": m, "within_mean": float(np.mean(within)), "within_min": float(np.min(within)),
            "within_sd": float(np.std(within, ddof=1)) if len(within) > 1 else float("nan"),
            "threshold": lb, "pass": bool(m >= lb)}


def eval_annotation(mean_d: float, vals_a, floor: float = ARI_FLOOR_ANNOT) -> dict:
    """Metric 1: ``|mean(D) - mean(A)| <= max(range(A), floor)``."""
    a = np.asarray(vals_a, float)
    rng = float(a.max() - a.min())
    tol = max(rng, floor)
    diff = float(mean_d - a.mean())
    return {"mean_crest": float(mean_d), "mean_scanpy": float(a.mean()), "range_scanpy": rng,
            "diff": diff, "threshold": tol, "pass": bool(abs(diff) <= tol)}


def modularity(graph, labels) -> float:
    """Weighted modularity (resolution 1) by igraph. ``graph`` is an ``igraph.Graph`` with a ``weight`` edge attribute."""
    return float(graph.modularity(np.asarray(labels).astype(int).tolist(), weights=graph.es["weight"], resolution=1.0))


def igraph_from_connectivities(C):
    """Undirected weighted igraph graph of a symmetric scipy connectivity matrix (each edge once, weight C_ij)."""
    import igraph as ig
    import scipy.sparse as sp
    return ig.Graph.Weighted_Adjacency(sp.triu(C, 1).tocsr(), mode="undirected")


def eval_modularity(q_crest, q_ref, tol: float = MOD_TOL) -> dict:
    gap = float(np.mean(q_crest) - np.mean(q_ref))
    return {"mean_q_crest": float(np.mean(q_crest)), "mean_q_reference": float(np.mean(q_ref)), "gap": gap,
            "threshold": -tol, "pass": bool(gap >= -tol), "n_seeds": int(len(q_crest))}


def match_clusters(la, lb, min_jaccard: float = DE_MIN_JACCARD_PAIR):
    """One-to-one pairing of two partitions of the same cells by maximal total Jaccard index of the
    cell sets (Hungarian assignment). Returns (pairs, jaccards, unpaired_a, unpaired_b), where pairs
    are (label_a, label_b) with Jaccard >= ``min_jaccard``; unpaired lists hold the other labels."""
    from scipy.optimize import linear_sum_assignment
    ua, ia = np.unique(la, return_inverse=True)
    ub, ib = np.unique(lb, return_inverse=True)
    M = np.zeros((len(ua), len(ub)), np.int64)
    np.add.at(M, (ia, ib), 1)
    sa, sb = M.sum(1), M.sum(0)
    J = M / (sa[:, None] + sb[None, :] - M)
    r, c = linear_sum_assignment(-J)
    pairs, jac = [], []
    for i, j in zip(r, c):
        if J[i, j] >= min_jaccard:
            pairs.append((ua[i], ub[j]))
            jac.append(float(J[i, j]))
    used_a, used_b = {p[0] for p in pairs}, {p[1] for p in pairs}
    return pairs, jac, [x for x in ua if x not in used_a], [x for x in ub if x not in used_b]


def _jaccard_sets(x: np.ndarray, y: np.ndarray) -> float:
    u = int((x | y).sum())
    return float((x & y).sum() / u) if u else float("nan")


def de_group_metrics(sa, la, pa, sb, lb, pb) -> dict:
    """Agreement of one group's Wilcoxon results of two tools over the same genes (arrays aligned by gene):
    Spearman correlation of scores and of log2 fold changes (all genes), Jaccard index of the top-50 genes by score
    and of the significant sets (adjusted p < 0.05, log2 fold change > 0.5, score > 0)."""
    from scipy.stats import spearmanr
    sa, sb = np.nan_to_num(np.asarray(sa, float)), np.nan_to_num(np.asarray(sb, float))
    la, lb = np.nan_to_num(np.asarray(la, float)), np.nan_to_num(np.asarray(lb, float))
    top = lambda s: np.isin(np.arange(len(s)), np.argsort(-s, kind="stable")[:DE_TOP])  # noqa: E731
    sig = lambda s, l, p: (np.asarray(p) < DE_SIG_PADJ) & (l > DE_SIG_LFC) & (s > 0)  # noqa: E731
    return {"spearman_scores": float(spearmanr(sa, sb).statistic), "spearman_lfc": float(spearmanr(la, lb).statistic),
            "top50_jaccard": _jaccard_sets(top(sa), top(sb)),
            "sig_jaccard": _jaccard_sets(sig(sa, la, pa), sig(sb, lb, pb)),
            "n_sig_a": int(sig(sa, la, pa).sum()), "n_sig_b": int(sig(sb, lb, pb).sum())}


def eval_de(rows: list[dict], minimum: dict, n_pairs_required: int = DE_MIN_PAIRS) -> dict:
    """Median over groups of each DE agreement value against ``minimum``. Fewer than ``n_pairs_required``
    groups is inconclusive (pass is None). NaN values (both sets empty) are left out of that value's median."""
    if len(rows) < n_pairs_required:
        return {"n_groups": len(rows), "pass": None, "inconclusive": True, "median": {}, "min": {}, "threshold": minimum}
    med, mn, ok = {}, {}, True
    for key, thr in minimum.items():
        v = np.array([r[key] for r in rows], float)
        v = v[np.isfinite(v)]
        med[key] = float(np.median(v)) if len(v) else float("nan")
        mn[key] = float(v.min()) if len(v) else float("nan")
        ok &= bool(len(v) and med[key] >= thr)
    return {"n_groups": len(rows), "pass": bool(ok), "inconclusive": False, "median": med, "min": mn, "threshold": minimum}


def recall_per_query(X_ref, Q, nbr, k, self_idx=None, block_entries: int = 200_000_000):
    """Recall of neighbour lists against exact brute force, per query, in float64.

    ``X_ref`` (n, d): the points searched; ``Q`` (m, d): the queries; ``nbr`` (m, >=k) indices into ``X_ref`` (-1 pads);
    ``self_idx`` (m,): when the queries are rows of ``X_ref``, the row of each query, which is excluded from the
    search and must not be listed. A listed neighbour is correct when its distance is not larger than the exact
    k-th distance (ties at the boundary are not misses); duplicates in a list count once.
    Returns an array of m recalls (correct / k). Raises ``ValueError`` if a list holds its own query."""
    X = np.asarray(X_ref, np.float64)
    Q = np.asarray(Q, np.float64)
    n, m = X.shape[0], Q.shape[0]
    nbr = np.asarray(nbr)
    if nbr.shape[0] != m:
        raise ValueError("one neighbour list per query is required")
    if self_idx is not None and (nbr == np.asarray(self_idx)[:, None]).any():
        raise ValueError("a neighbour list holds the query cell itself (self must be excluded)")
    kk = min(k, n - (1 if self_idx is not None else 0))
    sq = np.einsum("ij,ij->i", X, X)
    out = np.empty(m)
    chunk = int(max(4, min(1024, block_entries // max(n, 1))))
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        q = Q[lo:hi]
        D = (q * q).sum(1)[:, None] - 2.0 * (q @ X.T) + sq[None, :]
        if self_idx is not None:
            D[np.arange(hi - lo), np.asarray(self_idx)[lo:hi]] = np.inf
        kth = np.partition(D, kk - 1, axis=1)[:, kk - 1]
        for i in range(hi - lo):
            row = np.unique(nbr[lo + i][nbr[lo + i] >= 0])
            hits = int((D[i, row] <= kth[i] * (1 + 1e-9) + 1e-12).sum()) if len(row) else 0
            out[lo + i] = min(hits, kk) / kk
    return out


def recall_summary(rec: np.ndarray) -> dict:
    return {"mean": float(np.mean(rec)), "frac_below_0.8": float(np.mean(rec < 0.8)),
            "frac_below_0.5": float(np.mean(rec < 0.5)), "n_query": int(len(rec))}


def eval_recall(summary: dict) -> dict:
    ok = (summary["n_query"] >= RECALL_MIN_QUERIES and summary["mean"] >= RECALL_MEAN_MIN
          and summary["frac_below_0.5"] <= RECALL_FRAC_BELOW_05_MAX and summary["frac_below_0.8"] <= RECALL_FRAC_BELOW_08_MAX)
    return {**summary, "threshold": {"mean_min": RECALL_MEAN_MIN, "frac_below_0.5_max": RECALL_FRAC_BELOW_05_MAX,
                                     "frac_below_0.8_max": RECALL_FRAC_BELOW_08_MAX, "n_query_min": RECALL_MIN_QUERIES},
            "pass": bool(ok)}


def trustworthiness_sampled(X_high, X_low, queries, k: int = TRUST_K, block_entries: int = 100_000_000) -> float:
    """Trustworthiness of the low-dimensional embedding ``X_low`` of ``X_high`` at ``k`` neighbours, neighbourhoods
    taken among all n cells, the rank-penalty sum estimated from the cells ``queries`` (rows). With every cell as a
    query it equals ``sklearn.manifold.trustworthiness``.
    T = 1 - 2 / (k (2n - 3k - 1)) * mean over queries of sum_{j in U_i} max(0, r(i, j) - k), with U_i the k nearest
    cells of i in the embedding and r(i, j) the rank of j among the others by distance in the original space."""
    Xh = np.asarray(X_high, np.float64)
    Xl = np.asarray(X_low, np.float64)
    n = Xh.shape[0]
    queries = np.asarray(queries)
    sh, sl = np.einsum("ij,ij->i", Xh, Xh), np.einsum("ij,ij->i", Xl, Xl)
    chunk = int(max(2, min(512, block_entries // max(n, 1))))
    pen = np.empty(len(queries))
    for lo in range(0, len(queries), chunk):
        q = queries[lo:lo + chunk]
        r = np.arange(len(q))
        Dh = sh[q][:, None] - 2.0 * (Xh[q] @ Xh.T) + sh[None, :]
        Dl = sl[q][:, None] - 2.0 * (Xl[q] @ Xl.T) + sl[None, :]
        Dh[r, q] = np.inf
        Dl[r, q] = np.inf
        U = np.argpartition(Dl, k - 1, axis=1)[:, :k]
        for i in range(len(q)):
            dj = Dh[i, U[i]]
            rank = (Dh[i][None, :] < dj[:, None]).sum(1) + 1
            pen[lo + i] = np.maximum(rank - k, 0).sum()
    return float(1.0 - 2.0 / (k * (2 * n - 3 * k - 1)) * pen.mean())


def centroid_distance_spearman(X_high, X_low, labels, min_cells: int = GLOBAL_MIN_CLUSTER) -> dict:
    """Spearman correlation between the pairwise distances of cluster centroids in two spaces (all cluster
    pairs of the clusters with at least ``min_cells`` cells)."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    lab = np.asarray(labels)
    keep = [c for c in np.unique(lab) if (lab == c).sum() >= min_cells]
    if len(keep) < 3:
        return {"rho": float("nan"), "n_clusters": len(keep)}
    ch = np.stack([np.asarray(X_high)[lab == c].mean(0) for c in keep])
    cl = np.stack([np.asarray(X_low)[lab == c].mean(0) for c in keep])
    return {"rho": float(spearmanr(pdist(ch), pdist(cl)).statistic), "n_clusters": len(keep)}


def eval_threshold_diff(crest_val: float, scanpy_val: float, tol: float) -> dict:
    d = float(crest_val - scanpy_val)
    return {"crest": float(crest_val), "scanpy": float(scanpy_val), "diff": d, "threshold": -tol, "pass": bool(d >= -tol)}


def jaccard_neighbor_rows(A: np.ndarray, B: np.ndarray) -> float:
    """Mean Jaccard index of the neighbour sets of two kNN graphs over the same cells (-1 = padding)."""
    out = []
    for x, y in zip(A, B):
        sx, sy = set(x[x >= 0].tolist()), set(y[y >= 0].tolist())
        u = len(sx | sy)
        out.append(len(sx & sy) / u if u else 1.0)
    return float(np.mean(out))


# =========================================================================== data and workflow helpers
def _meta(name: str, data_dir: Path) -> dict:
    import run_one
    return run_one._meta(name, data_dir / f"{name}.h5")


def _obs_table(name: str, data_dir: Path):
    import polars as pl
    p = data_dir / f"{name}.obs.parquet"
    return pl.read_parquet(p) if p.exists() else None


def _threads(n: int) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS",
              "POLARS_MAX_THREADS"):
        os.environ[v] = str(n)


def _versions() -> dict:
    from importlib import metadata
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for d in ("crest-sc", "scanpy", "anndata", "numpy", "scipy", "scikit-learn", "pynndescent", "umap-learn", "numba",
              "igraph", "leidenalg", "polars", "pandas", "h5py"):
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out


def _save(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _n_mod_seeds(n: int) -> int:
    return MOD_SEEDS_SMALL if n <= SMALL_MAX_CELLS else MOD_SEEDS_LARGE


def _n_umap_seeds(n: int) -> int:
    return UMAP_SEEDS_SMALL if n <= SMALL_MAX_CELLS else UMAP_SEEDS_LARGE


def _de_tables_scanpy(adata, key: str, genes: np.ndarray):
    """scanpy's Wilcoxon results as (groups, scores[G, V], lfc[G, V], padj[G, V]) aligned to ``genes``."""
    r = adata.uns["rank_genes_groups"]
    groups = list(r["names"].dtype.names)
    pos = {g: i for i, g in enumerate(genes)}
    S = np.full((len(groups), len(genes)), np.nan)
    L, P = S.copy(), S.copy()
    for gi, g in enumerate(groups):
        idx = np.array([pos[x] for x in r["names"][g]])
        S[gi, idx], L[gi, idx], P[gi, idx] = r["scores"][g], r["logfoldchanges"][g], r["pvals_adj"][g]
    return np.array(groups, dtype=str), S, L, P


def _de_tables_crest(table, genes: np.ndarray):
    """CREST's long Wilcoxon table (every gene per group) as (groups, scores, lfc, padj) aligned to ``genes``."""
    import polars as pl
    pos = {g: i for i, g in enumerate(genes)}
    groups = table["group"].unique(maintain_order=True).to_list()
    S = np.full((len(groups), len(genes)), np.nan)
    L, P = S.copy(), S.copy()
    for gi, g in enumerate(groups):
        t = table.filter(pl.col("group") == g)
        idx = np.array([pos[x] for x in t["names"].to_list()])
        S[gi, idx], L[gi, idx], P[gi, idx] = t["scores"].to_numpy(), t["logfoldchanges"].to_numpy(), t["pvals_adj"].to_numpy()
    return np.array(groups, dtype=str), S, L, P


_LG_GRAPH = {}


def _leidenalg_one(job):
    import leidenalg
    g = _LG_GRAPH["g"]
    s, n_iter = job
    p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=g.es["weight"],
                                 resolution_parameter=RESOLUTION, n_iterations=n_iter, seed=s)
    return np.asarray(p.membership)


def leidenalg_partitions(graph, seeds, n_iterations: int, workers: int = 1) -> np.ndarray:
    """leidenalg ``RBConfigurationVertexPartition`` partitions of ``graph`` (resolution 1), one per seed, as an array
    (seeds, n). Each seed is an independent single-threaded run; with ``workers > 1`` they run in forked processes that
    share the graph, and the partitions are the same as one by one."""
    jobs = [(int(s), n_iterations) for s in seeds]
    if workers <= 1 or len(jobs) == 1:
        _LG_GRAPH["g"] = graph
        return np.stack([_leidenalg_one(j) for j in jobs])
    import multiprocessing as mp
    _LG_GRAPH["g"] = graph
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        return np.stack(pool.map(_leidenalg_one, jobs))


def _lg_workers(n_cells: int) -> int:
    return 8 if n_cells > SMALL_MAX_CELLS else 1


from contextlib import contextmanager


@contextmanager
def _pca_in_place(enabled: bool = True):
    """Make scanpy's PCA centre the scaled matrix in place (scikit-learn's ``copy=False``) instead of on a copy.
    The scaled matrix is not used after the PCA, and the result is the same floating-point numbers; it only saves
    one dense copy (16 GB for 1,000,000 cells x 2,000 genes in float64)."""
    import sklearn.decomposition as dec
    original = dec.PCA

    class _InPlacePCA(original):
        def fit_transform(self, X, y=None):
            self.copy = False
            return super().fit_transform(X, y)

    if enabled:
        dec.PCA = _InPlacePCA
    try:
        yield
    finally:
        dec.PCA = original


# =========================================================================== stage: scanpy
def stage_scanpy(a) -> None:
    import anndata as ad
    import igraph  # noqa: F401
    import leidenalg
    import scanpy as sc
    import scipy.sparse as sp
    from crest import crest as native
    import run_one
    d = Path(a.out) / a.dataset
    d.mkdir(parents=True, exist_ok=True)
    data_dir = Path(a.data_dir)
    path = data_dir / f"{a.dataset}.h5"
    meta = _meta(a.dataset, data_dir)
    sc.settings.n_jobs = a.threads
    core = d / "scanpy_core.npz"
    _log(f"scanpy {sc.__version__}: reading {path}")
    adata = sc.read_10x_h5(path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=meta.get("min_genes", 200))
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=run_one.NORMALIZE_TARGET_SUM)
    sc.pp.log1p(adata)
    # log-normalised, all genes: what the differential expression tests (the workflow's ``use_raw=True``)
    lognorm = ad.AnnData(X=adata.X.copy(), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
    sc.pp.highly_variable_genes(adata, n_top_genes=run_one.N_HVG, flavor="seurat")
    hvg = np.asarray(adata.var_names[adata.var["highly_variable"].to_numpy()], dtype=str)
    genes = np.asarray(adata.var_names, dtype=str)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=run_one.SCALE_MAX_VALUE)
    with _pca_in_place(a.pca_in_place):
        sc.tl.pca(adata, n_comps=run_one.N_PCS, svd_solver="arpack")
    pca = np.asarray(adata.obsm["X_pca"], np.float32)
    barcodes = np.asarray(adata.obs_names, dtype=str)
    n = len(barcodes)
    del adata
    g = ad.AnnData(X=sp.csr_matrix((n, 1), dtype=np.float32), obs=lognorm.obs.copy())
    g.obsm["X_pca"] = pca
    _log(f"scanpy PCA done: {n} cells, {len(genes)} genes")
    seeds = list(range(N_SEEDS))
    nmod = _n_mod_seeds(n)
    knn_only = a.mode == "knn"

    def neighbors(s):
        # use_rep is explicit: the container's X has one column, which scanpy would otherwise search
        sc.pp.neighbors(g, n_neighbors=N_NEIGHBORS, n_pcs=run_one.N_PCS, use_rep="X_pca", random_state=s)

    def leiden(s, key="leiden"):
        sc.tl.leiden(g, resolution=RESOLUTION, flavor="igraph", n_iterations=N_ITER, random_state=s, key_added=key)
        return g.obs[key].astype(int).to_numpy()

    graph0 = d / "scanpy_graph0.npz"

    def seed0_graph():
        """The seed-0 graph, computed once and kept: a rerun uses the same graph (parallel neighbour search need not
        reproduce itself)."""
        if graph0.exists():
            z = np.load(graph0)
            g.obsp["distances"] = sp.csr_matrix((z["d_data"], z["d_indices"], z["d_indptr"]), shape=(n, n))
            g.obsp["connectivities"] = sp.csr_matrix((z["c_data"], z["c_indices"], z["c_indptr"]), shape=(n, n))
            g.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                                  "params": {"n_neighbors": N_NEIGHBORS, "method": "umap", "random_state": 0,
                                             "metric": "euclidean", "n_pcs": run_one.N_PCS}}
        else:
            neighbors(0)
            D_, C_ = g.obsp["distances"].tocsr(), g.obsp["connectivities"].tocsr()
            _save(graph0, d_data=D_.data, d_indices=D_.indices, d_indptr=D_.indptr,
                  c_data=C_.data, c_indices=C_.indices, c_indptr=C_.indptr)

    if not core.exists():
        t0 = time.time()
        seed0_graph()
        from accuracy import scanpy_knn_indices
        knn0 = scanpy_knn_indices(g.obsp["distances"], N_NEIGHBORS)
        out = {"barcodes": barcodes, "pca": pca, "hvg": hvg, "genes": genes, "knn0": knn0}
        _log(f"seed-0 graph in {time.time() - t0:.0f} s")
        if not knn_only:
            C = g.obsp["connectivities"].tocsr()
            ig_g = igraph_from_connectivities(C)
            U = sp.triu(C, 1).tocoo()
            r, c, w = U.row.astype(np.uint32), U.col.astype(np.uint32), U.data.astype(np.float32)
            parts = d / "scanpy_parts.npz"
            if parts.exists():
                z = np.load(parts)
                a_leiden, b, lg = z["A_leiden"], z["B"], z["lg"]
            else:
                a_leiden = np.stack([leiden(s) for s in range(nmod)])
                _log("A_leiden done")
                b = np.stack([np.asarray(native.leiden(n, r, c, w, RESOLUTION, N_ITER, s)).astype(np.int64) for s in range(nmod)])
                _log("B (CREST Leiden on scanpy's graph) done")
                lg = leidenalg_partitions(ig_g, range(nmod), N_ITER, _lg_workers(n))
                _save(parts, A_leiden=a_leiden, B=b, lg=lg)
                _log("leidenalg done")
            # leidenalg run to convergence (reported only): not computed above SMALL_MAX_CELLS, where a run takes hours
            n_conv = 5 if n <= SMALL_MAX_CELLS else 0
            lg_conv = leidenalg_partitions(ig_g, range(n_conv), -1, _lg_workers(n)) if n_conv else np.zeros((0, n), np.int64)
            out.update({
                "A_leiden": a_leiden, "B": b,
                "q_common_scanpy": np.array([modularity(ig_g, x) for x in a_leiden]),
                "q_common_crest": np.array([modularity(ig_g, x) for x in b]),
                "q_common_leidenalg": np.array([modularity(ig_g, x) for x in lg]),
                "q_common_leidenalg_converged": np.array([modularity(ig_g, x) for x in lg_conv]),
            })
            del ig_g, C, U
        _save(core, **out)
        _log("scanpy_core.npz written")
    if knn_only:
        _log("mode knn: scanpy stage finished")
        return

    # arm A with every seed: the graph is rebuilt per seed (seed 0 is A_leiden[0])
    afull = d / "scanpy_afull.npz"
    if not afull.exists():
        runs = [np.load(core)["A_leiden"][0]]  # seed 0: the seed-0 graph clustered above
        for s in seeds[1:]:
            neighbors(s)
            runs.append(leiden(s))
            _log(f"A seed {s}: {len(np.unique(runs[-1]))} clusters")
        _save(afull, A=np.stack(runs))
    runs = np.load(afull)["A"]
    seed0 = runs[0]

    # UMAP and differential expression on the seed-0 graph
    seed0_graph()
    for s in range(_n_umap_seeds(n)):
        f = d / f"scanpy_umap_s{s}.npy"
        if not f.exists():
            sc.tl.umap(g, random_state=s)
            np.save(f, np.asarray(g.obsm["X_umap"], np.float32))
            _log(f"UMAP seed {s}")
    ref_seeds = [0] + (list(range(1, 1 + DE_REF_SEEDS_SMALL)) if n <= SMALL_MAX_CELLS else [])
    for s in ref_seeds:
        f = d / f"scanpy_de_s{s}.npz"
        if f.exists():
            continue
        lab = seed0 if s == 0 else np.load(d / "scanpy_core.npz")["A_leiden"][s]
        lognorm.obs["leiden"] = np.asarray(lab).astype(str)
        lognorm.obs["leiden"] = lognorm.obs["leiden"].astype("category")
        sc.tl.rank_genes_groups(lognorm, "leiden", method="wilcoxon", n_genes=lognorm.n_vars)
        groups, S, L, P = _de_tables_scanpy(lognorm, "leiden", genes)
        _save(f, groups=groups, scores=S, lfc=L, padj=P, labels=np.asarray(lab))
        _log(f"scanpy Wilcoxon on the partition of seed {s}")
    json.dump({"tool": "scanpy", "dataset": a.dataset, "n_cells": n, "n_genes": len(genes), "versions": _versions(),
               "threads": a.threads, "n_mod_seeds": nmod, "mode": a.mode, "pca_in_place": bool(a.pca_in_place)},
              open(d / "scanpy_meta.json", "w"), indent=1)
    _log("scanpy stage finished")


# =========================================================================== stage: crest
def _crest_graph_from_embedding(X, seed: int):
    """CREST's kNN search and connectivities on an embedding, as ``crest.pp.neighbors`` computes them."""
    from crest import crest as native
    idx, dist = native.knn_graph(np.ascontiguousarray(X, np.float32), K, exact=None, seed=seed)
    rows, cols, w = native.connectivities(idx, dist, N_NEIGHBORS)
    return idx, rows, cols, w


def stage_crest(a) -> None:
    import polars as pl
    import crest
    from crest import crest as native
    import run_one
    d = Path(a.out) / a.dataset
    d.mkdir(parents=True, exist_ok=True)
    data_dir = Path(a.data_dir)
    path = data_dir / f"{a.dataset}.h5"
    meta = _meta(a.dataset, data_dir)
    side = _obs_table(a.dataset, data_dir)
    knn_only = a.mode == "knn"
    have_scanpy = (d / "scanpy_core.npz").exists()
    backed = str(d / "_ooc") if a.ooc else None
    _log(f"crest {crest.__version__}: reading {path} (backed={bool(backed)})")
    b = crest.read_10x_h5(path, backed=backed)
    if side is not None:
        b.obs = b.obs.with_columns([side[c] for c in side.columns if c != "barcode"])
    b = crest.pp.filter_cells(b, min_genes=meta.get("min_genes", 200))
    b = crest.pp.filter_genes(b, min_cells=3)
    crest.pp.normalize_total(b, run_one.NORMALIZE_TARGET_SUM)
    crest.pp.log1p(b)
    crest.pp.highly_variable_genes(b, n_top_genes=run_one.N_HVG, flavor="seurat")
    hvg = np.asarray(b.var_names, dtype=str)[b.var["highly_variable"].to_numpy()]
    crest.pp.scale(b, max_value=run_one.SCALE_MAX_VALUE)
    crest.tl.pca(b, n_comps=run_one.N_PCS)
    pca = np.asarray(b.obsm["X_pca"], np.float32)
    barcodes = b.obs["barcode"].to_numpy().astype(str)
    genes = np.asarray(b.var_names, dtype=str)
    n = len(barcodes)
    _log(f"CREST PCA done: {n} cells, {len(genes)} genes")
    nmod = _n_mod_seeds(n)
    core = d / "crest_core.npz"
    if not core.exists():
        out = {"barcodes": barcodes, "pca": pca, "hvg": hvg, "genes": genes}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crest.pp.neighbors(b, n_neighbors=N_NEIGHBORS, random_state=0)
        out["knn0"] = np.asarray(b.uns["neighbors"]["indices"]).astype(np.int32)
        params = b.uns["neighbors"]["params"]
        _log(f"seed-0 graph: {params}")
        if have_scanpy:  # arm C seed 0: CREST's kNN on scanpy's PCA
            sp_ = np.load(d / "scanpy_core.npz")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idx, *_ = _crest_graph_from_embedding(sp_["pca"], 0)
            out["knn0_on_scanpy_pca"] = idx.astype(np.int32)
        if not knn_only:
            from crest.pp import connectivities_matrix
            C = connectivities_matrix(b)
            ig_g = igraph_from_connectivities(C)
            d_leiden = []
            for s in range(N_SEEDS):
                crest.tl.leiden(b, resolution=RESOLUTION, n_iterations=N_ITER, random_state=s)
                d_leiden.append(b.obs["leiden"].cast(int).to_numpy())
            seed0 = d_leiden[0]
            q_leidenalg, q_crest = [], []
            lg = leidenalg_partitions(ig_g, range(nmod), N_ITER, _lg_workers(n))
            q_leidenalg = [modularity(ig_g, x) for x in lg]
            for s in range(nmod):
                crest.tl.leiden(b, resolution=RESOLUTION, n_iterations=N_ITER, random_state=s)
                q_crest.append(modularity(ig_g, b.obs["leiden"].cast(int).to_numpy()))
            crest.tl.leiden(b, resolution=RESOLUTION, n_iterations=N_ITER, random_state=0)
            out.update({"D_leiden_graph0": np.stack(d_leiden), "q_own_crest": np.array(q_crest),
                        "q_own_leidenalg": np.array(q_leidenalg)})
            del ig_g, C
            _log("seed-0 graph clustering and modularity done")
        _save(core, **out)
        _log("crest_core.npz written")
    if have_scanpy and "knn0_on_scanpy_pca" not in np.load(core, allow_pickle=False).files:
        # the core was written before the scanpy stage had run: add arm C's seed-0 graph now
        saved = dict(np.load(core, allow_pickle=False))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx, *_ = _crest_graph_from_embedding(np.load(d / "scanpy_core.npz")["pca"], 0)
        saved["knn0_on_scanpy_pca"] = idx.astype(np.int32)
        _save(core, **saved)
        _log("crest_core.npz: added the seed-0 graph on scanpy's PCA")
    if knn_only:
        _log("mode knn: crest stage finished")
        return

    # arm D (every seed rebuilds CREST's graph) and arm C (scanpy's PCA)
    arms = d / "crest_arms.npz"
    if not arms.exists():
        if not have_scanpy:
            raise SystemExit("arm C needs the scanpy stage first (scanpy_core.npz)")
        sp_pca = np.load(d / "scanpy_core.npz")["pca"]
        D, Cc = [], []
        for s in range(N_SEEDS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crest.pp.neighbors(b, n_neighbors=N_NEIGHBORS, random_state=s)
            crest.tl.leiden(b, resolution=RESOLUTION, n_iterations=N_ITER, random_state=s)
            D.append(b.obs["leiden"].cast(int).to_numpy())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idx, rows, cols, w = _crest_graph_from_embedding(sp_pca, s)
            Cc.append(np.asarray(native.leiden(sp_pca.shape[0], rows, cols, w, RESOLUTION, N_ITER, s)).astype(np.int64))
            _log(f"D / C seed {s}: {len(np.unique(D[-1]))} / {len(np.unique(Cc[-1]))} clusters")
        _save(arms, D=np.stack(D), C=np.stack(Cc))
    D = np.load(arms)["D"]

    # UMAP and differential expression on the seed-0 graph
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crest.pp.neighbors(b, n_neighbors=N_NEIGHBORS, random_state=0)
    for s in range(_n_umap_seeds(n)):
        f = d / f"crest_umap_s{s}.npy"
        if not f.exists():
            crest.tl.umap(b, random_state=s)
            np.save(f, np.asarray(b.obsm["X_umap"], np.float32))
            _log(f"UMAP seed {s}")
    f_own, f_same = d / "crest_de_own.npz", d / "crest_de_same.npz"
    if not f_own.exists():
        b.obs = b.obs.with_columns(pl.Series("leiden_d0", D[0].astype(str)))
        tab = crest.tl.rank_genes_groups(b, "leiden_d0", method="wilcoxon", n_genes=None)
        groups, S, L, P = _de_tables_crest(tab, genes)
        _save(f_own, groups=groups, scores=S, lfc=L, padj=P, labels=D[0])
        _log("CREST Wilcoxon on its own seed-0 partition")
    if not f_same.exists() and have_scanpy:
        sc_core = np.load(d / "scanpy_core.npz")
        sc_lab = np.load(d / "scanpy_afull.npz")["A"][0]
        pos = {x: i for i, x in enumerate(sc_core["barcodes"])}
        lab = np.array([str(sc_lab[pos[x]]) if x in pos else None for x in barcodes], dtype=object)
        n_missing = int(sum(v is None for v in lab))
        b.obs = b.obs.with_columns(pl.Series("leiden_scanpy0", lab.tolist(), dtype=pl.Utf8))
        tab = crest.tl.rank_genes_groups(b, "leiden_scanpy0", method="wilcoxon", n_genes=None)
        groups, S, L, P = _de_tables_crest(tab, genes)
        _save(f_same, groups=groups, scores=S, lfc=L, padj=P, n_cells_without_scanpy_label=n_missing)
        _log(f"CREST Wilcoxon on scanpy's seed-0 partition ({n_missing} cells without a scanpy label)")
    json.dump({"tool": "crest", "dataset": a.dataset, "n_cells": n, "n_genes": len(genes), "versions": _versions(),
               "threads": a.threads, "n_mod_seeds": nmod, "mode": a.mode, "backed": bool(backed),
               "neighbors_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in
                                    b.uns["neighbors"]["params"].items()}, "crest_sha": os.environ.get("CREST_SHA")},
              open(d / "crest_meta.json", "w"), indent=1, default=str)
    _log("crest stage finished")


# =========================================================================== stage: exact (post-hoc diagnostic)
def stage_exact(a) -> None:
    """Diagnostic added after the first results were seen (see Deviations in THRESHOLDS.md); no threshold.

    Arm E: scanpy's PCA, CREST's *exact* kNN search (``exact=True``), CREST's connectivities and Leiden with seeds
    ``0..N_SEEDS-1``. E is the clustering of the true 14-nearest-neighbour graph of the common PCA, against which the
    approximate graphs of both tools (arms A and C) can be compared."""
    from crest import crest as native
    d = Path(a.out) / a.dataset
    sc_ = np.load(d / "scanpy_core.npz")
    pca = sc_["pca"]
    f = d / "exact_arm.npz"
    if f.exists():
        _log("exact_arm.npz exists")
        return
    t0 = time.time()
    idx, dist = native.knn_graph(np.ascontiguousarray(pca, np.float32), K, exact=True, seed=0)
    rows, cols, w = native.connectivities(idx, dist, N_NEIGHBORS)
    _log(f"exact kNN graph in {time.time() - t0:.0f} s")
    E = np.stack([np.asarray(native.leiden(pca.shape[0], rows, cols, w, RESOLUTION, N_ITER, s)).astype(np.int64)
                  for s in range(N_SEEDS)])
    _save(f, E=E, knn_exact=idx.astype(np.int32))
    _log("exact arm written")


def _exact_arm_metrics(d: Path, sc_, cr) -> dict | None:
    """Mean ARI between the exact-graph arm E and A, C (and the within-E spread), and the recall of both
    approximate graphs against the exact graph's neighbour sets (an identical-set comparison)."""
    f = d / "exact_arm.npz"
    if not f.exists() or not (d / "crest_arms.npz").exists():
        return None
    E = np.load(f)["E"]
    A = np.load(d / "scanpy_afull.npz")["A"]
    C = np.load(d / "crest_arms.npz")["C"]
    out = {"note": "post-hoc diagnostic, no threshold (see Deviations in THRESHOLDS.md)",
           "mean_ari_E_vs_A": float(pair_aris(E, A).mean()), "mean_ari_E_vs_C": float(pair_aris(E, C).mean()),
           "mean_ari_A_vs_C": float(pair_aris(A, C).mean()), "within_E": float(pair_aris(E).mean()),
           "within_A": float(pair_aris(A).mean()), "within_C": float(pair_aris(C).mean())}
    ke = np.load(f)["knn_exact"]
    out["knn_jaccard_vs_exact_graph"] = {"scanpy": jaccard_neighbor_rows(ke, sc_["knn0"]),
                                         "crest_on_scanpy_pca": jaccard_neighbor_rows(ke, cr["knn0_on_scanpy_pca"])}
    return out


# =========================================================================== stage: compare
def _obs_lookup(obs, column: str, barcodes) -> np.ndarray:
    """Values of an observation-table column for ``barcodes`` (object array; None where the barcode is absent or the value
    is missing, i.e. null, NaN or the text "nan")."""
    table = dict(zip(obs["barcode"].to_list(), obs[column].to_list()))
    out = np.empty(len(barcodes), dtype=object)
    for i, b in enumerate(barcodes):
        v = table.get(str(b))
        out[i] = None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) == "nan" else v
    return out


def _positions(ref_barcodes, query_barcodes) -> np.ndarray:
    pos = {x: i for i, x in enumerate(ref_barcodes)}
    return np.array([pos[x] for x in query_barcodes])


def _query_barcodes(barcodes, n_query: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = np.sort(rng.choice(len(barcodes), min(n_query, len(barcodes)), replace=False))
    return np.asarray(barcodes)[q]


def stage_compare(a) -> None:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    d = Path(a.out) / a.dataset
    data_dir = Path(a.data_dir)
    meta = _meta(a.dataset, data_dir)
    have = lambda f: (d / f).exists()  # noqa: E731
    res: dict = {"dataset": a.dataset, "versions": _versions(), "crest_sha": os.environ.get("CREST_SHA"),
                 "metrics": {}, "diagnostics": {}}
    M = res["metrics"]
    cr = np.load(d / "crest_core.npz", allow_pickle=False) if have("crest_core.npz") else None
    sc_ = np.load(d / "scanpy_core.npz", allow_pickle=False) if have("scanpy_core.npz") else None
    res["n_cells"] = {"crest": int(len(cr["barcodes"])) if cr is not None else None,
                      "scanpy": int(len(sc_["barcodes"])) if sc_ is not None else None}

    # ---- metric 5: kNN recall
    rec = {}
    if cr is not None:
        common = np.intersect1d(cr["barcodes"], sc_["barcodes"]) if sc_ is not None else cr["barcodes"]
        res["n_common_cells"] = int(len(common))
        qb = _query_barcodes(common if sc_ is not None else cr["barcodes"], RECALL_QUERIES)
        qc = _positions(cr["barcodes"], qb)
        rec["5.1_crest_own_pca"] = eval_recall(recall_summary(recall_per_query(
            cr["pca"], cr["pca"][qc], cr["knn0"][qc], K, self_idx=qc)))
        if sc_ is not None:
            qs = _positions(sc_["barcodes"], qb)
            rec["5.2_scanpy_own_pca"] = recall_summary(recall_per_query(sc_["pca"], sc_["pca"][qs], sc_["knn0"][qs], K, self_idx=qs))
            if "knn0_on_scanpy_pca" in cr.files:
                rec["5.3_crest_on_scanpy_pca"] = eval_recall(recall_summary(recall_per_query(
                    sc_["pca"], sc_["pca"][qs], cr["knn0_on_scanpy_pca"][qs], K, self_idx=qs)))
        batch = meta.get("batch")
        obs = _obs_table(a.dataset, data_dir)
        if batch and obs is not None and batch in obs.columns:
            lab = np.array([str(v) for v in _obs_lookup(obs, batch, cr["barcodes"])])
            levels = sorted(set(lab))
            ref_levels = set(levels[: len(levels) // 2])
            is_ref = np.array([x in ref_levels for x in lab])
            if is_ref.any() and (~is_ref).any():
                from crest import crest as native
                ref_idx = np.flatnonzero(is_ref)
                rng = np.random.default_rng(0)
                qi = np.sort(rng.choice(np.flatnonzero(~is_ref), min(RECALL_QUERIES, int((~is_ref).sum())), replace=False))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    nb, _ = native.knn_query(np.ascontiguousarray(cr["pca"][ref_idx], np.float32),
                                             np.ascontiguousarray(cr["pca"][qi], np.float32), K)
                r54 = eval_recall(recall_summary(recall_per_query(cr["pca"][ref_idx], cr["pca"][qi], nb.astype(np.int64), K)))
                r54.update({"batch_column": batch, "reference_levels": sorted(ref_levels), "n_reference": int(len(ref_idx)),
                            "n_reference_above_exact_limit": bool(len(ref_idx) > 20_000)})
                rec["5.4_crest_query_other_batches"] = r54
    M["5_knn_recall"] = rec
    if a.mode == "knn" or not (have("crest_arms.npz") and sc_ is not None):
        res["note"] = "mode knn or arms missing: only metric 5 computed"
        _write(d, res)
        return

    # ---- metrics 1 and 2: partitions
    A = np.load(d / "scanpy_afull.npz")["A"]
    arms = np.load(d / "crest_arms.npz")
    D, Cc = arms["D"], arms["C"]
    ia = _positions(cr["barcodes"], np.intersect1d(cr["barcodes"], sc_["barcodes"]))
    isc = _positions(sc_["barcodes"], np.intersect1d(cr["barcodes"], sc_["barcodes"]))
    # arms in the common cell order (D lives in CREST's cell order; A, B, C in scanpy's)
    Dc, Ac, Cc_c = D[:, ia], A[:, isc], Cc[:, isc]
    Al, Bc = sc_["A_leiden"][:N_SEEDS][:, isc], sc_["B"][:N_SEEDS][:, isc]
    within_A, within_Al = pair_aris(Ac), pair_aris(Al)
    M["2a_whole_workflow"] = eval_cross_ari(pair_aris(Dc, Ac), within_A)
    M["2b_clustering_step"] = eval_cross_ari(pair_aris(Bc, Al), within_Al)
    M["2c_knn_step"] = eval_cross_ari(pair_aris(Cc_c, Ac), within_A)
    res["diagnostics"]["within_tool_mean_ari"] = {"A": float(within_A.mean()), "A_leiden": float(within_Al.mean()),
                                                  "D": float(pair_aris(Dc).mean()), "B": float(pair_aris(Bc).mean()),
                                                  "C": float(pair_aris(Cc_c).mean())}
    res["diagnostics"]["n_clusters"] = {k: [int(len(np.unique(x))) for x in v] for k, v in
                                        {"A": Ac, "D": Dc, "B": Bc, "C": Cc_c}.items()}
    res["diagnostics"]["seed0_ari_D_vs_A"] = float(adjusted_rand_score(Dc[0], Ac[0]))
    obs = _obs_table(a.dataset, data_dir)
    ann_col = meta.get("celltype")
    if obs is not None and ann_col and ann_col in obs.columns:
        bc_common = np.intersect1d(cr["barcodes"], sc_["barcodes"])
        ann = _obs_lookup(obs, ann_col, bc_common)
        ok = np.array([v is not None for v in ann])
        y = np.array([str(v) for v in ann[ok]])
        res["diagnostics"]["annotation"] = {"column": ann_col, "n_labelled": int(ok.sum()), "n_classes": int(len(np.unique(y)))}
        for name, fn in (("ari", adjusted_rand_score), ("nmi", normalized_mutual_info_score)):
            vd = np.array([fn(y, x[ok]) for x in Dc])
            va = np.array([fn(y, x[ok]) for x in Ac])
            M[f"1_annotation_{name}"] = {**eval_annotation(vd.mean(), va), "per_seed_crest": vd.tolist(), "per_seed_scanpy": va.tolist(),
                                         "n_labelled_cells": int(ok.sum()), "column": ann_col}
    # ---- metric 3: modularity
    M["3_modularity_common_graph"] = {**eval_modularity(sc_["q_common_crest"], sc_["q_common_leidenalg"]),
                                      "gap_to_scanpy_igraph_flavour": float(sc_["q_common_crest"].mean() - sc_["q_common_scanpy"].mean()),
                                      "mean_q_scanpy_igraph": float(sc_["q_common_scanpy"].mean()),
                                      "mean_q_leidenalg_converged": (float(sc_["q_common_leidenalg_converged"].mean())
                                                                     if len(sc_["q_common_leidenalg_converged"]) else None),
                                      "n_seeds_leidenalg_converged": int(len(sc_["q_common_leidenalg_converged"]))}
    if "q_own_crest" in cr.files:
        M["3_modularity_crest_graph"] = eval_modularity(cr["q_own_crest"], cr["q_own_leidenalg"])
    # ---- metric 4: differential expression
    M["4_de"] = _de_metrics(d, ia, isc, Dc[0], Ac[0], cr, sc_)
    # ---- metric 6: UMAP
    M["6_umap"] = _umap_metrics(d, cr, sc_, ia, isc, Ac[0])
    # ---- diagnostics without thresholds
    ex = _exact_arm_metrics(d, sc_, cr)
    if ex is not None:
        res["diagnostics"]["post_hoc_exact_graph_arm"] = ex
    res["diagnostics"]["hvg_jaccard"] = float(len(set(cr["hvg"]) & set(sc_["hvg"])) / len(set(cr["hvg"]) | set(sc_["hvg"])))
    Pa, Pb = cr["pca"][ia].astype(np.float64), sc_["pca"][isc].astype(np.float64)
    Qa, _ = np.linalg.qr(Pa - Pa.mean(0))
    Qb, _ = np.linalg.qr(Pb - Pb.mean(0))
    cos = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
    res["diagnostics"]["pca_subspace_cos_min"], res["diagnostics"]["pca_subspace_cos_mean"] = float(cos.min()), float(cos.mean())
    if "knn0_on_scanpy_pca" in cr.files:
        res["diagnostics"]["knn_jaccard_scanpy_vs_crest_on_scanpy_pca"] = jaccard_neighbor_rows(sc_["knn0"], cr["knn0_on_scanpy_pca"])
    _write(d, res)


def _de_metrics(d: Path, ia, isc, d0, a0, cr, sc_) -> dict:
    out = {}
    sa, so, ss = np.load(d / "scanpy_de_s0.npz"), np.load(d / "crest_de_own.npz"), None
    sm = np.load(d / "crest_de_same.npz") if (d / "crest_de_same.npz").exists() else None
    gs, gc = np.asarray(sa["groups"]), np.asarray(so["groups"])
    # gene alignment: tables are in each tool's own gene order; align by name
    genes_s, genes_c = sc_["genes"], cr["genes"]
    common = np.intersect1d(genes_s, genes_c)
    out["n_genes"] = {"scanpy": int(len(genes_s)), "crest": int(len(genes_c)), "common": int(len(common))}
    ps = _positions(genes_s, common)
    pc = _positions(genes_c, common)
    # 4a: own clusters, matched
    pairs, jac, un_a, un_b = match_clusters(d0, a0)  # labels of D (crest) and A (scanpy), common cell order
    rows = []
    for (lc, ls), j in zip(pairs, jac):
        gi_c = int(np.flatnonzero(gc == str(lc))[0]) if (gc == str(lc)).any() else None
        gi_s = int(np.flatnonzero(gs == str(ls))[0]) if (gs == str(ls)).any() else None
        if gi_c is None or gi_s is None:
            continue
        r = de_group_metrics(so["scores"][gi_c][pc], so["lfc"][gi_c][pc], so["padj"][gi_c][pc],
                             sa["scores"][gi_s][ps], sa["lfc"][gi_s][ps], sa["padj"][gi_s][ps])
        rows.append({"group_crest": str(lc), "group_scanpy": str(ls), "cell_jaccard": j, **r})
    n_cells_un = int(np.isin(d0, un_a).sum())
    out["4a_own_clusters"] = {**eval_de(rows, DE_OWN_MIN), "n_unpaired_crest": len(un_a), "n_unpaired_scanpy": len(un_b),
                              "cell_fraction_in_unpaired_crest_clusters": float(n_cells_un / len(d0)), "groups": rows}
    # 4b: same labels
    if sm is not None:
        gm = np.asarray(sm["groups"])
        rows = []
        for gi_s, g in enumerate(gs):
            if (gm == g).any():
                gi_c = int(np.flatnonzero(gm == g)[0])
                rows.append({"group": str(g), **de_group_metrics(sm["scores"][gi_c][pc], sm["lfc"][gi_c][pc], sm["padj"][gi_c][pc],
                                                                  sa["scores"][gi_s][ps], sa["lfc"][gi_s][ps], sa["padj"][gi_s][ps])})
        out["4b_same_labels"] = {**eval_de(rows, DE_SAME_MIN), "n_cells_without_scanpy_label": int(sm["n_cells_without_scanpy_label"]),
                                 "groups": rows}
    # context: scanpy seeds 1, 2 against seed 0
    ctx = {}
    for s in (1, 2):
        f = d / f"scanpy_de_s{s}.npz"
        if f.exists():
            z = np.load(f)
            lab0, labs = np.asarray(sa["labels"]), np.asarray(z["labels"])
            pr, jc, _, _ = match_clusters(lab0, labs)
            rr = []
            for (l0, ls_) in pr:
                i0, i1 = int(np.flatnonzero(gs == str(l0))[0]), int(np.flatnonzero(np.asarray(z["groups"]) == str(ls_))[0])
                rr.append(de_group_metrics(sa["scores"][i0], sa["lfc"][i0], sa["padj"][i0], z["scores"][i1], z["lfc"][i1], z["padj"][i1]))
            ctx[f"scanpy_seed0_vs_seed{s}"] = {**eval_de(rr, DE_OWN_MIN), "note": "context only, not a registered threshold"}
    if ctx:
        out["context_within_scanpy"] = ctx
    out["pass"] = None if out["4a_own_clusters"]["pass"] is None else bool(
        out["4a_own_clusters"]["pass"] and (out.get("4b_same_labels", {}).get("pass") is True))
    return out


def _umap_metrics(d: Path, cr, sc_, ia, isc, common_labels) -> dict:
    pca_common = sc_["pca"][isc].astype(np.float64)  # one common space for both tools: scanpy's PCA
    n = len(isc)
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(n, min(TRUST_QUERIES, n), replace=False))
    out = {"space": "scanpy PCA", "n_cells": int(n), "n_query": int(len(q)), "k": TRUST_K}
    vals = {"crest": {"trust": [], "rho": []}, "scanpy": {"trust": [], "rho": []}}
    ncl = None
    for tool, pre, idx in (("crest", "crest", ia), ("scanpy", "scanpy", isc)):
        for s in range(UMAP_SEEDS_SMALL):
            f = d / f"{pre}_umap_s{s}.npy"
            if not f.exists():
                continue
            U = np.load(f)[idx]
            vals[tool]["trust"].append(trustworthiness_sampled(pca_common, U, q, TRUST_K))
            g = centroid_distance_spearman(pca_common, U, common_labels)
            vals[tool]["rho"].append(g["rho"])
            ncl = g["n_clusters"]
    if not vals["crest"]["trust"] or not vals["scanpy"]["trust"]:
        return {**out, "pass": None, "note": "UMAP files missing"}
    out["n_umap_seeds"] = {t: len(v["trust"]) for t, v in vals.items()}
    out["per_seed"] = vals
    out["local_trustworthiness"] = eval_threshold_diff(np.mean(vals["crest"]["trust"]), np.mean(vals["scanpy"]["trust"]), TRUST_TOL)
    out["global_centroid_distance_spearman"] = {**eval_threshold_diff(np.mean(vals["crest"]["rho"]), np.mean(vals["scanpy"]["rho"]), GLOBAL_TOL),
                                                "n_clusters": ncl, "clusters": "scanpy seed-0 partition, >= %d cells" % GLOBAL_MIN_CLUSTER}
    out["pass"] = bool(out["local_trustworthiness"]["pass"] and out["global_centroid_distance_spearman"]["pass"])
    return out


def _flat_rows(res: dict) -> list[tuple]:
    """(metric, value text, threshold text, status) rows of the result, for the table."""
    rows = []
    f = lambda x: "nan" if x is None else f"{x:.4g}"  # noqa: E731
    st = lambda p: "INCONCLUSIVE" if p is None else ("PASS" if p else "FAIL")  # noqa: E731
    for k, v in res["metrics"].items():
        if k.startswith("1_"):
            rows.append((k, f"crest {f(v['mean_crest'])} scanpy {f(v['mean_scanpy'])} diff {f(v['diff'])}", f"|diff| <= {f(v['threshold'])}", st(v["pass"])))
        elif k.startswith("2"):
            rows.append((k, f"cross {f(v['cross_mean'])} within {f(v['within_mean'])} (min {f(v['within_min'])})", f">= {f(v['threshold'])}", st(v["pass"])))
        elif k.startswith("3"):
            rows.append((k, f"Q crest {f(v['mean_q_crest'])} ref {f(v['mean_q_reference'])} gap {f(v['gap'])} ({v['n_seeds']} seeds)", f"gap >= {f(v['threshold'])}", st(v["pass"])))
        elif k == "4_de":
            for sub in ("4a_own_clusters", "4b_same_labels"):
                if sub in v:
                    s = v[sub]
                    rows.append((sub, f"{s['n_groups']} groups; median " + ", ".join(f"{a} {f(b)}" for a, b in s["median"].items()),
                                 ", ".join(f"{a} >= {b}" for a, b in s["threshold"].items()), st(s["pass"])))
        elif k == "5_knn_recall":
            for sub, s in v.items():
                rows.append((sub, f"mean {f(s['mean'])}; <0.8 {f(100 * s['frac_below_0.8'])}%; <0.5 {f(100 * s['frac_below_0.5'])}% ({s['n_query']} queries)",
                             "mean >= 0.99; <0.5 <= 0.5%; <0.8 <= 1%" if "pass" in s else "none (reported)", st(s["pass"]) if "pass" in s else "reported"))
        elif k == "6_umap" and "local_trustworthiness" in v:
            for sub in ("local_trustworthiness", "global_centroid_distance_spearman"):
                s = v[sub]
                rows.append((f"6_{sub}", f"crest {f(s['crest'])} scanpy {f(s['scanpy'])} diff {f(s['diff'])}", f"diff >= {f(s['threshold'])}", st(s["pass"])))
    return rows


def _write(d: Path, res: dict) -> None:
    (d / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    lines = [f"# Agreement results: {res['dataset']}", "", f"CREST SHA: {res.get('crest_sha')}; cells (crest / scanpy / common): "
             f"{res['n_cells']['crest']} / {res['n_cells']['scanpy']} / {res.get('n_common_cells')}", "",
             "| metric | value | threshold | status |", "|---|---|---|---|"]
    lines += [f"| {m} | {v} | {t} | {s} |" for m, v, t, s in _flat_rows(res)]
    (d / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# =========================================================================== command line
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["scanpy", "crest", "compare", "exact"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--mode", choices=["full", "knn"], default="full")
    ap.add_argument("--pca-in-place", action="store_true",
                    help="scanpy stage: centre the scaled matrix in place in the PCA (same numbers, one dense copy less)")
    ap.add_argument("--ooc", action="store_true", help="CREST reads the matrix out of core (datasets that do not fit in memory)")
    a = ap.parse_args(argv)
    _threads(a.threads)
    stage = {"scanpy": stage_scanpy, "crest": stage_crest, "compare": stage_compare, "exact": stage_exact}[a.stage]
    scanpy_mod = sys.modules.get("scanpy")
    n_jobs_before = scanpy_mod.settings.n_jobs if scanpy_mod is not None else 1   # scanpy's default
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stage(a)
    finally:  # a caller that runs stages in its own process (the tests) gets scanpy's settings back
        if "scanpy" in sys.modules:
            sys.modules["scanpy"].settings.n_jobs = n_jobs_before


if __name__ == "__main__":
    main()
