"""Scrublet doublet detection (Wolock et al. 2019), as ``scanpy.pp.scrublet``.

Pipeline per batch: filter genes/cells, select highly variable genes on the
normalised log data, simulate doublets by summing the raw counts of random
cell pairs, normalise observed and simulated profiles to 1e6, z-score with the
observed gene means/stds, PCA fitted on the observed cells, then a kNN
classifier on the union: the doublet score of a cell is the Bayesian estimate
of its doublet likelihood from the fraction of simulated doublets among its
neighbours. The threshold is the minimum between the two modes of the
simulated-doublet score histogram (``skimage.filters.threshold_minimum``).

CREST differences from scanpy: PCA and projection stream the sparse matrix
through the native Gram/projection kernels (scanpy densifies the observed and
simulated matrices, 3 × cells × HVGs), the kNN search is CREST's, and doublet
parents are drawn with numpy's generator as distinct cell pairs. Scores are
therefore statistically, not bitwise, equivalent; see ``bench/doublets``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from .core import BioFrame
from . import crest as _native

__all__ = ["scrublet", "threshold_minimum"]


def threshold_minimum(values: np.ndarray, nbins: int = 256, max_num_iter: int = 10000) -> Optional[float]:
    """``skimage.filters.threshold_minimum``: smooth the histogram with a 3-bin
    running mean until it has two maxima; return the lowest bin centre between
    them (``None`` if the histogram never becomes bimodal)."""
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0 or v.min() == v.max():
        return None
    counts, edges = np.histogram(v, bins=nbins, range=(v.min(), v.max()))
    centers = (edges[:-1] + edges[1:]) / 2
    h = counts.astype(np.float32)

    def maxima(x):
        out, direction = [], 1
        for i in range(len(x) - 1):
            if direction > 0:
                if x[i + 1] < x[i]:
                    direction = -1
                    out.append(i)
            elif x[i + 1] > x[i]:
                direction = 1
        return out

    idx = []
    for _ in range(max_num_iter):
        p = np.concatenate([h[:1], h, h[-1:]]).astype(np.float64)  # scipy "reflect" for size 3
        h = ((p[:-2] + p[1:-1] + p[2:]) / 3.0).astype(np.float32)
        idx = maxima(h)
        if len(idx) < 3:
            break
    if len(idx) != 2:
        return None
    return float(centers[idx[0] + int(np.argmin(h[idx[0]:idx[1] + 1]))])


def _pca_obs_sim(Xo, Xs, n_comps: int):
    """PCA of the z-scored observed matrix; both matrices projected (sparse, streamed)."""
    from .tl import pca as _pca

    bo = BioFrame.from_scipy(Xo)
    bo.uns["scale"] = {"max_value": 0.0}
    _pca(bo, n_comps=n_comps, use_highly_variable=False, scale=True, max_value=0.0)
    load = bo.varm["PCs"].astype(np.float64)
    from .pp import gene_stats, _mean_var
    st = gene_stats(bo)
    mean, var = _mean_var(st[:, 0], st[:, 1], bo.n_obs)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    mean, std = np.ascontiguousarray(mean), np.ascontiguousarray(std)
    # The kernels project only the sparse part S of the scaled matrix (zeros map to
    # a per-gene constant that centring removes), so centre with the observed
    # mean of S, as tl.pca does: S = x / std on non-zeros.
    shift = ((st[:, 0] / bo.n_obs) / std) @ load
    bs = BioFrame.from_scipy(Xs)
    k = load.shape[1]
    Ys = np.empty((bs.n_obs, k), np.float32)
    Ys[:] = -shift.astype(np.float32)
    gmap = np.arange(bs.n_vars, dtype=np.int32)
    L = np.ascontiguousarray(load)
    for ctx in bs.iter_ctx():
        _native.project(*ctx, gmap, L, Ys, True, mean, std, 0.0)
    return bo.obsm["X_pca"], Ys


def _scrublet_one(bf: BioFrame, sim_doublet_ratio, expected_doublet_rate, stdev_doublet_rate, n_neighbors,
                  n_prin_comps, threshold, random_state, min_counts_genes):
    import scipy.sparse as sp
    from . import pp

    sub = pp.filter_genes(bf, min_cells=min_counts_genes)
    sub = pp.filter_cells(sub, min_genes=min_counts_genes)
    kept_cells = sub.obs["cell_id"].to_numpy()
    work = sub.copy()
    work.ops = []
    obs_qc, _ = pp.calculate_qc_metrics(work, inplace=False)
    tot = obs_qc["total_counts"].to_numpy()
    pp.normalize_total(work, float(np.median(tot[tot > 0])))
    pp.log1p(work)
    pp.highly_variable_genes(work)
    hv = work.var["highly_variable"].to_numpy()
    if hv.sum() < 2:
        raise ValueError("scrublet: fewer than 2 highly variable genes")
    raw = sub._subset(var_mask=hv)
    raw.ops = []
    Xo = raw.to_scipy(transform=False).astype(np.float64).tocsr()
    n = Xo.shape[0]
    n_sim = int(n * sim_doublet_ratio)
    rng = np.random.default_rng(random_state)
    i = rng.integers(0, n, n_sim)
    j = rng.integers(0, n - 1, n_sim)
    j = j + (j >= i)
    Xs = (Xo[i] + Xo[j]).tocsr()

    def norm1e6(X):
        s = np.asarray(X.sum(1)).ravel()
        f = np.divide(1e6, s, out=np.zeros_like(s), where=s > 0)
        return sp.csr_matrix(sp.diags(f) @ X, dtype=np.float32)

    Po, Ps = _pca_obs_sim(norm1e6(Xo), norm1e6(Xs), n_prin_comps)
    if n_neighbors is None:
        n_neighbors = int(round(0.5 * np.sqrt(n)))
    k_adj = int(round(n_neighbors * (1 + n_sim / float(n))))
    M = np.ascontiguousarray(np.vstack([Po, Ps]), dtype=np.float32)
    # k_adj is large (~3 * 0.5 * sqrt(n)): tiled exact search is faster than
    # NN-descent refinement up to ~150k points
    idx, _ = _native.knn_graph(M, k_adj, exact=len(M) <= 150_000, seed=random_state)
    nd = (idx >= n).sum(1).astype(np.float64)
    rho, r, kk = expected_doublet_rate, n_sim / float(n), float(idx.shape[1])
    q = (nd + 1) / (kk + 2)
    ld = q * rho / r / (1 - rho - q * (1 - rho - rho / r))
    se_q = np.sqrt(q * (1 - q) / (kk + 3))
    se_ld = (q * rho / r / (1 - rho - q * (1 - rho - rho / r)) ** 2
             * np.sqrt((se_q / q * (1 - rho)) ** 2 + (stdev_doublet_rate / rho * (1 - q)) ** 2))
    s_obs, s_sim, e_obs = ld[:n], ld[n:], se_ld[:n]
    thr = threshold if threshold is not None else threshold_minimum(s_sim)
    uns = {"doublet_scores_sim": s_sim, "doublet_parents": np.column_stack([i, j]),
           "parameters": {"expected_doublet_rate": expected_doublet_rate, "sim_doublet_ratio": sim_doublet_ratio,
                          "n_neighbors": n_neighbors, "random_state": random_state, "n_hvg": int(hv.sum())}}
    pred = np.zeros(n, bool)
    if thr is not None:
        uns["threshold"] = thr
        pred = s_obs > thr
        uns["detected_doublet_rate"] = float(pred.mean())
        uns["detectable_doublet_fraction"] = float((s_sim > thr).mean())
        uns["overall_doublet_rate"] = uns["detected_doublet_rate"] / max(uns["detectable_doublet_fraction"], 1e-12)
    return kept_cells, s_obs, pred, (s_obs - thr) / e_obs if thr is not None else np.full(n, np.nan), uns


def scrublet(bf: BioFrame, batch_key: Optional[str] = None, sim_doublet_ratio: float = 2.0,
             expected_doublet_rate: float = 0.05, stdev_doublet_rate: float = 0.02,
             n_neighbors: Optional[int] = None, n_prin_comps: int = 30, threshold: Optional[float] = None,
             random_state: int = 0) -> BioFrame:
    """Predict doublets from raw counts (``scanpy.pp.scrublet`` defaults).

    Adds obs ``doublet_score``, ``predicted_doublet`` (and ``doublet_z``) and
    ``uns['scrublet']`` (simulated scores, threshold, parameters; per batch under
    ``uns['scrublet']['batches']`` with ``batch_key``). Cells removed by the
    internal ``min_genes=3`` filter get a null score. The recorded
    normalisation of ``bf`` is ignored: Scrublet works on raw counts.
    """
    if batch_key is None:
        parts = [(None, bf)]
    else:
        labels = bf.obs[batch_key].cast(pl.Utf8)
        parts = [(b, bf.filter_cells(pl.col(batch_key).cast(pl.Utf8) == b))
                 for b in labels.unique().sort().to_list()]
    pos = {c: k for k, c in enumerate(bf.obs["cell_id"].to_list())}
    score = np.full(bf.n_obs, np.nan)
    z = np.full(bf.n_obs, np.nan)
    pred = np.zeros(bf.n_obs, bool)
    uns = {}
    for b, part in parts:
        cells, s, p, zz, u = _scrublet_one(part, sim_doublet_ratio, expected_doublet_rate, stdev_doublet_rate,
                                           n_neighbors, n_prin_comps, threshold, random_state, 3)
        rows = np.array([pos[c] for c in cells.tolist()], dtype=np.int64)
        score[rows], pred[rows], z[rows] = s, p, zz
        uns[b] = u
    bf.obs = bf.obs.with_columns(pl.Series("doublet_score", score).fill_nan(None),
                                 pl.Series("predicted_doublet", pred), pl.Series("doublet_z", z).fill_nan(None))
    bf.uns["scrublet"] = uns[None] if batch_key is None else {"batches": uns, "batched_by": batch_key}
    return bf
