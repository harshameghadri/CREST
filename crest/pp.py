"""Preprocessing (scanpy ``sc.pp`` equivalents) on :class:`~crest.core.BioFrame`.

Transforms are lazy: ``normalize_total``, ``log1p`` and ``scale`` are recorded
and applied chunk-by-chunk inside the native kernels of later steps, so the
count matrix is never copied, normalised in place or densified.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import polars as pl

from .core import BioFrame
from . import crest as _native

__all__ = [
    "calculate_qc_metrics", "filter_cells", "filter_genes", "normalize_total", "log1p", "scale",
    "highly_variable_genes", "neighbors", "gene_stats", "connectivities_matrix",
]


# --------------------------------------------------------------------------- QC
def calculate_qc_metrics(bf: BioFrame, mito_prefix: Optional[str] = "MT-", inplace: bool = True) -> Optional[tuple]:
    """Per-cell ``total_counts``, ``n_genes_by_counts``, ``pct_counts_mt`` and per-gene
    ``n_cells_by_counts``, ``total_counts`` (on raw counts of kept cells/genes)."""
    names = bf.var_names
    flag = np.array([bool(mito_prefix) and n.upper().startswith(mito_prefix.upper()) for n in names], dtype=bool)
    ct = np.zeros(bf.n_obs, np.float64)
    cn = np.zeros(bf.n_obs, np.uint32)
    cf = np.zeros(bf.n_obs, np.float64)
    gn = np.zeros(bf.n_vars, np.uint32)
    gt = np.zeros(bf.n_vars, np.float64)
    for r, g, v in bf.iter_chunks(transform=False):
        _native.qc_chunk(r, g, v, flag, ct, cn, cf, gn, gt)
    pct = np.divide(cf, ct, out=np.zeros_like(ct), where=ct > 0) * 100.0
    obs_cols = [pl.Series("n_genes_by_counts", cn.astype(np.int64)), pl.Series("total_counts", ct),
                pl.Series("pct_counts_mt", pct)]
    var_cols = [pl.Series("n_cells_by_counts", gn.astype(np.int64)), pl.Series("total_counts", gt)]
    if not inplace:
        return pl.DataFrame(obs_cols), pl.DataFrame(var_cols)
    bf.obs = bf.obs.with_columns(obs_cols)
    bf.var = bf.var.with_columns(var_cols)
    return None


def filter_cells(bf: BioFrame, min_genes: Optional[int] = None, min_counts: Optional[float] = None,
                 max_genes: Optional[int] = None, max_counts: Optional[float] = None,
                 max_pct_mt: Optional[float] = None) -> BioFrame:
    """Return a BioFrame keeping cells within the given thresholds (like ``sc.pp.filter_cells``)."""
    obs, _ = calculate_qc_metrics(bf, inplace=False)
    keep = np.ones(bf.n_obs, bool)
    ng, tc, mt = obs["n_genes_by_counts"].to_numpy(), obs["total_counts"].to_numpy(), obs["pct_counts_mt"].to_numpy()
    if min_genes is not None:
        keep &= ng >= min_genes
    if max_genes is not None:
        keep &= ng <= max_genes
    if min_counts is not None:
        keep &= tc >= min_counts
    if max_counts is not None:
        keep &= tc <= max_counts
    if max_pct_mt is not None:
        keep &= mt <= max_pct_mt
    return bf._subset(obs_mask=keep)


def filter_genes(bf: BioFrame, min_cells: Optional[int] = None, min_counts: Optional[float] = None,
                 max_cells: Optional[int] = None, max_counts: Optional[float] = None) -> BioFrame:
    """Return a BioFrame keeping genes within the given thresholds (counted over kept cells)."""
    _, var = calculate_qc_metrics(bf, inplace=False)
    keep = np.ones(bf.n_vars, bool)
    nc, tc = var["n_cells_by_counts"].to_numpy(), var["total_counts"].to_numpy()
    if min_cells is not None:
        keep &= nc >= min_cells
    if max_cells is not None:
        keep &= nc <= max_cells
    if min_counts is not None:
        keep &= tc >= min_counts
    if max_counts is not None:
        keep &= tc <= max_counts
    return bf._subset(var_mask=keep)


# --------------------------------------------------------------------------- transforms
def normalize_total(bf: BioFrame, target_sum: float = 1e4) -> BioFrame:
    """Scale each cell to ``target_sum`` total counts (applied lazily)."""
    if target_sum is None or target_sum <= 0:
        raise ValueError("target_sum must be positive")
    if any(op[0] == "log1p" for op in bf.ops):
        raise ValueError("normalize_total must come before log1p")
    bf.ops = [op for op in bf.ops if op[0] != "normalize_total"] + [("normalize_total", float(target_sum))]
    return bf


def log1p(bf: BioFrame) -> BioFrame:
    """Natural log(1 + x) (applied lazily)."""
    if not any(op[0] == "log1p" for op in bf.ops):
        bf.ops.append(("log1p",))
    return bf


def scale(bf: BioFrame, max_value: Optional[float] = 10.0) -> BioFrame:
    """Zero-centre and unit-variance scale with clipping at ±``max_value`` (as ``sc.pp.scale``).

    Recorded for :func:`crest.tl.pca`, which applies it exactly without densifying
    the matrix. Other steps (DE, scores) keep using the log-normalised values,
    as scanpy workflows do via ``adata.raw``.
    """
    bf.uns["scale"] = {"max_value": float(max_value) if max_value else 0.0}
    return bf


# --------------------------------------------------------------------------- gene statistics / HVG
def gene_stats(bf: BioFrame) -> np.ndarray:
    """(n_vars, 5) array: Σx, Σx², Σexpm1(x), Σexpm1(x)², nnz of the transformed matrix."""
    out = np.zeros((bf.n_vars, 5), np.float64)
    for r, g, v in bf.iter_chunks():
        _native.gene_stats(g, v, out)
    return out


def _mean_var(s1: np.ndarray, s2: np.ndarray, n: int):
    mean = s1 / n
    var = (s2 - n * mean * mean) / (n - 1) if n > 1 else np.zeros_like(mean)
    return mean, np.maximum(var, 0.0)


def _pd_cut_codes(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin codes identical to ``pandas.cut(x, n_bins)`` (right-closed, 0.1% lower pad)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        return np.zeros(len(x), np.int64)
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0] -= (hi - lo) * 0.001
    return np.clip(np.searchsorted(edges, x, side="left") - 1, 0, n_bins - 1)


def highly_variable_genes(bf: BioFrame, n_top_genes: Optional[int] = None, flavor: str = "seurat",
                          n_bins: int = 20, min_mean: float = 0.0125, max_mean: float = 3.0,
                          min_disp: float = 0.5, max_disp: float = np.inf, subset: bool = False) -> BioFrame:
    """Highly variable genes, reproducing ``sc.pp.highly_variable_genes`` for
    ``flavor="seurat"`` (expects log-normalised data) and ``"cell_ranger"``.

    Adds var columns ``means``, ``dispersions``, ``dispersions_norm``,
    ``highly_variable`` and caches per-gene log-scale mean/std for :func:`scale`.
    """
    n = bf.n_obs
    st = gene_stats(bf)
    mean_log, var_log = _mean_var(st[:, 0], st[:, 1], n)
    if flavor == "seurat":
        mean, var = _mean_var(st[:, 2], st[:, 3], n)
    elif flavor == "cell_ranger":
        mean, var = mean_log.copy(), var_log.copy()
    else:
        raise ValueError('flavor must be "seurat" or "cell_ranger"')
    mean = mean.copy()
    mean[mean == 0] = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = var / mean
    if flavor == "seurat":
        disp[disp == 0] = np.nan
        with np.errstate(divide="ignore"):
            disp = np.log(disp)
        mean = np.log1p(mean)
        codes = _pd_cut_codes(mean, n_bins)
        nb = n_bins
    else:
        edges = np.r_[-np.inf, np.percentile(mean, np.arange(10, 105, 5)), np.inf]
        codes = np.clip(np.searchsorted(edges, mean, side="left") - 1, 0, len(edges) - 2)
        nb = len(edges) - 1

    avg = np.full(nb, np.nan)
    dev = np.full(nb, np.nan)
    for b in range(nb):
        d = disp[(codes == b) & ~np.isnan(disp)]
        if flavor == "seurat":
            if len(d):
                avg[b] = d.mean()
            if len(d) > 1:
                dev[b] = d.std(ddof=1)
        elif len(d):
            med = np.median(d)
            avg[b] = med
            dev[b] = np.median(np.abs(d - med)) / 0.6744897501960817  # statsmodels mad
    if flavor == "seurat":  # single-gene bins: normalised dispersion set to 1
        one = np.isnan(dev)
        dev[one] = avg[one]
        avg[one] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        disp_norm = (disp - avg[codes]) / dev[codes]

    if n_top_genes is not None:
        x = disp_norm[~np.isnan(disp_norm)]
        k = min(n_top_genes, x.size, bf.n_vars)
        cutoff = np.sort(x)[::-1][k - 1]
        hv = np.nan_to_num(disp_norm, nan=-np.inf) >= cutoff
    else:
        dn = np.nan_to_num(disp_norm)
        hv = (mean > min_mean) & (mean < max_mean) & (dn > min_disp) & (dn < max_disp)

    std_log = np.sqrt(var_log)
    std_log[std_log == 0] = 1.0
    cols = [
        pl.Series("means", mean), pl.Series("dispersions", disp), pl.Series("dispersions_norm", disp_norm),
        pl.Series("highly_variable", hv), pl.Series("mean_log", mean_log), pl.Series("std_log", std_log),
        pl.Series("n_cells_expr", st[:, 4].astype(np.int64)),
    ]
    bf.var = bf.var.drop([c.name for c in cols if c.name in bf.var.columns]).with_columns(cols)
    bf.uns["hvg"] = {"flavor": flavor, "ops": list(bf.ops)}
    if subset:
        return bf._subset(var_mask=hv)
    return bf


# --------------------------------------------------------------------------- neighbours
def neighbors(bf: BioFrame, n_neighbors: int = 15, n_pcs: Optional[int] = None, use_rep: str = "X_pca",
              exact: Optional[bool] = None, random_state: int = 0) -> BioFrame:
    """kNN graph + UMAP fuzzy connectivities (``sc.pp.neighbors`` with method="umap").

    ``n_neighbors`` counts the cell itself, as in scanpy. ``exact=None`` uses exact
    search up to 20k cells and the IVF + NN-descent index above.
    """
    if use_rep not in bf.obsm:
        raise KeyError(f"{use_rep!r} not in obsm; run crest.tl.pca first")
    X = bf.obsm[use_rep]
    if n_pcs is not None:
        X = X[:, :n_pcs]
    X = np.ascontiguousarray(X, dtype=np.float32)
    idx, dist = _native.knn_graph(X, n_neighbors - 1, exact=exact, seed=random_state)
    rows, cols, w = _native.connectivities(idx, dist, n_neighbors)
    bf.uns["neighbors"] = {
        "params": {"n_neighbors": n_neighbors, "n_pcs": n_pcs, "use_rep": use_rep, "method": "umap"},
        "indices": idx, "distances": dist, "rows": rows, "cols": cols, "weights": w,
    }
    return bf


def connectivities_matrix(bf: BioFrame):
    """Symmetric scipy.sparse CSR connectivities (scanpy ``obsp['connectivities']``)."""
    import scipy.sparse as sp
    nb = bf.uns["neighbors"]
    n = bf.n_obs
    C = sp.coo_matrix((nb["weights"], (nb["rows"], nb["cols"])), shape=(n, n)).tocsr()
    return (C + C.T).tocsr()
