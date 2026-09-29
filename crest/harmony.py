"""Harmony batch integration on a low-dimensional embedding (Rust, harmony2 algorithm).

Same algorithm, defaults and outputs as ``harmonypy.run_harmony`` 2.x (what
``scanpy.external.pp.harmony_integrate`` calls) and R ``harmony`` >= 1.2.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import polars as pl

from .core import BioFrame
from . import crest as _native

__all__ = ["harmony", "run_harmony"]


def _codes(values) -> tuple[np.ndarray, int]:
    s = pl.Series(values)
    if s.null_count():
        raise ValueError("batch column contains missing values")
    uniq = s.unique().sort()
    code = {v: i for i, v in enumerate(uniq.to_list())}
    return np.array([code[v] for v in s.to_list()], dtype=np.uint32), len(uniq)


def _per_level(x, n_levels: Sequence[int], name: str) -> np.ndarray:
    total = int(sum(n_levels))
    if np.isscalar(x):
        return np.full(total, float(x))
    x = np.asarray(x, dtype=np.float64)
    if len(x) == len(n_levels):
        return np.repeat(x, n_levels)
    if len(x) == total:
        return x
    raise ValueError(f"{name} needs one value per covariate or per batch level")


def run_harmony(z: np.ndarray, batches: Sequence[Sequence], theta: Union[float, Sequence[float]] = 2.0,
                lamb: Optional[Union[float, Sequence[float]]] = None, sigma: float = 0.1,
                nclust: Optional[int] = None, tau: float = 0.0, block_size: float = 0.05,
                max_iter_harmony: int = 10, max_iter_kmeans: int = 4, epsilon_cluster: float = 1e-3,
                epsilon_harmony: float = 1e-2, alpha: float = 0.2, batch_prop_cutoff: float = 1e-5,
                random_state: int = 0) -> dict:
    """Harmony on an (N, d) matrix. ``batches`` is a list of per-cell label arrays
    (one per covariate). ``lamb=None`` estimates the ridge penalty per cluster as
    ``alpha * E`` (harmony2). Returns a dict with ``Z_corr`` (N, d), ``R`` (N, K),
    ``Y`` (K, d), the objectives and ``converged``."""
    z = np.ascontiguousarray(z, dtype=np.float32)
    n = z.shape[0]
    codes, levels, off = [], [], 0
    for b in batches:
        if len(b) != n:
            raise ValueError("each batch array needs one label per row of z")
        c, m = _codes(b)
        codes.append(c + off)
        levels.append(m)
        off += m
    batch = np.ascontiguousarray(np.vstack(codes), dtype=np.uint32)
    if nclust is None:
        nclust = int(min(round(n / 30.0), 100))
    nclust = max(2, min(nclust, n))
    th = _per_level(theta, levels, "theta")
    if tau > 0:
        n_b = np.bincount(batch.ravel(), minlength=off).astype(np.float64)
        th = th * (1 - np.exp(-(n_b / (nclust * tau)) ** 2))
    lam = None if lamb is None or (np.isscalar(lamb) and lamb == -1) else _per_level(lamb, levels, "lamb")
    return _native.harmony(z, batch, levels, nclust, th.tolist(), None if lam is None else lam.tolist(),
                           float(sigma), float(alpha), float(block_size), int(max_iter_harmony),
                           int(max_iter_kmeans), float(epsilon_cluster), float(epsilon_harmony),
                           float(batch_prop_cutoff), int(random_state))


def harmony(bf: BioFrame, key: Union[str, Sequence[str]], basis: str = "X_pca",
            adjusted_basis: str = "X_pca_harmony", **kwargs) -> BioFrame:
    """Integrate ``obsm[basis]`` over the batch column(s) ``key`` (like
    ``sc.external.pp.harmony_integrate``); writes ``obsm[adjusted_basis]`` and
    ``uns['harmony']`` (objectives, convergence). Keyword arguments go to
    :func:`run_harmony`. Follow with ``crest.pp.neighbors(bf, use_rep=adjusted_basis)``."""
    if basis not in bf.obsm:
        raise KeyError(f"{basis!r} not in obsm; run crest.tl.pca first")
    keys = [key] if isinstance(key, str) else list(key)
    out = run_harmony(bf.obsm[basis], [bf.obs[k].to_list() for k in keys], **kwargs)
    bf.obsm[adjusted_basis] = out["Z_corr"]
    bf.uns["harmony"] = {"key": keys, "basis": basis, "objective_harmony": out["objective_harmony"],
                         "kmeans_rounds": out["kmeans_rounds"], "converged": out["converged"],
                         "n_clusters": out["R"].shape[1]}
    return bf
