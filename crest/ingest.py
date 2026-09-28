"""Map query cells onto an annotated reference (``scanpy.tl.ingest`` equivalent).

The query is projected onto the reference PCA (same genes, same scaling
parameters as the reference; genes the query lacks count as unexpressed) and,
as in scanpy, centred on its own mean,
each query cell gets its k nearest reference cells, labels are transferred by
majority vote, and reference UMAP coordinates by the umap-learn
``transform`` initialisation (membership-weighted mean of the neighbours).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import warnings

import numpy as np
import polars as pl

from .core import BioFrame
from . import crest as _native

__all__ = ["ingest", "project_pca"]


def project_pca(query: BioFrame, ref: BioFrame, center: str = "query") -> np.ndarray:
    """Coordinates of the query cells in the reference PCA space.

    ``center="query"`` (scanpy ``ingest``) centres the query on its own mean,
    which also removes a global shift between query and reference;
    ``center="reference"`` uses the reference mean (the plain projection).
    """
    if "pca" not in ref.uns or "projection" not in ref.uns["pca"]:
        raise KeyError("reference has no PCA projection; run crest.tl.pca on it (CREST >= 0.3)")
    P = ref.uns["pca"]["projection"]
    if [o[0] for o in query.ops] != [o[0] for o in P["ops"]]:
        raise ValueError(f"query transforms {query.ops} differ from the reference's {P['ops']}; "
                         "apply the same normalize_total / log1p")
    pos = {g: i for i, g in enumerate(P["genes"])}
    names = query.var_names
    sub = np.array([pos.get(g, -1) for g in names], dtype=np.int32)
    found = int((sub >= 0).sum())
    if found == 0:
        raise ValueError("no reference PCA genes in the query")
    if found < 0.8 * len(pos):
        warnings.warn(f"only {found} of {len(pos)} reference PCA genes are in the query; missing genes count as zero",
                      stacklevel=3)
    L = np.ascontiguousarray(P["loadings"], np.float64)
    k = L.shape[1]
    scale = P["mean"] is not None
    X = np.empty((query.n_obs, k), np.float32)
    X[:] = -np.asarray(P["shift"], np.float32)
    mv = float(ref.uns["pca"]["params"].get("max_value") or 0.0)
    for ctx in query.iter_ctx():
        _native.project(*ctx, sub, L, X, scale, P["mean"], P["std"], mv)
    if center == "query":
        X -= X.mean(axis=0, dtype=np.float64).astype(np.float32)
    elif center != "reference":
        raise ValueError('center must be "query" or "reference"')
    return X


def _membership_weights(dist: np.ndarray) -> np.ndarray:
    """umap-learn smooth_knn_dist memberships exp(-(d - rho) / sigma), sum = log2(k)."""
    d = dist.astype(np.float64)
    k = d.shape[1]
    rho = d[:, :1]
    target = np.log2(k)
    lo, hi, sigma = np.zeros(len(d)), np.full(len(d), np.inf), np.ones(len(d))
    for _ in range(64):
        s = np.exp(-np.maximum(d - rho, 0) / sigma[:, None]).sum(1)
        big = s > target
        hi = np.where(big, sigma, hi)
        lo = np.where(big, lo, sigma)
        sigma = np.where(np.isinf(hi), sigma * 2, (lo + hi) / 2)
    return np.exp(-np.maximum(d - rho, 0) / sigma[:, None])


def ingest(query: BioFrame, ref: BioFrame, obs: Union[str, Sequence[str], None] = None,
           embedding_method: Sequence[str] = ("pca", "umap"), n_neighbors: Optional[int] = None,
           center: str = "query") -> BioFrame:
    """Transfer ``ref.obs[obs]`` labels and the PCA/UMAP embedding to ``query``.

    Adds ``query.obsm['X_pca']`` (reference PC space), ``query.obsm['X_umap']``
    (if the reference has one) and, per transferred column, ``obs[col]`` plus
    ``obs[col + '_confidence']`` (fraction of neighbours voting for the label).
    ``n_neighbors`` defaults to the reference's ``pp.neighbors`` setting;
    ``center`` see :func:`project_pca` (default as scanpy).
    """
    Xq = project_pca(query, ref, center)
    params = ref.uns.get("neighbors", {}).get("params", {})
    k = int(n_neighbors or params.get("n_neighbors", 15))
    n_pcs = params.get("n_pcs")
    Xr = ref.obsm["X_pca"] if n_pcs is None else ref.obsm["X_pca"][:, :n_pcs]
    Xq_n = Xq if n_pcs is None else Xq[:, :n_pcs]
    idx, dist = _native.knn_query(np.ascontiguousarray(Xr, np.float32), np.ascontiguousarray(Xq_n, np.float32), k)
    idx = idx.astype(np.int64)
    if "pca" in embedding_method:
        query.obsm["X_pca"] = Xq
    if "umap" in embedding_method and "X_umap" in ref.obsm:
        w = _membership_weights(dist)
        U = ref.obsm["X_umap"][idx]  # (n, k, 2)
        query.obsm["X_umap"] = (np.einsum("nk,nkc->nc", w, U) / w.sum(1, keepdims=True)).astype(np.float32)
    cols = [] if obs is None else ([obs] if isinstance(obs, str) else list(obs))
    new = []
    for c in cols:
        lab = ref.obs[c].cast(pl.Utf8).fill_null("NA").to_numpy()
        cats, codes = np.unique(lab, return_inverse=True)
        votes = codes[idx]
        counts = np.zeros((len(votes), len(cats)), np.int32)
        np.add.at(counts, (np.repeat(np.arange(len(votes)), votes.shape[1]), votes.ravel()), 1)
        win = counts.argmax(1)  # ties -> first label in sorted order (pandas .mode()[0])
        new += [pl.Series(c, cats[win]), pl.Series(f"{c}_confidence", counts.max(1) / votes.shape[1])]
    if new:
        query.obs = query.obs.drop([s.name for s in new if s.name in query.obs.columns]).with_columns(new)
    query.uns["ingest"] = {"n_neighbors": k, "obs": cols, "indices": idx.astype(np.uint32), "distances": dist}
    return query
