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


def _unique_positions(keys: Sequence[str]) -> dict:
    """``key -> position`` for the keys that occur exactly once."""
    seen: dict = {}
    for i, g in enumerate(keys):
        seen[g] = -1 if g in seen else i
    return {g: i for g, i in seen.items() if i >= 0}


def _match_genes(query: BioFrame, P: dict):
    """For each query gene, its row in the reference PCA loadings (``-1`` if absent).

    Genes are matched by their stable ids (``var['gene_ids']``) when the reference stored them and the
    query has them. Otherwise they are matched by name, and only names that occur once on each side:
    Cell Ranger feature tables repeat some symbols, and a repeated symbol cannot say which gene is
    meant, so those genes are left out (counted as not found) with a ``UserWarning``.
    Returns ``(rows, number of genes left out as ambiguous)``.
    """
    ref_ids, q_ids = P.get("gene_ids"), query.gene_ids
    if ref_ids is not None and q_ids is not None:
        ref_keys, q_keys, what = list(ref_ids), q_ids, "gene id"
    else:
        ref_keys, q_keys, what = list(P["genes"]), query.var_names, "gene name"
    pos = _unique_positions(ref_keys)
    q_unique = _unique_positions(q_keys)
    rows = np.full(len(q_keys), -1, dtype=np.int32)
    n_amb = 0
    ref_all = set(ref_keys)
    for i, g in enumerate(q_keys):
        if g in pos and g in q_unique:
            rows[i] = pos[g]
        elif g in ref_all:  # present in the reference but repeated on one side
            n_amb += 1
    if n_amb:
        warnings.warn(f"{n_amb} genes were left out of the projection because their {what} is not unique in the "
                      "query or in the reference PCA genes; read the data with gene ids (var['gene_ids']) in both "
                      "objects to match them exactly", UserWarning, stacklevel=4)
    return rows, n_amb


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
    sub, n_ambiguous = _match_genes(query, P)
    found = int((sub >= 0).sum())
    if found == 0:
        raise ValueError("no reference PCA genes in the query")
    n_ref = len(P["genes"])
    if found < 0.8 * n_ref:
        warnings.warn(f"only {found} of {n_ref} reference PCA genes are in the query; missing genes count as zero",
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
    ``n_neighbors`` and the number of PCs searched default to the reference's ``pp.neighbors``
    settings, which a cell filter of the reference keeps (``uns["neighbors_params"]``); without
    them, ``n_neighbors`` must be given (``ValueError`` otherwise) and all PCs are used.
    ``center`` see :func:`project_pca` (default as scanpy).
    """
    Xq = project_pca(query, ref, center)
    params = ref.uns.get("neighbors", {}).get("params") or ref.uns.get("neighbors_params", {}).get("params") or {}
    if n_neighbors is None and "n_neighbors" not in params:
        raise ValueError("the reference has no neighbour settings (crest.pp.neighbors has not been run on it); "
                         "run it, or pass n_neighbors (all principal components are then used)")
    k = int(n_neighbors or params["n_neighbors"])
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
