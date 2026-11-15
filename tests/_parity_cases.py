"""Measurements of CREST against scanpy on real data (PBMC 3k and PBMC 10k).

Every function that compares the two tools returns plain numbers in a dict ``{comparison id: value}``.
Each value is a non-negative deviation, so that a tolerance in ``tests/_tolerances.py`` is an upper bound
(a shortfall is ``max(0, reference - crest)``). ``measure`` runs all comparisons for one data set and
returns ``{"meta": ..., "metrics": ..., "errors": ...}``; a comparison that cannot run in the installed
scanpy is recorded under ``errors`` instead of stopping the others.

Both tools start from the same raw counts and the same unique gene names; scanpy never reads the
files itself. The module is used by ``tests/test_scanpy_parity_matrix.py`` and can be run directly to
produce the measurements that the tolerances were derived from::

    python tests/_parity_cases.py pbmc3k=/path/to/hg19 pbmc10k=/path/to/pbmc10k.h5 --out results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import polars as pl

import crest

# --------------------------------------------------------------------------- settings
SEEDS_LEIDEN = (0, 1, 2, 3, 4)
SEEDS_UMAP = (0, 1, 2)
SEEDS_SCRUBLET = (0, 1, 2, 3)
N_NEIGHBORS = 15
N_PCS_GRAPH = 40
N_COMPS = 50
N_TOP_GENES = 2000
RECALL_QUERIES = 5000
TRUST_CELLS = 5000
TRUST_K = 15
MIN_GENES, MIN_CELLS = 200, 3
# per data set: cell filter after the QC metrics (<= as in crest.pp.filter_cells), spiked doublets for Scrublet
DATASETS = {
    "pbmc3k": dict(max_genes=2500, max_pct_mt=5.0, n_spiked=150),
    "pbmc10k": dict(max_genes=7000, max_pct_mt=20.0, n_spiked=600),
}
B_MARKERS = ["CD79A", "MS4A1", "CD79B", "CD19", "BANK1", "IGHM", "CD74", "HLA-DRA", "HLA-DRB1", "CD37"]


# --------------------------------------------------------------------------- data
def unique_names(names):
    """scanpy's ``var_names_make_unique`` rule (``-1``, ``-2`` ...), applied before either tool sees the data."""
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}-{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    assert len(set(out)) == len(out)
    return out


def load_raw(path):
    """(raw counts CSR float32, unique gene names, gene ids) of a 10x directory (matrix.mtx) or .h5 file."""
    path = Path(path)
    bf = crest.read_10x_h5(path) if path.suffix == ".h5" else crest.read_10x_mtx(path)
    X = bf.to_scipy(transform=False).tocsr().astype(np.float32)
    X.sort_indices()
    ids = list(bf.var["gene_ids"].to_numpy()) if "gene_ids" in bf.var.columns else [f"G{i}" for i in range(X.shape[1])]
    return X, unique_names(bf.var_names), ids


def _frame(X, names, ids):
    return crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": list(names), "gene_ids": list(ids)}))


def _adata(X, names):
    import anndata as ad
    import pandas as pd
    a = ad.AnnData(X.copy().astype(np.float32), var=pd.DataFrame(index=list(names)))
    a.obs_names = [str(i) for i in range(a.n_obs)]
    return a


# --------------------------------------------------------------------------- small numeric helpers
def max_abs(a, b):
    """max |a - b| over the entries where b (scanpy's value) is finite; inf if a is not finite there."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(b)
    if not np.isfinite(a[ok]).all():
        return float("inf")
    return float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0


def max_rel(a, b, floor=0.0):
    """max |a - b| / |b| over entries with |b| > floor (b is the scanpy value)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > floor)
    return float((np.abs(a[m] - b[m]) / np.abs(b[m])).max()) if m.any() else 0.0


def sparse_same(A, B):
    A, B = A.tocsr().astype(np.float32), B.tocsr().astype(np.float32)
    A.sort_indices(); B.sort_indices()
    pattern = np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
    return pattern, A, B


def sparse_diff(A, B):
    """(pattern differences, max |a-b|, max |a-b|/|b|) of two sparse matrices (B is scanpy's)."""
    pattern, A, B = sparse_same(A, B)
    if not pattern:
        d = abs(A - B)
        return int((A != B).sum()), float(d.max()), float("inf")
    return 0, float(np.abs(A.data - B.data).max()), max_rel(A.data, B.data)


def principal_angle_deg(A, B):
    from scipy.linalg import subspace_angles
    return float(np.degrees(subspace_angles(np.asarray(A, np.float64), np.asarray(B, np.float64)).max()))


def sign_aligned_rel(A, B, k):
    """Largest |score difference| / std of the PC, after giving each of the first k PCs the same sign."""
    out = 0.0
    for j in range(k):
        a, b = np.asarray(A[:, j], np.float64), np.asarray(B[:, j], np.float64)
        s = 1.0 if a @ b >= 0 else -1.0
        out = max(out, float(np.abs(s * a - b).max() / b.std()))
    return out


def scanpy_knn(distances, n_neighbors):
    """scanpy's kNN as (idx, dist), each (n, n_neighbors - 1): the cell itself is dropped, nearest first."""
    D = distances.tocsr()
    n, k = D.shape[0], int(n_neighbors) - 1
    rows = np.repeat(np.arange(n), np.diff(D.indptr))
    keep = D.indices != rows
    rows, cols, dist = rows[keep], D.indices[keep], D.data[keep]
    order = np.lexsort((cols, dist, rows))
    rows, cols, dist = rows[order], cols[order], dist[order]
    rank = np.arange(len(rows)) - np.searchsorted(rows, np.arange(n))[rows]
    idx = np.full((n, k), -1, np.int32)
    dst = np.full((n, k), np.inf, np.float32)
    sel = rank < k
    idx[rows[sel], rank[sel]] = cols[sel]
    dst[rows[sel], rank[sel]] = dist[sel]
    return idx, dst


def knn_recall(X, knn, n_query=RECALL_QUERIES, seed=0):
    """Recall of a kNN graph (n, k; self excluded) against float64 brute force, for up to n_query query cells."""
    X = np.asarray(X, np.float64)
    n, k = knn.shape
    assert not (knn == np.arange(n)[:, None]).any(), "a row lists its own cell"
    chunk = int(max(16, min(512, 50_000_000 // max(n, 1))))
    q = np.sort(np.random.default_rng(seed).choice(n, min(n_query, n), replace=False))
    sq = np.einsum("ij,ij->i", X, X)
    hits = 0
    for lo in range(0, len(q), chunk):
        rows = q[lo:lo + chunk]
        D = sq[rows, None] - 2.0 * X[rows] @ X.T + sq[None, :]
        D[np.arange(len(rows)), rows] = np.inf
        exact = np.argpartition(D, k - 1, axis=1)[:, :k]
        for r, e in zip(rows, exact):
            hits += len(set(knn[r][knn[r] >= 0].tolist()) & set(e.tolist()))
    return hits / (len(q) * k)


def knn_overlap(A, B):
    """Mean fraction of a row of A that is also in the same row of B."""
    k = A.shape[1]
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(A, B)]))


def scanpy_ingest_centering():
    """'reference' when scanpy's ingest centres the query on the reference mean (1.12.3 and later), else 'query'."""
    from scanpy.tools._ingest import Ingest
    init = getattr(Ingest, "_init_pca", None)
    return "reference" if init is not None and "_pca_mean" in init.__code__.co_names else "query"


_UMAP_SCRIPT = """
import sys
import numpy as np
import scipy.sparse as sp
import crest
x = np.ascontiguousarray(np.load(sys.argv[1]), dtype=np.float32)
bf = crest.BioFrame.from_scipy(sp.csr_matrix((len(x), 1), dtype=np.float32))
bf.obsm["X_pca"] = x
crest.pp.neighbors(bf, n_neighbors=int(sys.argv[2]), n_pcs=int(sys.argv[3]), exact=True)
for s in map(int, sys.argv[5:]):
    crest.tl.umap(bf, min_dist=0.5, spread=1.0, random_state=s)
    np.save(f"{sys.argv[4]}_{s}.npy", np.asarray(bf.obsm["X_umap"]))
"""


# --------------------------------------------------------------------------- the comparisons
class Measure:
    def __init__(self, name, X, names, ids, spec):
        import scanpy as sc
        self.sc, self.name, self.spec = sc, name, spec
        self.X, self.names, self.ids = X, names, ids
        self.m, self.errors, self.meta, self.seconds = {}, {}, {}, {}

    def put(self, key, value):
        self.m[key] = float(value)

    def section(self, fn):
        t = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn()
        except Exception:  # noqa: BLE001 - recorded, the other sections still run
            self.errors[fn.__name__] = traceback.format_exc()[-1500:]
        self.seconds[fn.__name__] = round(time.time() - t, 1)

    # ---- filters and QC
    def qc(self):
        sc, spec = self.sc, self.spec
        a = _adata(self.X, self.names)
        sc.pp.filter_cells(a, min_genes=MIN_GENES)
        sc.pp.filter_genes(a, min_cells=MIN_CELLS)
        a.var["mt"] = a.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True)
        bf = crest.pp.filter_cells(_frame(self.X, self.names, self.ids), min_genes=MIN_GENES)
        bf = crest.pp.filter_genes(bf, min_cells=MIN_CELLS)
        crest.pp.calculate_qc_metrics(bf)
        same_cells = np.array_equal(bf.obs["cell_id"].to_numpy().astype(int), a.obs_names.astype(int).to_numpy())
        same_genes = list(bf.var_names) == list(a.var_names)
        self.put("qc.cells_after_min_genes.n_different", 0 if same_cells else 1)
        self.put("qc.genes_after_min_cells.n_different", 0 if same_genes else 1)
        assert same_cells and same_genes, "the two tools keep different cells or genes"
        self.put("qc.n_genes_by_counts.max_abs", max_abs(bf.obs["n_genes_by_counts"].to_numpy(), a.obs["n_genes_by_counts"].to_numpy()))
        self.put("qc.total_counts.max_rel", max_rel(bf.obs["total_counts"].to_numpy(), a.obs["total_counts"].to_numpy()))
        self.put("qc.pct_counts_mt.max_abs", max_abs(bf.obs["pct_counts_mt"].to_numpy(), a.obs["pct_counts_mt"].to_numpy()))
        keep = (a.obs["n_genes_by_counts"].to_numpy() <= spec["max_genes"]) & (a.obs["pct_counts_mt"].to_numpy() <= spec["max_pct_mt"])
        bf = crest.pp.filter_cells(bf, max_genes=spec["max_genes"], max_pct_mt=spec["max_pct_mt"])
        a = a[keep].copy()
        same = np.array_equal(bf.obs["cell_id"].to_numpy().astype(int), a.obs_names.astype(int).to_numpy())
        self.put("qc.cells_after_max_filters.n_different", 0 if same else 1)
        assert same, "the two tools keep different cells after the max_genes / pct_mt filter"
        self.a_raw, self.a, self.bf = a.copy(), a, bf
        self.meta.update(n_cells_raw=int(self.X.shape[0]), n_genes_raw=int(self.X.shape[1]),
                         n_cells=int(a.n_obs), n_genes=int(a.n_vars))

    # ---- normalisation, log1p
    def normalize(self):
        sc, a, bf = self.sc, self.a, self.bf
        sc.pp.normalize_total(a, target_sum=1e4)
        crest.pp.normalize_total(bf, 1e4)
        p, d, r = sparse_diff(bf.to_scipy(), a.X)
        self.put("normalize_total.target_1e4.pattern_differences", p)
        self.put("normalize_total.target_1e4.max_abs", d)
        self.put("normalize_total.target_1e4.max_rel", r)
        scanpy_default = self.a_raw.copy()
        sc.pp.normalize_total(scanpy_default)
        crest_default = _frame(self.a_raw.X, self.a_raw.var_names, self.ids_kept())
        crest.pp.normalize_total(crest_default)
        p, d, r = sparse_diff(crest_default.to_scipy(), scanpy_default.X)
        self.put("normalize_total.default.pattern_differences", p)
        self.put("normalize_total.default.max_abs", d)
        self.put("normalize_total.default.max_rel", r)
        sc.pp.log1p(a)
        crest.pp.log1p(bf)
        p, d, r = sparse_diff(bf.to_scipy(), a.X)
        self.put("log1p.pattern_differences", p)
        self.put("log1p.max_abs", d)
        self.put("log1p.max_rel", r)
        self.a_log_X = a.X.copy()

    def ids_kept(self):
        gid = dict(zip(self.names, self.ids))
        return [gid[n] for n in self.a.var_names]

    # ---- highly variable genes
    def hvg(self):
        sc, a, bf = self.sc, self.a, self.bf

        def compare(tag, flavor, hv_kwargs, var_pairs, frame, adata):
            sc.pp.highly_variable_genes(adata, flavor=flavor, **hv_kwargs)
            crest.pp.highly_variable_genes(frame, flavor=flavor, **hv_kwargs)
            hs, hc = adata.var["highly_variable"].to_numpy().astype(bool), frame.var["highly_variable"].to_numpy().astype(bool)
            self.put(f"hvg.{tag}.n_genes_different", int((hs != hc).sum()))
            for col, kind in var_pairs:
                f = max_rel if kind == "rel" else max_abs
                self.put(f"hvg.{tag}.{col}.max_{kind}", f(frame.var[col].to_numpy(), adata.var[col].to_numpy()))
            return hs

        seurat_cols = [("means", "rel"), ("dispersions", "rel"), ("dispersions_norm", "abs")]
        compare("seurat_ntop", "seurat", dict(n_top_genes=N_TOP_GENES), seurat_cols, bf, a)
        compare("cell_ranger_ntop", "cell_ranger", dict(n_top_genes=N_TOP_GENES), seurat_cols, bf, a)
        self.hv = compare("seurat_tutorial", "seurat", dict(min_mean=0.0125, max_mean=3, min_disp=0.5), seurat_cols, bf, a)
        self.meta["n_hvg_tutorial"] = int(self.hv.sum())
        # seurat_v3 works on raw counts
        braw = _frame(self.a_raw.X, self.a_raw.var_names, self.ids_kept())
        ar = self.a_raw.copy()
        sc.pp.highly_variable_genes(ar, flavor="seurat_v3", n_top_genes=N_TOP_GENES)
        crest.pp.highly_variable_genes(braw, flavor="seurat_v3", n_top_genes=N_TOP_GENES)
        hs, hc = ar.var["highly_variable"].to_numpy().astype(bool), braw.var["highly_variable"].to_numpy().astype(bool)
        self.put("hvg.seurat_v3.n_genes_different", int((hs != hc).sum()))
        self.put("hvg.seurat_v3.variances_norm.max_rel", max_rel(braw.var["variances_norm"].to_numpy(), ar.var["variances_norm"].to_numpy()))
        rs, rc = ar.var["highly_variable_rank"].to_numpy(), braw.var["highly_variable_rank"].fill_null(np.nan).to_numpy()
        self.put("hvg.seurat_v3.rank.n_different", int(np.sum(~(np.isnan(rs) & np.isnan(rc)) & (rs != rc))))
        # CREST uses scanpy's tutorial gene set from here on, so the later comparisons do not depend on the HVG step
        a.var["highly_variable"] = self.hv
        bf.var = bf.var.with_columns(pl.Series("highly_variable", self.hv))

    # ---- scale + PCA
    def pca(self):
        sc, a, bf = self.sc, self.a, self.bf
        ah = a[:, self.hv].copy()
        sc.pp.scale(ah, max_value=10)
        sc.pp.pca(ah, n_comps=N_COMPS, svd_solver="arpack")
        crest.pp.scale(bf, max_value=10)
        crest.tl.pca(bf, n_comps=N_COMPS, use_highly_variable=True)
        A, B = bf.obsm["X_pca"], ah.obsm["X_pca"]
        self.put("pca.scale_max10.angle_top10_deg", principal_angle_deg(A[:, :10], B[:, :10]))
        self.put("pca.scale_max10.angle_all50_deg", principal_angle_deg(A, B))
        self.put("pca.scale_max10.pc_scores_top10.max_rel_std", sign_aligned_rel(A, B, 10))
        self.put("pca.scale_max10.variance_ratio.max_abs", max_abs(bf.uns["pca"]["variance_ratio"], ah.uns["pca"]["variance_ratio"]))
        self.pca_sc = ah.obsm["X_pca"].astype(np.float32)
        self.ah = ah
        # scale() without arguments does not clip, in both tools
        an = a[:, self.hv].copy()
        sc.pp.scale(an)
        sc.pp.pca(an, n_comps=10, svd_solver="arpack")
        crest.pp.scale(bf)
        crest.tl.pca(bf, n_comps=10, use_highly_variable=True)
        self.put("pca.scale_noclip.angle_top10_deg", principal_angle_deg(bf.obsm["X_pca"], an.obsm["X_pca"]))
        # restore the clipped PCA for the graph steps
        crest.pp.scale(bf, max_value=10)
        crest.tl.pca(bf, n_comps=N_COMPS, use_highly_variable=True)

    # ---- neighbours
    def neighbors(self):
        sc, a, bf = self.sc, self.a, self.bf
        a.obsm["X_pca"] = self.pca_sc
        sc.pp.neighbors(a, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_GRAPH, random_state=0)
        idx_s, dist_s = scanpy_knn(a.obsp["distances"], N_NEIGHBORS)
        X40 = np.asarray(self.pca_sc[:, :N_PCS_GRAPH], np.float64)
        # CREST's search on scanpy's embedding
        import scipy.sparse as sp
        t = crest.BioFrame.from_scipy(sp.csr_matrix((a.n_obs, 1), dtype=np.float32))
        t.obsm["X_pca"] = self.pca_sc
        crest.pp.neighbors(t, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_GRAPH, exact=True)
        idx_c = np.asarray(t.uns["neighbors"]["indices"])
        self.put("knn.crest_exact.recall_shortfall", 1.0 - knn_recall(X40, idx_c))
        self.put("knn.scanpy.recall_shortfall", 1.0 - knn_recall(X40, idx_s))
        self.m["knn.scanpy.recall"] = 1.0 - self.m["knn.scanpy.recall_shortfall"]
        self.put("knn.crest_vs_scanpy.overlap_shortfall", 1.0 - knn_overlap(idx_c, idx_s))
        self.meta["scanpy_knn_columns_in_distances"] = int(np.diff(a.obsp["distances"].tocsr().indptr).max())
        # connectivities from the same neighbour lists: CREST's weights on scanpy's kNN against scanpy's connectivities
        from crest import crest as native
        rows, cols, w = native.connectivities(np.ascontiguousarray(idx_s, np.uint32), np.ascontiguousarray(dist_s, np.float32), N_NEIGHBORS)
        self.edge_types = (rows.dtype, cols.dtype, w.dtype)
        C = sp.coo_matrix((w, (rows, cols)), shape=(a.n_obs, a.n_obs)).tocsr()
        C = (C + C.T).tocsr()
        S = a.obsp["connectivities"].tocsr()
        p, d, r = sparse_diff(C, S)
        self.put("connectivities.same_knn.pattern_differences", p)
        self.put("connectivities.same_knn.max_abs", d)
        self.S, self.idx_s = S, idx_s
        self.rgc = (t, X40)

    # ---- Leiden on scanpy's graph
    def leiden(self):
        import igraph as ig
        import leidenalg as la
        import scipy.sparse as sp
        from sklearn.metrics import adjusted_rand_score
        from crest import crest as native
        sc, a = self.sc, self.a
        S = sp.triu(self.S, k=1).tocoo()
        rows, cols, w = (S.row.astype(self.edge_types[0]), S.col.astype(self.edge_types[1]), S.data.astype(self.edge_types[2]))
        g = ig.Graph(n=a.n_obs, edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
        g.es["weight"] = w.astype(float).tolist()
        q = lambda lab: g.modularity(list(map(int, lab)), weights="weight")  # noqa: E731
        res = {"crest": [], "leidenalg": [], "igraph": []}
        ari = []
        for s in SEEDS_LEIDEN:
            lc = np.asarray(native.leiden(a.n_obs, rows, cols, w, 1.0, 2, s))
            ll = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0,
                                   n_iterations=-1, seed=s).membership
            sc.tl.leiden(a, resolution=1.0, random_state=s, flavor="igraph", n_iterations=2, directed=False, key_added="leiden_ig")
            li = a.obs["leiden_ig"].to_numpy().astype(int)
            for k, lab in (("crest", lc), ("leidenalg", ll), ("igraph", li)):
                res[k].append((q(lab), len(set(map(int, lab)))))
            ari.append(adjusted_rand_score(lc, li))
            if s == 0:
                self.labels = np.array([str(x) for x in li])
        mean = {k: np.mean(v, axis=0) for k, v in res.items()}
        self.put("leiden.modularity_shortfall_vs_leidenalg", max(0.0, mean["leidenalg"][0] - mean["crest"][0]))
        self.put("leiden.modularity_shortfall_vs_scanpy_igraph", max(0.0, mean["igraph"][0] - mean["crest"][0]))
        self.put("leiden.n_clusters_mean_diff_vs_leidenalg", abs(mean["crest"][1] - mean["leidenalg"][1]))
        self.put("leiden.n_clusters_mean_diff_vs_scanpy_igraph", abs(mean["crest"][1] - mean["igraph"][1]))
        self.meta["leiden"] = {k: {"modularity": float(v[0]), "n_clusters": float(v[1])} for k, v in mean.items()}
        self.meta["leiden_ari_crest_vs_scanpy_igraph_mean"] = float(np.mean(ari))
        a.obs["g"] = self.labels

    # ---- UMAP trustworthiness (CREST runs in a one-thread subprocess: its parallel SGD uses another schedule)
    def umap(self):
        import os
        import subprocess
        import tempfile
        from sklearn.manifold import trustworthiness
        sc, a = self.sc, self.a
        t, X40 = self.rgc
        sub = np.sort(np.random.default_rng(0).choice(a.n_obs, min(TRUST_CELLS, a.n_obs), replace=False))
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "x.npy", self.pca_sc)
            env = dict(os.environ, RAYON_NUM_THREADS="1", PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
            subprocess.run([sys.executable, "-c", _UMAP_SCRIPT, str(Path(d) / "x.npy"), str(N_NEIGHBORS), str(N_PCS_GRAPH),
                            str(Path(d) / "u"), *map(str, SEEDS_UMAP)], check=True, env=env)
            tc = [trustworthiness(X40[sub], np.load(Path(d) / f"u_{s}.npy")[sub], n_neighbors=TRUST_K) for s in SEEDS_UMAP]
        ts = []
        for s in SEEDS_UMAP:
            sc.tl.umap(a, random_state=s)
            ts.append(trustworthiness(X40[sub], np.asarray(a.obsm["X_umap"])[sub], n_neighbors=TRUST_K))
        self.put("umap.trustworthiness_gap", abs(np.mean(tc) - np.mean(ts)))
        self.put("umap.trustworthiness_shortfall", max(0.0, np.mean(ts) - np.mean(tc)))
        self.meta["umap_trustworthiness"] = {"crest": [float(x) for x in tc], "scanpy": [float(x) for x in ts]}

    # ---- differential expression on the shared labels
    def rank_genes(self):
        import anndata as ad_mod
        sc, a, bf = self.sc, self.a, self.bf
        g = self.labels
        bf.obs = bf.obs.with_columns(pl.Series("g", g))
        a.obs["g"] = g
        for path in ("identical", "lazy"):
            if path == "identical":
                ad_in = ad_mod.AnnData(bf.to_scipy(), var=a.var[[]].copy())
                ad_in.obs["g"] = g
                ad_in.uns["log1p"] = {"base": None}
            else:
                ad_in = ad_mod.AnnData(self.a_raw.X.copy(), var=a.var[[]].copy())
                ad_in.obs["g"] = g
                sc.pp.normalize_total(ad_in, target_sum=1e4)
                sc.pp.log1p(ad_in)
            for method in ("t-test", "wilcoxon"):
                sc.tl.rank_genes_groups(ad_in, "g", method=method, n_genes=ad_in.n_vars, use_raw=False, reference="rest")
                res = crest.tl.rank_genes_groups(bf, "g", method=method, n_genes=None)
                rg = ad_in.uns["rank_genes_groups"]
                worst = dict(scores=0.0, logfoldchanges=0.0, pvals=0.0, pvals_adj=0.0)
                top50 = 1.0
                for grp in rg["names"].dtype.names:
                    ref = {n: i for i, n in enumerate(rg["names"][grp])}
                    sub = res.filter(pl.col("group") == grp)
                    order = np.array([ref[n] for n in sub["names"]])
                    for col, key in (("scores", "scores"), ("logfoldchanges", "logfoldchanges"), ("pvals", "pvals"), ("pvals_adj", "pvals_adj")):
                        worst[key] = max(worst[key], max_abs(sub[col].to_numpy(), np.asarray(rg[key][grp])[order]))
                    top50 = min(top50, len(set(sub["names"].to_list()[:50]) & set(rg["names"][grp][:50].tolist())) / 50)
                tag = f"rank_genes_groups.{method.replace('-', '')}.{path}"
                for key, v in worst.items():
                    self.put(f"{tag}.{key}.max_abs", v)
                self.put(f"{tag}.top50_overlap_shortfall", 1.0 - top50)

    # ---- score_genes
    def score_genes(self):
        sc, a, bf = self.sc, self.a, self.bf
        present = [g for g in B_MARKERS if g in set(a.var_names)]
        rng = np.random.default_rng(0)
        pool = [n for n in a.var_names if not n[:3] == "MT-"]
        sets = {"b_markers": present, "random_25": list(rng.choice(pool, 25, replace=False))}
        for tag, genes in sets.items():
            sc.tl.score_genes(a, genes, random_state=0, score_name="score_sc")
            crest.tl.score_genes(bf, genes, random_state=0)
            self.put(f"score_genes.{tag}.max_abs", max_abs(bf.obs["score"].to_numpy(), a.obs["score_sc"].to_numpy()))
        self.meta["score_genes_b_markers_present"] = len(present)

    # ---- Scrublet on raw counts with spiked doublets
    def scrublet(self):
        import scipy.sparse as sp
        from scipy import stats
        from sklearn.metrics import roc_auc_score
        sc, X = self.sc, self.X
        n, n_spiked = X.shape[0], self.spec["n_spiked"]
        ratios, rhos, thr, rate, auc_d = [], [], [], [], []
        for seed in SEEDS_SCRUBLET:
            rng = np.random.default_rng(100 + seed)
            i = rng.integers(0, n, n_spiked)
            j = (i + rng.integers(1, n, n_spiked)) % n
            Y = sp.vstack([X, X[i] + X[j]]).tocsr()
            truth = np.r_[np.zeros(n), np.ones(n_spiked)]
            bf = _frame(Y, self.names, self.ids)
            crest.pp.scrublet(bf, random_state=seed)
            a = _adata(Y, self.names)
            sc.pp.scrublet(a, random_state=seed)
            c, s = bf.obs["doublet_score"].to_numpy().astype(float), a.obs["doublet_score"].to_numpy().astype(float)
            ok = ~np.isnan(c) & ~np.isnan(s)
            ratios.append(c[ok].mean() / s[ok].mean())
            rhos.append(stats.spearmanr(c[ok], s[ok])[0])
            thr.append(abs(float(bf.uns["scrublet"]["threshold"]) - float(a.uns["scrublet"]["threshold"])))
            rate.append(abs(float(np.mean(bf.obs["predicted_doublet"].to_numpy())) - float(np.mean(a.obs["predicted_doublet"].to_numpy()))))
            auc_d.append(abs(roc_auc_score(truth[ok], c[ok]) - roc_auc_score(truth[ok], s[ok])))
        self.put("scrublet.mean_score_ratio_dev", abs(np.mean(ratios) - 1.0))
        self.put("scrublet.spearman_shortfall_worst_seed", 1.0 - min(rhos))
        self.put("scrublet.threshold_abs_diff_max", max(thr))
        self.put("scrublet.predicted_rate_abs_diff_max", max(rate))
        self.put("scrublet.auroc_spiked_abs_diff_max", max(auc_d))
        self.meta["scrublet"] = {"ratio": [float(x) for x in ratios], "spearman": [float(x) for x in rhos]}

    # ---- ingest: reference = random half, query = other half
    def ingest(self):
        sc, a = self.sc, self.a
        from scipy.sparse import issparse
        rng = np.random.default_rng(0)
        perm = rng.permutation(a.n_obs)
        iq, ir = np.sort(perm[: a.n_obs // 2]), np.sort(perm[a.n_obs // 2:])
        labels = self.labels
        Xr = self.a_raw.X
        hv = self.hv
        names = list(self.a.var_names)
        ids = self.ids_kept()
        centre_matching = scanpy_ingest_centering()
        self.meta["scanpy_ingest_centering"] = centre_matching

        def prep_frame(rows):
            b = _frame(Xr[rows], names, ids)
            crest.pp.normalize_total(b, 1e4)
            crest.pp.log1p(b)
            b.var = b.var.with_columns(pl.Series("highly_variable", hv))
            return b

        def prep_adata(rows):
            x = _adata(Xr[rows], names)
            sc.pp.normalize_total(x, target_sum=1e4)
            sc.pp.log1p(x)
            return x

        ref_s = prep_adata(ir)
        ref_s = ref_s[:, hv].copy()
        sc.pp.scale(ref_s, max_value=10)
        sc.pp.pca(ref_s, n_comps=N_COMPS, svd_solver="arpack")
        sc.pp.neighbors(ref_s, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_GRAPH, random_state=0)
        ref_s.obs["leiden"] = [labels[i] for i in ir]
        ref_s.obs["leiden"] = ref_s.obs["leiden"].astype("category")
        ref_c = prep_frame(ir)
        crest.pp.scale(ref_c, max_value=10)
        crest.tl.pca(ref_c, n_comps=N_COMPS, use_highly_variable=True)
        crest.pp.neighbors(ref_c, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_GRAPH, exact=True)
        ref_c.obs = ref_c.obs.with_columns(pl.Series("leiden", [labels[i] for i in ir]))
        qry_c = prep_frame(iq)
        # the query scaled with the reference statistics, handed to scanpy as it is (scanpy projects X as supplied)
        P = ref_c.uns["pca"]["projection"]
        Q = qry_c.to_scipy().tocsc()[:, np.flatnonzero(hv)].toarray().astype(np.float64)
        Qs = np.clip((Q - np.asarray(P["mean"])) / np.asarray(P["std"]), -10, 10).astype(np.float32)
        import anndata as ad
        import pandas as pd
        qry_s0 = ad.AnnData(Qs, var=pd.DataFrame(index=list(ref_s.var_names)))
        truth = np.array([labels[i] for i in iq])
        accs = {}
        for centre in ("query", "reference"):
            qry_s = qry_s0.copy()
            sc.tl.ingest(qry_s, ref_s, obs="leiden", embedding_method="pca")
            out = crest.ingest.ingest(qry_c, ref_c, obs="leiden", embedding_method=("pca",), center=centre)
            pc_s, pc_c = np.asarray(qry_s.obsm["X_pca"]), np.asarray(out.obsm["X_pca"])
            lab_s, lab_c = qry_s.obs["leiden"].astype(str).to_numpy(), out.obs["leiden"].to_numpy().astype(str)
            tag = f"ingest.centre_{centre}"
            self.put(f"{tag}.pca_top10.max_rel_std", sign_aligned_rel(pc_c, pc_s, 10))
            self.put(f"{tag}.label_disagreement", np.mean(lab_s != lab_c))
            self.put(f"{tag}.label_accuracy_abs_diff", abs(np.mean(lab_s == truth) - np.mean(lab_c == truth)))
            accs[centre] = (float(np.mean(lab_s == truth)), float(np.mean(lab_c == truth)))
        self.meta["ingest_label_accuracy_scanpy_crest"] = accs
        # the comparison that matches the installed scanpy
        for k in ("pca_top10.max_rel_std", "label_disagreement", "label_accuracy_abs_diff"):
            self.m[f"ingest.matching_centre.{k}"] = self.m[f"ingest.centre_{centre_matching}.{k}"]

    def run(self):
        self.meta.update(dataset=self.name, **versions())
        for fn in (self.qc, self.normalize, self.hvg, self.pca, self.neighbors, self.leiden, self.umap,
                   self.rank_genes, self.score_genes, self.scrublet, self.ingest):
            self.section(fn)
            if fn.__name__ == "qc" and "qc" in self.errors:
                break
        self.meta["seconds"] = self.seconds
        return {"meta": self.meta, "metrics": self.m, "errors": self.errors}


def versions():
    from importlib import metadata

    def v(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    out = {"crest": v("crest-sc"), "python": platform.python_version(), "machine": platform.machine(), "system": platform.system()}
    for pkg in ("scanpy", "anndata", "numpy", "scipy", "pandas", "polars", "scikit-learn", "numba", "pynndescent",
                "umap-learn", "igraph", "leidenalg", "scikit-misc", "scikit-image", "h5py"):
        out[pkg] = v(pkg)
    return out


def measure(name, path):
    X, names, ids = load_raw(path)
    return Measure(name, X, names, ids, DATASETS[name]).run()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="+", help="name=path, name is pbmc3k or pbmc10k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = {}
    for item in args.datasets:
        name, path = item.split("=", 1)
        print(f"measuring {name}", flush=True)
        result[name] = measure(name, path)
        print(f"  errors: {list(result[name]['errors'])}  seconds: {result[name]['meta']['seconds']}", flush=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
