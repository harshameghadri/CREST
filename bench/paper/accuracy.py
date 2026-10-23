"""Agreement between CREST and scanpy outputs of the same dataset.

Cells are matched by barcode. Metrics:

* leiden_ari / leiden_nmi   - clusterings (both at resolution 1), each on its own graph
* leiden_ari_seeds_crest / _scanpy / _cross_mean - mean pairwise ARI over 3 seeds within each
  tool and across tools: the seed-to-seed yardstick for leiden_ari
* same_graph_leiden_ari, same_graph_modularity_crest / _scanpy - CREST's Leiden and scanpy's
  Leiden (flavor="igraph", n_iterations=2) on the SAME graph (scanpy's connectivities); both
  partitions scored by igraph's Graph.modularity (resolution 1), independent of either tool
* knn_graph_jaccard         - mean Jaccard of each cell's neighbour set in the kNN graphs the
                              two tools actually built (CREST's neighbour graph vs scanpy's
                              obsp['distances'], n_neighbors - 1 = 14 neighbours, self excluded)
* knn_recall_crest / _scanpy - recall@k of each tool's kNN graph against brute-force exact
                              neighbours in that tool's own PCA space, for up to 10,000 random
                              query cells against ALL cells. CREST is exact up to 20k cells, so
                              1.0 there is by construction
* pca_knn15_jaccard         - mean Jaccard of 15-NN sets recomputed by CREST's kNN in each tool's
                              PCA space. NOT a comparison of the tools' graphs: it measures how
                              similar the two PCA embeddings are as neighbourhoods (above 20k
                              cells both sides use the approximate index)
* pca_subspace_cos_min/mean - cosines of the principal angles between the two
                              50-d PC score subspaces (1 = same subspace)
* pca_abs_corr_pc1..5       - |Pearson r| of matched PCs (sign is arbitrary)
* hvg_jaccard, hvg_v3_jaccard
* umap_trustworthiness_*    - sklearn trustworthiness of each tool's UMAP w.r.t.
                              its own PCA (15-NN, 5,000-cell subsample)
* de_top50_overlap          - mean overlap of top-50 Wilcoxon markers of matched clusters.
                              Clusters are paired greedily: each CREST cluster with the scanpy
                              cluster sharing the most cells (not a one-to-one assignment; two
                              CREST clusters can map to the same scanpy cluster). Kept as is,
                              documented here
* doublet_spearman, harmony_space_knn15_jaccard (recomputed 15-NN in the two Harmony
  embeddings), sweep_ari_r*  (full profile)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RECALL_QUERIES = 10_000


def _align(a: dict, b: dict):
    ia = {x: i for i, x in enumerate(a["barcodes"])}
    common = [x for x in b["barcodes"] if x in ia]
    ib = {x: i for i, x in enumerate(b["barcodes"])}
    return np.array([ia[x] for x in common]), np.array([ib[x] for x in common])


def _knn(X, k=15):
    import crest
    idx, _ = crest.crest.knn_graph(np.ascontiguousarray(X, np.float32), k, exact=None, seed=0)
    return idx


def _jaccard_rows(A, B):
    """Mean Jaccard of row-wise neighbour sets; -1 entries are padding and ignored."""
    out = []
    for x, y in zip(A, B):
        sx, sy = set(int(v) for v in x if v >= 0), set(int(v) for v in y if v >= 0)
        u = len(sx | sy)
        out.append(len(sx & sy) / u if u else 1.0)
    return float(np.mean(out)) if out else float("nan")


def _sets_jaccard(x, y):
    x, y = set(map(str, x)), set(map(str, y))
    return len(x & y) / max(len(x | y), 1)


def _jaccard_common(A, B):
    out = []
    for x, y in zip(A, B):
        # -1 = padding (ignored); -2 = neighbour outside the common cells (counted, never shared)
        sx = {int(v) for v in x if v >= 0}
        sy = {int(v) for v in y if v >= 0}
        nx = len(sx) + int(np.sum(x < -1))
        ny = len(sy) + int(np.sum(y < -1))
        inter = len(sx & sy)
        u = nx + ny - inter
        out.append(inter / u if u else 1.0)
    return float(np.mean(out)) if out else float("nan")


def knn_recall(X, knn, n_query=RECALL_QUERIES, seed=0) -> dict:
    """Recall@k of a kNN graph (n, k) against exact brute-force neighbours (self excluded)
    for up to ``n_query`` random query cells against all cells, in float64 (distance blocks
    of at most ~50M entries, i.e. ~0.4 GB, at any n)."""
    X = np.asarray(X, np.float64)
    n, k = knn.shape
    chunk = int(max(16, min(512, 50_000_000 // max(n, 1))))
    rng = np.random.default_rng(seed)
    q = np.sort(rng.choice(n, min(n_query, n), replace=False))
    sq = np.einsum("ij,ij->i", X, X)
    hits, total = 0, 0
    for lo in range(0, len(q), chunk):
        rows = q[lo:lo + chunk]
        D = sq[rows, None] - 2.0 * X[rows] @ X.T + sq[None, :]
        D[np.arange(len(rows)), rows] = np.inf  # exclude self
        exact = np.argpartition(D, k - 1, axis=1)[:, :k]
        for r, e in zip(rows, exact):
            g = knn[r]
            g = g[g >= 0]
            hits += len(set(g.tolist()) & set(e.tolist()))
            total += k
    return {"recall": hits / total if total else float("nan"), "n_query": int(len(q)), "k": int(k)}


def compare(crest_npz: Path, scanpy_npz: Path) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.manifold import trustworthiness
    a, b = dict(np.load(crest_npz, allow_pickle=False)), dict(np.load(scanpy_npz, allow_pickle=False))
    ia, ib = _align(a, b)
    m = {"n_common_cells": int(len(ia)), "n_cells_crest": int(len(a["barcodes"])), "n_cells_scanpy": int(len(b["barcodes"]))}
    la, lb = a["leiden"][ia], b["leiden"][ib]
    m["leiden_ari"] = float(adjusted_rand_score(la, lb))
    m["leiden_nmi"] = float(normalized_mutual_info_score(la, lb))
    m["n_clusters_crest"], m["n_clusters_scanpy"] = int(len(np.unique(la))), int(len(np.unique(lb)))
    # seed baseline: how much each tool disagrees with itself on its own graph. CREST vs
    # scanpy agreement is only meaningful relative to this.
    if "leiden_seeds" in a and "leiden_seeds" in b:
        ra = [la] + [x[ia] for x in a["leiden_seeds"]]
        rb = [lb] + [x[ib] for x in b["leiden_seeds"]]
        pair = lambda runs: [adjusted_rand_score(runs[i], runs[j])  # noqa: E731
                             for i in range(len(runs)) for j in range(i + 1, len(runs))]
        m["leiden_ari_seeds_crest"] = float(np.mean(pair(ra)))
        m["leiden_ari_seeds_scanpy"] = float(np.mean(pair(rb)))
        m["leiden_ari_cross_mean"] = float(np.mean([adjusted_rand_score(x, y) for x in ra for y in rb]))
    # same graph (scanpy's connectivities), both Leiden implementations, igraph modularity
    if "same_graph_leiden_crest" in b:
        sgc, sgs = b["same_graph_leiden_crest"], b["same_graph_leiden_scanpy"]
        m["same_graph_leiden_ari"] = float(adjusted_rand_score(sgc, sgs))
        m["same_graph_n_clusters_crest"] = int(len(np.unique(sgc)))
        m["same_graph_n_clusters_scanpy"] = int(len(np.unique(sgs)))
        m["same_graph_modularity_crest"] = float(b["same_graph_modularity_crest"])
        m["same_graph_modularity_scanpy"] = float(b["same_graph_modularity_scanpy"])
        m["same_graph_modularity_diff"] = m["same_graph_modularity_crest"] - m["same_graph_modularity_scanpy"]
    # the kNN graphs the tools actually built
    if "knn_indices" in a and "knn_indices" in b:
        m["knn_graph_jaccard"] = _jaccard_common(*_remap_pair(a["knn_indices"], b["knn_indices"], ia, ib))
        m["knn_graph_k"] = int(a["knn_indices"].shape[1])
        rc = knn_recall(a["pca"], a["knn_indices"])
        rs = knn_recall(b["pca"], b["knn_indices"])
        m["knn_recall_crest"], m["knn_recall_scanpy"] = rc["recall"], rs["recall"]
        m["knn_recall_n_query"] = rc["n_query"]
    Pa, Pb = a["pca"][ia].astype(np.float64), b["pca"][ib].astype(np.float64)
    m["pca_knn15_jaccard"] = _jaccard_rows(_knn(Pa), _knn(Pb))
    Qa, _ = np.linalg.qr(Pa - Pa.mean(0))
    Qb, _ = np.linalg.qr(Pb - Pb.mean(0))
    cos = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
    m["pca_subspace_cos_min"], m["pca_subspace_cos_mean"] = float(cos.min()), float(cos.mean())
    for j in range(min(5, Pa.shape[1])):
        m[f"pca_abs_corr_pc{j + 1}"] = float(abs(np.corrcoef(Pa[:, j], Pb[:, j])[0, 1]))
    m["hvg_jaccard"] = _sets_jaccard(a["hvg"], b["hvg"])
    if "hvg_v3" in a and "hvg_v3" in b:
        m["hvg_v3_jaccard"] = _sets_jaccard(a["hvg_v3"], b["hvg_v3"])
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ia), min(5000, len(ia)), replace=False)
    m["umap_trustworthiness_crest"] = float(trustworthiness(Pa[sub], a["umap"][ia][sub], n_neighbors=15))
    m["umap_trustworthiness_scanpy"] = float(trustworthiness(Pb[sub], b["umap"][ib][sub], n_neighbors=15))
    # DE: pair clusters greedily by maximal cell overlap (see module docstring), compare top-50
    ga, gb = [str(x) for x in a["de_groups"]], [str(x) for x in b["de_groups"]]
    ov = []
    for i, g in enumerate(ga):
        ca = la == int(g)
        if not ca.any():
            continue
        best = max(range(len(gb)), key=lambda j: np.sum(ca & (lb == int(gb[j]))))
        ov.append(len(set(a["de_top"][i]) & set(b["de_top"][best])) / 50)
    m["de_top50_overlap"] = float(np.mean(ov)) if ov else None
    if "doublet_score" in a and "doublet_score" in b:
        from scipy.stats import spearmanr
        sa, sb = a["doublet_score"], b["doublet_score"]
        # scrublet ran before QC in neither tool; scores align with the post-QC cells
        if len(sa) == len(a["barcodes"]) and len(sb) == len(b["barcodes"]):
            sa, sb = sa[ia], sb[ib]
            ok = np.isfinite(sa) & np.isfinite(sb)
            m["doublet_spearman"] = float(spearmanr(sa[ok], sb[ok]).statistic)
    if "harmony" in a and "harmony" in b:
        m["harmony_space_knn15_jaccard"] = _jaccard_rows(_knn(a["harmony"][ia]), _knn(b["harmony"][ib]))
    if "sweep" in a and "sweep" in b:
        for r, (x, y) in enumerate(zip(a["sweep"], b["sweep"])):
            m[f"sweep_ari_{r}"] = float(adjusted_rand_score(x[ia], y[ib]))
    return m


def _remap_pair(knn_a, knn_b, ia, ib):
    """Map each tool's neighbour indices (into its own cell order) to positions in the common,
    barcode-aligned order. -1 stays padding; a neighbour outside the common cells becomes -2."""
    def remap(knn, idx):
        pos = np.full(len(knn), -2, np.int64)
        pos[idx] = np.arange(len(idx))
        K = knn[idx].astype(np.int64)
        return np.where(K >= 0, pos[np.clip(K, 0, len(knn) - 1)], -1)
    return remap(knn_a, ia), remap(knn_b, ib)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="directory with *.outputs.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.raw)
    rows = []
    for f in sorted(raw.glob("crest__*__r0.outputs.npz")):
        _, ds, prof, th, _ = f.name.split(".")[0].split("__")
        g = raw / f"scanpy__{ds}__{prof}__{th}__r0.outputs.npz"
        if g.exists():
            print(f"accuracy: {ds}")
            rows.append({"dataset": ds, "profile": prof, "threads": th, **compare(f, g)})
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
