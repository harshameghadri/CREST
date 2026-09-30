"""Agreement between CREST and scanpy outputs of the same dataset.

Cells are matched by barcode. Metrics:

* leiden_ari / leiden_nmi   - clusterings (both at resolution 1)
* leiden_ari_seeds_crest / _scanpy / _cross_mean - mean pairwise ARI over 3 seeds within each
  tool and across tools (same graph): the seed-to-seed yardstick for leiden_ari
* knn15_jaccard             - mean Jaccard of each cell's 15 nearest neighbours in
                              the two PCA spaces (CREST's exact kNN on both)
* pca_subspace_cos_min/mean - cosines of the principal angles between the two
                              50-d PC score subspaces (1 = same subspace)
* pca_abs_corr_pc1..5       - |Pearson r| of matched PCs (sign is arbitrary)
* hvg_jaccard, hvg_v3_jaccard
* umap_trustworthiness_*    - sklearn trustworthiness of each tool's UMAP w.r.t.
                              its own PCA (15-NN, 5,000-cell subsample)
* de_top50_overlap          - mean overlap of top-50 Wilcoxon markers of matched
                              clusters (clusters paired by maximum overlap)
* doublet_spearman, harmony_knn15_jaccard, sweep_ari_r*  (full profile)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
    k = A.shape[1]
    return float(np.mean([len(set(x) & set(y)) / (2 * k - len(set(x) & set(y))) for x, y in zip(A, B)]))


def _sets_jaccard(x, y):
    x, y = set(map(str, x)), set(map(str, y))
    return len(x & y) / max(len(x | y), 1)


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
    # seed baseline: how much each tool disagrees with itself on the same graph. CREST vs
    # scanpy agreement is only meaningful relative to this.
    if "leiden_seeds" in a and "leiden_seeds" in b:
        ra = [la] + [x[ia] for x in a["leiden_seeds"]]
        rb = [lb] + [x[ib] for x in b["leiden_seeds"]]
        pair = lambda runs: [adjusted_rand_score(runs[i], runs[j])  # noqa: E731
                             for i in range(len(runs)) for j in range(i + 1, len(runs))]
        m["leiden_ari_seeds_crest"] = float(np.mean(pair(ra)))
        m["leiden_ari_seeds_scanpy"] = float(np.mean(pair(rb)))
        m["leiden_ari_cross_mean"] = float(np.mean([adjusted_rand_score(x, y) for x in ra for y in rb]))
    Pa, Pb = a["pca"][ia].astype(np.float64), b["pca"][ib].astype(np.float64)
    m["knn15_jaccard"] = _jaccard_rows(_knn(Pa), _knn(Pb))
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
    # DE: pair clusters by maximal cell overlap, compare their top-50 marker lists
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
        m["harmony_knn15_jaccard"] = _jaccard_rows(_knn(a["harmony"][ia]), _knn(b["harmony"][ib]))
    if "sweep" in a and "sweep" in b:
        for r, (x, y) in enumerate(zip(a["sweep"], b["sweep"])):
            m[f"sweep_ari_{r}"] = float(adjusted_rand_score(x[ia], y[ib]))
    return m


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
            rows.append({"dataset": ds, **compare(f, g)})
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
