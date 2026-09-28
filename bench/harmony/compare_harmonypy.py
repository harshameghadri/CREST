"""CREST Harmony vs harmonypy 2.x on Kang et al. 2018 (ctrl vs stim PBMCs).

Both get the identical PCA (CREST pipeline, 2000 HVG, 50 PCs) and the same
parameters. Harmony's k-means initialisation is random, so the comparison is
on integration quality (LISI, ARI of Leiden vs annotated cell types, kNN
agreement between the two corrected embeddings), not bitwise equality.
Seeds 0..S-1 are run for both tools to show the seed-to-seed spread.

    python bench/harmony/compare_harmonypy.py [--seeds 3] [--out bench/harmony/results]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deseq2"))
import crest  # noqa: E402
from kang_pseudobulk import fetch, load  # noqa: E402


def lisi(X, labels, perplexity=30):
    """LISI (Korsunsky et al. 2019): inverse Simpson index of the labels in each
    cell's 3*perplexity neighbourhood, Gaussian weights calibrated to the
    perplexity by bisection (as harmonypy's compute_lisi), vectorised."""
    k = 3 * perplexity
    idx, dist = crest.crest.knn_graph(np.ascontiguousarray(X, np.float32), k, exact=None, seed=0)
    D = dist.astype(np.float64) ** 2
    target = np.log(perplexity)
    beta = np.ones(len(D)); lo = np.full(len(D), -np.inf); hi = np.full(len(D), np.inf)
    for _ in range(50):
        P = np.exp(-(D - D[:, :1]) * beta[:, None])
        sP = P.sum(1)
        H = np.log(sP) + beta * ((D - D[:, :1]) * P).sum(1) / sP
        too_high = H > target  # entropy too high -> sharper kernel (larger beta)
        lo = np.where(too_high, beta, lo); hi = np.where(too_high, hi, beta)
        beta = np.where(too_high, np.where(np.isinf(hi), beta * 2, (beta + hi) / 2),
                        np.where(np.isinf(lo), beta / 2, (beta + lo) / 2))
    P = np.exp(-(D - D[:, :1]) * beta[:, None]); P /= P.sum(1, keepdims=True)
    codes = np.unique(labels, return_inverse=True)[1]
    nbr = codes[idx]
    simpson = np.zeros(len(D))
    for c in range(codes.max() + 1):
        simpson += ((nbr == c) * P).sum(1) ** 2
    return 1.0 / simpson


def knn_sets(X, k=15):
    idx, _ = crest.crest.knn_graph(np.ascontiguousarray(X, np.float32), k, exact=None, seed=0)
    return idx


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def leiden_labels(X, obs):
    bf = crest.BioFrame.from_scipy(__import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix((X.shape[0], 1)), obs=obs)
    bf.obsm["X"] = np.ascontiguousarray(X, np.float32)
    crest.pp.neighbors(bf, n_neighbors=15, use_rep="X")
    crest.tl.leiden(bf, resolution=0.5)
    return bf.obs["leiden"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/kang")
    ap.add_argument("--out", default="bench/harmony/results")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch(Path(a.data))
    bf = load(Path(a.data))
    bf = crest.pp.filter_genes(bf, min_cells=10)
    crest.pp.normalize_total(bf, 1e4); crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=2000)
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=50)
    X = bf.obsm["X_pca"]
    batch = bf.obs["stim"].to_list()
    ctype = bf.obs["cell"].to_numpy()
    obs = bf.obs.select("stim", "cell")
    print(f"{X.shape[0]} cells, 50 PCs")

    import harmonypy
    rows, emb = [], {}
    base = {"tool": "uncorrected", "seed": -1, "time_s": 0.0, "Z": X}
    for tool in ("crest", "harmonypy"):
        for s in range(a.seeds):
            t0 = time.perf_counter()
            if tool == "crest":
                Z = crest.harmony.run_harmony(X, [batch], random_state=s)["Z_corr"]
            else:
                import pandas as pd
                Z = harmonypy.run_harmony(X, pd.DataFrame({"stim": batch}), "stim", random_state=s, verbose=False).Z_corr
            dt = time.perf_counter() - t0
            emb[(tool, s)] = np.asarray(Z, np.float32)
            rows.append({"tool": tool, "seed": s, "time_s": dt, "Z": np.asarray(Z, np.float32)})
            print(tool, s, f"{dt:.2f}s")
    res = []
    for r in [base] + rows:
        Z = r["Z"]
        il = lisi(Z, np.array(batch)); cl = lisi(Z, ctype)
        lab = leiden_labels(Z, obs)
        res.append({"tool": r["tool"], "seed": r["seed"], "time_s": r["time_s"],
                    "iLISI_median": float(np.median(il)), "cLISI_median": float(np.median(cl)),
                    "ARI_leiden_vs_celltype": ari(lab, ctype)})
        print(res[-1])
    # agreement of neighbourhoods between the tools (seed 0) and seed-to-seed within a tool
    def overlap(A, B, k=15):
        ia, ib = knn_sets(A, k), knn_sets(B, k)
        return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib)]))
    agree = {"kNN15_overlap_crest0_vs_harmonypy0": overlap(emb[("crest", 0)], emb[("harmonypy", 0)])}
    if a.seeds > 1:
        agree["kNN15_overlap_crest0_vs_crest1"] = overlap(emb[("crest", 0)], emb[("crest", 1)])
        agree["kNN15_overlap_harmonypy0_vs_harmonypy1"] = overlap(emb[("harmonypy", 0)], emb[("harmonypy", 1)])
    print(agree)
    df = pl.DataFrame(res)
    df.write_csv(out / "kang_harmony.csv")
    summ = df.group_by("tool", maintain_order=True).agg(pl.all().exclude("seed").median())
    with open(out / "kang_harmony.md", "w") as fh:
        fh.write(f"Kang 2018: {X.shape[0]} cells, 50 PCs, batch = stim; median over {a.seeds} seeds.\n\n")
        fh.write("| " + " | ".join(summ.columns) + " |\n|" + "---|" * len(summ.columns) + "\n")
        for row in summ.iter_rows():
            fh.write("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row) + " |\n")
        fh.write("\nNeighbourhood agreement (mean fraction of shared 15-NN):\n\n")
        for k_, v in agree.items():
            fh.write(f"* {k_}: {v:.3f}\n")


if __name__ == "__main__":
    main()
