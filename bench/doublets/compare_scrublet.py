"""CREST Scrublet vs scanpy.pp.scrublet on Kang et al. 2018 with demuxlet truth.

Kang et al. multiplexed 8 donors per lane; demuxlet flags droplets holding
cells of two donors ("doublet"). Those are the ground truth here (inter-donor
doublets only, so recall is bounded by what expression can reveal). Both tools
run per lane-condition batch (``stim``) with default parameters.

    python bench/doublets/compare_scrublet.py [--out bench/doublets/results]
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


def auc(score, truth):
    from sklearn.metrics import roc_auc_score, average_precision_score
    ok = np.isfinite(score)
    return roc_auc_score(truth[ok], score[ok]), average_precision_score(truth[ok], score[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/kang")
    ap.add_argument("--out", default="bench/doublets/results")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch(Path(a.data))
    bf = load(Path(a.data), singlets_only=False)
    bf = bf.filter_cells(pl.col("multiplets").is_in(["singlet", "doublet"]))
    truth = (bf.obs["multiplets"] == "doublet").to_numpy()
    print(f"{bf.n_obs} droplets, {truth.sum()} demuxlet doublets ({truth.mean():.1%})")

    import scanpy as sc
    rows = []
    for s in range(a.seeds):
        t0 = time.perf_counter()
        b = bf.copy()
        crest.pp.scrublet(b, batch_key="stim", random_state=s)
        tc = time.perf_counter() - t0
        sc_c = b.obs["doublet_score"].fill_null(np.nan).to_numpy()
        pred_c = b.obs["predicted_doublet"].to_numpy()

        ad = bf.to_anndata()
        t0 = time.perf_counter()
        sc.pp.scrublet(ad, batch_key="stim", random_state=s, verbose=False)
        ts = time.perf_counter() - t0
        sc_s = ad.obs["doublet_score"].to_numpy(float)
        pred_s = ad.obs["predicted_doublet"].to_numpy(bool)
        for tool, score, pred, t in (("crest", sc_c, pred_c, tc), ("scanpy", sc_s, pred_s, ts)):
            roc, pr = auc(score, truth)
            rows.append({"tool": tool, "seed": s, "time_s": t, "AUROC": roc, "AUPRC": pr,
                         "called": int(pred.sum()), "recall": float((pred & truth).sum() / truth.sum()),
                         "precision": float((pred & truth).sum() / max(pred.sum(), 1))})
            print(rows[-1])
        ok = np.isfinite(sc_c) & np.isfinite(sc_s)
        from scipy.stats import spearmanr
        rows[-1]["spearman_vs_crest"] = float(spearmanr(sc_c[ok], sc_s[ok]).statistic)
        print("spearman crest vs scanpy scores:", rows[-1]["spearman_vs_crest"])
    df = pl.DataFrame(rows)
    df.write_csv(out / "kang_scrublet.csv")
    summ = df.drop("seed").group_by("tool", maintain_order=True).median()
    with open(out / "kang_scrublet.md", "w") as fh:
        fh.write(f"Kang 2018: {bf.n_obs} droplets, {truth.sum()} demuxlet doublets; batch_key='stim'; "
                 f"median over {a.seeds} seeds.\n\n")
        fh.write("| " + " | ".join(summ.columns) + " |\n|" + "---|" * len(summ.columns) + "\n")
        for row in summ.iter_rows():
            fh.write("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row) + " |\n")


if __name__ == "__main__":
    main()
