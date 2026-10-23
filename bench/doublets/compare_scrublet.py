"""CREST Scrublet vs scanpy.pp.scrublet on Kang et al. 2018 with demuxlet truth.

Kang et al. multiplexed 8 donors per lane; demuxlet flags droplets holding
cells of two donors ("doublet"). Those are the ground truth here (inter-donor
doublets only, so recall is bounded by what expression can reveal). Both tools
run per lane-condition batch (``stim``) with default parameters.

Every seed is reported, with each batch's threshold and calls
(``kang_scrublet_batches.csv``): the automatic ``threshold_minimum`` threshold
is unstable on these data in both tools (a batch can get a threshold in the
tail of the simulated-score histogram and call ~0 doublets), so a median over
a few seeds would hide failed runs. The CREST and scanpy random streams differ,
so seed ``s`` of one tool is not comparable to seed ``s`` of the other; compare
the per-tool rates of failed batches over many seeds.

    python bench/doublets/compare_scrublet.py [--out bench/doublets/results] [--seeds 10]
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
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch(Path(a.data))
    bf = load(Path(a.data), singlets_only=False)
    bf = bf.filter_cells(pl.col("multiplets").is_in(["singlet", "doublet"]))
    truth = (bf.obs["multiplets"] == "doublet").to_numpy()
    print(f"{bf.n_obs} droplets, {truth.sum()} demuxlet doublets ({truth.mean():.1%})")

    import scanpy as sc
    rows, brows = [], []
    batch = bf.obs["stim"].cast(pl.Utf8).to_numpy()

    def per_batch(tool, s, pred, uns_batches):
        for b, u in sorted(uns_batches.items()):
            sim = np.asarray(u["doublet_scores_sim"])
            thr = u.get("threshold")
            det = float((sim > thr).mean()) if thr is not None else float("nan")
            m = batch == str(b)
            if thr is None:
                status = "failed"  # threshold_minimum found no two maxima: nothing is called
            elif det < 0.01:
                status = "degenerate"  # threshold in the tail: <1% of simulated doublets above it
            else:
                status = "ok"
            brows.append({"tool": tool, "seed": s, "batch": str(b), "threshold": thr, "status": status,
                          "detectable_doublet_fraction": det, "called": int(pred[m].sum()),
                          "recall": float((pred[m] & truth[m]).sum() / max(truth[m].sum(), 1))})
            print(brows[-1])

    for s in range(a.seeds):
        t0 = time.perf_counter()
        b = bf.copy()
        crest.pp.scrublet(b, batch_key="stim", random_state=s)
        tc = time.perf_counter() - t0
        sc_c = b.obs["doublet_score"].fill_null(np.nan).to_numpy()
        pred_c = b.obs["predicted_doublet"].to_numpy()
        per_batch("crest", s, pred_c, b.uns["scrublet"]["batches"])

        ad = bf.to_anndata()
        t0 = time.perf_counter()
        sc.pp.scrublet(ad, batch_key="stim", random_state=s, verbose=False)
        ts = time.perf_counter() - t0
        sc_s = ad.obs["doublet_score"].to_numpy(float)
        pred_s = ad.obs["predicted_doublet"].to_numpy(bool)
        per_batch("scanpy", s, pred_s, ad.uns["scrublet"]["batches"])
        for tool, score, pred, t in (("crest", sc_c, pred_c, tc), ("scanpy", sc_s, pred_s, ts)):
            roc, pr = auc(score, truth)
            nfail = sum(r["status"] != "ok" for r in brows if r["tool"] == tool and r["seed"] == s)
            rows.append({"tool": tool, "seed": s, "time_s": t, "AUROC": roc, "AUPRC": pr,
                         "called": int(pred.sum()), "recall": float((pred & truth).sum() / truth.sum()),
                         "precision": float((pred & truth).sum() / max(pred.sum(), 1)),
                         "failed_batches": nfail, "spearman_vs_crest": None})
            print(rows[-1])
        ok = np.isfinite(sc_c) & np.isfinite(sc_s)
        from scipy.stats import spearmanr
        rows[-1]["spearman_vs_crest"] = float(spearmanr(sc_c[ok], sc_s[ok]).statistic)
        print("spearman crest vs scanpy scores:", rows[-1]["spearman_vs_crest"])
    df = pl.DataFrame(rows)
    bdf = pl.DataFrame(brows)
    df.write_csv(out / "kang_scrublet.csv")
    bdf.write_csv(out / "kang_scrublet_batches.csv")

    def table(fh, d):
        fh.write("| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n")
        for row in d.iter_rows():
            fh.write("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row) + " |\n")

    fail = (bdf.group_by("tool", "batch", maintain_order=True)
            .agg(pl.len().alias("runs"), (pl.col("status") != "ok").sum().alias("failed"),
                 pl.col("threshold").median().alias("median_threshold"))
            .sort("tool", "batch"))
    summ = df.drop("seed").group_by("tool", maintain_order=True).median()
    with open(out / "kang_scrublet.md", "w") as fh:
        fh.write(f"Kang 2018: {bf.n_obs} droplets, {truth.sum()} demuxlet doublets; batch_key='stim'; "
                 f"{a.seeds} seeds per tool.\n\n")
        fh.write("A batch *fails* when the automatic threshold calls (almost) nothing: threshold_minimum found "
                 "no two maxima, or fewer than 1% of the simulated doublets score above its threshold.\n\n"
                 "Failed batches per tool:\n\n")
        table(fh, fail)
        fh.write("\nEvery run:\n\n")
        table(fh, df)
        fh.write("\nPer batch:\n\n")
        table(fh, bdf)
        fh.write(f"\nMedian over {a.seeds} seeds (hides failed runs; see above):\n\n")
        table(fh, summ)


if __name__ == "__main__":
    main()
