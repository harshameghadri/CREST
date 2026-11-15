"""Where do CREST's and R's fitted dispersions differ in one stratum, and which gene-wise inputs account for it?

    python bench/deseq2/covid_trend_gap.py --h5 <covid_stephenson.h5> --obs <covid_stephenson.obs.parquet> --stratum "IgA plasma cell" --out <dir>

Fits CREST and R DESeq2 on the stratum of ``covid_concordance.py`` and writes ``covid_trend_gap_<stratum>.json``:

* the distribution over genes of |CREST dispFit / R dispFit - 1|, and the genes above 0.1 with their base means (which differ
  between the tools for genes whose counts are replaced) and their replacement flags;
* the number of first-pass gene-wise dispersions that differ by more than 1e-6 (relative) between the tools (outlier replacement
  switched off in CREST and not run in R: the trend is fitted before any replacement), and, through R's ``localDispersionFit``,
  how far R's trend moves when CREST's gene-wise values are used instead of R's, and how far it still is from R's own trend when
  R's values are put back at the differing genes only.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import covid_trend_inputs as ti  # noqa: E402
import kang_concordance as kc  # noqa: E402

R_TRENDS = """
suppressMessages(library(DESeq2)); d <- commandArgs(TRUE)[1]; fn <- getFromNamespace("localDispersionFit", "DESeq2")
q <- read.csv(file.path(d, "q.csv"))$q
for (nm in c("R", "CREST", "hybrid")) { x <- read.csv(file.path(d, paste0(nm, ".csv"))); write.csv(data.frame(t = fn(x$mean, x$disp, 1e-8)(q)), file.path(d, paste0(nm, "_trend.csv")), row.names = FALSE) }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--stratum", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    import crest
    genes, tabs = ti.stratum_tables(a.h5, a.obs)
    counts, obs = tabs[a.stratum]
    kw = dict(ref_levels={"Status": "Healthy"}, var_names=list(genes), quiet=True)
    f = crest.tl.DESeq2(counts, obs.select("Status"), "~ Status", **kw).fit
    r, diag = kc.run_r(counts, obs.select("Status"), genes, "~ Status", ("Status", "Covid", "Healthy"), {"Status": ["Healthy", "Covid"]})
    live = ~np.asarray(f["allZero"], bool)
    c, rr = np.asarray(f["dispFit"], float), r["dispFit"].to_numpy().astype(float)
    rb, cb = r["baseMean"].to_numpy().astype(float), np.asarray(f["baseMean"], float)
    ok = live & np.isfinite(c) & np.isfinite(rr)
    rel = np.abs(c / rr - 1)
    rep_r = r["replace"].to_list()
    out = {"stratum": a.stratum, "R_fitType": diag["fitType"], "CREST_fitType": f["fitType"], "genes_compared": int(ok.sum()),
           "dispFit_rel_diff": {"median": float(np.median(rel[ok])), "p90": float(np.percentile(rel[ok], 90)), "p99": float(np.percentile(rel[ok], 99)),
                                "max": float(rel[ok].max()), **{f"n_above_{t}": int((ok & (rel > t)).sum()) for t in (2e-4, 1e-3, 1e-2, 0.1)}}}
    out["genes_above_0.1"] = [{"gene": genes[j], "rel_diff": float(rel[j]), "baseMean_CREST": float(cb[j]), "baseMean_R": float(rb[j]),
                               "dispFit_CREST": float(c[j]), "dispFit_R": float(rr[j]), "replaced_CREST": bool(f["replace"][j]), "replaced_R": bool(rep_r[j])}
                              for j in np.flatnonzero(ok & (rel > 0.1))]
    fp, _ = ti.r_first_pass(genes, counts, obs)
    fp_m, fp_g = fp["baseMean"].to_numpy().astype(float), fp["dispGeneEst"].to_numpy().astype(float)
    f0 = crest.tl.DESeq2(counts, obs.select("Status"), "~ Status", min_replicates_for_replace=float("inf"), **kw).fit
    g0 = np.asarray(f0["dispGeneEst"], float)
    cut = 100 * ti.MIN_DISP
    diff = live & np.isfinite(fp_g) & np.isfinite(g0) & (np.abs(g0 / fp_g - 1) > 1e-6)
    hyb = g0.copy()
    hyb[diff] = fp_g[diff]
    d = Path(tempfile.mkdtemp())
    pl.DataFrame({"q": fp_m[live]}).write_csv(d / "q.csv")
    for nm, g in (("R", fp_g), ("CREST", g0), ("hybrid", hyb)):
        u = live & np.isfinite(g) & (g > cut)
        pl.DataFrame({"mean": fp_m[u], "disp": g[u]}).write_csv(d / f"{nm}.csv")
    (d / "run.R").write_text(R_TRENDS)
    p = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    T = {nm: pl.read_csv(d / f"{nm}_trend.csv")["t"].to_numpy() for nm in ("R", "CREST", "hybrid")}
    out["first_pass_gene_wise"] = {"n_differ_above_1e-6": int(diff.sum()), "n_differ_entering_or_leaving_the_fit": int((diff & ((g0 > cut) != (fp_g > cut))).sum()),
                                   "R_trend_on_CREST_values_vs_R_values": {"median": float(np.median(np.abs(T["CREST"] / T["R"] - 1))), "max": float(np.max(np.abs(T["CREST"] / T["R"] - 1)))},
                                   "R_trend_on_hybrid_values_vs_R_values_max": float(np.max(np.abs(T["hybrid"] / T["R"] - 1)))}
    Path(a.out, f"covid_trend_gap_{a.stratum.replace(' ', '_')}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
