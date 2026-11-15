"""Where do CREST's and R's fitted dispersions differ in one stratum, and what accounts for it?

    python bench/deseq2/covid_trend_gap.py --h5 <covid_stephenson.h5> --obs <covid_stephenson.obs.parquet> --stratum "IgA plasma cell" --out <dir>

Fits CREST and R DESeq2 on the stratum of ``covid_concordance.py`` and writes ``covid_trend_gap_<stratum>.json`` with

* ``trend_function``: CREST's ``dispFit`` against R's ``localDispersionFit`` given CREST's first-pass gene-wise dispersions (outlier
  replacement switched off: the trend is fitted before any replacement), over ALL genes with counts and over the genes that
  receive a p-value, evaluated at two sets of points. ``at_first_pass_baseMean`` is the comparison that looks alarming
  (maximum 5.4e5 in IgA plasma cells): CREST reports ``dispFit`` for a gene whose counts were replaced at its base mean AFTER the
  replacement (R does the same: the two tools' reported base means agree for all but two genes), which is far below the range of
  the fitted means for some genes (0.0065 for a gene whose first-pass base mean is 0.66, the smallest mean in the fit being 0.037), where the curve is locfit's quadratic extrapolation, so evaluating R's
  trend at the first-pass base mean compares different points. ``at_reported_baseMean`` evaluates R's trend at the base mean CREST
  used and is the like-for-like comparison of the two predictions; the genes of the first comparison are listed;
* ``reported_dispFit_vs_R``: the relative difference of the two tools' reported ``dispFit`` over the genes with a p-value, its
  distribution, the genes above 0.1 and how much of it the base mean explains (R's trend at CREST's and at R's base mean);
* ``first_pass_gene_wise``: how many first-pass gene-wise dispersions differ by more than 1e-6 (relative) and how far R's trend moves
  (at the first-pass base means) when CREST's values are used, and when R's values are put back at the differing genes only;
* ``downstream``: the differences of ``dispersion`` and of the p-value over the genes with a p-value in both tools, over all of
  them and over those with equal base means, and the genes with the largest p-value differences (dispersion, gene-wise estimate
  and fold change shown).
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
suppressMessages(library(DESeq2)); d <- commandArgs(TRUE)[1]; fn0 <- getFromNamespace("localDispersionFit", "DESeq2")
q <- read.csv(file.path(d, "q.csv"))
pr <- function(fn, v) { o <- rep(NA_real_, length(v)); k <- which(is.finite(v) & v > 0); o[k] <- fn(v[k]); o }
out <- list()
for (nm in c("R", "CREST", "hybrid")) {
  x <- read.csv(file.path(d, paste0(nm, ".csv"))); fn <- fn0(x$mean, x$disp, 1e-8)
  for (pt in c("first_pass", "crest_reported", "R_reported")) out[[paste(nm, pt, sep = "__")]] <- pr(fn, q[[pt]])
}
write.csv(as.data.frame(out), file.path(d, "trends.csv"), row.names = FALSE)
"""


def _rel(a, b, mask):
    ok = mask & np.isfinite(a) & np.isfinite(b)
    r = np.abs(a[ok] / b[ok] - 1)
    return {"n": int(ok.sum()), "max": float(r.max()) if ok.any() else None, "median": float(np.median(r)) if ok.any() else None}


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
    contrast = ("Status", "Covid", "Healthy")
    ds = crest.tl.DESeq2(counts, obs.select("Status"), "~ Status", **kw)
    f, res = ds.fit, ds.results(contrast)
    r, diag = kc.run_r(counts, obs.select("Status"), genes, "~ Status", contrast, {"Status": ["Healthy", "Covid"]})
    A = lambda x: np.asarray(x, float)
    live = ~np.asarray(f["allZero"], bool)
    c_fit, c_bm, c_disp = A(f["dispFit"]), A(f["baseMean"]), A(f["dispersion"])
    r_fit, r_bm, r_disp = (r[k].to_numpy().astype(float) for k in ("dispFit", "baseMean", "dispersion"))
    c_p, r_p = res["pvalue"].to_numpy().astype(float), r["pvalue"].to_numpy().astype(float)
    c_lfc, r_lfc = res["log2FoldChange"].to_numpy().astype(float), r["log2FoldChange"].to_numpy().astype(float)
    r_rep, c_rep = np.array(r["replace"].to_list(), dtype=bool), np.asarray(f["replace"], bool)
    has_p = live & np.isfinite(c_p) & np.isfinite(r_p)
    out = {"stratum": a.stratum, "R_fitType": diag["fitType"], "CREST_fitType": f["fitType"], "genes_with_counts": int(live.sum()),
           "genes_with_a_p_value_in_both": int(has_p.sum()), "replaced_CREST": int(c_rep.sum()), "replaced_R": int(r_rep.sum())}

    # first-pass inputs of both tools
    fp, _ = ti.r_first_pass(genes, counts, obs)
    fp_m, fp_g = fp["baseMean"].to_numpy().astype(float), fp["dispGeneEst"].to_numpy().astype(float)
    f0 = crest.tl.DESeq2(counts, obs.select("Status"), "~ Status", min_replicates_for_replace=float("inf"), **kw).fit
    g0, m0 = A(f0["dispGeneEst"]), A(f0["baseMean"])
    assert np.allclose(m0[live], fp_m[live], rtol=1e-12)
    cut = 100 * ti.MIN_DISP
    diff = live & np.isfinite(fp_g) & np.isfinite(g0) & (np.abs(g0 / fp_g - 1) > 1e-6)
    hyb = g0.copy()
    hyb[diff] = fp_g[diff]
    d = Path(tempfile.mkdtemp())
    pl.DataFrame({"first_pass": fp_m, "crest_reported": np.where(c_bm > 0, c_bm, np.nan), "R_reported": np.where(r_bm > 0, r_bm, np.nan)}).write_csv(d / "q.csv")
    for nm, g in (("R", fp_g), ("CREST", g0), ("hybrid", hyb)):
        u = live & np.isfinite(g) & (g > cut)
        pl.DataFrame({"mean": fp_m[u], "disp": g[u]}).write_csv(d / f"{nm}.csv")
    (d / "run.R").write_text(R_TRENDS)
    p = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    T = pl.read_csv(d / "trends.csv", null_values="NA")
    t = {k: T[k].to_numpy().astype(float) for k in T.columns}

    # 1. the trend function, like for like and as first compared
    changed = live & (np.abs(c_bm / np.where(fp_m > 0, fp_m, np.nan) - 1) > 1e-9)
    lo = float(np.log(fp_m[live & np.isfinite(fp_g) & (fp_g > cut)].min()))
    out["trend_function"] = {
        "at_reported_baseMean": {"all_genes": _rel(c_fit, t["CREST__crest_reported"], live), "genes_with_a_p_value": _rel(c_fit, t["CREST__crest_reported"], has_p)},
        "at_first_pass_baseMean": {"all_genes": _rel(c_fit, t["CREST__first_pass"], live), "genes_with_a_p_value": _rel(c_fit, t["CREST__first_pass"], has_p)},
        "genes_whose_baseMean_changed_with_replacement": int(changed.sum()),
        "of_which_replaced_in_CREST": int((changed & c_rep).sum()), "of_which_with_a_p_value": int((changed & has_p).sum()),
        "reported_baseMean_CREST_vs_R_over_genes_with_a_p_value": {"n_differing_above_1e-9": int((has_p & (np.abs(c_bm / r_bm - 1) > 1e-9)).sum())},
        "smallest_fitted_baseMean": float(np.exp(lo)),
        "R_reported_dispFit_vs_R_localDispersionFit_on_R_values_at_R_baseMean": _rel(r_fit, t["R__R_reported"], live)}
    rel_first = np.where(live & np.isfinite(t["CREST__first_pass"]), np.abs(c_fit / t["CREST__first_pass"] - 1), -1)
    out["trend_function"]["genes_with_the_largest_first_pass_difference"] = [
        {"gene": genes[j], "baseMean_first_pass": float(fp_m[j]), "baseMean_reported": float(c_bm[j]), "first_pass_gene_wise_dispersion": float(g0[j]),
         "CREST_dispFit": float(c_fit[j]), "R_prediction_at_first_pass_baseMean": float(t["CREST__first_pass"][j]),
         "R_prediction_at_reported_baseMean": float(t["CREST__crest_reported"][j]), "R_reported_dispFit": float(r_fit[j]),
         "relative_difference_first_pass": float(rel_first[j]), "replaced_CREST": bool(c_rep[j]), "p_value_CREST": float(c_p[j])}
        for j in np.argsort(-rel_first)[:8]]

    # 2. reported dispFit against R's, and what the base mean explains
    rel = np.abs(c_fit / r_fit - 1)
    out["reported_dispFit_vs_R"] = {"genes_with_a_p_value": _rel(c_fit, r_fit, has_p),
                                    "p99_over_genes_with_a_p_value": float(np.percentile(rel[has_p], 99)),
                                    **{f"n_above_{x}": int((has_p & (rel > x)).sum()) for x in (2e-4, 1e-3, 1e-2, 0.1)},
                                    "over_genes_with_equal_baseMean": _rel(c_fit, r_fit, has_p & (np.abs(c_bm / r_bm - 1) < 1e-9))}
    out["reported_dispFit_vs_R"]["genes_above_0.1"] = [
        {"gene": genes[j], "rel_diff": float(rel[j]), "baseMean_CREST": float(c_bm[j]), "baseMean_R": float(r_bm[j]), "dispFit_CREST": float(c_fit[j]),
         "dispFit_R": float(r_fit[j]), "R_trend_on_R_values_at_CREST_baseMean": float(t["R__crest_reported"][j]),
         "R_trend_on_R_values_at_R_baseMean": float(t["R__R_reported"][j]), "replaced_CREST": bool(c_rep[j]), "replaced_R": bool(r_rep[j])}
        for j in np.flatnonzero(has_p & (rel > 0.1))]

    # 3. first-pass gene-wise values
    out["first_pass_gene_wise"] = {
        "n_differ_above_1e-6": int(diff.sum()), "n_differ_entering_or_leaving_the_fit": int((diff & ((g0 > cut) != (fp_g > cut))).sum()),
        "R_trend_on_CREST_values_vs_R_values": {"median": float(np.nanmedian(np.abs(t["CREST__first_pass"] / t["R__first_pass"] - 1))),
                                                "max": float(np.nanmax(np.abs(t["CREST__first_pass"] / t["R__first_pass"] - 1)))},
        "R_trend_on_hybrid_values_vs_R_values_max": float(np.nanmax(np.abs(t["hybrid__first_pass"] / t["R__first_pass"] - 1)))}

    # 4. what reaches the dispersions and the p-values
    same = has_p & (np.abs(c_bm / r_bm - 1) < 1e-9)
    rel_p = np.where(has_p, np.abs(c_p / r_p - 1), -1)
    ge_r = r["dispGeneEst"].to_numpy().astype(float)
    out["downstream"] = {"dispersion_over_genes_with_a_p_value": _rel(c_disp, r_disp, has_p), "dispersion_over_genes_with_equal_baseMean": _rel(c_disp, r_disp, same),
                         "p_value_over_genes_with_a_p_value": _rel(c_p, r_p, has_p), "p_value_over_genes_with_equal_baseMean": _rel(c_p, r_p, same),
                         "genes_with_the_largest_p_value_difference": [
                             {"gene": genes[j], "rel_diff": float(rel_p[j]), "p_CREST": float(c_p[j]), "p_R": float(r_p[j]), "baseMean": float(c_bm[j]),
                              "dispersion_CREST": float(c_disp[j]), "dispersion_R": float(r_disp[j]), "dispGeneEst_CREST": float(A(f["dispGeneEst"])[j]),
                              "dispGeneEst_R": float(ge_r[j]), "lfc_CREST": float(c_lfc[j]), "lfc_R": float(r_lfc[j]), "replaced_CREST": bool(c_rep[j]), "replaced_R": bool(r_rep[j])}
                             for j in np.argsort(-rel_p)[:8]]}
    Path(a.out, f"covid_trend_gap_{a.stratum.replace(' ', '_')}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("trend_function", "downstream")}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
