"""Concordance of CREST's pseudobulk DESeq2 with R DESeq2 on every stratum of a real data set.

    python bench/deseq2/kang_concordance.py [--data bench_data/kang] [--out bench/deseq2/results]

For every Kang et al. 2018 cell type (design ``~ ind + stim``, stim vs ctrl) the script fits R DESeq2 and CREST on the
same pseudobulk counts and records

* the significant sets at adjusted p < 0.05 and < 0.1 (CREST, R, shared) from the ``alpha = 0.1`` results table, and the
  same counts when both tools are asked for ``alpha = 0.05`` (the independent-filtering target moves with ``alpha``);
* for every gene called significant by exactly one tool: both adjusted p-values and p-values, both dispersions, the
  side of the threshold, and a cause class (``cooks``, ``outlier_refit``, ``filter_cut``, ``threshold_adjacent``,
  ``other``; see ``classify``);
* the independent-filtering threshold of both tools, the discordance with independent filtering switched off in both,
  and the band of significant-set sizes obtained from R's own p-values under a multiplicative p-value jitter;
* ``dispGeneIter`` agreement, and the type of dispersion trend each tool ended up with (R falls back from the
  parametric fit to a local fit, CREST to the mean).

``concordance_stratum`` takes any (samples x genes) count table, so the same code runs on other data sets.
Nothing here is a timing result.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crest  # noqa: E402
from crest.deseq2 import _independent_filtering  # noqa: E402  (the port that reproduces R's filter, used for the jitter band)

ALPHAS = (0.05, 0.1)
JITTER_SD = 0.003          # relative (multiplicative) p-value jitter for R's reference band
JITTER_DRAWS = 200
JITTER_SEED = 1
FLOOR = 1e-8               # minDisp of DESeq2
FLOOR_TOL = 1e-6           # relative slack when deciding that a gene-wise dispersion sits at the floor
THRESH_RTOL = 1e-9         # two filter thresholds are the same if they agree to this
TAU_QUANTILE = 0.99        # "within the measured tolerance": this quantile of |p_CREST / p_R - 1| over genes with p < 0.1 in both
MAXIT = 100
DISP_REPORT_RTOL = 1e-3    # genes whose final dispersion differs by more than this are counted ...
DISP_REPORT_MAX = 200      # ... and the largest differences are listed, at most this many per stratum

R_SCRIPT = r"""
suppressMessages({library(DESeq2); library(jsonlite)})
d <- commandArgs(TRUE)[1]
cfg <- fromJSON(file.path(d, "config.json"), simplifyVector = TRUE)
cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
cd <- read.csv(file.path(d, "coldata.csv"), row.names = 1, colClasses = "character")
for (v in names(cfg$levels)) cd[[v]] <- factor(cd[[v]], levels = cfg$levels[[v]])
msgs <- character()
withCallingHandlers({
  dds <- DESeqDataSetFromMatrix(round(cts), cd, as.formula(cfg$design))
  dds <- DESeq(dds, quiet = TRUE)
}, message = function(m) { msgs <<- c(msgs, trimws(conditionMessage(m))); invokeRestart("muffleMessage") },
   warning = function(w) { msgs <<- c(msgs, paste("warning:", trimws(conditionMessage(w)))); invokeRestart("muffleWarning") })
res <- function(alpha, filt = TRUE) results(dds, contrast = cfg$contrast, alpha = alpha, independentFiltering = filt)
r10 <- res(0.1); r05 <- res(0.05); rnf <- res(0.1, FALSE)
mc <- mcols(dds)
col <- function(x) if (x %in% names(mc)) mc[[x]] else rep(NA, nrow(dds))
write.csv(data.frame(baseMean = r10$baseMean, log2FoldChange = r10$log2FoldChange, lfcSE = r10$lfcSE,
                     pvalue = r10$pvalue, padj = r10$padj, padj_a05 = r05$padj, padj_nofilter = rnf$padj,
                     dispGeneEst = col("dispGeneEst"), dispGeneIter = col("dispGeneIter"), dispFit = col("dispFit"),
                     dispMAP = col("dispMAP"), dispersion = col("dispersion"), dispIter = col("dispIter"),
                     dispOutlier = col("dispOutlier"), betaConv = col("betaConv"), maxCooks = col("maxCooks"),
                     replace = col("replace"), allZero = col("allZero")),
          file.path(d, "r.csv"), row.names = FALSE)
dfun <- dispersionFunction(dds)
filt <- function(r) list(threshold = metadata(r)$filterThreshold, theta = metadata(r)$filterTheta,
                         numRej = metadata(r)$filterNumRej$numRej)
diag <- list(fitType = attr(dfun, "fitType"),
  trend = if (identical(attr(dfun, "fitType"), "parametric")) unname(attr(dfun, "coefficients")) else NULL,
  dispPriorVar = attr(dfun, "dispPriorVar"), messages = msgs,
  filter_a10 = filt(r10), filter_a05 = filt(r05), cooksCutoff = qf(0.99, ncol(model.matrix(design(dds), colData(dds))), ncol(dds) - ncol(model.matrix(design(dds), colData(dds)))),
  deseq2 = as.character(packageVersion("DESeq2")), r_version = R.version.string)
writeLines(toJSON(diag, auto_unbox = TRUE, digits = NA, null = "null"), file.path(d, "diag.json"))
"""


def run_r(counts, obs, genes, design, contrast, levels):
    """Serial R DESeq2 on the table; returns (per-gene polars frame, diagnostics dict)."""
    d = Path(tempfile.mkdtemp())
    names = [f"s{i}" for i in range(counts.shape[0])]
    pl.DataFrame({"gene": genes}).hstack(pl.DataFrame(np.ascontiguousarray(np.asarray(counts).T), schema=names)).write_csv(d / "counts.csv")
    obs.with_columns(pl.Series("sample", names)).select("sample", *[c for c in obs.columns]).write_csv(d / "coldata.csv")
    (d / "config.json").write_text(json.dumps({"design": design, "contrast": list(contrast), "levels": levels}))
    (d / "run.R").write_text(R_SCRIPT)
    env = {**os.environ, **{v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    p = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Rscript failed ({p.returncode}):\n{p.stderr[-3000:]}")
    return pl.read_csv(d / "r.csv", null_values="NA", infer_schema_length=None), json.loads((d / "diag.json").read_text())


def bh(p):
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = len(q)
    if n:
        o = np.argsort(q)[::-1]
        adj = np.minimum.accumulate(q[o] * n / np.arange(n, 0, -1))
        r = np.empty(n)
        r[o] = np.minimum(adj, 1.0)
        out[ok] = r
    return out


def padj_at_threshold(pvalue, base_mean, threshold):
    """BH over the genes whose baseMean passes ``threshold`` (DESeq2's filter rule), p-values as given."""
    use = base_mean >= threshold
    out = np.full(len(pvalue), np.nan)
    out[use] = bh(pvalue[use])
    return out


def _arr(s):
    return s.cast(pl.Float64).fill_null(np.nan).to_numpy() if isinstance(s, pl.Series) else np.asarray(s, float)


def classify(i, thr, c, r, f, info_c, diag_r, tau):
    """Cause class of gene ``i`` called significant (padj < thr) by exactly one tool.

    cooks: the p-value is missing in one tool only (Cook's distance flag differs);
    outlier_refit: the replaced-outlier flag differs;
    filter_cut: the two independent-filtering thresholds differ and the gene's call follows R's threshold when CREST's
        p-values are filtered at R's threshold (or the reverse);
    threshold_adjacent: the two adjusted p-values agree within ``tau`` (relative), i.e. the gene sits on the line;
    other: none of the above (a note says whether a gene-wise dispersion is at the floor).
    """
    pc, pr = c["pvalue"][i], r["pvalue"][i]
    if np.isnan(pc) != np.isnan(pr):
        return "cooks"
    rep_r = bool(r["replace"][i]) if r["replace"][i] is not None else False
    if bool(f["replace"][i]) != rep_r:
        return "outlier_refit"
    tc, tr = info_c["filterThreshold"], diag_r["filter_a10"]["threshold"]
    if abs(tc - tr) > THRESH_RTOL * max(abs(tr), 1e-300):
        # counterfactual: CREST p-values, R's threshold; and R p-values, CREST's threshold
        cf_c = padj_at_threshold(_arr(c["pvalue"]), _arr(f["baseMean"]), tr)[i]
        cf_r = padj_at_threshold(_arr(r["pvalue"]), _arr(f["baseMean"]), tc)[i]
        sig_c, sig_r = c["padj"][i] < thr, r["padj"][i] < thr
        if (np.nan_to_num(cf_c, nan=1.0) < thr) == sig_r or (np.nan_to_num(cf_r, nan=1.0) < thr) == sig_c:
            return "filter_cut"
    ac, ar = c["padj"][i], r["padj"][i]
    if np.isfinite(ac) and np.isfinite(ar) and ar > 0 and abs(ac / ar - 1) <= tau:
        return "threshold_adjacent"
    return "other"


def _floor(x):
    return np.asarray(x, float) <= FLOOR * (1 + FLOOR_TOL)


def jitter_band(r, alpha, rng):
    """Significant-set sizes from R's p-values and baseMean under a multiplicative p-value jitter."""
    p0, bm = _arr(r["pvalue"]), _arr(r["baseMean"])
    out = []
    for _ in range(JITTER_DRAWS):
        p = np.clip(p0 * np.exp(rng.normal(0, JITTER_SD, len(p0))), 0, 1)
        padj, _info = _independent_filtering(p, bm, alpha)
        out.append(int((np.nan_to_num(padj, nan=1.0) < alpha).sum()))
    return min(out), max(out)


def concordance_stratum(name, counts, obs, genes, design, contrast, ref_levels, levels):
    """Fit CREST and R on one stratum. Returns (summary row, discordant-gene rows, fit row, genes whose dispersion differs)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = crest.tl.DESeq2(counts, obs, design, ref_levels=ref_levels, var_names=list(genes), quiet=True)
        c10 = ds.results(contrast, alpha=0.1)
        info_c = dict(ds.last_results_info)
        c05 = ds.results(contrast, alpha=0.05)
        info_c05 = dict(ds.last_results_info)
        cnf = ds.results(contrast, alpha=0.1, independent_filtering=False)
    f = ds.fit
    r, diag = run_r(counts, obs, genes, design, contrast, levels)
    pad_c, pad_r = _arr(c10["padj"]), _arr(r["padj"])
    row = {"stratum": name, "samples": int(counts.shape[0]), "genes_tested": int((~np.asarray(f["allZero"], bool)).sum())}
    for a in ALPHAS:
        sc, sr = np.nan_to_num(pad_c, nan=1.0) < a, np.nan_to_num(pad_r, nan=1.0) < a
        row.update({f"sig_CREST_{a}": int(sc.sum()), f"sig_R_{a}": int(sr.sum()), f"shared_{a}": int((sc & sr).sum()),
                    f"R_only_{a}": int((sr & ~sc).sum()), f"CREST_only_{a}": int((sc & ~sr).sum())})
    # alpha = 0.05 passed to both tools
    sc, sr = np.nan_to_num(_arr(c05["padj"]), nan=1.0) < 0.05, np.nan_to_num(_arr(r["padj_a05"]), nan=1.0) < 0.05
    row.update({"matched_alpha_0.05_sig_CREST": int(sc.sum()), "matched_alpha_0.05_sig_R": int(sr.sum()),
                "matched_alpha_0.05_shared": int((sc & sr).sum())})
    # independent filtering switched off
    for a in ALPHAS:
        sc, sr = np.nan_to_num(_arr(cnf["padj"]), nan=1.0) < a, np.nan_to_num(_arr(r["padj_nofilter"]), nan=1.0) < a
        row.update({f"nofilter_discordant_{a}": int((sc != sr).sum()), f"nofilter_sig_CREST_{a}": int(sc.sum()),
                    f"nofilter_sig_R_{a}": int(sr.sum())})
    # tolerance for "threshold adjacent": quantile of the relative p-value deviation among genes with p < 0.1 in both
    pc, pr = _arr(c10["pvalue"]), _arr(r["pvalue"])
    both = (pc < 0.1) & (pr < 0.1) & (pr > 0)
    rel = np.abs(pc[both] / pr[both] - 1)
    tau = float(np.quantile(rel, TAU_QUANTILE)) if both.any() else 0.0
    row["tau_p_rel"] = tau
    row["p_rel_dev_max"] = float(rel.max()) if both.any() else 0.0
    # discordant genes
    cdict = {k: _arr(c10[k]) for k in ("pvalue", "padj", "log2FoldChange", "lfcSE")}
    rdict = {"pvalue": pr, "padj": pad_r, "log2FoldChange": _arr(r["log2FoldChange"]), "lfcSE": _arr(r["lfcSE"]),
             "replace": r["replace"].to_list()}
    disc = []
    for a in ALPHAS:
        sc, sr = np.nan_to_num(pad_c, nan=1.0) < a, np.nan_to_num(pad_r, nan=1.0) < a
        for gi in np.flatnonzero(sc != sr):
            cause = classify(int(gi), a, cdict, rdict, f, info_c, diag, tau)
            floor_c, floor_r = bool(_floor(f["dispGeneEst"][gi])), bool(r["dispGeneEst"][int(gi)] <= FLOOR * (1 + FLOOR_TOL))
            disc.append({"stratum": name, "gene": genes[gi], "threshold": a, "called_by": "R_only" if sr[gi] else "CREST_only",
                         "padj_CREST": pad_c[gi], "padj_R": pad_r[gi], "pvalue_CREST": pc[gi], "pvalue_R": pr[gi],
                         "lfc_CREST": cdict["log2FoldChange"][gi], "lfc_R": rdict["log2FoldChange"][gi],
                         "lfcSE_CREST": cdict["lfcSE"][gi], "lfcSE_R": rdict["lfcSE"][gi], "baseMean": float(f["baseMean"][gi]),
                         "dispersion_CREST": float(f["dispersion"][gi]), "dispersion_R": r["dispersion"][int(gi)],
                         "dispGeneEst_CREST": float(f["dispGeneEst"][gi]), "dispGeneEst_R": r["dispGeneEst"][int(gi)],
                         "dispGeneIter_CREST": float(f["dispGeneIter"][gi]), "dispGeneIter_R": r["dispGeneIter"][int(gi)],
                         "maxCooks_CREST": float(f["maxCooks"][gi]), "maxCooks_R": r["maxCooks"][int(gi)],
                         "filterThreshold_CREST": info_c["filterThreshold"], "filterThreshold_R": diag["filter_a10"]["threshold"],
                         "cause": cause, "dispGeneEst_at_floor_CREST": floor_c, "dispGeneEst_at_floor_R": floor_r})
    # fit diagnostics
    nz = ~np.asarray(f["allZero"], bool)
    it_c, it_r = np.asarray(f["dispGeneIter"], float), _arr(r["dispGeneIter"])
    ge_c, ge_r = np.asarray(f["dispGeneEst"], float), _arr(r["dispGeneEst"])
    diff = nz & (it_c != it_r)
    fl_c, fl_r = _floor(ge_c), _floor(ge_r)
    rng = np.random.default_rng(JITTER_SEED)
    # the filtering code on R's own p-values: threshold and adjusted p-values against R's
    filt_check = {}
    for a_, key, pad_key, info_key in ((0.1, "a0.1", "padj", "filter_a10"), (0.05, "a0.05", "padj_a05", "filter_a05")):
        padj_x, info_x = _independent_filtering(_arr(r["pvalue"]), _arr(r["baseMean"]), a_)
        pr_x = _arr(r[pad_key])
        both_x = np.isfinite(padj_x) & np.isfinite(pr_x)
        filt_check[f"CREST_filter_on_R_pvalues_threshold_diff_{key}"] = float(info_x["filterThreshold"] - diag[info_key]["threshold"])
        filt_check[f"CREST_filter_on_R_pvalues_padj_maxreldiff_{key}"] = float(np.max(np.abs(padj_x[both_x] / pr_x[both_x] - 1))) if both_x.any() else 0.0
        filt_check[f"CREST_filter_on_R_pvalues_na_mismatch_{key}"] = int((np.isfinite(padj_x) != np.isfinite(pr_x)).sum())
    num_c = {"a0.1": info_c["filterNumRej"]["numRej"].to_numpy(), "a0.05": info_c05["filterNumRej"]["numRej"].to_numpy()}
    num_r = {"a0.1": np.asarray(diag["filter_a10"]["numRej"], float), "a0.05": np.asarray(diag["filter_a05"]["numRej"], float)}
    numrej = {}
    for key in num_c:
        numrej[f"filter_numRej_n_theta_differ_{key}"] = int((num_c[key] != num_r[key]).sum())
        numrej[f"filter_numRej_max_abs_diff_{key}"] = float(np.max(np.abs(num_c[key] - num_r[key])))
        numrej[f"filter_numRej_max_R_{key}"] = float(num_r[key].max())
        numrej[f"filter_numRej_max_CREST_{key}"] = float(num_c[key].max())
    # genes whose final dispersion differs by more than DISP_REPORT_RTOL
    disp_c, disp_r = np.asarray(f["dispersion"], float), _arr(r["dispersion"])
    dd = np.where(nz & np.isfinite(disp_r), np.abs(disp_c / disp_r - 1), 0.0)
    disp_out = [{"stratum": name, "gene": genes[gi], "baseMean": float(f["baseMean"][gi]), "dispersion_rel_diff": float(dd[gi]),
                 "dispersion_CREST": float(disp_c[gi]), "dispersion_R": float(disp_r[gi]),
                 "dispGeneEst_CREST": float(ge_c[gi]), "dispGeneEst_R": float(ge_r[gi]),
                 "dispGeneIter_CREST": float(it_c[gi]), "dispGeneIter_R": float(it_r[gi]),
                 "dispFit_CREST": float(f["dispFit"][gi]), "dispFit_R": float(_arr(r["dispFit"])[gi]),
                 "dispOutlier_R": bool(r["dispOutlier"][int(gi)]) if r["dispOutlier"][int(gi)] is not None else None,
                 "padj_CREST": pad_c[gi], "padj_R": pad_r[gi]}
                for gi in np.flatnonzero(dd > DISP_REPORT_RTOL)[np.argsort(-dd[dd > DISP_REPORT_RTOL])[:DISP_REPORT_MAX]]]
    n_disp_above = int((dd > DISP_REPORT_RTOL).sum())
    jb = {a: jitter_band(r, a, rng) for a in ALPHAS}
    nz_r = nz & np.isfinite(_arr(r["dispersion"]))
    fit = {"stratum": name, "R_fitType": diag["fitType"], "CREST_fitType": f["fitType"],
           "CREST_fit_substituted": bool(f.get("fit_substituted", False)), "R_messages": " | ".join([diag["messages"]] if isinstance(diag["messages"], str) else diag["messages"]),
           "CREST_messages": " | ".join(f["messages"]),
           "R_trend": json.dumps(diag.get("trend")), "CREST_trend": json.dumps([float(x) for x in f["trend"]]) if "trend" in f and f["trend"] is not None else None,
           "R_dispPriorVar": diag["dispPriorVar"], "CREST_dispPriorVar": float(f["dispPriorVar"]),
           "filterThreshold_R_a0.1": diag["filter_a10"]["threshold"], "filterThreshold_CREST_a0.1": info_c["filterThreshold"],
           "filterThreshold_R_a0.05": diag["filter_a05"]["threshold"], "filterThreshold_CREST_a0.05": info_c05["filterThreshold"],
           "dispGeneIter_differs": int(diff.sum()), "dispGeneIter_differs_both_at_floor": int((diff & fl_c & fl_r).sum()),
           "genes_at_floor_both": int((nz & fl_c & fl_r).sum()), "genes_at_floor_either": int((nz & (fl_c | fl_r)).sum()),
           "dispGeneIter_eq1_CREST": int((nz & (it_c == 1)).sum()), "dispGeneIter_eq1_R": int((nz & (it_r == 1)).sum()),
           "dispGeneIter_ge_maxit_CREST": int((nz & (it_c >= MAXIT)).sum()), "dispGeneIter_ge_maxit_R": int((nz & (it_r >= MAXIT)).sum()),
           "dispersion_maxreldiff": float(np.max(np.abs(np.asarray(f["dispersion"], float)[nz_r] / _arr(r["dispersion"])[nz_r] - 1))),
           "dispFit_maxreldiff": float(np.max(np.abs(np.asarray(f["dispFit"], float)[nz_r] / _arr(r["dispFit"])[nz_r] - 1))),
           "dispFit_median_reldiff": float(np.median(np.abs(np.asarray(f["dispFit"], float)[nz_r] / _arr(r["dispFit"])[nz_r] - 1))),
           "jitter_min_0.05": jb[0.05][0], "jitter_max_0.05": jb[0.05][1], "jitter_min_0.1": jb[0.1][0], "jitter_max_0.1": jb[0.1][1],
           "R_betaNonConv": int((nz & ~(r["betaConv"].fill_null(True).to_numpy().astype(bool))).sum()),
           "CREST_betaNonConv": int((nz & ~np.asarray(f["betaConv"], bool)).sum()),
           # genes whose counts were replaced in each tool, and genes on which the two tools disagree about it
           "R_replaced": int(r["replace"].fill_null(False).to_numpy().astype(bool).sum()),
           "CREST_replaced": int(np.asarray(f["replace"], bool).sum()),
           "replaced_flag_mismatch": int((r["replace"].fill_null(False).to_numpy().astype(bool) != np.asarray(f["replace"], bool)).sum()),
           "n_genes_dispersion_rel_diff_above_report_tol": n_disp_above,
           **filt_check, **numrej,
           "deseq2": diag["deseq2"], "r_version": diag["r_version"]}
    for a_ in ALPHAS:
        row[f"jitter_min_{a_}"], row[f"jitter_max_{a_}"] = jb[a_]
        # the band is made at alpha = a_, so it is compared with the counts of tables made at alpha = a_
        n_c = row["sig_CREST_0.1"] if a_ == 0.1 else row["matched_alpha_0.05_sig_CREST"]
        n_r = row["sig_R_0.1"] if a_ == 0.1 else row["matched_alpha_0.05_sig_R"]
        row[f"CREST_in_jitter_band_{a_}"] = bool(jb[a_][0] <= n_c <= jb[a_][1])
        row[f"R_in_jitter_band_{a_}"] = bool(jb[a_][0] <= n_r <= jb[a_][1])
    return row, disc, fit, disp_out


def kang_strata(data: Path):
    import kang_pseudobulk as kp
    kp.fetch(data)
    bf = kp.load(data)
    pb = crest.tl.pseudobulk(bf, ["ind", "stim"], groupby="cell", min_cells=10)
    genes = list(bf.var["gene_ids"])
    out = []
    for ct in pb.obs["cell"].unique(maintain_order=True).to_list():
        sub = pb.subset((pb.obs["cell"] == ct).to_numpy())
        keep = sub.obs.group_by("ind").agg(pl.col("stim").n_unique().alias("k")).filter(pl.col("k") == 2)["ind"]
        sub = sub.subset(sub.obs["ind"].is_in(keep.implode()).to_numpy())
        if sub.obs.height >= 6:
            out.append((ct, np.asarray(sub.counts), sub.obs.select(pl.col("ind").cast(pl.Utf8), pl.col("stim").cast(pl.Utf8))))
    return genes, out, bf.n_obs


def write_report(out: Path, prefix: str, rows, disc, fits, dispo, meta):
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", rows), ("fit", fits)):
        df = pl.DataFrame(data, infer_schema_length=None)
        df.write_csv(out / f"{prefix}_{name}.csv")
    if disc:
        pl.DataFrame(disc, infer_schema_length=None).write_csv(out / f"{prefix}_discordant_genes.csv")
    if dispo:
        pl.DataFrame(dispo, infer_schema_length=None).write_csv(out / f"{prefix}_dispersion_differences.csv")
    (out / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=1, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/kang")
    ap.add_argument("--out", default="bench/deseq2/results")
    a = ap.parse_args()
    genes, strata, n_cells = kang_strata(Path(a.data))
    rows, disc, fits, dispo = [], [], [], []
    for name, counts, obs in strata:
        row, d, fit, dout = concordance_stratum(name, counts, obs, genes, "~ ind + stim", ("stim", "stim", "ctrl"),
                                          {"stim": "ctrl"}, {"stim": ["ctrl", "stim"], "ind": sorted(obs["ind"].unique().to_list())})
        rows.append(row)
        disc += d
        dispo += dout
        fits.append(fit)
        print({k: row[k] for k in ("stratum", "sig_CREST_0.1", "sig_R_0.1", "shared_0.1", "sig_CREST_0.05", "sig_R_0.05", "shared_0.05")},
              fit["R_fitType"], fit["CREST_fitType"], flush=True)
    write_report(Path(a.out), "kang_concordance", rows, disc, fits, dispo,
                 {"dataset": "Kang 2018 GSE96583 batch 2", "cells": n_cells, "design": "~ ind + stim", "contrast": "stim vs ctrl",
                  "crest_version": crest.__version__, "jitter": {"sd": JITTER_SD, "draws": JITTER_DRAWS, "seed": JITTER_SEED}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
