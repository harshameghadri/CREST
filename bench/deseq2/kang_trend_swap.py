"""Which stage of the fit makes CREST and R disagree on a stratum? Feed R CREST's gene-wise dispersions.

    python bench/deseq2/kang_trend_swap.py [--data bench_data/kang] [--out bench/deseq2/results] [--strata "CD4 T cells" ...]

R DESeq2 is run three times on the same pseudobulk counts:

* E0  unchanged;
* E1  after the gene-wise dispersion step, ``dispGeneEst`` is replaced by CREST's value for the genes where the two differ
      by more than ``SWAP_RTOL`` (relative); the trend, the MAP step and the Wald test then run in R as usual;
* E2  ``dispGeneEst`` is replaced by CREST's value for every gene.

For each run the script records R's trend coefficients and prior variance next to CREST's, the independent-filtering
threshold, and the genes called significant by exactly one tool at padj < 0.05 and < 0.1. If the discordance and the trend
difference vanish in E1, the genes that were swapped are the cause; if they vanish only in E2, the gene-wise step as a
whole is. ``kang_trend_swap_genes.csv`` lists the swapped genes with both values and the adjusted log-likelihood at each
(the gene-wise step maximises it), which shows how flat the likelihood is between the two values.
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
import kang_concordance as kc  # noqa: E402

SWAP_RTOL = 1e-6
STRATA = ("CD4 T cells", "FCGR3A+ Monocytes")

R_SCRIPT = r"""
suppressMessages({library(DESeq2); library(jsonlite)})
d <- commandArgs(TRUE)[1]
cfg <- fromJSON(file.path(d, "config.json"), simplifyVector = TRUE)
cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
cd <- read.csv(file.path(d, "coldata.csv"), row.names = 1, colClasses = "character")
for (v in names(cfg$levels)) cd[[v]] <- factor(cd[[v]], levels = cfg$levels[[v]])
ge_crest <- read.csv(file.path(d, "crest_ge.csv"))$dispGeneEst
# CR-adjusted NB log-likelihood of one gene at dispersion alpha, with the fitted means of the model (what the gene-wise step maximises)
cr_loglik <- function(y, mu, alpha, X) {
  w <- mu / (1 + alpha * mu)
  sum(dnbinom(y, mu = mu, size = 1 / alpha, log = TRUE)) - 0.5 * as.numeric(determinant(t(X) %*% (X * w), logarithm = TRUE)$modulus)
}
swapped <- NULL
run <- function(mode) {
  dds <- DESeqDataSetFromMatrix(round(cts), cd, as.formula(cfg$design))
  if (mode == "E0") {
    dds <- suppressMessages(DESeq(dds, quiet = TRUE))
    ge <- mcols(dds)$dispGeneEst
    idx <- which(is.finite(ge) & is.finite(ge_crest) & abs(ge_crest / ge - 1) > cfg$swap_rtol)
    X <- model.matrix(design(dds), colData(dds))
    swapped <<- data.frame(gene = rownames(dds)[idx], baseMean = mcols(dds)$baseMean[idx], dispGeneEst_R = ge[idx], dispGeneEst_CREST = ge_crest[idx],
                           dispGeneIter_R = mcols(dds)$dispGeneIter[idx],
                           loglik_at_R = sapply(idx, function(g) cr_loglik(counts(dds)[g, ], assays(dds)[["mu"]][g, ], ge[g], X)),
                           loglik_at_CREST = sapply(idx, function(g) cr_loglik(counts(dds)[g, ], assays(dds)[["mu"]][g, ], ge_crest[g], X)))
    write.csv(swapped, file.path(d, "swapped_genes.csv"), row.names = FALSE)
  } else {
    dds <- suppressMessages(estimateSizeFactors(dds))
    dds <- suppressMessages(estimateDispersionsGeneEst(dds))
    ge <- mcols(dds)$dispGeneEst
    ok <- is.finite(ge) & is.finite(ge_crest)
    idx <- if (mode == "E2") which(ok) else which(ok & abs(ge_crest / ge - 1) > cfg$swap_rtol)
    mcols(dds)$dispGeneEst[idx] <- ge_crest[idx]
    dds <- suppressMessages(estimateDispersionsFit(dds))
    dds <- suppressMessages(estimateDispersionsMAP(dds))
    dds <- suppressMessages(nbinomWaldTest(dds))
    attr(dds, "n_swapped") <- length(idx)
  }
  r10 <- results(dds, contrast = cfg$contrast, alpha = 0.1)
  r05 <- results(dds, contrast = cfg$contrast, alpha = 0.05)
  dfun <- dispersionFunction(dds)
  write.csv(data.frame(padj = r10$padj, padj_a05 = r05$padj, pvalue = r10$pvalue, dispersion = mcols(dds)$dispersion),
            file.path(d, paste0("r_", mode, ".csv")), row.names = FALSE)
  list(n_swapped = if (is.null(attr(dds, "n_swapped"))) 0 else attr(dds, "n_swapped"), fitType = attr(dfun, "fitType"),
       trend = unname(attr(dfun, "coefficients")), priorVar = attr(dfun, "dispPriorVar"),
       threshold_a10 = metadata(r10)$filterThreshold, threshold_a05 = metadata(r05)$filterThreshold)
}
out <- list(E0 = run("E0"), E1 = run("E1"), E2 = run("E2"), deseq2 = as.character(packageVersion("DESeq2")), r_version = R.version.string)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), file.path(d, "swap.json"))
"""


def discordant(pc, pr, a):
    return int(((np.nan_to_num(pc, nan=1.0) < a) != (np.nan_to_num(pr, nan=1.0) < a)).sum())


def one(name, counts, obs, genes, out_rows, swapped_all):
    design, contrast = "~ ind + stim", ("stim", "stim", "ctrl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = crest.tl.DESeq2(counts, obs, design, ref_levels={"stim": "ctrl"}, var_names=list(genes), quiet=True)
        c10 = ds.results(contrast, alpha=0.1)
        info10 = dict(ds.last_results_info)
        c05 = ds.results(contrast, alpha=0.05)
        info05 = dict(ds.last_results_info)
    d = Path(tempfile.mkdtemp())
    names = [f"s{i}" for i in range(counts.shape[0])]
    pl.DataFrame({"gene": genes}).hstack(pl.DataFrame(np.ascontiguousarray(counts.T), schema=names)).write_csv(d / "counts.csv")
    obs.with_columns(pl.Series("sample", names)).select("sample", *obs.columns).write_csv(d / "coldata.csv")
    pl.DataFrame({"dispGeneEst": np.asarray(ds.fit["dispGeneEst"], float)}).write_csv(d / "crest_ge.csv")
    levels = {"stim": ["ctrl", "stim"], "ind": sorted(obs["ind"].unique().to_list())}
    (d / "config.json").write_text(json.dumps({"design": design, "contrast": list(contrast), "levels": levels, "swap_rtol": SWAP_RTOL}))
    (d / "run.R").write_text(R_SCRIPT)
    env = {**os.environ, **{v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    p = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])
    sw = json.loads((d / "swap.json").read_text())
    sg = pl.read_csv(d / "swapped_genes.csv").with_columns(pl.lit(name).alias("stratum"))
    swapped_all.append(sg)
    pc10, pc05 = kc._arr(c10["padj"]), kc._arr(c05["padj"])
    trend_c = [float(x) for x in ds.fit["trend"]]
    for mode in ("E0", "E1", "E2"):
        r = pl.read_csv(d / f"r_{mode}.csv", null_values="NA")
        s = sw[mode]
        out_rows.append({
            "stratum": name, "run": mode, "genes_swapped": s["n_swapped"], "R_fitType": s["fitType"],
            "R_trend_a": s["trend"][0], "R_trend_b": s["trend"][1], "CREST_trend_a": trend_c[0], "CREST_trend_b": trend_c[1],
            "trend_a_rel_diff": abs(trend_c[0] / s["trend"][0] - 1), "trend_b_rel_diff": abs(trend_c[1] / s["trend"][1] - 1),
            "R_priorVar": s["priorVar"], "CREST_priorVar": float(ds.fit["dispPriorVar"]),
            "R_filterThreshold_0.1": s["threshold_a10"], "CREST_filterThreshold_0.1": info10["filterThreshold"],
            "discordant_0.1": discordant(pc10, kc._arr(r["padj"]), 0.1), "discordant_0.05": discordant(pc10, kc._arr(r["padj"]), 0.05),
            "matched_alpha_0.05_discordant": discordant(pc05, kc._arr(r["padj_a05"]), 0.05),
            "R_filterThreshold_0.05": s["threshold_a05"], "CREST_filterThreshold_0.05": info05["filterThreshold"],
            "max_rel_dispersion_diff": float(np.nanmax(np.abs(np.asarray(ds.fit["dispersion"], float) / kc._arr(r["dispersion"]) - 1)))})
    return sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/kang")
    ap.add_argument("--out", default="bench/deseq2/results")
    ap.add_argument("--strata", nargs="+", default=list(STRATA))
    a = ap.parse_args()
    genes, strata, _ = kc.kang_strata(Path(a.data))
    rows, swapped_all = [], []
    for name, counts, obs in strata:
        if name in a.strata:
            one(name, counts, obs, genes, rows, swapped_all)
            print(rows[-3:], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(Path(a.out) / "kang_trend_swap.csv")
    pl.concat(swapped_all).write_csv(Path(a.out) / "kang_trend_swap_genes.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
