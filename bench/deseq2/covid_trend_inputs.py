"""Dispersion-trend inputs from real strata of the Stephenson et al. 2021 COVID-19 atlas, for the local-fit fixtures.

    python bench/deseq2/covid_trend_inputs.py --h5 <covid_stephenson.h5> --obs <covid_stephenson.obs.parquet> --out tests/data

For each stratum named in ``LOCAL_STRATA`` (cell types where R's parametric trend fails and DESeq2 substitutes its
local regression) it builds the donor-level pseudobulk of ``covid_concordance.py``, runs R's ``estimateSizeFactors`` +
``estimateDispersionsGeneEst``, and writes ``deseq2_local_fit_covid_cases.csv`` (case, mean, disp): ``SUBSAMPLE`` genes
drawn with a fixed seed from the genes R would use for the trend (gene-wise dispersion above ``10 * minDisp``), with
R's base means and gene-wise dispersions. ``make_local_fit_fixtures.R`` then fits R's ``localDispersionFit`` to them.

For ``MDC`` (R's parametric fit succeeds, CREST's gene-wise values make it fail in R as well) it writes
``deseq2_trend_mdc_inputs.csv`` (mean, disp_R, disp_CREST): the genes with either gene-wise estimate above
``100 * minDisp``, with R's and with CREST's first-pass gene-wise dispersions (outlier replacement switched off, as
the trend is fitted before any replacement), and ``deseq2_trend_mdc_R.json`` with R's parametric coefficients and the
outcome of R's ``parametricDispersionFit`` on each set. Nothing written by this script is a reference result of CREST.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import covid_concordance as cc  # noqa: E402

LOCAL_STRATA = ["B cell", "IgA plasma cell", "T follicular helper cell", "effector memory CD4-positive, alpha-beta T cell",
                "natural killer cell", "plasmacytoid dendritic cell"]
MDC = "myeloid dendritic cell"
SUBSAMPLE = 1500
SEED = 20261115
MIN_DISP = 1e-8

R_FIRST_PASS = r"""
suppressMessages({library(DESeq2); library(jsonlite)})
d <- commandArgs(TRUE)[1]
cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
cd <- read.csv(file.path(d, "coldata.csv"), row.names = 1, colClasses = "character")
cd$Status <- factor(cd$Status, levels = c("Healthy", "Covid"))
dds <- DESeqDataSetFromMatrix(round(cts), cd, ~ Status)
dds <- estimateSizeFactors(dds)
dds <- estimateDispersionsGeneEst(dds, quiet = TRUE)
mc <- mcols(dds)
write.csv(data.frame(gene = rownames(dds), baseMean = mc$baseMean, dispGeneEst = mc$dispGeneEst, allZero = mc$allZero),
          file.path(d, "r_first_pass.csv"), row.names = FALSE)
"""

R_MDC = r"""
suppressMessages({library(DESeq2); library(jsonlite)})
d <- commandArgs(TRUE)[1]
x <- read.csv(file.path(d, "mdc_inputs.csv"), na.strings = "NA")
pf <- getFromNamespace("parametricDispersionFit", "DESeq2")
fit <- function(m, g) {
  k <- is.finite(g) & g > 100 * 1e-8
  r <- tryCatch(pf(m[k], g[k]), error = function(e) conditionMessage(e))
  if (is.function(r)) list(ok = TRUE, n = sum(k), coefficients = unname(attr(r, "coefficients"))) else list(ok = FALSE, n = sum(k), error = r)
}
out <- list(R_gene_wise = fit(x$mean, x$disp_R), CREST_gene_wise = fit(x$mean, x$disp_CREST),
            versions = list(R = R.version.string, DESeq2 = as.character(packageVersion("DESeq2"))))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(d, "mdc_R.json"))
"""


def stratum_tables(h5, obs):
    genes, strata, info = cc.build_strata(h5, obs)
    return genes, {n: (c, o) for n, c, o in strata}


def r_first_pass(genes, counts, obs):
    d = Path(tempfile.mkdtemp())
    names = [f"s{i}" for i in range(counts.shape[0])]
    pl.DataFrame({"gene": genes}).hstack(pl.DataFrame(np.ascontiguousarray(np.asarray(counts).T), schema=names)).write_csv(d / "counts.csv")
    obs.select("Status").with_columns(pl.Series("sample", names)).select("sample", "Status").write_csv(d / "coldata.csv")
    (d / "run.R").write_text(R_FIRST_PASS)
    env = {**os.environ, **{v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    p = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return pl.read_csv(d / "r_first_pass.csv", null_values="NA", infer_schema_length=None), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", default="tests/data")
    a = ap.parse_args()
    out = Path(a.out)
    import crest
    genes, tabs = stratum_tables(a.h5, a.obs)
    rng = np.random.default_rng(SEED)
    rows = []
    for name in LOCAL_STRATA:
        counts, obs = tabs[name]
        r, _ = r_first_pass(genes, counts, obs)
        m, g, az = r["baseMean"].to_numpy(), r["dispGeneEst"].to_numpy(), r["allZero"].to_numpy().astype(bool)
        use = np.flatnonzero(~az & np.isfinite(g) & (g >= 10 * MIN_DISP))
        pick = np.sort(rng.choice(use, min(SUBSAMPLE, len(use)), replace=False))
        print(name, counts.shape, "genes in the trend fit:", len(use), "written:", len(pick), flush=True)
        rows += [(name, float(m[i]), float(g[i])) for i in pick]
    pl.DataFrame(rows, schema=["case", "mean", "disp"], orient="row").write_csv(out / "deseq2_local_fit_covid_cases.csv")

    counts, obs = tabs[MDC]
    r, d = r_first_pass(genes, counts, obs)
    ds = crest.tl.DESeq2(counts, obs.select("Status"), "~ Status", ref_levels={"Status": "Healthy"}, var_names=list(genes), quiet=True,
                         min_replicates_for_replace=float("inf"))
    f = ds.fit
    rm, rg = r["baseMean"].to_numpy(), r["dispGeneEst"].to_numpy()
    cm, cg = np.asarray(f["baseMean"], float), np.asarray(f["dispGeneEst"], float)
    assert np.allclose(rm, cm, rtol=1e-12, equal_nan=True)
    keep = np.flatnonzero((np.nan_to_num(rg) > 100 * MIN_DISP) | (np.nan_to_num(cg) > 100 * MIN_DISP))
    tab = pl.DataFrame({"mean": rm[keep], "disp_R": rg[keep], "disp_CREST": cg[keep]})
    tab.write_csv(out / "deseq2_trend_mdc_inputs.csv")
    tab.write_csv(d / "mdc_inputs.csv")
    (d / "run_mdc.R").write_text(R_MDC)
    p = subprocess.run(["Rscript", str(d / "run_mdc.R"), str(d)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    meta = json.loads((d / "mdc_R.json").read_text())
    meta.update({"stratum": MDC, "samples": int(counts.shape[0]), "genes": int(len(genes)), "rows": int(len(keep)),
                 "generated_by": "bench/deseq2/covid_trend_inputs.py", "crest_version": crest.__version__})
    (out / "deseq2_trend_mdc_R.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
