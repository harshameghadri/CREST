"""Compare crest.tl.DESeq2 with R DESeq2 on simulated and pseudobulk count data.

    python bench/deseq2/compare_r.py [--genes 5000] [--out bench/deseq2/results]

Needs Rscript with DESeq2 installed (e.g. ``apt install r-bioc-deseq2`` or
BiocManager). For each scenario the same counts/design go through R
``DESeq()`` + ``results()`` and through CREST; the script reports the largest
differences and wall times, and writes ``comparison.md``.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

import crest

R_SCRIPT = r"""
suppressMessages(library(DESeq2))
args <- commandArgs(TRUE)
d <- args[1]
cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
cd <- read.csv(file.path(d, "coldata.csv"), row.names = 1, stringsAsFactors = TRUE)
cfg <- jsonlite::fromJSON(file.path(d, "cfg.json"))
for (v in names(cfg$levels)) cd[[v]] <- factor(as.character(cd[[v]]), levels = cfg$levels[[v]])
dds <- DESeqDataSetFromMatrix(round(cts), cd, as.formula(cfg$design))
t0 <- proc.time()[["elapsed"]]
dds <- DESeq(dds, sfType = cfg$sf_type, quiet = TRUE)
res <- if (is.null(cfg$contrast)) results(dds) else results(dds, contrast = cfg$contrast)
el <- proc.time()[["elapsed"]] - t0
mc <- mcols(dds)
out <- data.frame(baseMean = res$baseMean, log2FoldChange = res$log2FoldChange, lfcSE = res$lfcSE,
                  stat = res$stat, pvalue = res$pvalue, padj = res$padj,
                  dispGeneEst = mc$dispGeneEst, dispFit = mc$dispFit, dispersion = mc$dispersion,
                  dispMAP = mc$dispMAP)
write.csv(out, file.path(d, "r_results.csv"), row.names = FALSE)
writeLines(jsonlite::toJSON(list(elapsed = el, sf = unname(sizeFactors(dds)),
  priorVar = attr(dispersionFunction(dds), "dispPriorVar"),
  trend = unname(attr(dispersionFunction(dds), "coefficients")),
  fitType = attr(dispersionFunction(dds), "fitType")), auto_unbox = TRUE, digits = NA),
  file.path(d, "r_meta.json"))
"""


def simulate(n_genes, groups, reps, rng, n_batch=0, outliers=0):
    """NB counts with a realistic mean-dispersion trend and 10% DE genes."""
    levels = [f"g{i}" for i in range(groups)]
    cond = np.repeat(levels, reps)
    m = len(cond)
    mu0 = np.exp(rng.normal(4.0, 2.0, n_genes))
    disp = (0.05 + 1.0 / mu0) * np.exp(rng.normal(0, 0.5, n_genes))
    lfc = np.zeros((n_genes, groups))
    de = rng.random(n_genes) < 0.1
    for k in range(1, groups):
        lfc[de, k] = rng.normal(0, 1.5, de.sum())
    sf = np.exp(rng.normal(0, 0.3, m))
    gi = np.array([levels.index(c) for c in cond])
    mu = mu0[:, None] * np.exp(lfc[:, gi]) * sf[None, :]
    obs = {"condition": cond}
    if n_batch:
        batch = np.array([f"b{i % n_batch}" for i in range(m)])
        bfx = rng.normal(0, 0.5, (n_genes, n_batch))
        mu = mu * np.exp(bfx[:, [int(b[1:]) for b in batch]])
        obs["batch"] = batch
    r = 1.0 / disp[:, None]
    counts = rng.negative_binomial(r, r / (r + mu)).astype(np.float64)
    if outliers:
        rows = rng.choice(n_genes, outliers, replace=False)
        cols = rng.integers(0, m, outliers)
        counts[rows, cols] = counts[rows, cols] * 20 + 100
    counts[:20] = 0  # some all-zero genes
    return counts.T, pl.DataFrame(obs)


def run(name, counts, obs, design, contrast, levels, sf_type="ratio", workdir=None):
    d = Path(tempfile.mkdtemp(dir=workdir))
    genes = [f"gene{i}" for i in range(counts.shape[1])]
    samples = [f"s{i}" for i in range(counts.shape[0])]
    pl.DataFrame({"gene": genes}).hstack(pl.DataFrame(counts.T, schema=samples)).write_csv(d / "counts.csv")
    obs.insert_column(0, pl.Series("sample", samples)).write_csv(d / "coldata.csv")
    (d / "cfg.json").write_text(json.dumps({"design": design, "contrast": contrast, "levels": levels, "sf_type": sf_type}))
    (d / "run.R").write_text(R_SCRIPT)
    subprocess.run(["Rscript", str(d / "run.R"), str(d)], check=True)
    r = pl.read_csv(d / "r_results.csv", null_values="NA")
    rmeta = json.loads((d / "r_meta.json").read_text())

    t0 = time.perf_counter()
    ds = crest.tl.DESeq2(counts, obs, design, ref_levels={k: v[0] for k, v in levels.items()}, sf_type=sf_type, quiet=True)
    res = ds.results(tuple(contrast) if contrast else None)
    t_crest = time.perf_counter() - t0

    def maxrel(a, b, floor=1e-8):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), floor))) if ok.any() else 0.0

    def q99rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.quantile(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-8), 0.99))

    f = ds.fit
    na_r, na_c = r["padj"].is_null().to_numpy(), res["padj"].is_null().to_numpy()
    sig_r = set(np.flatnonzero((r["padj"].fill_null(1) < 0.1).to_numpy()))
    sig_c = set(np.flatnonzero((res["padj"].fill_null(1) < 0.1).to_numpy()))
    lp_r, lp_c = -np.log10(r["pvalue"].to_numpy()), -np.log10(res["pvalue"].fill_null(np.nan).to_numpy())
    out = {
        "scenario": name, "samples": counts.shape[0], "genes": counts.shape[1],
        "sizeFactors": maxrel(f["size_factors"], rmeta["sf"]),
        "dispGeneEst_q99": q99rel(f["dispGeneEst"], r["dispGeneEst"]),
        "dispFit": maxrel(f["dispFit"], r["dispFit"]),
        "priorVar": abs(f["dispPriorVar"] - rmeta["priorVar"]) / rmeta["priorVar"],
        "dispersion_q99": q99rel(f["dispersion"], r["dispersion"]),
        "log2FC_q99": q99rel(res["log2FoldChange"], r["log2FoldChange"]),
        "log2FC_max_abs": float(np.nanmax(np.abs(res["log2FoldChange"].fill_null(np.nan).to_numpy() - r["log2FoldChange"].fill_null(np.nan).to_numpy()))),
        "log10p_max_abs": float(np.nanmax(np.abs(lp_c - lp_r))),
        "padj_NA_mismatch": int(np.sum(na_r != na_c)),
        "sig_R": len(sig_r), "sig_CREST": len(sig_c), "sig_overlap": len(sig_r & sig_c),
        "time_R_s": rmeta["elapsed"], "time_CREST_s": t_crest,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=5000)
    ap.add_argument("--out", default="bench/deseq2/results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    G = a.genes
    c, o = simulate(G, 2, 3, rng)
    rows.append(run("2 groups x 3", c, o, "~ condition", None, {"condition": ["g0", "g1"]}))
    c, o = simulate(G, 2, 2, rng)
    rows.append(run("2 groups x 2 (df=2)", c, o, "~ condition", None, {"condition": ["g0", "g1"]}))
    c, o = simulate(G, 2, 4, rng, n_batch=2)
    rows.append(run("batch + condition, 2 x 4", c, o, "~ batch + condition", None,
                    {"condition": ["g0", "g1"], "batch": ["b0", "b1"]}))
    c, o = simulate(G, 3, 3, rng)
    rows.append(run("3 levels, g2 vs g1", c, o, "~ condition", ["condition", "g2", "g1"],
                    {"condition": ["g0", "g1", "g2"]}))
    c, o = simulate(G, 2, 8, rng, outliers=200)
    rows.append(run("2 x 8 + outliers (replacement)", c, o, "~ condition", None, {"condition": ["g0", "g1"]}))
    c, o = simulate(G, 2, 12, rng, outliers=100)
    rows.append(run("2 x 12 + outliers", c, o, "~ condition", ["condition", "g0", "g1"], {"condition": ["g0", "g1"]}))
    df = pl.DataFrame(rows)
    pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
    print(df)
    df.write_csv(out / "comparison.csv")
    with open(out / "comparison.md", "w") as fh:
        fh.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for row in df.iter_rows():
            fh.write("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row) + " |\n")


if __name__ == "__main__":
    main()
