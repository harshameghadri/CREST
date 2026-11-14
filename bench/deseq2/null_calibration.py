"""Null calibration of pseudobulk DESeq2 on Kang et al. 2018 (PBMCs, 8 donors x ctrl / stim).

The stim / ctrl label of every cell is permuted *within its donor*, so that condition is unrelated to
expression while donor structure, cell numbers and library sizes stay as they are. For every permutation
the counts are summed per donor x (permuted) condition within each cell type and tested with
``~ ind + stim`` (stim vs ctrl) by CREST, R DESeq2 and, when installed, pydeseq2. Nothing is truly
differentially expressed, so the fraction of genes with an adjusted p-value below alpha estimates the
false-positive rate at that threshold (and the fraction of (permutation, cell type) fits with at least
one call estimates the family-wise error of the BH procedure). This is a record of a run, not a test.

    python bench/deseq2/null_calibration.py --h5 kang.h5 --obs kang.obs.parquet --work WORKDIR \
        --out bench/deseq2/results [--perms 20] [--seed 1] [--jobs 12] [--methods crest r pydeseq2]
        [--cell-types "B cells" ...]

Input: the CREST-format Kang matrix (genes x cells CSC in an HDF5 group ``matrix``) and the cell
annotation table (``barcode, ind, stim, cell, multiplets``) written by ``bench/paper/datasets.py``.
Only annotated singlets are used, and pseudobulk samples with fewer than 10 cells are dropped, as in
``kang_pseudobulk.py``. The unpermuted labels are also fitted once as a positive control.
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp

MIN_CELLS = 10
ALPHAS = (0.05, 0.1)
BASEMEAN_STRATA = (10, 100)   # raw p-value rate is also reported among genes with baseMean >= these values

R_SCRIPT = r"""
suppressMessages({library(DESeq2); library(parallel)})
args <- commandArgs(TRUE); d <- args[1]; cores <- as.integer(args[2])
tasks <- sub("\\.counts\\.csv$", "", list.files(d, pattern = "\\.counts\\.csv$"))
run1 <- function(t) {
  cts <- as.matrix(read.csv(file.path(d, paste0(t, ".counts.csv")), row.names = 1, check.names = FALSE))
  cd <- read.csv(file.path(d, paste0(t, ".coldata.csv")), row.names = 1, colClasses = "character")
  cd$ind <- factor(cd$ind); cd$stim <- factor(cd$stim, levels = c("ctrl", "stim"))
  dds <- DESeqDataSetFromMatrix(cts, cd, ~ ind + stim)
  dds <- DESeq(dds, quiet = TRUE)
  res <- results(dds, contrast = c("stim", "stim", "ctrl"))
  write.csv(data.frame(gene = rownames(res), baseMean = res$baseMean, pvalue = res$pvalue, padj = res$padj, log2FoldChange = res$log2FoldChange),
            file.path(d, paste0(t, ".r.csv")), row.names = FALSE)
  attr(dispersionFunction(dds), "fitType")
}
fits <- mclapply(tasks, function(t) tryCatch(run1(t), error = function(e) paste("ERROR", conditionMessage(e))), mc.cores = cores)
writeLines(c("task,fitType", paste(tasks, unlist(fits), sep = ",")), file.path(d, "r_fit_types.csv"))
writeLines(as.character(packageVersion("DESeq2")), file.path(d, "r_version.txt"))
"""


def load(h5, obs_path):
    """cells x genes CSR counts and the annotation of the annotated singlets (rows aligned)."""
    with h5py.File(h5, "r") as f:
        m = f["matrix"]
        shape = tuple(int(v) for v in m["shape"][:])           # genes x cells
        x = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape).T.tocsr()
        barcodes = [b.decode() for b in m["barcodes"][:]]
    obs = pl.read_parquet(obs_path)
    if obs.height != x.shape[0] or obs["barcode"].to_list() != barcodes:
        raise RuntimeError("annotation rows do not match the matrix barcodes")
    keep = ((obs["multiplets"] == "singlet") & obs["cell"].is_not_null() & (obs["cell"] != "NA")).to_numpy()
    return x[np.flatnonzero(keep)], obs.filter(pl.Series(keep))


def permute_within_donor(ind, stim, rng):
    out = stim.copy()
    for d in np.unique(ind):
        idx = np.flatnonzero(ind == d)
        out[idx] = rng.permutation(stim[idx])
    return out


def pseudobulk_tasks(x, obs, stim, cell_types):
    """{cell type: (counts samples x genes, coldata)} for one labelling; donors seen in only one
    condition are dropped, samples need >= MIN_CELLS cells."""
    ind, cell = obs["ind"].to_numpy(), obs["cell"].to_numpy()
    out = {}
    for ct in cell_types:
        sel = np.flatnonzero(cell == ct)
        keys = np.char.add(np.char.add(ind[sel].astype(str), "|"), stim[sel].astype(str))
        uniq, code, n = np.unique(keys, return_inverse=True, return_counts=True)
        ok = n >= MIN_CELLS
        uniq, remap = uniq[ok], np.cumsum(ok) - 1
        keep_cells = ok[code]
        g = sp.csr_matrix((np.ones(keep_cells.sum()), (remap[code[keep_cells]], np.flatnonzero(keep_cells))),
                          shape=(len(uniq), len(sel)))
        counts = np.asarray((g @ x[sel]).todense())
        co = pl.DataFrame({"sample": [u.replace("|", "_") for u in uniq], "ind": [u.split("|")[0] for u in uniq],
                           "stim": [u.split("|")[1] for u in uniq]})
        donors = co.group_by("ind").agg(pl.col("stim").n_unique().alias("k")).filter(pl.col("k") == 2)["ind"]
        m = co["ind"].is_in(donors.to_list()).to_numpy()
        if m.sum() < 6 or len(donors) < 3:
            continue
        counts = counts[m]
        nz = counts.sum(axis=0) > 0
        out[ct] = (counts[:, nz], co.filter(pl.Series(m)), np.flatnonzero(nz))
    return out


def _pydeseq2_one(args):
    task, counts, coldata, nz_index = args
    import pandas as pd
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from threadpoolctl import threadpool_limits
    df = pd.DataFrame(counts.astype(int), index=coldata["sample"].to_list(), columns=[f"g{i}" for i in nz_index])
    meta = pd.DataFrame({"ind": coldata["ind"].to_list(), "stim": coldata["stim"].to_list()}, index=df.index)
    try:
        with threadpool_limits(limits=1):
            inf = DefaultInference(n_cpus=1)
            dds = DeseqDataSet(counts=df, metadata=meta, design="~ind + stim", quiet=True, inference=inf)
            dds.deseq2()
            st = DeseqStats(dds, contrast=["stim", "stim", "ctrl"], alpha=0.1, inference=inf, quiet=True)
            st.summary()
        r = st.results_df
        return task, pl.DataFrame({"gene": [int(g[1:]) for g in r.index], "baseMean": r["baseMean"].to_numpy(), "pvalue": r["pvalue"].to_numpy(),
                                   "padj": r["padj"].to_numpy()})
    except Exception as e:  # a failed fit is recorded, not hidden
        return task, f"ERROR {type(e).__name__}: {e}"


def report(df, run, calls, out):
    """Write null_calibration.md from the per-fit table."""
    ok = df.filter(pl.col("status") == "ok")
    null = ok.filter(pl.col("labelling") != "observed")
    obs = ok.filter(pl.col("labelling") == "observed")
    methods = [m for m in ("crest", "r", "pydeseq2") if m in df["method"].unique().to_list()]
    label = {"crest": "CREST", "r": "R DESeq2", "pydeseq2": "pydeseq2"}
    L = ["# Null calibration of pseudobulk DESeq2 on Kang et al. 2018", "",
         f"Condition (stim / ctrl) permuted within donor at the cell level, {run['perms']} permutations "
         f"(seed {run['seed']}); counts summed per donor x condition within each of {len(run['cell_types'])} cell types "
         f"(annotated singlets, samples with >= {MIN_CELLS} cells, donors seen in both conditions); design `~ ind + stim`, "
         "stim vs ctrl; all genes with a non-zero count. Software: " + ", ".join(f"{k} {v}" for k, v in run["versions"].items()) + ". "
         "Adjusted p-values come from each tool's own `results()` / `summary()` with alpha = 0.1 for independent filtering. "
         "Nothing is differentially expressed under this null, so every call is a false positive. "
         "Evidence from one run (one seed); wall times are not a result of this study.", ""]
    L += ["## All cell types pooled", "",
          "| method | fits | genes tested | FPR padj < 0.05 | FPR padj < 0.1 | fits with >= 1 call (0.05) | fits with >= 1 call (0.1) | raw p < 0.05 | raw p < 0.05, baseMean >= 10 | raw p < 0.05, baseMean >= 100 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        d = null.filter(pl.col("method") == m)
        n_pad = d["n_padj"].sum()
        L.append(f"| {label[m]} | {d.height} | {d['n_tested'].sum():,} | "
                 f"{d['n_padj_lt_0.05'].sum() / n_pad:.2e} ({d['n_padj_lt_0.05'].sum()}) | {d['n_padj_lt_0.1'].sum() / n_pad:.2e} ({d['n_padj_lt_0.1'].sum()}) | "
                 f"{(d['n_padj_lt_0.05'] > 0).sum()} / {d.height} | {(d['n_padj_lt_0.1'] > 0).sum()} / {d.height} | "
                 f"{d['n_p_lt_0.05'].sum() / d['n_tested'].sum():.4f} | "
                 f"{d['n_p_lt_0.05_bm10'].sum() / max(d['n_tested_bm10'].sum(), 1):.4f} ({d['n_tested_bm10'].sum():,} genes) | "
                 f"{d['n_p_lt_0.05_bm100'].sum() / max(d['n_tested_bm100'].sum(), 1):.4f} ({d['n_tested_bm100'].sum():,} genes) |")
    L += ["", "FPR = genes with padj below the threshold / genes with a non-missing adjusted p-value, summed over fits (calls in brackets). "
          "Under a calibrated test the raw-p columns are 0.05 and BH keeps the share of fits with any call at or below the threshold. "
          "Counts of 0-3 give a discrete Wald statistic that rarely exceeds 1.96, so the raw-p rate over all genes is "
          "expected to sit below 0.05; the baseMean strata show the rate for genes with enough counts.", "",
          "## By cell type (padj < 0.1: calls summed over permutations / fits with a call; raw p < 0.05)", ""]
    cts = sorted(null["cell_type"].unique().to_list())
    L += ["| cell type | fits | " + " | ".join(label[m] for m in methods) + " |", "|---|---|" + "---|" * len(methods)]
    for ct in cts:
        cells = []
        for m in methods:
            d = null.filter((pl.col("method") == m) & (pl.col("cell_type") == ct))
            cells.append(f"{d['n_padj_lt_0.1'].sum()} / {(d['n_padj_lt_0.1'] > 0).sum()}; {d['n_p_lt_0.05'].sum() / max(d['n_tested'].sum(), 1):.3f}")
        L.append(f"| {ct} | {null.filter((pl.col('method') == methods[0]) & (pl.col('cell_type') == ct)).height} | " + " | ".join(cells) + " |")
    L += ["", "## Positive control: the observed labels (padj < 0.1)", "",
          "| cell type | " + " | ".join(label[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    for ct in cts:
        row = []
        for m in methods:
            d = obs.filter((pl.col("method") == m) & (pl.col("cell_type") == ct))
            row.append(str(d["n_padj_lt_0.1"].sum()) if d.height else "-")
        L.append(f"| {ct} | " + " | ".join(row) + " |")
    shared = {}
    for a_i, ma in enumerate(methods):
        for mb in methods[a_i + 1:]:
            n_a = n_b = n_ab = 0
            for t, ca in calls.get(ma, {}).items():
                cb = calls.get(mb, {}).get(t)
                if ca is None or cb is None or t.startswith("observed"):
                    continue
                n_a, n_b, n_ab = n_a + len(ca), n_b + len(cb), n_ab + len(set(ca) & set(cb))
            shared[f"{label[ma]} / {label[mb]}"] = (n_a, n_b, n_ab)
    L += ["", "## Calls shared between methods under the null (padj < 0.1)", "", "| pair | calls in first | calls in second | in both |", "|---|---|---|---|"]
    for k, (a_, b_, ab) in shared.items():
        L.append(f"| {k} | {a_} | {b_} | {ab} |")
    bad = df.filter(pl.col("status") != "ok")
    L += ["", "## Fits and fit types", "", f"Fits per method: {run['n_fits_per_method']}. Failed fits: {bad.height}."]
    for r in bad.iter_rows(named=True):
        L.append(f"- {r['method']} {r['labelling']} {r['cell_type']}: {r['status']}")
    for m, ft in run["fit_types"].items():
        L.append("")
        L.append(f"Dispersion trend used by {label[m]} (fits): " + ", ".join(f"{k or 'n/a'}: {v}" for k, v in ft.items())
                 + (". R substitutes a local regression when the parametric fit fails; CREST substitutes the mean (`fit_substituted`)." if m == "r" else "."))
    # the null calibration split by whether R fell back to a local trend
    if "r" in run["fit_types"] and "crest" in run["fit_types"]:
        r_loc = {(x["labelling"], x["cell_type"]) for x in df.filter((pl.col("method") == "r") & (pl.col("fit_type") == "local")).iter_rows(named=True)}
        L += ["", "### Fits where R fell back to a local trend", ""]
        if not r_loc:
            L.append("None.")
        else:
            L += ["| method | fits | calls padj < 0.1 | raw p < 0.05 |", "|---|---|---|---|"]
            for m in methods:
                d = null.filter(pl.col("method") == m)
                d = d.filter(pl.struct("labelling", "cell_type").map_elements(lambda r_: (r_["labelling"], r_["cell_type"]) in r_loc, return_dtype=pl.Boolean))
                L.append(f"| {label[m]} | {d.height} | {d['n_padj_lt_0.1'].sum()} | {d['n_p_lt_0.05'].sum() / max(d['n_tested'].sum(), 1):.4f} |")
    (out / "null_calibration.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--work", required=True, help="scratch directory for the count tables and per-fit results")
    ap.add_argument("--out", default="bench/deseq2/results")
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--methods", nargs="+", default=["crest", "r", "pydeseq2"])
    ap.add_argument("--cell-types", nargs="*", default=None)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if a.perms < 20:
        print(f"note: {a.perms} permutations (< 20), the report is a smoke test", file=sys.stderr)

    x, obs = load(a.h5, a.obs)
    cell_types = a.cell_types or sorted(obs["cell"].unique().to_list())
    ind, stim0 = obs["ind"].to_numpy().astype(str), obs["stim"].to_numpy().astype(str)
    labellings = {"observed": stim0}
    for p in range(a.perms):
        labellings[f"perm{p:02d}"] = permute_within_donor(ind, stim0, np.random.default_rng(a.seed * 1000 + p))
    # one task per (labelling, cell type)
    tasks = {}
    for lab, stim in labellings.items():
        for ct, (counts, co, nz) in pseudobulk_tasks(x, obs, stim, cell_types).items():
            tasks[f"{lab}__{ct.replace(' ', '_').replace('+', 'p')}"] = (lab, ct, counts, co, nz)
    print(f"{len(tasks)} fits per method ({len(labellings)} labellings x up to {len(cell_types)} cell types)", flush=True)

    results = {m: {} for m in a.methods}
    fit_type_of = {m: {} for m in a.methods}   # dispersion trend actually used, per method and fit
    versions = {}
    if "crest" in a.methods:
        import crest
        versions["crest"] = getattr(crest, "__version__", "unknown")
        for t, (lab, ct, counts, co, nz) in tasks.items():
            ds = crest.tl.DESeq2(counts, co.select("ind", "stim"), "~ ind + stim", quiet=True)
            r = ds.results(("stim", "stim", "ctrl"))
            fit_type_of["crest"][t] = ds.fit["fitType"]
            results["crest"][t] = pl.DataFrame({"gene": nz, "baseMean": r["baseMean"], "pvalue": r["pvalue"], "padj": r["padj"]})
        print("crest done", flush=True)
    if "r" in a.methods:
        rd = work / "r_tasks"
        rd.mkdir(exist_ok=True)
        for f in rd.glob("*"):
            f.unlink()
        for t, (lab, ct, counts, co, nz) in tasks.items():
            pl.DataFrame({"gene": [f"g{i}" for i in nz]}).hstack(pl.DataFrame(counts.T.astype(np.int64), schema=co["sample"].to_list())) \
                .write_csv(rd / f"{t}.counts.csv")
            co.write_csv(rd / f"{t}.coldata.csv")
        (rd / "run.R").write_text(R_SCRIPT)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        t0 = time.time()
        subprocess.run(["Rscript", str(rd / "run.R"), str(rd), str(a.jobs)], check=True, env=env)
        versions["R DESeq2"] = (rd / "r_version.txt").read_text().strip()
        r_fit = dict(l.split(",", 1) for l in (rd / "r_fit_types.csv").read_text().splitlines()[1:])
        for t in tasks:
            f = rd / f"{t}.r.csv"
            if f.exists():
                d = pl.read_csv(f, null_values="NA")
                results["r"][t] = d.with_columns(pl.col("gene").str.slice(1).cast(pl.Int64))
            else:
                results["r"][t] = r_fit.get(t, "ERROR no output")
        print(f"R done in {time.time() - t0:.0f} s", flush=True)
        fit_type_of["r"] = {t: r_fit.get(t, "") for t in tasks}
    if "pydeseq2" in a.methods:
        import pydeseq2
        versions["pydeseq2"] = pydeseq2.__version__
        with Pool(a.jobs) as pool:
            for t, res in pool.imap_unordered(_pydeseq2_one, [(t, c, co, nz) for t, (lab, ct, c, co, nz) in tasks.items()]):
                results["pydeseq2"][t] = res
        print("pydeseq2 done", flush=True)

    rows = []
    for m, per_task in results.items():
        for t, res in per_task.items():
            lab, ct = tasks[t][0], tasks[t][1]
            if isinstance(res, str):
                rows.append({"method": m, "labelling": lab, "cell_type": ct, "status": res, "n_samples": tasks[t][2].shape[0]})
                continue
            p, padj = res["pvalue"].to_numpy(), res["padj"].to_numpy()
            tested = ~np.isnan(p)
            row = {"method": m, "labelling": lab, "cell_type": ct, "status": "ok", "n_samples": tasks[t][2].shape[0],
                   "fit_type": fit_type_of[m].get(t, ""),
                   "n_genes": int(len(p)), "n_tested": int(tested.sum()), "n_padj": int((~np.isnan(padj)).sum()),
                   "n_p_lt_0.05": int((p[tested] < 0.05).sum())}
            bm = res["baseMean"].to_numpy()
            for lo in BASEMEAN_STRATA:
                s_ = tested & (bm >= lo)
                row[f"n_tested_bm{lo}"], row[f"n_p_lt_0.05_bm{lo}"] = int(s_.sum()), int((p[s_] < 0.05).sum())
            for al in ALPHAS:
                row[f"n_padj_lt_{al}"] = int((padj[~np.isnan(padj)] < al).sum())
            rows.append(row)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(out / "null_calibration.csv")
    (out / "null_calibration_run.json").write_text(json.dumps({
        "versions": versions, "perms": a.perms, "seed": a.seed, "cell_types": cell_types,
        "n_fits_per_method": len(tasks), "fit_types": {m: dict(sorted(collections.Counter(v.values()).items())) for m, v in fit_type_of.items() if v}},
        indent=1))
    # raw calls per fit, for set comparisons between methods
    calls = {m: {t: (res.filter(pl.col("padj") < 0.1)["gene"].to_list() if not isinstance(res, str) else None)
                 for t, res in per.items()} for m, per in results.items()}
    (work / "calls_padj_0.1.json").write_text(json.dumps(calls))
    report(df, json.loads((out / "null_calibration_run.json").read_text()), calls, out)
    print(df.filter(pl.col("labelling") != "observed").group_by("method").agg(
        pl.col("n_padj_lt_0.1").sum(), pl.col("n_tested").sum()))


if __name__ == "__main__":
    main()
