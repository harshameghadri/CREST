"""Pseudobulk DE on real data: Kang et al. 2018 IFN-β PBMCs (GEO GSE96583, batch 2).

8 donors × {ctrl, stim}, ~29k cells, 8 annotated cell types. For every cell
type the pseudobulk counts (sum over cells of a donor × condition) are tested
with the paired design ``~ ind + stim`` by CREST, R DESeq2 and pydeseq2, and
the results / wall times compared.

    python bench/deseq2/kang_pseudobulk.py [--data bench_data/kang] [--out bench/deseq2/results]
        [--skip-r] [--skip-pydeseq2] [--threads 1 2 4 8 16 32 64] [--repeats 3]

Timings are thread-matched: CREST at RAYON_NUM_THREADS=k (in a fresh process), R DESeq2
serial (k = 1) or with BiocParallel MulticoreParam(k), pydeseq2 with n_cpus=k, for each k
in --threads; the report gives the speed-up at every matched k and best vs best, the
headline being the minimum. CREST's and R's fit diagnostics are logged per cell type, and
every gene called significant by only one of them is written out with both fits' details.

R is timed at every k so that R's own best thread count is reported (R DESeq2 does not always
speed up with more BiocParallel workers). A failed R or pydeseq2 call at one k is reported on
stderr and in the output, the other k values still run, and the script exits with status 2 after
writing everything it has. R's results are kept for every k and compared with the first
successful k (``R_padj_ndiff_t<k>`` genes with a different padj, ``sig_R_diff_t<k>`` genes with
a different significance call). With more than one BiocParallel worker the genes are reordered
before the matrix is handed to R (see ``gene_order_for_workers``); no gene is removed and the
results are restored to the input order.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import scipy.io
import scipy.sparse as sp

import crest

GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
FILES = {
    "GSM2560248_2.1.mtx.gz": f"{GEO}/samples/GSM2560nnn/GSM2560248/suppl/GSM2560248_2.1.mtx.gz",
    "GSM2560248_barcodes.tsv.gz": f"{GEO}/samples/GSM2560nnn/GSM2560248/suppl/GSM2560248_barcodes.tsv.gz",
    "GSM2560249_2.2.mtx.gz": f"{GEO}/samples/GSM2560nnn/GSM2560249/suppl/GSM2560249_2.2.mtx.gz",
    "GSM2560249_barcodes.tsv.gz": f"{GEO}/samples/GSM2560nnn/GSM2560249/suppl/GSM2560249_barcodes.tsv.gz",
    "GSE96583_batch2.genes.tsv.gz": f"{GEO}/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.genes.tsv.gz",
    "GSE96583_batch2.total.tsne.df.tsv.gz": f"{GEO}/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.total.tsne.df.tsv.gz",
}


def fetch(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        if not (d / name).exists():
            print(f"  [download] {name}")
            urllib.request.urlretrieve(url, d / name)


def load(d: Path, singlets_only: bool = True) -> crest.BioFrame:
    """Cells × genes BioFrame with obs columns ind (donor), stim, cell (type),
    multiplets (demuxlet call). ``singlets_only`` keeps annotated singlets."""
    mats, bcs = [], []
    for mtx, bc in [("GSM2560248_2.1.mtx.gz", "GSM2560248_barcodes.tsv.gz"),
                    ("GSM2560249_2.2.mtx.gz", "GSM2560249_barcodes.tsv.gz")]:
        mats.append(scipy.io.mmread(gzip.open(d / mtx)).T.tocsr())  # genes × cells on disk
        bcs += gzip.open(d / bc).read().decode().split()
    X = sp.vstack(mats).tocsr()
    ann = pl.read_csv(d / "GSE96583_batch2.total.tsne.df.tsv.gz", separator="\t", has_header=False, skip_rows=1,
                      new_columns=["barcode", "tsne1", "tsne2", "ind", "stim", "cluster", "cell", "multiplets"])
    # the annotation renames stim barcodes that collide with ctrl ones "-11"; rows are in matrix order
    stem = [b.rsplit("-", 1)[0] for b in ann["barcode"].to_list()]
    if ann.height != X.shape[0] or stem != [b.rsplit("-", 1)[0] for b in bcs]:
        raise RuntimeError("annotation rows do not match the matrix barcodes")
    genes = pl.read_csv(d / "GSE96583_batch2.genes.tsv.gz", separator="\t", has_header=False,
                        new_columns=["gene_ids", "gene_name"])
    obs = ann.select(pl.col("barcode"), pl.col("ind").cast(pl.Utf8), "stim", "cell", "multiplets")
    bf = crest.BioFrame.from_scipy(X, obs=obs, var=genes.with_columns(
        pl.col("gene_name").is_duplicated().alias("_dup")))
    if not singlets_only:
        return bf
    return bf.filter_cells((pl.col("multiplets") == "singlet") & pl.col("cell").is_not_null())


R_SCRIPT = r"""
suppressMessages(library(DESeq2))
args <- commandArgs(TRUE)
d <- args[1]
workers <- as.integer(args[2])
cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
cd <- read.csv(file.path(d, "coldata.csv"), row.names = 1, colClasses = "character")
cd$ind <- factor(cd$ind); cd$stim <- factor(cd$stim, levels = c("ctrl", "stim"))
par <- workers > 1
if (par) {
  suppressMessages(library(BiocParallel))
  bp <- MulticoreParam(workers = workers)
}
# timing window (same as compare_r.py): DESeqDataSetFromMatrix + DESeq + results
t0 <- proc.time()[["elapsed"]]
dds <- DESeqDataSetFromMatrix(round(cts), cd, ~ ind + stim)
if (par) {
  dds <- DESeq(dds, quiet = TRUE, parallel = TRUE, BPPARAM = bp)
  res <- results(dds, contrast = c("stim", "stim", "ctrl"), parallel = TRUE, BPPARAM = bp)
} else {
  dds <- DESeq(dds, quiet = TRUE)
  res <- results(dds, contrast = c("stim", "stim", "ctrl"))
}
el <- proc.time()[["elapsed"]] - t0
mc <- mcols(dds)
col <- function(x) if (x %in% names(mc)) mc[[x]] else rep(NA, nrow(dds))
write.csv(data.frame(log2FoldChange = res$log2FoldChange, pvalue = res$pvalue, padj = res$padj,
                     baseMean = mc$baseMean, dispersion = mc$dispersion, dispGeneEst = col("dispGeneEst"),
                     dispGeneIter = col("dispGeneIter"), dispFit = col("dispFit"), dispMAP = col("dispMAP"),
                     dispIter = col("dispIter"), dispOutlier = col("dispOutlier"), betaConv = col("betaConv"),
                     maxCooks = col("maxCooks"), replace = col("replace")),
          file.path(d, "r.csv"), row.names = FALSE)
nz <- !col("allZero")
gi <- col("dispGeneIter"); ge <- col("dispGeneEst")
diag <- list(elapsed = el, workers = workers,
  fitType = attr(dispersionFunction(dds), "fitType"),
  n_beta_nonconverged = sum(!col("betaConv")[nz], na.rm = TRUE),
  # rows whose gene-wise dispersion fell back to the grid search (estimateDispersionsGeneEst:
  # not converged, i.e. iter >= maxit (100) or iter == 1, and estimate > 10 * minDisp)
  n_grid_gene = sum(((gi >= 100) | (gi == 1)) & ge > 1e-7 & nz, na.rm = TRUE),
  # rows whose MAP dispersion hit maxit (refit by grid search in estimateDispersionsMAP)
  n_grid_map = sum(col("dispIter") >= 100 & nz, na.rm = TRUE),
  n_disp_outlier = sum(col("dispOutlier")[nz], na.rm = TRUE),
  n_replaced = if ("replace" %in% names(mc)) sum(mc$replace, na.rm = TRUE) else 0,
  deseq2 = as.character(packageVersion("DESeq2")),
  biocparallel = as.character(packageVersion("BiocParallel")),
  r_version = R.version.string)
writeLines(jsonlite::toJSON(diag, auto_unbox = TRUE, digits = NA), file.path(d, "diag.json"))
"""

THREAD_VARS = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
MAXIT = 100  # DESeq2 / CREST dispersion maxit
MIN_DISP = 1e-8


def _thread_env(t: int) -> dict:
    env = dict(os.environ)
    for v in THREAD_VARS:
        env[v] = str(t)
    return env


class RRunError(RuntimeError):
    """Rscript exited with a non-zero status; the message carries the tail of its stderr."""


R_STDERR_TAIL = 3000  # characters of R's stderr kept in an error message


def worker_chunks(n_genes: int, workers: int) -> list:
    """Row ranges that BiocParallel workers receive in ``DESeq(parallel = TRUE)``: DESeq2
    splits the rows into ``workers`` consecutive blocks of (almost) equal size."""
    workers = max(1, min(workers, n_genes))
    base, extra = divmod(n_genes, workers)
    out, lo = [], 0
    for i in range(workers):
        hi = lo + base + (1 if i < extra else 0)
        out.append(range(lo, hi))
        lo = hi
    return out


def gene_order_for_workers(counts, workers: int) -> np.ndarray:
    """Order in which genes are handed to R. ``counts`` is samples x genes.

    With one worker the order is unchanged. With several, all-zero genes (which are typically
    stored together at the end of a gene list) are interleaved evenly with the expressed
    genes, so that every consecutive block of rows that BiocParallel hands to one worker holds
    expressed genes. Every gene stays in the model, so the baseMean quantiles behind
    ``results()``' independent filtering and the set of tests adjusted together are the same
    as in the serial run: dropping the all-zero genes instead would change which genes pass
    that filter. DESeq2 fits every gene independently, so the order does not change a
    gene's result."""
    counts = np.asarray(counts)
    n = counts.shape[1]
    if workers <= 1 or n == 0:
        return np.arange(n)
    zero = counts.sum(axis=0) == 0
    nz_idx, z_idx = np.flatnonzero(~zero), np.flatnonzero(zero)
    if not len(z_idx) or not len(nz_idx):
        return np.arange(n)
    # evenly spaced sort keys inside each class; a stable sort merges the two classes
    keys = np.concatenate([(np.arange(len(nz_idx)) + 0.5) / len(nz_idx), (np.arange(len(z_idx)) + 0.5) / len(z_idx)])
    return np.concatenate([nz_idx, z_idx])[np.argsort(keys, kind="stable")]


def run_r(counts, obs, genes, workers: int = 1):
    """R DESeq2 in a fresh Rscript; BLAS threads limited to ``workers`` as well. Raises
    ``RRunError`` (with R's stderr) if Rscript fails. Results are in the input gene order."""
    d = Path(tempfile.mkdtemp())
    order = gene_order_for_workers(counts, workers)
    names = obs["pseudobulk"].to_list()
    gene_col = [genes[i] for i in order]
    mat = np.ascontiguousarray(np.asarray(counts)[:, order].T)
    pl.DataFrame({"gene": gene_col}).hstack(pl.DataFrame(mat, schema=names)).write_csv(d / "counts.csv")
    obs.select(pl.col("pseudobulk").alias("sample"), "ind", "stim").write_csv(d / "coldata.csv")
    (d / "run.R").write_text(R_SCRIPT)
    proc = subprocess.run(["Rscript", str(d / "run.R"), str(d), str(workers)], capture_output=True, text=True,
                          env=_thread_env(workers))
    if proc.returncode != 0:
        raise RRunError(f"Rscript failed with exit status {proc.returncode} (workers = {workers}); "
                        f"stderr (tail):\n{(proc.stderr or '')[-R_STDERR_TAIL:]}")
    res = pl.read_csv(d / "r.csv", null_values="NA")
    return res[np.argsort(order)], json.loads((d / "diag.json").read_text())


def timed_over_threads(run, threads: list, label: str) -> dict:
    """``{k: (result, None)}`` or ``{k: (None, error text)}`` for ``run(k)`` at every k. A
    failure at one k is printed to stderr and does not stop the other values of k."""
    out = {}
    for t in threads:
        try:
            out[t] = (run(t), None)
        except Exception as e:  # noqa: BLE001 - one failed configuration must not lose the others
            msg = f"{type(e).__name__}: {e}"
            print(f"  [{label} @ {t}] FAILED: {msg}", file=sys.stderr, flush=True)
            out[t] = (None, msg)
    return out


def padj_ndiff(ref: pl.DataFrame, other: pl.DataFrame, rtol: float = 1e-8) -> int:
    """Number of genes whose padj differs between two R runs (a missing value equals a
    missing value only)."""
    a, b = ref["padj"].to_numpy(allow_copy=True).astype(float), other["padj"].to_numpy(allow_copy=True).astype(float)
    na, nb = np.isnan(a), np.isnan(b)
    both = ~na & ~nb
    close = np.zeros(len(a), bool)
    close[both] = np.isclose(a[both], b[both], rtol=rtol, atol=0.0)
    return int((~((na & nb) | close)).sum())


def sig_ndiff(ref: pl.DataFrame, other: pl.DataFrame, alpha: float = 0.1) -> int:
    """Number of genes whose significance call (padj < alpha) differs between two R runs."""
    s = lambda r: r["padj"].fill_null(1).to_numpy() < alpha  # noqa: E731
    return int((s(ref) != s(other)).sum())


def run_pydeseq2(counts, obs, genes, n_cpus: int = 1):
    """pydeseq2 with ``n_cpus`` joblib workers, BLAS limited to the same count, alpha = 0.1
    (DESeq2's default; pydeseq2's DeseqStats defaults to 0.05)."""
    import pandas as pd
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats
    from threadpoolctl import threadpool_limits

    df = pd.DataFrame(counts.astype(int), index=obs["pseudobulk"].to_list(), columns=genes)
    meta = pd.DataFrame({"ind": obs["ind"].to_list(), "stim": obs["stim"].to_list()})
    meta.index = df.index
    with threadpool_limits(limits=n_cpus):
        t0 = time.perf_counter()
        inf = DefaultInference(n_cpus=n_cpus)
        dds = DeseqDataSet(counts=df, metadata=meta, design="~ind + stim", quiet=True, inference=inf)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["stim", "stim", "ctrl"], alpha=0.1, inference=inf, quiet=True)
        st.summary()
        el = time.perf_counter() - t0
    return st.results_df, el


def crest_diagnostics(ds) -> dict:
    """CREST's fit diagnostics, comparable to R's: fit type actually used, beta rows that
    did not converge, gene-wise dispersion rows that fell back to the grid search (same rule
    as src/deseq: iter >= maxit or iter == 1, estimate > 10 * min_disp), dispersion outliers,
    Cook's replacements, warnings. CREST does not expose its MAP iteration counts, so the
    MAP-stage grid refits cannot be counted (R's n_grid_map has no CREST counterpart)."""
    f = ds.fit
    nz = ~np.asarray(f["allZero"], bool)
    it, ge = np.asarray(f["dispGeneIter"], float), np.asarray(f["dispGeneEst"], float)
    return {"fitType": f["fitType"], "n_beta_nonconverged": int((~np.asarray(f["betaConv"], bool) & nz).sum()),
            "n_grid_gene": int((((it >= MAXIT) | (it == 1)) & (ge > 10 * MIN_DISP) & nz).sum()),
            "n_disp_outlier": int((np.asarray(f["dispOutlier"], bool) & nz).sum()),
            "n_replaced": int(np.asarray(f["replace"], bool).sum()), "messages": " | ".join(f["messages"])}


# --------------------------------------------------------------------------- CREST timing worker
def crest_worker(npz: Path, out: Path, repeats: int) -> None:
    """Fit every cell type ``repeats`` times in this process (thread count fixed by
    RAYON_NUM_THREADS, set by the parent before the rayon pool exists); first fit untimed."""
    z = np.load(npz, allow_pickle=False)
    cts = [str(x) for x in z["cell_types"]]
    times = {}
    for i, ct in enumerate(cts):
        counts = z[f"counts_{i}"]
        obs = pl.DataFrame({"ind": z[f"ind_{i}"].astype(str), "stim": z[f"stim_{i}"].astype(str)})
        if i == 0:  # warm-up: thread-pool start-up, page faults
            crest.tl.DESeq2(counts, obs, "~ ind + stim", quiet=True).results(("stim", "stim", "ctrl"))
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            crest.tl.DESeq2(counts, obs, "~ ind + stim", quiet=True).results(("stim", "stim", "ctrl"))
            ts.append(time.perf_counter() - t0)
        times[ct] = ts
    out.write_text(json.dumps({"rayon_threads": os.environ.get("RAYON_NUM_THREADS"), "times": times}))


def crest_times(subs: list, threads: int, repeats: int) -> dict:
    d = Path(tempfile.mkdtemp())
    arrays = {"cell_types": np.array([ct for ct, _ in subs], dtype=str)}
    for i, (_, sub) in enumerate(subs):
        arrays[f"counts_{i}"] = np.asarray(sub.counts)
        arrays[f"ind_{i}"] = sub.obs["ind"].cast(pl.Utf8).to_numpy().astype(str)
        arrays[f"stim_{i}"] = sub.obs["stim"].cast(pl.Utf8).to_numpy().astype(str)
    np.savez(d / "pb.npz", **arrays)
    subprocess.run([sys.executable, __file__, "--crest-worker", str(d / "pb.npz"), str(d / "t.json"),
                    "--repeats", str(repeats)], check=True, env=_thread_env(threads))
    return json.loads((d / "t.json").read_text())["times"]


def _speedups(df: pl.DataFrame, threads: list, other: str) -> list[dict]:
    """Total time over cell types, CREST vs ``other`` (R / pydeseq2): matched at each thread
    count, and best vs best; the headline is the minimum."""
    rows = []
    # a thread count is used only if every cell type has a time for it (a failed run is a null,
    # and a sum over the remaining cell types would understate the total)
    tot = lambda tool, t: (float(df[f"time_{tool}_t{t}"].sum()) if f"time_{tool}_t{t}" in df.columns  # noqa: E731
                           and df[f"time_{tool}_t{t}"].null_count() == 0 else None)
    for t in threads:
        c, o = tot("CREST", t), tot(other, t)
        if c and o:
            rows.append({"vs": other, "basis": f"matched @ {t}", "CREST_s": c, "other_s": o, "speedup": o / c})
    cs = {t: tot("CREST", t) for t in threads if tot("CREST", t)}
    os_ = {t: tot(other, t) for t in threads if tot(other, t)}
    if cs and os_:
        bc, bo = min(cs, key=cs.get), min(os_, key=os_.get)
        rows.append({"vs": other, "basis": f"best vs best (CREST @ {bc}, {other} @ {bo})", "CREST_s": cs[bc],
                     "other_s": os_[bo], "speedup": os_[bo] / cs[bc]})
    if rows:
        h = min(rows, key=lambda r: r["speedup"])
        rows.append({**h, "basis": f"HEADLINE = min ({h['basis']})"})
    return rows


def _md_table(df: pl.DataFrame) -> str:
    s = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    for row in df.iter_rows():
        s += "| " + " | ".join("" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)) for v in row) + " |\n"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/kang")
    ap.add_argument("--out", default="bench/deseq2/results")
    ap.add_argument("--skip-r", action="store_true")
    ap.add_argument("--skip-pydeseq2", action="store_true")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64],
                    help="thread counts: CREST RAYON_NUM_THREADS, R BiocParallel MulticoreParam workers "
                         "(1 = serial), pydeseq2 n_cpus; capped at the logical core count")
    ap.add_argument("--repeats", type=int, default=3, help="CREST timing repeats per cell type (median)")
    ap.add_argument("--crest-worker", nargs=2, metavar=("NPZ", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.crest_worker:
        crest_worker(Path(a.crest_worker[0]), Path(a.crest_worker[1]), a.repeats)
        return 0
    ncpu = os.cpu_count() or 1
    threads = sorted({min(t, ncpu) for t in a.threads})
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch(data)
    t0 = time.perf_counter()
    bf = load(data)
    t_load = time.perf_counter() - t0
    print(f"{bf.n_obs} cells × {bf.n_vars} genes (load {t_load:.1f}s); thread counts {threads}")

    t0 = time.perf_counter()
    pb = crest.tl.pseudobulk(bf, ["ind", "stim"], groupby="cell", min_cells=10)
    t_pb = time.perf_counter() - t0
    t0 = time.perf_counter()
    crest.tl.pseudobulk_de(bf, ["ind", "stim"], design="~ ind + stim", contrast=("stim", "stim", "ctrl"),
                           groupby="cell", min_cells=10)
    t_crest_all = time.perf_counter() - t0
    print(f"pseudobulk: {pb} in {t_pb:.2f}s; pseudobulk_de over all cell types {t_crest_all:.2f}s (all cores)")

    genes = list(bf.var["gene_ids"])
    subs = []
    for ct in pb.obs["cell"].unique(maintain_order=True).to_list():
        sub = pb.subset((pb.obs["cell"] == ct).to_numpy())
        keep = sub.obs.group_by("ind").agg(pl.col("stim").n_unique().alias("k")).filter(pl.col("k") == 2)["ind"]
        sub = sub.subset(sub.obs["ind"].is_in(keep.implode()).to_numpy())  # donors with both conditions
        if sub.obs.height >= 6:
            subs.append((ct, sub))

    ctimes = {t: crest_times(subs, t, a.repeats) for t in threads}
    rows, diags, disc, failures, r_info = [], [], [], [], {}
    for ct, sub in subs:
        ds = crest.tl.DESeq2(sub, design="~ ind + stim", quiet=True)
        res = ds.results(("stim", "stim", "ctrl"))
        sig_c = res["padj"].fill_null(1).to_numpy() < 0.1
        row = {"cell_type": ct, "samples": sub.obs.height, "genes_tested": int((~ds.fit["allZero"]).sum()),
               "sig_CREST": int(sig_c.sum())}
        for t in threads:
            row[f"time_CREST_t{t}"] = float(np.median(ctimes[t][ct]))
        dg = {"cell_type": ct, **{f"CREST_{k}": v for k, v in crest_diagnostics(ds).items()}}
        lp_c = -np.log10(res["pvalue"].fill_null(np.nan).to_numpy())
        r = None
        if not a.skip_r:
            runs = timed_over_threads(lambda t: run_r(sub.counts, sub.obs, genes, workers=t), threads, f"R, {ct}")
            ok_k = [t for t in threads if runs[t][0] is not None]
            for t in threads:
                row[f"time_R_t{t}"] = runs[t][0][1]["elapsed"] if runs[t][0] is not None else None
                if runs[t][1] is not None:
                    failures.append({"cell_type": ct, "tool": "R", "threads": t, "error": runs[t][1]})
            if ok_k:  # reference results and diagnostics: the first thread count that worked (serial if it did)
                ref_k = ok_k[0]
                r, rdiag = runs[ref_k][0]
                row["R_reference_threads"] = ref_k
                r_info = {k: rdiag.get(k) for k in ("r_version", "deseq2", "biocparallel")}
                for t in ok_k:  # R's own results at every k, against the reference
                    rt = runs[t][0][0]
                    row[f"sig_R_t{t}"] = int((rt["padj"].fill_null(1).to_numpy() < 0.1).sum())
                    row[f"R_padj_ndiff_t{t}"] = padj_ndiff(r, rt)
                    row[f"sig_R_diff_t{t}"] = sig_ndiff(r, rt)
        if r is not None:
            lp_r = -np.log10(r["pvalue"].fill_null(np.nan).to_numpy())
            d_lfc = np.abs(res["log2FoldChange"].fill_null(np.nan).to_numpy() - r["log2FoldChange"].fill_null(np.nan).to_numpy())
            sig_r = r["padj"].fill_null(1).to_numpy() < 0.1
            row.update({"sig_R": int(sig_r.sum()), "sig_overlap_R": int((sig_r & sig_c).sum()),
                        "lfc_maxdiff_R": float(np.nanmax(d_lfc)), "log10p_maxdiff_R": float(np.nanmax(np.abs(lp_c - lp_r)))})
            dg.update({f"R_{k}": rdiag[k] for k in ("fitType", "n_beta_nonconverged", "n_grid_gene", "n_grid_map",
                                                    "n_disp_outlier", "n_replaced", "deseq2")})
            # every gene called significant by exactly one of CREST and R, with both fits' details
            f = ds.fit
            for gi in np.flatnonzero(sig_r != sig_c):
                disc.append({"cell_type": ct, "gene": genes[gi], "sig_CREST": bool(sig_c[gi]), "sig_R": bool(sig_r[gi]),
                             "padj_CREST": res["padj"][int(gi)], "padj_R": r["padj"][int(gi)],
                             "pvalue_CREST": res["pvalue"][int(gi)], "pvalue_R": r["pvalue"][int(gi)],
                             "lfc_CREST": res["log2FoldChange"][int(gi)], "lfc_R": r["log2FoldChange"][int(gi)],
                             "baseMean": float(f["baseMean"][gi]),
                             "dispersion_CREST": float(f["dispersion"][gi]), "dispersion_R": r["dispersion"][int(gi)],
                             "dispGeneEst_CREST": float(f["dispGeneEst"][gi]), "dispGeneEst_R": r["dispGeneEst"][int(gi)],
                             "dispGeneIter_CREST": float(f["dispGeneIter"][gi]), "dispGeneIter_R": r["dispGeneIter"][int(gi)],
                             "dispOutlier_CREST": bool(f["dispOutlier"][gi]), "dispOutlier_R": r["dispOutlier"][int(gi)],
                             "betaConv_CREST": bool(f["betaConv"][gi]), "betaConv_R": r["betaConv"][int(gi)],
                             "maxCooks_CREST": float(f["maxCooks"][gi]), "maxCooks_R": r["maxCooks"][int(gi)]})
        if not a.skip_pydeseq2:
            pruns = timed_over_threads(lambda t: run_pydeseq2(sub.counts, sub.obs, genes, n_cpus=t), threads,
                                       f"pydeseq2, {ct}")
            pr = None
            for t in threads:
                row[f"time_pydeseq2_t{t}"] = pruns[t][0][1] if pruns[t][0] is not None else None
                if pruns[t][1] is not None:
                    failures.append({"cell_type": ct, "tool": "pydeseq2", "threads": t, "error": pruns[t][1]})
                elif pr is None:
                    pr = pruns[t][0][0]
            if pr is not None:
                sig_p = pr["padj"].fillna(1).to_numpy() < 0.1
                row.update({"sig_pydeseq2": int(sig_p.sum()), "sig_overlap_pydeseq2": int((sig_p & sig_c).sum())})
        rows.append(row)
        diags.append(dg)
        print(row, dg, flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    dgf = pl.DataFrame(diags, infer_schema_length=None)
    sp_rows = (_speedups(df, threads, "R") if not a.skip_r else []) + \
              (_speedups(df, threads, "pydeseq2") if not a.skip_pydeseq2 else [])
    spf = pl.DataFrame(sp_rows) if sp_rows else pl.DataFrame()
    df.write_csv(out / "kang_pseudobulk.csv")
    dgf.write_csv(out / "kang_deseq2_diagnostics.csv")
    if spf.height:
        spf.write_csv(out / "kang_deseq2_speedups.csv")
    if disc:
        pl.DataFrame(disc, infer_schema_length=None).write_csv(out / "kang_discordant_genes.csv")
    with open(out / "kang_pseudobulk.md", "w") as fh:
        fh.write(f"Kang 2018 (GSE96583 batch 2): {bf.n_obs} cells, design `~ ind + stim`, stim vs ctrl, alpha 0.1 "
                 f"for every tool (pydeseq2's DeseqStats is given alpha=0.1; its default is 0.05).\n\n")
        fh.write(f"CREST pseudobulk aggregation {t_pb:.2f} s; `pseudobulk_de` over all cell types {t_crest_all:.2f} s "
                 f"(all cores).\n\n")
        fh.write(f"Timing, per cell type: `time_CREST_t<k>` = CREST at RAYON_NUM_THREADS=k (fresh process, median of "
                 f"{a.repeats}, after one untimed fit); `time_R_t<k>` = R DESeq2, serial for k = 1, "
                 f"`DESeq(..., parallel=TRUE, BPPARAM=MulticoreParam(k))` + `results(..., parallel=TRUE)` otherwise, "
                 f"BLAS threads limited to k; the window is DESeqDataSetFromMatrix + DESeq + results (as in "
                 f"compare_r.py; CREST's constructor + results is the same work); `time_pydeseq2_t<k>` = pydeseq2 "
                 f"with `n_cpus=k` (DeseqDataSet + deseq2 + DeseqStats.summary). Thread counts: {threads}.\n\n")
        if r_info:
            fh.write(f"R reference: {r_info.get('r_version')}, DESeq2 {r_info.get('deseq2')}, BiocParallel "
                     f"{r_info.get('biocparallel')}.\n\n")
        fh.write(_md_table(df))
        if failures:
            fh.write(f"\n**{len(failures)} failed run(s)** (null times above; the other thread counts are unaffected):\n\n"
                     + "".join(f"* {f['cell_type']}, {f['tool']} @ {f['threads']}: {f['error'][:300]!r}\n"
                               for f in failures))
        if spf.height:
            fh.write("\nTotal over cell types, CREST vs each tool: matched at every thread count and best vs best; the "
                     "headline is the minimum.\n\n" + _md_table(spf))
        fh.write("\nFit diagnostics per cell type (to investigate discordant calls): fit type used, beta rows not "
                 "converged, gene-wise dispersion rows that fell back to the grid search, MAP rows that hit maxit "
                 "(R only: CREST does not expose MAP iterations), dispersion outliers, Cook's replacements.\n\n")
        fh.write(_md_table(dgf))
        if disc:
            fh.write(f"\n{len(disc)} genes are significant (padj < 0.1) in exactly one of CREST and R; per-gene fit "
                     f"details in `kang_discordant_genes.csv`.\n")
    print(json.dumps({"totals": {c: (float(df[c].sum()) if df[c].null_count() == 0 else None)  # null = a failed run
                          for c in df.columns if c.startswith("time_")},
                      "failed_runs": len(failures)}, indent=1))
    if failures:
        print(f"{len(failures)} run(s) failed (listed in kang_pseudobulk.md); exit status 2", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
