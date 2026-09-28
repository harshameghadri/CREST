"""Pseudobulk DE on real data: Kang et al. 2018 IFN-β PBMCs (GEO GSE96583, batch 2).

8 donors × {ctrl, stim}, ~29k cells, 8 annotated cell types. For every cell
type the pseudobulk counts (sum over cells of a donor × condition) are tested
with the paired design ``~ ind + stim`` by CREST, R DESeq2 and pydeseq2, and
the results / wall times compared.

    python bench/deseq2/kang_pseudobulk.py [--data bench_data/kang] [--out bench/deseq2/results]
        [--skip-r] [--skip-pydeseq2]
"""

import argparse
import gzip
import json
import subprocess
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


def load(d: Path) -> crest.BioFrame:
    """Cells × genes BioFrame with obs columns ind (donor), stim, cell (type)."""
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
    return bf.filter_cells((pl.col("multiplets") == "singlet") & pl.col("cell").is_not_null())


R_SCRIPT = r"""
suppressMessages(library(DESeq2))
d <- commandArgs(TRUE)[1]
cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
cd <- read.csv(file.path(d, "coldata.csv"), row.names = 1, colClasses = "character")
cd$ind <- factor(cd$ind); cd$stim <- factor(cd$stim, levels = c("ctrl", "stim"))
t0 <- proc.time()[["elapsed"]]
dds <- DESeq(DESeqDataSetFromMatrix(round(cts), cd, ~ ind + stim), quiet = TRUE)
res <- results(dds, contrast = c("stim", "stim", "ctrl"))
el <- proc.time()[["elapsed"]] - t0
write.csv(data.frame(log2FoldChange = res$log2FoldChange, pvalue = res$pvalue, padj = res$padj,
                     dispersion = mcols(dds)$dispersion), file.path(d, "r.csv"), row.names = FALSE)
writeLines(as.character(el), file.path(d, "time.txt"))
"""


def run_r(counts, obs, genes):
    d = Path(tempfile.mkdtemp())
    names = obs["pseudobulk"].to_list()
    pl.DataFrame({"gene": genes}).hstack(pl.DataFrame(counts.T, schema=names)).write_csv(d / "counts.csv")
    obs.select(pl.col("pseudobulk").alias("sample"), "ind", "stim").write_csv(d / "coldata.csv")
    (d / "run.R").write_text(R_SCRIPT)
    subprocess.run(["Rscript", str(d / "run.R"), str(d)], check=True, capture_output=True)
    return pl.read_csv(d / "r.csv", null_values="NA"), float((d / "time.txt").read_text())


def run_pydeseq2(counts, obs, genes):
    import pandas as pd
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    df = pd.DataFrame(counts.astype(int), index=obs["pseudobulk"].to_list(), columns=genes)
    meta = pd.DataFrame({"ind": obs["ind"].to_list(), "stim": obs["stim"].to_list()})
    meta.index = df.index
    t0 = time.perf_counter()
    dds = DeseqDataSet(counts=df, metadata=meta, design="~ind + stim", quiet=True,
                       inference=DefaultInference(n_cpus=None))
    dds.deseq2()
    st = DeseqStats(dds, contrast=["stim", "stim", "ctrl"], quiet=True)
    st.summary()
    return st.results_df, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/kang")
    ap.add_argument("--out", default="bench/deseq2/results")
    ap.add_argument("--skip-r", action="store_true")
    ap.add_argument("--skip-pydeseq2", action="store_true")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch(data)
    t0 = time.perf_counter()
    bf = load(data)
    t_load = time.perf_counter() - t0
    print(f"{bf.n_obs} cells × {bf.n_vars} genes (load {t_load:.1f}s)")

    t0 = time.perf_counter()
    pb = crest.tl.pseudobulk(bf, ["ind", "stim"], groupby="cell", min_cells=10)
    t_pb = time.perf_counter() - t0
    t0 = time.perf_counter()
    table = crest.tl.pseudobulk_de(bf, ["ind", "stim"], design="~ ind + stim", contrast=("stim", "stim", "ctrl"),
                                   groupby="cell", min_cells=10)
    t_crest_all = time.perf_counter() - t0
    print(f"pseudobulk: {pb} in {t_pb:.2f}s; pseudobulk_de over all cell types {t_crest_all:.2f}s")

    genes = list(bf.var["gene_ids"])
    rows = []
    for ct in pb.obs["cell"].unique(maintain_order=True).to_list():
        sub = pb.subset((pb.obs["cell"] == ct).to_numpy())
        keep = sub.obs.group_by("ind").agg(pl.col("stim").n_unique().alias("k")).filter(pl.col("k") == 2)["ind"]
        sub = sub.subset(sub.obs["ind"].is_in(keep.implode()).to_numpy())  # donors with both conditions
        if sub.obs.height < 6:
            continue
        t0 = time.perf_counter()
        ds = crest.tl.DESeq2(sub, design="~ ind + stim", quiet=True)
        res = ds.results(("stim", "stim", "ctrl"))
        t_c = time.perf_counter() - t0
        row = {"cell_type": ct, "samples": sub.obs.height, "genes_tested": int((~ds.fit["allZero"]).sum()),
               "sig_CREST": int((res["padj"].fill_null(1) < 0.1).sum()), "time_CREST_s": t_c}
        lp_c = -np.log10(res["pvalue"].fill_null(np.nan).to_numpy())
        if not a.skip_r:
            r, t_r = run_r(sub.counts, sub.obs, genes)
            lp_r = -np.log10(r["pvalue"].fill_null(np.nan).to_numpy())
            d_lfc = np.abs(res["log2FoldChange"].fill_null(np.nan).to_numpy() - r["log2FoldChange"].fill_null(np.nan).to_numpy())
            sig_r = r["padj"].fill_null(1).to_numpy() < 0.1
            sig_c = res["padj"].fill_null(1).to_numpy() < 0.1
            row.update({"sig_R": int(sig_r.sum()), "sig_overlap_R": int((sig_r & sig_c).sum()),
                        "lfc_maxdiff_R": float(np.nanmax(d_lfc)), "log10p_maxdiff_R": float(np.nanmax(np.abs(lp_c - lp_r))),
                        "time_R_s": t_r})
        if not a.skip_pydeseq2:
            pr, t_p = run_pydeseq2(sub.counts, sub.obs, genes)
            sig_p = pr["padj"].fillna(1).to_numpy() < 0.1
            row.update({"sig_pydeseq2": int(sig_p.sum()),
                        "sig_overlap_pydeseq2": int((sig_p & (res["padj"].fill_null(1).to_numpy() < 0.1)).sum()),
                        "time_pydeseq2_s": t_p})
        rows.append(row)
        print(row)
    df = pl.DataFrame(rows)
    df.write_csv(out / "kang_pseudobulk.csv")
    with open(out / "kang_pseudobulk.md", "w") as fh:
        fh.write(f"Kang 2018 (GSE96583 batch 2): {bf.n_obs} cells, design `~ ind + stim`, stim vs ctrl.\n\n")
        fh.write(f"CREST pseudobulk aggregation {t_pb:.2f} s; `pseudobulk_de` over all cell types {t_crest_all:.2f} s.\n\n")
        fh.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for row in df.iter_rows():
            fh.write("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row) + " |\n")
    print(json.dumps({"totals": {c: float(df[c].sum()) for c in df.columns if c.startswith("time_")}}, indent=1))


if __name__ == "__main__":
    main()
