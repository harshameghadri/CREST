"""CREST ingest vs scanpy.tl.ingest on Kang et al. 2018: reference = control
PBMCs, query = IFN-beta stimulated PBMCs (a real distribution shift). Both tools
get the same reference (2000 HVG, scaled, 50 PCs, 15-NN, UMAP) and transfer the
published cell-type annotation; accuracy is measured against the query cells'
own annotation.

    python bench/ingest/compare_scanpy_ingest.py [--out bench/ingest/results]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/kang")
    ap.add_argument("--out", default="bench/ingest/results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch(Path(a.data))
    bf = crest.pp.filter_genes(load(Path(a.data)), min_cells=10)
    ref = bf.filter_cells(pl.col("stim") == "ctrl")
    qry = bf.filter_cells(pl.col("stim") == "stim")
    truth = qry.obs["cell"].to_numpy()

    # ---- CREST
    for b in (ref, qry):
        crest.pp.normalize_total(b, 1e4)
        crest.pp.log1p(b)
    crest.pp.highly_variable_genes(ref, n_top_genes=2000)
    crest.pp.scale(ref, max_value=10)
    crest.tl.pca(ref, n_comps=50)
    crest.pp.neighbors(ref, n_neighbors=15)
    crest.tl.umap(ref)
    t0 = time.perf_counter()
    crest.tl.ingest(qry, ref, obs="cell")
    t_crest = time.perf_counter() - t0
    acc_crest = float((qry.obs["cell"].to_numpy() == truth).mean())

    # ---- scanpy on the same genes and the same reference preprocessing
    import scanpy as sc
    import anndata as ad
    hv = ref.var["highly_variable"].to_numpy()
    A = ref._subset(var_mask=hv).to_anndata(transform=True)
    B = qry._subset(var_mask=hv).to_anndata(transform=True)
    A.obs["cell"] = ref.obs["cell"].to_list()
    sc.pp.scale(A, max_value=10)
    # ingest applies the reference PCA to the query as given: scale it with the reference parameters
    mu, sd = np.asarray(A.var["mean"]), np.asarray(A.var["std"])
    Bx = (B.X.toarray() - mu) / sd
    B = ad.AnnData(np.clip(Bx, -10, 10), obs=B.obs, var=B.var)
    sc.tl.pca(A, n_comps=50)
    sc.pp.neighbors(A, n_neighbors=15)
    sc.tl.umap(A)
    t0 = time.perf_counter()
    sc.tl.ingest(B, A, obs="cell")
    t_sc = time.perf_counter() - t0
    acc_sc = float((B.obs["cell"].astype(str).to_numpy() == truth).mean())
    agree = float((B.obs["cell"].astype(str).to_numpy() == qry.obs["cell"].to_numpy()).mean())

    rows = [{"tool": "crest", "time_s": t_crest, "accuracy": acc_crest},
            {"tool": "scanpy", "time_s": t_sc, "accuracy": acc_sc}]
    print(rows, "label agreement crest vs scanpy:", agree)
    df = pl.DataFrame(rows)
    df.write_csv(out / "kang_ingest.csv")
    per = pl.DataFrame({"truth": truth, "crest": qry.obs["cell"].to_numpy(),
                        "scanpy": B.obs["cell"].astype(str).to_numpy()})
    by = per.group_by("truth").agg(pl.len().alias("n"), (pl.col("crest") == pl.col("truth")).mean().alias("crest_acc"),
                                   (pl.col("scanpy") == pl.col("truth")).mean().alias("scanpy_acc")).sort("n", descending=True)
    with open(out / "kang_ingest.md", "w") as fh:
        fh.write(f"Kang 2018: reference = {ref.n_obs} control cells, query = {qry.n_obs} stimulated cells; "
                 f"label = published cell type.\n\n| tool | time_s | accuracy |\n|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['tool']} | {r['time_s']:.2f} | {r['accuracy']:.4f} |\n")
        fh.write(f"\nLabel agreement CREST vs scanpy: {agree:.4f}\n\nPer cell type:\n\n| truth | n | crest | scanpy |\n|---|---|---|---|\n")
        for t, n, c, s in by.iter_rows():
            fh.write(f"| {t} | {n} | {c:.3f} | {s:.3f} |\n")


if __name__ == "__main__":
    main()
