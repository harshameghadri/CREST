"""CREST ingest vs scanpy.tl.ingest on Kang et al. 2018: reference = control
PBMCs, query = IFN-beta stimulated PBMCs (a real distribution shift). Both tools
get the same reference (2000 HVG, scaled, 50 PCs, 15-NN, UMAP) and transfer the
published cell-type annotation; accuracy is measured against the query cells'
own annotation.

Timing is like for like: both tools are warmed up first (scanpy's pynndescent / numba
code is JIT-compiled on the first call; CREST gets the same untimed call), both run with
the same thread count (scanpy ``sc.settings.n_jobs``; CREST's rayon pool), and the headline
times ``embedding_method=("pca",)`` for both: the label transfer itself, no UMAP
projection. A separate row times both with ``("pca", "umap")`` (scanpy then also runs its
UMAP transform, CREST its neighbour-weighted UMAP average; not the same algorithm). Times
are the median of ``--repeats`` calls. The accuracy difference (CREST - scanpy) gets a
paired, seeded bootstrap 95% interval over query cells.

    python bench/ingest/compare_scanpy_ingest.py [--out bench/ingest/results] [--repeats 3]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deseq2"))
import crest  # noqa: E402
from kang_pseudobulk import fetch, load  # noqa: E402


def _timed(make, fn, repeats: int):
    """Median time of fn(make()) over repeats; the input copy make() is not timed."""
    ts = []
    for _ in range(repeats):
        x = make()
        t0 = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), ts


def paired_bootstrap(ok_a: np.ndarray, ok_b: np.ndarray, n: int = 2000, seed: int = 0):
    """95% percentile interval of mean(ok_a) - mean(ok_b), resampling query cells (paired)."""
    rng = np.random.default_rng(seed)
    d = ok_a.astype(float) - ok_b.astype(float)
    idx = rng.integers(0, len(d), (n, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/kang")
    ap.add_argument("--out", default="bench/ingest/results")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("RAYON_NUM_THREADS") or os.cpu_count() or 1)
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
    crest.tl.ingest(qry.copy(), ref, obs="cell", embedding_method=("pca",))  # warm-up, untimed
    t_crest, ts_crest = _timed(qry.copy, lambda q: crest.tl.ingest(q, ref, obs="cell", embedding_method=("pca",)),
                               a.repeats)
    t_crest_u, ts_crest_u = _timed(qry.copy, lambda q: crest.tl.ingest(q, ref, obs="cell",
                                                                       embedding_method=("pca", "umap")), a.repeats)
    crest.tl.ingest(qry, ref, obs="cell", embedding_method=("pca",))
    lab_crest = qry.obs["cell"].to_numpy()
    acc_crest = float((lab_crest == truth).mean())

    # ---- scanpy on the same genes and the same reference preprocessing
    import scanpy as sc
    import anndata as ad
    sc.settings.n_jobs = threads
    hv = ref.var["highly_variable"].to_numpy()
    A = ref._subset(var_mask=hv).to_anndata(transform=True)
    B = qry._subset(var_mask=hv).to_anndata(transform=True)
    A.obs["cell"] = ref.obs["cell"].to_list()
    sc.pp.scale(A, max_value=10)
    # ingest applies the reference PCA to the query as given: scale it with the reference parameters
    mu, sd = np.asarray(A.var["mean"]), np.asarray(A.var["std"])
    Bx = (B.X.toarray() - mu) / sd
    B = ad.AnnData(np.clip(Bx, -10, 10), obs=B.obs.drop(columns=["cell"], errors="ignore"), var=B.var)
    sc.tl.pca(A, n_comps=50)
    sc.pp.neighbors(A, n_neighbors=15)
    sc.tl.umap(A)
    sc.tl.ingest(B[:500].copy(), A, obs="cell", embedding_method="umap")  # warm-up: numba JIT, untimed
    t_sc, ts_sc = _timed(B.copy, lambda q: sc.tl.ingest(q, A, obs="cell", embedding_method="pca"), a.repeats)
    t_sc_u, ts_sc_u = _timed(B.copy, lambda q: sc.tl.ingest(q, A, obs="cell", embedding_method=["umap", "pca"]),
                             a.repeats)
    sc.tl.ingest(B, A, obs="cell", embedding_method="pca")
    lab_sc = B.obs["cell"].astype(str).to_numpy()
    acc_sc = float((lab_sc == truth).mean())
    agree = float((lab_sc == lab_crest).mean())
    diff, lo, hi = paired_bootstrap(lab_crest == truth, lab_sc == truth, n=a.bootstrap)

    rows = [{"tool": "crest", "embedding": "pca", "time_s": t_crest, "time_min_s": min(ts_crest),
             "time_max_s": max(ts_crest), "accuracy": acc_crest, "headline": True},
            {"tool": "scanpy", "embedding": "pca", "time_s": t_sc, "time_min_s": min(ts_sc),
             "time_max_s": max(ts_sc), "accuracy": acc_sc, "headline": True},
            {"tool": "crest", "embedding": "pca+umap", "time_s": t_crest_u, "time_min_s": min(ts_crest_u),
             "time_max_s": max(ts_crest_u), "accuracy": acc_crest, "headline": False},
            {"tool": "scanpy", "embedding": "pca+umap", "time_s": t_sc_u, "time_min_s": min(ts_sc_u),
             "time_max_s": max(ts_sc_u), "accuracy": acc_sc, "headline": False}]
    print(rows, "label agreement crest vs scanpy:", agree, "accuracy diff", diff, (lo, hi))
    df = pl.DataFrame(rows)
    df.write_csv(out / "kang_ingest.csv")
    per = pl.DataFrame({"truth": truth, "crest": lab_crest, "scanpy": lab_sc})
    by = per.group_by("truth").agg(pl.len().alias("n"), (pl.col("crest") == pl.col("truth")).mean().alias("crest_acc"),
                                   (pl.col("scanpy") == pl.col("truth")).mean().alias("scanpy_acc")).sort("n", descending=True)
    with open(out / "kang_ingest.md", "w") as fh:
        fh.write(f"Kang 2018: reference = {ref.n_obs} control cells, query = {qry.n_obs} stimulated cells; "
                 f"label = published cell type. Both tools warmed up first, {threads} threads, median of "
                 f"{a.repeats} timed calls [min-max]. Headline: `embedding_method=(\"pca\",)` for both (label "
                 f"transfer only).\n\n| tool | embedding | time_s | accuracy |\n|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['tool']} | {r['embedding']}{' (headline)' if r['headline'] else ''} | {r['time_s']:.3f} "
                     f"[{r['time_min_s']:.3f}-{r['time_max_s']:.3f}] | {r['accuracy']:.4f} |\n")
        fh.write(f"\nSpeed-up (headline, pca only): {t_sc / t_crest:.2f}x; with UMAP projection: {t_sc_u / t_crest_u:.2f}x "
                 f"(not the same UMAP algorithm, see the script docstring).\n")
        fh.write(f"\nAccuracy difference CREST - scanpy: {diff:+.4f} (95% paired bootstrap interval over query cells, "
                 f"{a.bootstrap} resamples, seed 0: {lo:+.4f} to {hi:+.4f}).\n")
        fh.write(f"\nLabel agreement CREST vs scanpy: {agree:.4f}\n\nPer cell type:\n\n| truth | n | crest | scanpy |\n|---|---|---|---|\n")
        for t, n, c, s in by.iter_rows():
            fh.write(f"| {t} | {n} | {c:.3f} | {s:.3f} |\n")


if __name__ == "__main__":
    main()
