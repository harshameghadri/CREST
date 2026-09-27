"""Time and memory of the standard single-cell pipeline, one tool per process.

    python bench/whitepaper/run_pipeline.py --tool scanpy --data data.h5 --out results/
    python bench/whitepaper/run_pipeline.py --tool crest  --data data.h5 --out results/
    python bench/whitepaper/run_pipeline.py --tool crest-ooc --data data.h5 --out results/

Pipeline (scanpy tutorial): read → QC filter → normalize_total(1e4) + log1p →
HVG (seurat, 2000) → scale(max_value=10) + PCA(50) → neighbors(15) → Leiden →
UMAP → rank_genes_groups (t-test, Wilcoxon).

Per step: wall time and peak RSS (sampled every 10 ms by a background thread).
JIT-compiled code (numba in scanpy/umap/pynndescent) is warmed up on a
20,000-cell subset first (pynndescent compiles extra code paths as n grows), so compile time is not charged to scanpy.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import psutil

warnings.filterwarnings("ignore")


class Sampler:
    """Background RSS sampler; `peak()` returns the max RSS since the last reset."""

    def __init__(self, interval=0.01):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self._peak = 0
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop:
            self._peak = max(self._peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def reset(self):
        gc.collect()
        self._peak = self.proc.memory_info().rss

    def peak(self):
        self._peak = max(self._peak, self.proc.memory_info().rss)
        return self._peak

    def stop(self):
        self._stop = True


def scanpy_pipeline(path, timer, warm=False):
    import scanpy as sc

    with timer("read"):
        a = sc.read_10x_h5(path)
        a.var_names_make_unique()
    if warm:
        a = a[:20000].copy()
    with timer("qc_filter"):
        sc.pp.filter_cells(a, min_genes=200)
        sc.pp.filter_genes(a, min_cells=3)
    with timer("normalize_log1p"):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.raw = a
    with timer("hvg"):
        sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    with timer("scale_pca"):
        a = a[:, a.var.highly_variable].copy()
        sc.pp.scale(a, max_value=10)
        sc.tl.pca(a, n_comps=50, svd_solver="arpack")
    with timer("neighbors"):
        sc.pp.neighbors(a, n_neighbors=15, n_pcs=50, random_state=0)
    with timer("leiden"):
        sc.tl.leiden(a, flavor="igraph", n_iterations=2, random_state=0)
    with timer("umap"):
        sc.tl.umap(a, random_state=0)
    with timer("de_ttest"):
        sc.tl.rank_genes_groups(a, "leiden", method="t-test", use_raw=True)
    with timer("de_wilcoxon"):
        sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon", use_raw=True)
    return {"pca": a.obsm["X_pca"], "umap": a.obsm["X_umap"], "leiden": a.obs["leiden"].astype(int).values,
            "barcodes": a.obs_names.values.astype(str)}


def crest_pipeline(path, timer, backed=None, warm=False):
    import crest

    with timer("read"):
        b = crest.read_10x_h5(path, backed=backed)
    if warm:
        b = b.filter_cells(np.arange(b.n_obs) < 20000)
    with timer("qc_filter"):
        b = crest.pp.filter_cells(b, min_genes=200)
        b = crest.pp.filter_genes(b, min_cells=3)
    with timer("normalize_log1p"):
        crest.pp.normalize_total(b, 1e4)
        crest.pp.log1p(b)
    with timer("hvg"):
        crest.pp.highly_variable_genes(b, n_top_genes=2000, flavor="seurat")
    with timer("scale_pca"):
        crest.pp.scale(b, max_value=10)
        crest.tl.pca(b, n_comps=50)
    with timer("neighbors"):
        crest.pp.neighbors(b, n_neighbors=15)
    with timer("leiden"):
        crest.tl.leiden(b)
    with timer("umap"):
        crest.tl.umap(b)
    with timer("de_ttest"):
        crest.tl.rank_genes_groups(b, "leiden", method="t-test")
    with timer("de_wilcoxon"):
        crest.tl.rank_genes_groups(b, "leiden", method="wilcoxon")
    return {"pca": b.obsm["X_pca"], "umap": b.obsm["X_umap"],
            "leiden": b.obs["leiden"].cast(int).to_numpy(), "barcodes": b.obs["barcode"].to_numpy().astype(str)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["scanpy", "crest", "crest-ooc"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sampler = Sampler()
    steps = {}

    class timer:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            sampler.reset()
            self.t = time.perf_counter()

        def __exit__(self, *exc):
            if exc[0] is None:
                steps[self.name] = {"seconds": time.perf_counter() - self.t, "peak_rss_gb": sampler.peak() / 1e9}

    backed = str(out / "crest_parquet") if args.tool == "crest-ooc" else None
    run = (lambda t, warm=False: scanpy_pipeline(args.data, t, warm)) if args.tool == "scanpy" else \
          (lambda t, warm=False: crest_pipeline(args.data, t, backed, warm))

    if not args.no_warmup:
        class null:
            def __init__(self, name): pass
            def __enter__(self): pass
            def __exit__(self, *exc): pass
        run(null, warm=True)
        gc.collect()

    baseline = psutil.Process(os.getpid()).memory_info().rss
    t0 = time.perf_counter()
    res = run(timer)
    total = time.perf_counter() - t0
    sampler.stop()

    tag = f"{args.tool}_{Path(args.data).stem}"
    np.savez_compressed(out / f"{tag}_outputs.npz", **res)
    report = {
        "tool": args.tool, "data": Path(args.data).name, "n_cells": int(len(res["leiden"])),
        "total_seconds": total, "baseline_rss_gb": baseline / 1e9,
        "peak_rss_gb": max(s["peak_rss_gb"] for s in steps.values()), "steps": steps,
        "machine": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                    "ram_gb": psutil.virtual_memory().total / 1e9, "python": platform.python_version()},
    }
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}))
    for k, v in steps.items():
        print(f"  {k:16s} {v['seconds']:8.2f}s  peak {v['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
