"""One benchmark run: one tool on one dataset, in a fresh process.

    python bench/paper/run_one.py --tool crest --data bench_data/pbmc68k.h5 --out results/raw \
        --profile full --threads 8 --repeat 0

Pipeline (scanpy tutorial defaults), each step timed and monitored:

  core : read, qc_filter, normalize_log1p, hvg (seurat, 2000), scale_pca (50 PCs),
         neighbors (15), leiden, umap, de_ttest, de_wilcoxon
  full : core + hvg_seurat_v3 and scrublet (on raw counts), leiden_sweep
         (5 resolutions), harmony (if the dataset has a batch column),
         pseudobulk_deseq2 (CREST only; if it has sample + condition columns)

JIT-compiled code (numba in scanpy / umap-learn / pynndescent) is warmed up on
a 20,000-cell subset first so compile time is not charged to scanpy; CREST
gets the same warm-up (it only warms the file cache). Writes
``<tag>.json`` (steps, resources, environment), ``<tag>.timeline.csv`` and
``<tag>.outputs.npz`` (embeddings/labels/gene lists for the accuracy analysis).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import Monitor  # noqa: E402
from datasets import REGISTRY  # noqa: E402

RESOLUTIONS = [0.2, 0.5, 1.0, 1.5, 2.0]


class Null:
    def track(self, name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _meta(name: str) -> dict:
    return REGISTRY.get(name, {})


def _obs_sidecar(path: Path):
    side = path.with_suffix(".obs.parquet")
    if side.exists():
        import polars as pl
        return pl.read_parquet(side)
    return None


# --------------------------------------------------------------------------- scanpy
def scanpy_pipeline(path: Path, mon, profile: str, warm: bool = False) -> dict:
    import scanpy as sc
    meta, out = _meta(path.stem), {}
    side = _obs_sidecar(path)
    with mon.track("read"):
        a = sc.read_10x_h5(path)
        a.var_names_make_unique()
        if side is not None:
            for c in side.columns:
                if c != "barcode":
                    a.obs[c] = side[c].to_numpy()
    if warm:
        a = a[: min(20000, a.n_obs)].copy()
    with mon.track("qc_filter"):
        sc.pp.filter_cells(a, min_genes=200)
        sc.pp.filter_genes(a, min_cells=3)
    batch = meta.get("batch") if meta.get("batch") in a.obs else None
    if profile == "full":
        with mon.track("hvg_seurat_v3"):
            v3 = sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=2000, inplace=False)
        out["hvg_v3"] = np.asarray(a.var_names[v3["highly_variable"].to_numpy()], dtype=str)
        with mon.track("scrublet"):
            sc.pp.scrublet(a, batch_key=batch, random_state=0, verbose=False)
        out["doublet_score"] = a.obs["doublet_score"].to_numpy(float)
    with mon.track("normalize_log1p"):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.raw = a
    with mon.track("hvg"):
        sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    out["hvg"] = np.asarray(a.var_names[a.var["highly_variable"].to_numpy()], dtype=str)
    with mon.track("scale_pca"):
        a = a[:, a.var.highly_variable].copy()
        sc.pp.scale(a, max_value=10)
        sc.tl.pca(a, n_comps=50, svd_solver="arpack")
    with mon.track("neighbors"):
        sc.pp.neighbors(a, n_neighbors=15, n_pcs=50, random_state=0)
    with mon.track("leiden"):
        sc.tl.leiden(a, flavor="igraph", n_iterations=2, random_state=0)
    with mon.track("umap"):
        sc.tl.umap(a, random_state=0)
    with mon.track("de_ttest"):
        sc.tl.rank_genes_groups(a, "leiden", method="t-test", use_raw=True)
    with mon.track("de_wilcoxon"):
        sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon", use_raw=True)
    names = a.uns["rank_genes_groups"]["names"]
    out["de_groups"] = np.array(list(names.dtype.names), dtype=str)
    out["de_top"] = np.array([[str(x) for x in names[g][:50]] for g in names.dtype.names], dtype=str)
    if profile == "full":
        with mon.track("leiden_sweep"):
            for r in RESOLUTIONS:
                sc.tl.leiden(a, resolution=r, flavor="igraph", n_iterations=2, random_state=0, key_added=f"leiden_{r:g}")
        out["sweep"] = np.stack([a.obs[f"leiden_{r:g}"].astype(int).to_numpy() for r in RESOLUTIONS])
        if batch:
            with mon.track("harmony"):
                # what sc.external.pp.harmony_integrate wraps; called directly because
                # scanpy <= 1.11 transposes harmonypy 2.x's (cells x PCs) output
                import harmonypy
                a.obsm["X_pca_harmony"] = np.asarray(
                    harmonypy.run_harmony(a.obsm["X_pca"], a.obs, batch, verbose=False).Z_corr)
            out["harmony"] = a.obsm["X_pca_harmony"].astype(np.float32)
    out.update({"pca": a.obsm["X_pca"], "umap": a.obsm["X_umap"], "leiden": a.obs["leiden"].astype(int).to_numpy(),
                "barcodes": np.asarray(a.obs_names, dtype=str)})
    return out


# --------------------------------------------------------------------------- CREST
def crest_pipeline(path: Path, mon, profile: str, backed=None, warm: bool = False) -> dict:
    import crest
    import polars as pl
    meta, out = _meta(path.stem), {}
    side = _obs_sidecar(path)
    with mon.track("read"):
        b = crest.read_10x_h5(path, backed=backed)
        if side is not None:
            b.obs = b.obs.with_columns([side[c] for c in side.columns if c != "barcode"])
    if warm:
        b = b.filter_cells(np.arange(b.n_obs) < 20000)
    with mon.track("qc_filter"):
        b = crest.pp.filter_cells(b, min_genes=200)
        b = crest.pp.filter_genes(b, min_cells=3)
    batch = meta.get("batch") if meta.get("batch") in b.obs.columns else None
    if profile == "full":
        with mon.track("hvg_seurat_v3"):
            v3 = b.copy()
            crest.pp.highly_variable_genes(v3, flavor="seurat_v3", n_top_genes=2000)
        out["hvg_v3"] = np.asarray(v3.var_names, dtype=str)[v3.var["highly_variable"].to_numpy()]
        del v3
        with mon.track("scrublet"):
            crest.pp.scrublet(b, batch_key=batch, random_state=0)
        out["doublet_score"] = b.obs["doublet_score"].fill_null(np.nan).to_numpy()
    with mon.track("normalize_log1p"):
        crest.pp.normalize_total(b, 1e4)
        crest.pp.log1p(b)
    with mon.track("hvg"):
        crest.pp.highly_variable_genes(b, n_top_genes=2000, flavor="seurat")
    out["hvg"] = np.asarray(b.var_names, dtype=str)[b.var["highly_variable"].to_numpy()]
    with mon.track("scale_pca"):
        crest.pp.scale(b, max_value=10)
        crest.tl.pca(b, n_comps=50)
    with mon.track("neighbors"):
        crest.pp.neighbors(b, n_neighbors=15)
    with mon.track("leiden"):
        crest.tl.leiden(b)
    with mon.track("umap"):
        crest.tl.umap(b)
    with mon.track("de_ttest"):
        crest.tl.rank_genes_groups(b, "leiden", method="t-test")
    with mon.track("de_wilcoxon"):
        de = crest.tl.rank_genes_groups(b, "leiden", method="wilcoxon", n_genes=50)
    groups = de["group"].unique(maintain_order=True).to_list()
    out["de_groups"] = np.array(groups, dtype=str)
    out["de_top"] = np.array([de.filter(pl.col("group") == g)["names"].to_list() for g in groups], dtype=str)
    if profile == "full":
        with mon.track("leiden_sweep"):
            crest.tl.leiden_sweep(b, RESOLUTIONS)
        out["sweep"] = np.stack([b.obs[f"leiden_{r:g}"].cast(int).to_numpy() for r in RESOLUTIONS])
        if batch:
            with mon.track("harmony"):
                crest.tl.harmony(b, batch)
            out["harmony"] = b.obsm["X_pca_harmony"]
        samp, cond, con = meta.get("sample"), meta.get("condition"), meta.get("contrast")
        if samp and con and all(c in b.obs.columns for c in samp + [cond]) and meta.get("celltype") in b.obs.columns:
            with mon.track("pseudobulk_deseq2"):
                design = "~ " + " + ".join(dict.fromkeys([s for s in samp if s != cond] + [cond]))
                crest.tl.pseudobulk_de(b, samp, design=design, contrast=tuple(con),
                                       groupby=meta["celltype"], min_cells=10, quiet=True)
    out.update({"pca": b.obsm["X_pca"], "umap": b.obsm["X_umap"], "leiden": b.obs["leiden"].cast(int).to_numpy(),
                "barcodes": b.obs["barcode"].to_numpy().astype(str)})
    return out


def environment() -> dict:
    import psutil
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor(), "logical_cores": psutil.cpu_count(logical=True),
           "physical_cores": psutil.cpu_count(logical=False), "ram_gb": psutil.virtual_memory().total / 1e9,
           "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "thread_env": {k: os.environ.get(k) for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}}
    try:
        from threadpoolctl import threadpool_info
        env["threadpools"] = [{k: d.get(k) for k in ("internal_api", "version", "num_threads", "threading_layer")}
                              for d in threadpool_info()]
    except Exception:  # noqa: BLE001
        pass
    for mod in ("numpy", "scipy", "scanpy", "anndata", "sklearn", "umap", "pynndescent", "igraph", "leidenalg",
                "harmonypy", "crest", "polars"):
        try:
            m = __import__(mod)
            env.setdefault("versions", {})[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["scanpy", "crest", "crest-ooc"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", choices=["core", "full"], default="core")
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores (recorded only; set via env)")
    ap.add_argument("--pin", action="store_true", help="pin the process to the first --threads cores (Linux)")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--save-outputs", action="store_true")
    a = ap.parse_args()
    path, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.pin and a.threads and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[: a.threads]))
    tag = f"{a.tool}__{path.stem}__{a.profile}__t{a.threads}__r{a.repeat}"
    backed = str(out / f"_ooc_{tag}") if a.tool == "crest-ooc" else None
    run = (lambda m, warm=False: scanpy_pipeline(path, m, a.profile, warm)) if a.tool == "scanpy" else \
          (lambda m, warm=False: crest_pipeline(path, m, a.profile, backed, warm))

    if not a.no_warmup:
        try:
            run(Null(), warm=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    mon = Monitor()
    status, err, res = "ok", None, {}
    t0 = time.perf_counter()
    try:
        res = run(mon)
    except Exception as e:  # noqa: BLE001
        status, err = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    total = time.perf_counter() - t0
    usage = mon.stop()
    if backed:
        import shutil
        shutil.rmtree(backed, ignore_errors=True)
    mon.write_timeline(out / f"{tag}.timeline.csv")
    if res and a.save_outputs:
        np.savez_compressed(out / f"{tag}.outputs.npz", **res)
    report = {"tool": a.tool, "dataset": path.stem, "profile": a.profile, "threads": a.threads, "pinned": a.pin,
              "repeat": a.repeat, "status": status, "error": err, "n_cells": int(len(res["leiden"])) if res else None,
              "total_seconds": total, "peak_rss_gb": max((s["peak_rss_gb"] for s in mon.steps.values()), default=None),
              "usage": usage, "steps": mon.steps, "env": environment()}
    (out / f"{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("tool", "dataset", "threads", "repeat", "status", "total_seconds", "peak_rss_gb")}))
    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
