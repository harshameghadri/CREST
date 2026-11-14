"""One benchmark run: one tool on one dataset, in a fresh process.

    python bench/paper/run_one.py --tool crest --data bench_data/pbmc68k.h5 --out results/raw \
        --profile full --threads 8 --repeat 0

Pipeline (scanpy tutorial defaults), each step timed and monitored:

  core : read, qc_filter, normalize_log1p, hvg (seurat, 2000), scale_pca (50 PCs),
         neighbors (15), leiden, umap, de_ttest, de_wilcoxon
  full : core + hvg_seurat_v3 and scrublet (on raw counts; skipped above
         --scrublet-max-cells), leiden_sweep
         (5 resolutions), harmony (if the dataset has a batch column),
         pseudobulk_deseq2 (CREST only; if it has sample + condition columns)

JIT-compiled code (numba in scanpy / umap-learn / pynndescent) is warmed up on
a 20,000-cell subset first so compile time is not charged to scanpy; CREST
gets the same warm-up (it only warms the file cache). The warm-up's objects are
then released (``LAST.clear(); gc.collect()``, then ``malloc_trim`` on glibc /
``malloc_zone_pressure_relief`` on macOS so the allocator returns the freed pages) and
the process RSS is recorded as the *baseline* when monitoring starts, so memory
is reported both as the absolute peak and as peak minus baseline. Whether the
warm-up succeeded (and its error, if not) is recorded in the result JSON.

Both tools get the same thread count: ``RAYON_NUM_THREADS`` etc. are set by
run_all.py, and scanpy's own ``sc.settings.n_jobs`` (used by pynndescent's kNN)
is set to ``--threads`` here.

Writes ``<tag>.json`` (steps, resources, environment), ``<tag>.timeline.csv`` and
``<tag>.outputs.npz`` (embeddings/labels/gene lists, each tool's actual kNN
indices and, for scanpy, CREST's Leiden on scanpy's own graph, for the accuracy
analysis).
"""

from __future__ import annotations

import argparse
import gc
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

# --------------------------------------------------------------------------- shared parameters
# Passed identically to BOTH tools. They are kept at the values of the published runs for
# now. A separate PR switches CREST's defaults to scanpy's (normalize_total target_sum=None,
# i.e. the median library size, and scale max_value=None); when it lands, set both to None
# here (CREST currently raises on target_sum=None, so they cannot be switched before that).
NORMALIZE_TARGET_SUM = 1e4
SCALE_MAX_VALUE = 10
N_HVG = 2000
N_PCS = 50
N_NEIGHBORS = 15


class Null:
    def track(self, name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


SCRUBLET_MAX = 0  # set from --scrublet-max-cells; 0 = no limit
LAST = {}  # the final AnnData / BioFrame, for the untimed seed-baseline clusterings
SEED_BASELINE = (1, 2)  # extra Leiden seeds, run after timing, saved with --save-outputs
SKIPPED: list = []
THREADS = 0  # set from --threads (0 = all cores); scanpy's sc.settings.n_jobs


def _meta(name: str, path: Path | None = None) -> dict:
    """Registry metadata, overridden by a ``<data>.meta.json`` sidecar written at conversion
    (datasets whose obs columns are only known after reading the file, e.g. parse_pbmc)."""
    meta = dict(REGISTRY.get(name, {}))
    if path is not None and path.with_suffix(".meta.json").exists():
        meta.update(json.loads(path.with_suffix(".meta.json").read_text()))
    return meta


def _run_scrublet(n: int) -> bool:
    if SCRUBLET_MAX and n > SCRUBLET_MAX:
        if "scrublet" not in SKIPPED:
            SKIPPED.append("scrublet")
        return False
    return True


def _obs_sidecar(path: Path):
    side = path.with_suffix(".obs.parquet")
    if side.exists():
        import polars as pl
        return pl.read_parquet(side)
    return None


# --------------------------------------------------------------------------- scanpy
def scanpy_pipeline(path: Path, mon, profile: str, warm: bool = False) -> dict:
    import scanpy as sc
    sc.settings.n_jobs = THREADS  # pynndescent's kNN gets the same thread count as CREST's rayon pool
    meta, out = _meta(path.stem, path), {}
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
        sc.pp.filter_cells(a, min_genes=meta.get("min_genes", 200))
        sc.pp.filter_genes(a, min_cells=3)
    batch = meta.get("batch") if meta.get("batch") in a.obs else None
    if profile == "full":
        with mon.track("hvg_seurat_v3"):
            v3 = sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=N_HVG, inplace=False)
        out["hvg_v3"] = np.asarray(a.var_names[v3["highly_variable"].to_numpy()], dtype=str)
        if _run_scrublet(a.n_obs):
            with mon.track("scrublet"):
                sc.pp.scrublet(a, batch_key=batch, random_state=0, verbose=False)
            out["doublet_score"] = a.obs["doublet_score"].to_numpy(float)
    with mon.track("normalize_log1p"):
        sc.pp.normalize_total(a, target_sum=NORMALIZE_TARGET_SUM)
        sc.pp.log1p(a)
        a.raw = a
    with mon.track("hvg"):
        sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat")
    out["hvg"] = np.asarray(a.var_names[a.var["highly_variable"].to_numpy()], dtype=str)
    with mon.track("scale_pca"):
        a = a[:, a.var.highly_variable].copy()
        sc.pp.scale(a, max_value=SCALE_MAX_VALUE)
        sc.tl.pca(a, n_comps=N_PCS, svd_solver="arpack")
    with mon.track("neighbors"):
        sc.pp.neighbors(a, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=0)
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
    LAST.update(tool="scanpy", obj=a)
    return out


# --------------------------------------------------------------------------- CREST
def crest_pipeline(path: Path, mon, profile: str, backed=None, warm: bool = False) -> dict:
    import crest
    import polars as pl
    meta, out = _meta(path.stem, path), {}
    side = _obs_sidecar(path)
    with mon.track("read"):
        b = crest.read_10x_h5(path, backed=backed)
        if side is not None:
            b.obs = b.obs.with_columns([side[c] for c in side.columns if c != "barcode"])
    if warm:
        b = b.filter_cells(np.arange(b.n_obs) < 20000)
    with mon.track("qc_filter"):
        b = crest.pp.filter_cells(b, min_genes=meta.get("min_genes", 200))
        b = crest.pp.filter_genes(b, min_cells=3)
    batch = meta.get("batch") if meta.get("batch") in b.obs.columns else None
    if profile == "full":
        with mon.track("hvg_seurat_v3"):
            v3 = b.copy()
            crest.pp.highly_variable_genes(v3, flavor="seurat_v3", n_top_genes=N_HVG)
        out["hvg_v3"] = np.asarray(v3.var_names, dtype=str)[v3.var["highly_variable"].to_numpy()]
        del v3
        if _run_scrublet(b.n_obs):
            with mon.track("scrublet"):
                crest.pp.scrublet(b, batch_key=batch, random_state=0)
            out["doublet_score"] = b.obs["doublet_score"].fill_null(np.nan).to_numpy()
    with mon.track("normalize_log1p"):
        crest.pp.normalize_total(b, NORMALIZE_TARGET_SUM)
        crest.pp.log1p(b)
    with mon.track("hvg"):
        crest.pp.highly_variable_genes(b, n_top_genes=N_HVG, flavor="seurat")
    out["hvg"] = np.asarray(b.var_names, dtype=str)[b.var["highly_variable"].to_numpy()]
    with mon.track("scale_pca"):
        crest.pp.scale(b, max_value=SCALE_MAX_VALUE)
        crest.tl.pca(b, n_comps=N_PCS)
    with mon.track("neighbors"):
        crest.pp.neighbors(b, n_neighbors=N_NEIGHBORS)
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
                pb_bf = b
                if meta.get("pb_levels_only"):  # many conditions: test only the contrast's two levels
                    pb_bf = b.filter_cells(pl.col(cond).cast(pl.Utf8).is_in([str(con[1]), str(con[2])]))
                crest.tl.pseudobulk_de(pb_bf, samp, design=design, contrast=tuple(con),
                                       groupby=meta["celltype"], min_cells=10, quiet=True)
    out.update({"pca": b.obsm["X_pca"], "umap": b.obsm["X_umap"], "leiden": b.obs["leiden"].cast(int).to_numpy(),
                "barcodes": b.obs["barcode"].to_numpy().astype(str)})
    LAST.update(tool="crest", obj=b)
    return out


def seed_baseline() -> np.ndarray | None:
    """Leiden (resolution 1) of the final object with seeds SEED_BASELINE on the same graph,
    untimed: how much each tool disagrees with itself, the yardstick for CREST-vs-scanpy ARI."""
    obj, runs = LAST.get("obj"), []
    if obj is None:
        return None
    for s in SEED_BASELINE:
        if LAST["tool"] == "scanpy":
            import scanpy as sc
            sc.tl.leiden(obj, flavor="igraph", n_iterations=2, random_state=s, key_added=f"_seed{s}")
            runs.append(obj.obs[f"_seed{s}"].astype(int).to_numpy())
        else:
            import crest
            crest.tl.leiden(obj, random_state=s, key_added=f"_seed{s}")
            runs.append(obj.obs[f"_seed{s}"].cast(int).to_numpy())
    return np.stack(runs)


def knn_indices() -> np.ndarray | None:
    """The kNN graph each tool actually built in the timed run, as (n, k-1) neighbour
    indices (self excluded; -1 pads rows with fewer neighbours). scanpy: the non-zeros of
    ``obsp['distances']`` sorted by distance; CREST: ``uns['neighbors']['indices']``."""
    obj = LAST.get("obj")
    if obj is None:
        return None
    if LAST["tool"] == "scanpy":
        D = obj.obsp["distances"].tocsr()
        lens = np.diff(D.indptr)
        k = int(lens.max()) if D.shape[0] else 0
        if D.shape[0] and (lens == k).all():  # the usual case: every row has k - 1 neighbours
            dd, ii = D.data.reshape(-1, k), D.indices.reshape(-1, k)
            return np.take_along_axis(ii, np.argsort(dd, axis=1, kind="stable"), axis=1).astype(np.int32)
        out = np.full((D.shape[0], k), -1, np.int32)
        for i in range(D.shape[0]):
            lo, hi = D.indptr[i], D.indptr[i + 1]
            out[i, : hi - lo] = D.indices[lo:hi][np.argsort(D.data[lo:hi], kind="stable")]
        return out
    return np.asarray(obj.uns["neighbors"]["indices"]).astype(np.int32)


def same_graph_leiden() -> dict:
    """scanpy process only, untimed: CREST's Leiden on scanpy's own connectivities (resolution
    1, 2 iterations, seed 0), next to scanpy's Leiden (flavor="igraph", n_iterations=2) of the
    timed run on the same graph. Both partitions are scored by igraph's Graph.modularity, an
    implementation independent of both tools."""
    obj = LAST.get("obj")
    if obj is None or LAST["tool"] != "scanpy":
        return {}
    import igraph as ig
    import scipy.sparse as sp
    from crest import crest as native
    C = obj.obsp["connectivities"]
    U = sp.triu(C, 1).tocoo()
    lab_c = np.asarray(native.leiden(C.shape[0], U.row.astype(np.uint32), U.col.astype(np.uint32),
                                     U.data.astype(np.float32), 1.0, 2, 0)).astype(np.int64)
    lab_s = obj.obs["leiden"].astype(int).to_numpy()
    g = ig.Graph.Weighted_Adjacency(sp.triu(C, 1).tocsr(), mode="undirected")
    w = g.es["weight"]
    return {"same_graph_leiden_crest": lab_c, "same_graph_leiden_scanpy": lab_s,
            "same_graph_modularity_crest": np.float64(g.modularity(lab_c.tolist(), weights=w, resolution=1.0)),
            "same_graph_modularity_scanpy": np.float64(g.modularity(lab_s.tolist(), weights=w, resolution=1.0))}


def _release_memory() -> str | None:
    """Return freed heap pages to the OS, so the warm-up's freed memory is not counted in the
    baseline (the allocator otherwise keeps it mapped: measured ~0.25 GB on PBMC 3k on macOS).
    glibc: malloc_trim(0); macOS: malloc_zone_pressure_relief. Returns the method used."""
    gc.collect()
    import ctypes
    try:
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            return "malloc_trim"
        if sys.platform == "darwin":
            ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief(None, 0)
            return "malloc_zone_pressure_relief"
    except (OSError, AttributeError):
        pass
    return None


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
        # recorded after the run, so it lists every BLAS / OpenMP pool the tool loaded
        env["threadpools"] = [{k: d.get(k) for k in ("user_api", "internal_api", "prefix", "version", "num_threads",
                                                     "threading_layer")} for d in threadpool_info()]
    except Exception as e:  # noqa: BLE001
        env["threadpools"] = f"unavailable: {e}"
    if "scanpy" in sys.modules:
        env["scanpy_n_jobs"] = sys.modules["scanpy"].settings.n_jobs
    if "numba" in sys.modules:
        env["numba_num_threads"] = sys.modules["numba"].config.NUMBA_NUM_THREADS
    from importlib import metadata
    env["versions"] = {}
    for dist in ("crest-sc", "numpy", "scipy", "scanpy", "anndata", "scikit-learn", "umap-learn", "pynndescent",
                 "numba", "igraph", "leidenalg", "harmonypy", "pydeseq2", "polars", "pandas", "h5py",
                 "threadpoolctl"):
        try:
            env["versions"][dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            env["versions"][dist] = None
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
    ap.add_argument("--scrublet-max-cells", type=int, default=0, help="skip Scrublet above this many cells")
    a = ap.parse_args()
    global SCRUBLET_MAX, THREADS
    SCRUBLET_MAX = a.scrublet_max_cells
    THREADS = a.threads or (os.cpu_count() or 1)
    path, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.pin and a.threads and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[: a.threads]))
    tag = f"{a.tool}__{path.stem}__{a.profile}__t{a.threads}__r{a.repeat}"
    backed = str(out / f"_ooc_{tag}") if a.tool == "crest-ooc" else None
    run = (lambda m, warm=False: scanpy_pipeline(path, m, a.profile, warm)) if a.tool == "scanpy" else \
          (lambda m, warm=False: crest_pipeline(path, m, a.profile, backed, warm))

    warmup = {"status": "skipped", "error": None, "seconds": None}
    if not a.no_warmup:
        t0 = time.perf_counter()
        try:
            run(Null(), warm=True)
            warmup["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            warmup.update(status="failed", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        warmup["seconds"] = time.perf_counter() - t0
    # drop the warm-up's objects before measuring: otherwise its AnnData / BioFrame stays
    # alive during the timed run and inflates that tool's memory
    LAST.clear()
    SKIPPED.clear()
    warmup["memory_release"] = _release_memory()

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
        # all of this runs after timing and monitoring: never counted in the run
        extra = seed_baseline()
        if extra is not None:
            res["leiden_seeds"] = extra
        knn = knn_indices()
        if knn is not None:
            res["knn_indices"] = knn
        try:
            res.update(same_graph_leiden())
        except Exception as e:  # noqa: BLE001
            print(f"same-graph Leiden failed: {type(e).__name__}: {e}", file=sys.stderr)
        np.savez_compressed(out / f"{tag}.outputs.npz", **res)
    peak = max((s["peak_rss_gb"] for s in mon.steps.values()), default=None)
    report = {"tool": a.tool, "dataset": path.stem, "profile": a.profile, "threads": a.threads, "pinned": a.pin,
              "repeat": a.repeat, "status": status, "error": err, "n_cells": int(len(res["leiden"])) if res else None,
              "total_seconds": total, "warmup": warmup,
              "baseline_rss_gb": mon.baseline_rss / 1e9, "peak_rss_gb": peak,
              "peak_rss_minus_baseline_gb": None if peak is None else peak - mon.baseline_rss / 1e9,
              "usage": usage, "steps": mon.steps, "skipped_steps": SKIPPED, "env": environment(),
              "params": {"normalize_target_sum": NORMALIZE_TARGET_SUM, "scale_max_value": SCALE_MAX_VALUE,
                         "n_hvg": N_HVG, "n_pcs": N_PCS, "n_neighbors": N_NEIGHBORS}}
    (out / f"{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("tool", "dataset", "threads", "repeat", "status", "total_seconds", "peak_rss_gb")}))
    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
