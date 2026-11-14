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
is set to ``--n-jobs`` (default ``--threads``) here. ``--variant`` and ``--control`` only label
the run (the settings themselves come from the environment); a control run's files carry a suffix.

Endpoints. Everything the tool needs for the workflow is imported first, in a timed block
(``import_seconds``); ``startup_seconds`` is the time from the start of the process (the parent's spawn timestamp when run_all.py
launched it, else the operating system's process creation time) to the end of that block;
``total_seconds`` is the timed pipeline (every step plus the garbage collections and result
extraction between them); ``end_to_end_seconds = startup_seconds + total_seconds`` is the primary
endpoint. The sum of the step timers (``core_seconds`` in the report) is secondary, and
``untracked_seconds = total_seconds - sum(step seconds)`` is recorded for every run. The in-process
warm-up (numba JIT, file cache) and the untimed accuracy extras are not part of any endpoint.

Pinning. With ``--pin`` the process is pinned to the first ``--threads`` logical CPUs in
physical-core-first order (``protocol.pin_cpus``); the CPUs, the SMT sibling map and the numba
threading layer actually used are recorded.

Memory. Peak RSS is sampled every ``--sample-interval`` seconds (default 0.05), per step and for the run;
the baseline is the RSS after the imports and the warm-up's release (``rss_after_imports_gb`` is the
RSS right after the imports, before the warm-up). A peak shorter than the interval can be missed; the
lifetime ``ru_maxrss`` is recorded as well but includes the warm-up.

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
from monitor import Monitor, foreign_busy_stats  # noqa: E402
import protocol  # noqa: E402

# Modules each tool loads for the workflow. They are imported in the timed start-up block, so the
# end-to-end time includes them (the warm-up would otherwise import the lazily loaded ones for free).
TOOL_IMPORTS = {
    "scanpy": ["scanpy", "anndata", "numba", "pynndescent", "umap", "igraph"],
    "crest": ["crest", "polars"],
}
FULL_PROFILE_IMPORTS = {"scanpy": ["harmonypy", "skmisc"], "crest": []}

RESOLUTIONS = [0.2, 0.5, 1.0, 1.5, 2.0]

# --------------------------------------------------------------------------- shared parameters
# Passed identically to BOTH tools, and explicitly: both tools default to the median library
# size (target_sum=None) and no clipping (max_value=None), but the published runs used 1e4 and 10,
# so these values stay as they are and the timings are unaffected.
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
N_JOBS = 0  # set from --n-jobs (0 = THREADS)


def _meta(name: str, path: Path | None = None) -> dict:
    """Registry metadata, overridden by a ``<data>.meta.json`` sidecar written at conversion
    (datasets whose obs columns are only known after reading the file, e.g. parse_pbmc)."""
    from datasets import REGISTRY
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
def scanpy_pipeline(path: Path, mon, profile: str, warm: bool = False, lean: bool = False) -> dict:
    """The scanpy workflow. ``lean`` is the memory-lean profile: the only difference is that the
    highly-variable-gene subset is taken in place instead of through a view and a copy (no
    computed value changes), so the full-gene matrix is not duplicated while the subset is made."""
    import scanpy as sc
    sc.settings.n_jobs = N_JOBS or THREADS  # pynndescent's kNN gets the same thread count as CREST's rayon pool
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
        if lean and hasattr(a, "_inplace_subset_var"):
            a._inplace_subset_var(a.var["highly_variable"].to_numpy())
        else:
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
    """The kNN graph each tool actually built in the timed run, as (n, n_neighbors - 1)
    neighbour indices (self excluded; -1 pads rows with fewer neighbours). scanpy: the
    non-zeros of ``obsp['distances']`` without the cell itself, sorted by distance and cut to
    ``n_neighbors - 1`` (see ``accuracy.scanpy_knn_indices``); CREST: ``uns['neighbors']['indices']``."""
    obj = LAST.get("obj")
    if obj is None:
        return None
    if LAST["tool"] == "scanpy":
        from accuracy import scanpy_knn_indices
        params = (obj.uns.get("neighbors") or {}).get("params") or {}
        return scanpy_knn_indices(obj.obsp["distances"], int(params.get("n_neighbors", N_NEIGHBORS)))
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


def _rss_gb() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:  # noqa: BLE001
        return None


def _dataset_meta(path: Path) -> dict:
    """Provenance fields of the dataset file (kind, parent, seed) from its ``.meta.json`` sidecar, and
    from the name for real subsamples and synthetic sets."""
    meta = _meta(path.stem, path)
    kind = meta.get("kind")
    sub = protocol.parse_subset_name(path.stem)
    if kind is None:
        kind = "synthetic" if path.stem.startswith("synth_") else ("real-subsample" if sub or meta.get("subset_of") else "real")
    return {"kind": kind, "parent": meta.get("subset_of") or (sub[0] if sub else None), "seed": meta.get("subset_seed")}


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
           "thread_env": {k: os.environ.get(k) for k in (*protocol.THREAD_VARS, "NUMBA_THREADING_LAYER")}}
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
        nb = sys.modules["numba"]
        env["numba_num_threads"] = nb.config.NUMBA_NUM_THREADS
        env["numba_threading_layer_requested"] = os.environ.get("NUMBA_THREADING_LAYER")
        try:  # defined only once a parallel function has been compiled in this process
            env["numba_threading_layer"] = nb.threading_layer()
            env["numba_get_num_threads"] = nb.get_num_threads()
        except Exception as e:  # noqa: BLE001
            env["numba_threading_layer"] = f"unavailable: {type(e).__name__}"
    env["smt_siblings"] = {str(c): list(sib) for c, sib in protocol.read_smt_siblings().items()} or None
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


def import_tool(tool: str, profile: str) -> dict:
    """Import every module the workflow of ``tool`` loads (the timed start-up block). Optional modules
    that are not installed are listed, not fatal. Returns per-module seconds and the total."""
    import importlib
    names = TOOL_IMPORTS[protocol.base_tool(tool)] + (FULL_PROFILE_IMPORTS[protocol.base_tool(tool)] if profile == "full" else [])
    per, errors = {}, {}
    t0 = time.perf_counter()
    for n in names:
        t1 = time.perf_counter()
        try:
            importlib.import_module(n)
        except Exception as e:  # noqa: BLE001
            errors[n] = f"{type(e).__name__}: {e}"
        per[n] = time.perf_counter() - t1
    return {"seconds": time.perf_counter() - t0, "per_module": per, "errors": errors}


def _process_age_seconds() -> float | None:
    """Seconds since this process was started: since the parent's spawn timestamp (``CREST_BENCH_SPAWN_TIME``,
    set by run_all.py just before it starts the process; accurate to milliseconds), else since the operating
    system's process creation time (psutil; on Linux its resolution is limited by the boot time's one-second
    rounding). None if neither is available."""
    spawn = os.environ.get("CREST_BENCH_SPAWN_TIME")
    if spawn:
        try:
            return max(0.0, time.time() - float(spawn))
        except ValueError:
            pass
    try:
        import psutil
        return max(0.0, time.time() - psutil.Process(os.getpid()).create_time())
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["scanpy", "scanpy-lean", "crest", "crest-ooc"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", choices=["core", "full"], default="core")
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores (recorded only; set via env)")
    ap.add_argument("--pin", action="store_true",
                    help="pin the process to the first --threads logical CPUs, physical cores first (Linux)")
    ap.add_argument("--n-jobs", type=int, default=0, help="scanpy's sc.settings.n_jobs (0 = --threads)")
    ap.add_argument("--variant", default=protocol.DEFAULT_VARIANT, help="label of the thread-settings variant (recorded only)")
    ap.add_argument("--control", choices=protocol.CONTROLS, default=None,
                    help="label of a control run (never part of the headline); its files carry a suffix")
    ap.add_argument("--pilot", action="store_true", help="tuning-pilot run: files carry the variant in their name")
    ap.add_argument("--sample-interval", type=float, default=0.05, help="memory / load sampling interval (s)")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--save-outputs", action="store_true")
    ap.add_argument("--scrublet-max-cells", type=int, default=0, help="skip Scrublet above this many cells")
    a = ap.parse_args()
    global SCRUBLET_MAX, THREADS, N_JOBS
    SCRUBLET_MAX = a.scrublet_max_cells
    THREADS = a.threads or (os.cpu_count() or 1)
    N_JOBS = a.n_jobs
    path, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pin_cpus = None
    if a.pin and a.threads and hasattr(os, "sched_setaffinity"):
        pin_cpus = protocol.pin_cpus(a.threads, os.sched_getaffinity(0), protocol.read_smt_siblings())
        os.sched_setaffinity(0, set(pin_cpus))
    tag = protocol.run_tag(a.tool, path.stem, a.profile, a.threads, a.repeat, a.control, a.variant if a.pilot else None)
    backed = str(out / f"_ooc_{tag}") if a.tool == "crest-ooc" else None
    scanpy_like = protocol.base_tool(a.tool) == "scanpy"
    run = (lambda m, warm=False: scanpy_pipeline(path, m, a.profile, warm, lean=a.tool == "scanpy-lean")) if scanpy_like else \
          (lambda m, warm=False: crest_pipeline(path, m, a.profile, backed, warm))

    # timed start-up block: everything the workflow imports (part of the end-to-end endpoint)
    imports = import_tool(a.tool, a.profile)
    startup = _process_age_seconds()  # process start -> end of the imports
    startup_source = "spawn-time" if os.environ.get("CREST_BENCH_SPAWN_TIME") else "process-create-time"
    rss_after_imports = _rss_gb()
    _meta(path.stem, path)  # loads the registry module now, so its import is not charged to a step

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

    mon = Monitor(interval=a.sample_interval)
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
    step_sum = sum(v["seconds"] for v in mon.steps.values())
    quiet = foreign_busy_stats(mon.samples)
    env_rec = environment()
    report = {"tool": a.tool, "dataset": path.stem, "profile": a.profile, "threads": a.threads, "pinned": a.pin,
              "pin_cpus": pin_cpus, "variant": a.variant, "control": a.control, "n_jobs": N_JOBS or THREADS,
              "repeat": a.repeat, "status": status, "error": err, "n_cells": int(len(res["leiden"])) if res else None,
              "total_seconds": total, "startup_seconds": startup, "startup_source": startup_source,
              "import_seconds": imports["seconds"],
              "import_breakdown": imports["per_module"], "import_errors": imports["errors"],
              "end_to_end_seconds": protocol.end_to_end_seconds(startup, total) if status == "ok" else None,
              "untracked_seconds": total - step_sum, "quiet": quiet, "sample_interval_s": a.sample_interval,
              "rss_after_imports_gb": rss_after_imports, "dataset_meta": _dataset_meta(path), "warmup": warmup,
              "baseline_rss_gb": mon.baseline_rss / 1e9, "peak_rss_gb": peak,
              "peak_rss_minus_baseline_gb": None if peak is None else peak - mon.baseline_rss / 1e9,
              "usage": usage, "steps": mon.steps, "skipped_steps": SKIPPED, "env": env_rec,
              "params": {"normalize_target_sum": NORMALIZE_TARGET_SUM, "scale_max_value": SCALE_MAX_VALUE,
                         "n_hvg": N_HVG, "n_pcs": N_PCS, "n_neighbors": N_NEIGHBORS}}
    (out / f"{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("tool", "dataset", "threads", "repeat", "status", "total_seconds", "peak_rss_gb")}))
    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
