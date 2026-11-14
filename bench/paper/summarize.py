"""Statistics, tables and figures from a benchmark results directory.

    python bench/paper/summarize.py RESULTS_DIR

Reads RESULTS_DIR/raw/*.json (+ timelines, accuracy.json, env/) and writes
RESULTS_DIR/tables/*.csv|.tex, RESULTS_DIR/figures/*.pdf|.png and
RESULTS_DIR/report.md.

Reporting rules ("honest benchmarks"):
  * the primary endpoint is the end-to-end process wall time: process start, the imports the workflow
    needs and the timed pipeline (``end_to_end_seconds``); the sum of the step timers (``core_seconds``)
    is the secondary endpoint, and ``total_seconds`` / ``untracked_seconds`` / ``import_seconds`` are
    reported per tool in ``tables/endpoints.csv``. ``main_table(endpoint="core")`` selects the secondary one;
  * control runs (unpinned, default-settings) and tuning-pilot runs are never part of any statistic below;
    ``tables/controls.csv`` compares each control with its main cell;
  * per-step times are reported as FUSED STAGES (normalize + log1p + HVG; the lazily applied
    normalisation lands in the next kernel of CREST), never as the ratio of two lazy raw steps;
  * a step that gets slower with more threads (>= 1.5x and >= 10 s) is listed with its thread counts;
  * a results directory with a missing required result file (``protocol.missing_result_files``) does not
    summarise unless ``--allow-incomplete`` is given, and the report then carries an INCOMPLETE banner;
  * the headline compares the CORE workflow (read ... Wilcoxon); optional modules
    (seurat_v3 HVG, Scrublet, resolution sweep, Harmony, pseudobulk DESeq2) are a
    separate table;
  * headline candidates = the --main-threads set, identical for every dataset; thread-scan
    counts appear only in the thread-scaling table (and in informational per-thread
    columns). Within that set the speedup is computed at EVERY thread count both tools
    were measured at and best vs best (each tool at its fastest main thread count); the
    headline is the minimum of all of them. A row is flagged when only one common thread count was measured, and a
    tool's best thread count is flagged when it is at the edge of the scanned range;
  * a table lists every step where CREST is not clearly faster (< 1.2x);
  * every failed attempt is listed, from the append-only attempts.jsonl (a resumed run
    that later succeeded still shows its earlier failure).

Statistics (per dataset, tool pair CREST vs scanpy, n = repeats, typically 5):
  median, IQR, min and max of wall time and peak RSS (absolute and minus the RSS at the
  start of the timed run); speedup = median(scanpy) / median(CREST) with its min-max
  envelope [min(scanpy)/max(CREST), max(scanpy)/min(CREST)]. No bootstrap CIs: with 5
  repeats they are not meaningful. Two-sided Mann-Whitney U test and the Hodges-Lehmann
  estimate of the time difference, on the two samples behind the headline. Memory ratio
  likewise. Scaling with cells: OLS fit of log time on log cells over the synthetic series
  only, each tool at its best thread count per dataset (exponent alpha with a 95%
  t-interval and the number of points). Thread scaling: speedup T1/Tp, parallel efficiency,
  and the serial fraction f of Amdahl's law (least squares on 1/S = f + (1 - f)/p), fitted
  only up to the peak speedup and the physical core count, with residuals; no f when the
  speedup regresses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol  # noqa: E402

CORE_STEPS = ["read", "qc_filter", "normalize_log1p", "hvg", "scale_pca", "neighbors", "leiden", "umap",
              "de_ttest", "de_wilcoxon"]
MODULE_STEPS = ["hvg_seurat_v3", "scrublet", "leiden_sweep", "harmony", "pseudobulk_deseq2"]
COLORS = {"crest": "#2a78d6", "scanpy": "#eb6834", "crest-ooc": "#1baf7a"}  # validated categorical slots 1-3
LABELS = {"crest": "CREST", "scanpy": "scanpy", "crest-ooc": "CREST (out-of-core)"}
TOOLS = ["crest", "crest-ooc", "scanpy"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Fused stages of the core workflow, as (stage, raw steps). CREST records normalize_total / log1p and
# applies them inside the next kernel, so its ``normalize_log1p`` step takes ~0 s and the cost lands in
# ``hvg`` (and scale in ``scale_pca``): only the sum is comparable between the tools.
STAGES = [("read", ["read"]), ("qc_filter", ["qc_filter"]), ("norm+hvg", ["normalize_log1p", "hvg"]),
          ("scale_pca", ["scale_pca"]), ("neighbors", ["neighbors"]), ("leiden", ["leiden"]), ("umap", ["umap"]),
          ("de_ttest", ["de_ttest"]), ("de_wilcoxon", ["de_wilcoxon"])]
STAGE_LABELS = {"norm+hvg": "normalize+log1p+HVG (fused)"}
TIMER_FLOOR_S = 0.01     # below this a step time is timer noise or a deferred cost: no ratio is formed
MAX_STEP_RATIO = 1000.0  # no per-stage ratio above this is ever printed
ENDPOINT_COLUMNS = {"end_to_end": "end_to_end_seconds", "core": "core_seconds"}


# --------------------------------------------------------------------------- loading
def load_runs(raw: Path) -> pl.DataFrame:
    rows = []
    for f in sorted(raw.glob("*.json")):
        r = json.loads(f.read_text())
        if "tool" not in r:
            continue
        row = {k: r.get(k) for k in ("tool", "dataset", "profile", "threads", "pinned", "repeat", "status",
                                     "n_cells", "total_seconds", "peak_rss_gb", "baseline_rss_gb",
                                     "peak_rss_minus_baseline_gb")}
        for k in ("variant", "control", "n_jobs", "startup_seconds", "import_seconds", "end_to_end_seconds",
                  "sample_interval_s", "rss_after_imports_gb"):
            row[k] = r.get(k)
        q = r.get("quiet") or {}
        row["quiet_median_foreign_cores"] = q.get("median_foreign_busy_cores")
        row["quiet_violated"] = q.get("violated")
        row["quiet_flag"] = q.get("flag")
        row["process_wall_seconds"] = (r.get("harness") or {}).get("process_wall_seconds")
        dm = r.get("dataset_meta") or {}
        row["dataset_kind"], row["dataset_parent"] = dm.get("kind"), dm.get("parent")
        env = r.get("env") or {}
        row["numba_threading_layer"] = env.get("numba_threading_layer")
        row["numba_num_threads"] = env.get("numba_num_threads")
        row["pin_cpus"] = " ".join(map(str, r["pin_cpus"])) if r.get("pin_cpus") else None
        row["warmup_status"] = (r.get("warmup") or {}).get("status")
        row["warmup_error"] = (r.get("warmup") or {}).get("error")
        u = r.get("usage") or {}
        row["cpu_seconds"] = (u.get("ru_utime_s") or 0) + (u.get("ru_stime_s") or 0)
        row["skipped_steps"] = ",".join(r.get("skipped_steps") or []) or None
        row["energy_j"] = sum(s.get("energy_j", 0) for s in r.get("steps", {}).values()) or None
        for step, s in (r.get("steps") or {}).items():
            row[f"step:{step}"] = s["seconds"]
            row[f"par:{step}"] = s.get("parallelism")
            row[f"mem:{step}"] = s.get("peak_rss_gb")
        steps = r.get("steps") or {}
        row["core_seconds"] = (sum(steps[s]["seconds"] for s in CORE_STEPS)
                               if all(s in steps for s in CORE_STEPS) else None)
        # untracked time: the timed block minus the step timers (garbage collection between steps, result
        # extraction); recorded by run_one, recomputed when an older run JSON lacks it
        row["untracked_seconds"] = r.get("untracked_seconds")
        if row["untracked_seconds"] is None and steps and r.get("total_seconds") is not None:
            row["untracked_seconds"] = r["total_seconds"] - sum(v["seconds"] for v in steps.values())
        # memory of the core workflow is only clean in core-profile runs (in full runs the
        # optional modules run first and the allocator keeps their memory)
        row["core_peak_rss_gb"] = (max(steps[s].get("peak_rss_gb") or 0 for s in CORE_STEPS)
                                   if r.get("profile") == "core" and row["core_seconds"] is not None else None)
        row["core_peak_delta_gb"] = (row["core_peak_rss_gb"] - r["baseline_rss_gb"]
                                     if row["core_peak_rss_gb"] is not None and r.get("baseline_rss_gb") is not None
                                     else None)
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# --------------------------------------------------------------------------- statistics
def iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    return q3 - q1


def ratio_range(a, b):
    """median(b) / median(a) and its extreme range [min(b)/max(a), max(b)/min(a)].

    With n = 5 repeats per tool a bootstrap CI is not meaningful (the resampling
    distribution of a median of 5 has only a handful of distinct values), so the
    spread is reported as the full min-max envelope instead: every pair of repeats
    gives a ratio inside it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.median(b) / np.median(a)), float(b.min() / a.max()), float(b.max() / a.min())


def mann_whitney(a, b):
    from scipy.stats import mannwhitneyu
    if len(a) < 2 or len(b) < 2:
        return None
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def hodges_lehmann(a, b):
    d = np.subtract.outer(np.asarray(b, float), np.asarray(a, float)).ravel()
    return float(np.median(d))


def loglog_fit(n, t):
    from scipy import stats
    x, y = np.log10(n), np.log10(t)
    res = stats.linregress(x, y)
    dfree = len(x) - 2
    half = stats.t.ppf(0.975, dfree) * res.stderr if dfree > 0 else np.nan
    return {"alpha": res.slope, "alpha_ci95_lo": res.slope - half, "alpha_ci95_hi": res.slope + half,
            "n_points": int(len(x)), "intercept": res.intercept, "r2": res.rvalue ** 2}


def amdahl(p, s):
    """Least-squares serial fraction f of Amdahl's law, 1/S = f + (1 - f)/p."""
    p, s = np.asarray(p, float), np.asarray(s, float)
    x, y = 1 - 1 / p, 1 / s - 1 / p   # 1/S - 1/p = f (1 - 1/p)
    f = float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")
    return min(max(f, 0.0), 1.0)


def amdahl_speedup(f, p):
    return 1.0 / (f + (1.0 - f) / np.asarray(p, float))


def _main_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Runs used for the per-step breakdown: the richest profile, all cores."""
    df = _no_controls(df)
    prof = "full" if "full" in df["profile"].drop_nulls().to_list() else "core"
    d = df.filter(pl.col("profile") == prof)
    return d.filter(pl.col("threads") == d["threads"].max())


def _no_controls(df: pl.DataFrame) -> pl.DataFrame:
    """Runs of the main plan only: control runs (unpinned, default-settings) are never statistics."""
    return df.filter(pl.col("control").is_null()) if "control" in df.columns else df


def _core_samples(df: pl.DataFrame, endpoint: str = "core") -> dict:
    """{(dataset, tool, threads): {"t": core times, "mem": peak RSS, "dmem": peak - baseline, "src": profile}}.

    Core-profile runs are used wherever a (dataset, tool) has any; only a (dataset, tool) with
    no core-profile run at all falls back to the core steps of its full-profile runs (time
    only: their memory includes the optional modules, which also run first and can perturb
    the core timings), so headline numbers never mix the two profiles.

    ``endpoint``: ``"core"`` (sum of the step timers) or ``"end_to_end"`` (process start to the last
    step, imports included; core-profile runs only, a full-profile run has no clean end-to-end time).
    Control runs are excluded."""
    col = ENDPOINT_COLUMNS[endpoint]
    if col not in df.columns or df[col].drop_nulls().len() == 0:
        raise ValueError(f"the runs have no {col} column (written by the current run_one.py); "
                         f"summarise an older results directory with endpoint='core'")
    df = _no_controls(df)
    if endpoint == "end_to_end":
        df = df.filter(pl.col("profile") == "core")
    ok = df.filter((pl.col("status") == "ok") & pl.col(col).is_not_null())
    has_core = set(ok.filter(pl.col("profile") == "core").select("dataset", "tool").unique().iter_rows())
    ok = ok.filter((pl.col("profile") == "core") | ~pl.struct("dataset", "tool").map_elements(
        lambda r: (r["dataset"], r["tool"]) in has_core, return_dtype=pl.Boolean))
    out = {}
    for (ds, tool, th), g in ok.group_by(["dataset", "tool", "threads"], maintain_order=True):
        core = g.filter(pl.col("profile") == "core")
        use = core if core.height else g
        mem = core["core_peak_rss_gb"].drop_nulls().to_numpy() if core.height else np.array([])
        dmem = core["core_peak_delta_gb"].drop_nulls().to_numpy() if core.height else np.array([])
        out[(ds, tool, int(th))] = {"t": use[col].to_numpy(), "mem": mem if len(mem) else None,
                                    "dmem": dmem if len(dmem) else None, "src": "core" if core.height else "full"}
    return out


def main_thread_sets(res: Path, samp: dict, ncpu: int | None) -> dict:
    """{tool: thread counts of the MAIN runs} - the only counts the headline may use.

    Read from plan.json (the resolved ``main_threads`` / ``ooc_threads``; for an older plan the
    ``args``: ``main_threads`` or ``thread_grid``), clamped to the core count exactly as
    run_all.plan() does. The thread-scan counts (--threads-scan on --thread-dataset) are
    not in it, so every dataset's headline answers the same question. Without plan.json:
    the thread counts present on every dataset the tool was run on."""
    plan = res / "plan.json"
    if plan.exists():
        pj = json.loads(plan.read_text())
        args = pj.get("args", {})
        n = ncpu or 1
        if pj.get("main_threads"):   # resolved by the planner (thread grid, clamped to the CPUs)
            main = sorted(int(q) for q in pj["main_threads"])
            ooc = sorted(int(q) for q in (pj.get("ooc_threads") or main))
        else:
            main = sorted({min(int(q), n) for q in (args.get("main_threads") or args.get("thread_grid") or [n])})
            ooc = sorted({min(int(q), n) for q in (args.get("ooc_threads") or main)})
        return {"crest": set(main), "scanpy": set(main), "scanpy-lean": set(main), "crest-ooc": set(ooc)}
    out = {}
    for t in TOOLS:
        per_ds = [{k[2] for k in samp if k[0] == ds and k[1] == t} for ds in {k[0] for k in samp if k[1] == t}]
        out[t] = set.intersection(*per_ds) if per_ds else set()
    return out


def _best_threads(samp: dict, ds: str, tool: str, allowed: set | None = None):
    """Fastest measured thread count of ``tool`` on ``ds`` among ``allowed`` (None = any)."""
    th = sorted(k[2] for k in samp if k[0] == ds and k[1] == tool and (allowed is None or k[2] in allowed))
    if not th:
        return None, th
    med = {p: float(np.median(samp[(ds, tool, p)]["t"])) for p in th}
    return min(med, key=med.get), th


def _edge(best, th, ncpu) -> str | None:
    """Is the best thread count at the edge of what was scanned? (Then the true best may lie
    outside the scanned range; at the top, only if fewer than all logical cores were scanned.)"""
    if best is None or len(th) < 2:
        return None
    if best == th[0] and th[0] > 1:
        return f"lowest scanned ({best}); fewer threads untested"
    if best == th[-1]:
        return f"highest scanned ({best})" + ("" if ncpu and best >= ncpu else "; more threads untested")
    return None


def main_table(df: pl.DataFrame, ncpu: int | None = None, main_sets: dict | None = None,
               endpoint: str = "core") -> pl.DataFrame:
    """Headline: core workflow, CREST vs scanpy.

    Headline candidates = the --main-threads set (``main_sets``), identical for every
    dataset; thread-scan counts appear only in the thread-scaling table (and, for
    information, in the per-thread time columns and ``{tool}_best_threads_any``). The
    speed-up is computed at EVERY main thread count both tools were measured at (matched)
    and best vs best (each tool at its fastest main thread count). The headline is the
    minimum of all of these, and the Mann-Whitney test and Hodges-Lehmann shift are
    computed on the two samples behind the headline.

    ``endpoint`` is the time that is compared: ``"end_to_end"`` (the primary endpoint of the report) or
    ``"core"`` (sum of the step timers, the default of this function); the row says which one."""
    samp = _core_samples(df, endpoint)
    main_sets = main_sets or {t: None for t in TOOLS}
    rows = []
    datasets = _no_controls(df).group_by("dataset").agg(pl.col("n_cells").max()).sort("n_cells")
    for ds, n in datasets.iter_rows():
        row = {"dataset": ds, "n_cells": int(n or 0), "endpoint": endpoint}
        th = {}
        for t in TOOLS:
            fail = _no_controls(df).filter((pl.col("dataset") == ds) & (pl.col("tool") == t) & (pl.col("status") != "ok")).height
            row[f"{t}_failed"] = fail
            best, th[t] = _best_threads(samp, ds, t, main_sets.get(t))
            best_any, th_any = _best_threads(samp, ds, t)
            if not th[t]:
                continue
            top = max(th[t])
            x = samp[(ds, t, top)]["t"]
            bx = samp[(ds, t, best)]["t"]
            row.update({f"{t}_n": len(x), f"{t}_threads_main": " ".join(map(str, th[t])),
                        f"{t}_threads_measured": " ".join(map(str, th_any)),
                        f"{t}_best_threads_any": best_any,  # informational only, never the headline
                        f"{t}_time_median": float(np.median(x)), f"{t}_time_min": float(x.min()),
                        f"{t}_time_max": float(x.max()), f"{t}_time_iqr": iqr(x), f"{t}_source": samp[(ds, t, top)]["src"],
                        f"{t}_best_threads": best, f"{t}_best_time": float(np.median(bx)),
                        f"{t}_best_time_min": float(bx.min()), f"{t}_best_time_max": float(bx.max()),
                        f"{t}_best_edge": _edge(best, th[t], ncpu)})
            # the other endpoints of the same runs, so one table shows both and the untracked time
            sel = _no_controls(df).filter((pl.col("dataset") == ds) & (pl.col("tool") == t) & (pl.col("threads") == top)
                                          & (pl.col("status") == "ok") & (pl.col("profile") == samp[(ds, t, top)]["src"]))
            for c in ("core_seconds", "total_seconds", "end_to_end_seconds", "untracked_seconds", "import_seconds"):
                v = sel[c].drop_nulls().to_numpy() if c in sel.columns else np.array([])
                row[f"{t}_{c}_median"] = float(np.median(v)) if len(v) else None
            for p in th_any:  # every measured count, scan counts included (information)
                row[f"{t}_time_t{p}"] = float(np.median(samp[(ds, t, p)]["t"]))
        common = sorted(set(th["crest"]) & set(th["scanpy"]))
        if not common:
            rows.append(row)
            continue
        cands = []  # (speedup, lo, hi, basis, crest key, scanpy key)
        for p in common:
            s, lo, hi = ratio_range(samp[(ds, "crest", p)]["t"], samp[(ds, "scanpy", p)]["t"])
            row.update({f"speedup_t{p}": s, f"speedup_t{p}_min": lo, f"speedup_t{p}_max": hi})
            cands.append((s, lo, hi, f"matched @ {p}", (ds, "crest", p), (ds, "scanpy", p)))
        kc, ks = (ds, "crest", row["crest_best_threads"]), (ds, "scanpy", row["scanpy_best_threads"])
        s2, lo2, hi2 = ratio_range(samp[kc]["t"], samp[ks]["t"])
        row.update({"speedup_best": s2, "speedup_best_min": lo2, "speedup_best_max": hi2})
        cands.append((s2, lo2, hi2, "best vs best", kc, ks))
        m_s = [c for c in cands if c[3].startswith("matched")]
        lo_m = min(m_s, key=lambda c: c[0])
        row.update({"matched_threads": " ".join(map(str, common)), "speedup_matched_min_over_threads": lo_m[0],
                    "speedup_matched_min_threads": lo_m[4][2]})
        h = min(cands, key=lambda c: c[0])
        a, b = samp[h[4]]["t"], samp[h[5]]["t"]
        row.update({"speedup": h[0], "speedup_min": h[1], "speedup_max": h[2], "speedup_basis": h[3],
                    "headline_crest_threads": h[4][2], "headline_scanpy_threads": h[5][2],
                    "mannwhitney_p": mann_whitney(a, b), "hodges_lehmann_s": hodges_lehmann(a, b),
                    "n_crest": len(a), "n_scanpy": len(b)})
        row["single_thread_count"] = len(common) < 2
        # memory on the same basis as the headline: absolute peak, and peak minus the RSS at
        # the start of the timed run (interpreter + libraries)
        for key, col in (("mem", "memory_ratio"), ("dmem", "memory_ratio_minus_baseline")):
            ma, mb = samp[h[4]][key], samp[h[5]][key]
            if ma is not None and mb is not None:
                m, mlo, mhi = ratio_range(ma, mb)
                row.update({col: m, f"{col}_min": mlo, f"{col}_max": mhi,
                            f"crest_{key}_gb": float(np.median(ma)), f"scanpy_{key}_gb": float(np.median(mb))})
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def module_table(df: pl.DataFrame) -> pl.DataFrame:
    """Optional modules, from full-profile runs at the highest thread count."""
    ok = df.filter((pl.col("status") == "ok") & (pl.col("profile") == "full"))
    if not ok.height:
        return pl.DataFrame()
    ok = ok.filter(pl.col("threads") == ok["threads"].max())
    rows = []
    for ds in ok.sort("n_cells")["dataset"].unique(maintain_order=True).to_list():
        for s in MODULE_STEPS:
            col = f"step:{s}"
            if col not in ok.columns:
                continue
            v = {t: ok.filter((pl.col("dataset") == ds) & (pl.col("tool") == t))[col].drop_nulls().to_numpy()
                 for t in ("crest", "scanpy")}
            if not len(v["crest"]):
                continue
            c = float(np.median(v["crest"]))
            sc = float(np.median(v["scanpy"])) if len(v["scanpy"]) else None
            rows.append({"dataset": ds, "module": s, "crest_s": c, "scanpy_s": sc,
                         "speedup": sc / c if sc else None, "n": len(v["crest"])})
    return pl.DataFrame(rows, infer_schema_length=None)


FUSED_PARTS = {x for _, parts in STAGES if len(parts) > 1 for x in parts}


def not_faster_table(steps: pl.DataFrame, threshold: float = 1.2) -> pl.DataFrame:
    """Every (dataset, step) where CREST is not clearly faster than scanpy. Feed it the fused stage table
    (``stage_table``): a raw step whose cost is deferred in one tool (normalize_log1p, hvg) is skipped,
    because the ratio of such steps compares different amounts of work."""
    rows = []
    for (ds, st), g in steps.group_by(["dataset", "step"], maintain_order=True):
        if st in FUSED_PARTS:
            continue
        c = g.filter(pl.col("tool") == "crest")["median_s"].to_numpy()
        s = g.filter(pl.col("tool") == "scanpy")["median_s"].to_numpy()
        if len(c) and len(s) and c[0] > 0.005 and s[0] / c[0] < threshold:
            rows.append({"dataset": ds, "step": st, "crest_s": float(c[0]), "scanpy_s": float(s[0]),
                         "speedup": float(s[0] / c[0])})
    return pl.DataFrame(rows).sort(["speedup"]) if rows else pl.DataFrame()


# A step counts as slower with more threads only if it is at least 1.5x slower AND at least
# 10 s slower: the ratio alone also flags short steps whose few-second timings differ by noise.
THREAD_REGRESSION_RATIO = 1.5
THREAD_REGRESSION_MIN_SECONDS = 10.0


def is_thread_regression(t_few: float, t_many: float, ratio: float = THREAD_REGRESSION_RATIO,
                         min_seconds: float = THREAD_REGRESSION_MIN_SECONDS) -> bool:
    """Is a step slower with more threads: ``t_many >= ratio * t_few`` and
    ``t_many - t_few >= min_seconds`` (both must hold)."""
    return bool(t_many >= ratio * t_few and t_many - t_few >= min_seconds)


def safe_ratio(slow: float | None, fast: float | None, floor: float = TIMER_FLOOR_S,
               cap: float = MAX_STEP_RATIO) -> float | None:
    """``slow / fast``, or None when either time is below the timer floor (the step is deferred or
    below the resolution) or the ratio exceeds ``cap``: such a number compares different amounts of work."""
    if slow is None or fast is None or fast < floor or slow < floor:
        return None
    r = slow / fast
    return None if r > cap else float(r)


def stage_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per-stage time of the core workflow, with the lazily deferred steps fused: for every run the raw
    steps of a stage are summed first (``STAGES``), then the median, IQR and range over the repeats are
    taken. Same runs as ``step_table`` (the richest profile at the highest thread count)."""
    ok = _main_runs(df).filter(pl.col("status") == "ok")
    rows = []
    for (ds, tool), g in ok.group_by(["dataset", "tool"], maintain_order=True):
        for stage, parts in STAGES:
            cols = [f"step:{x}" for x in parts]
            if not all(c in g.columns for c in cols):
                continue
            sub = g.drop_nulls(cols)
            if not sub.height:
                continue
            v = sub.select(pl.sum_horizontal(cols).alias("v"))["v"].to_numpy()
            rows.append({"dataset": ds, "tool": tool, "step": stage, "median_s": float(np.median(v)), "iqr_s": iqr(v),
                         "min_s": float(v.min()), "max_s": float(v.max()), "n": int(len(v)), "raw_steps": "+".join(parts)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def stage_speedup_table(stages: pl.DataFrame) -> pl.DataFrame:
    """scanpy / CREST per (dataset, stage), as ``safe_ratio``: a stage below the timer floor, or a ratio
    above ``MAX_STEP_RATIO``, gets no ratio. Never contains a ratio above 1000."""
    rows = []
    for (ds, st), g in stages.group_by(["dataset", "step"], maintain_order=True):
        c = g.filter(pl.col("tool") == "crest")["median_s"].to_numpy()
        sc = g.filter(pl.col("tool") == "scanpy")["median_s"].to_numpy()
        if not (len(c) and len(sc)):
            continue
        r = safe_ratio(float(sc[0]), float(c[0]))
        rows.append({"dataset": ds, "stage": STAGE_LABELS.get(st, st), "crest_s": float(c[0]), "scanpy_s": float(sc[0]),
                     "speedup": r, "note": None if r is not None else "no ratio: below the timer floor or not comparable"})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def thread_regression_table(df: pl.DataFrame, main_sets: dict | None = None, ratio: float = THREAD_REGRESSION_RATIO,
                            min_seconds: float = THREAD_REGRESSION_MIN_SECONDS) -> pl.DataFrame:
    """Steps that get SLOWER with more threads. For every (dataset, tool, step) the median time at the highest
    main thread count is compared with the fastest lower main thread count; ``flagged`` is
    ``is_thread_regression`` (>= ratio x AND >= min_seconds slower). Core-profile main runs only."""
    d = _no_controls(df).filter((pl.col("status") == "ok") & (pl.col("profile") == "core"))
    step_cols = [c for c in d.columns if c.startswith("step:")]
    rows = []
    for (ds, tool), g in d.group_by(["dataset", "tool"], maintain_order=True):
        allowed = (main_sets or {}).get(tool)
        ths = sorted(t for t in g["threads"].unique().to_list() if allowed is None or t in allowed)
        if len(ths) < 2:
            continue
        hi = ths[-1]
        for col in step_cols:
            med = {}
            for p in ths:
                v = g.filter(pl.col("threads") == p)[col].drop_nulls().to_numpy()
                if len(v):
                    med[p] = float(np.median(v))
            lower = {p: t for p, t in med.items() if p < hi}
            if hi not in med or not lower:
                continue
            ref = min(lower, key=lower.get)
            rows.append({"dataset": ds, "tool": tool, "step": col[5:], "ref_threads": ref, "ref_s": lower[ref],
                         "threads": hi, "s": med[hi], "ratio": med[hi] / lower[ref] if lower[ref] > 0 else None,
                         "extra_s": med[hi] - lower[ref], "flagged": is_thread_regression(lower[ref], med[hi], ratio, min_seconds)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def endpoint_table(df: pl.DataFrame) -> pl.DataFrame:
    """Medians of every endpoint per (dataset, tool, threads) over the main core-profile runs: the primary
    ``end_to_end_seconds`` (imports and the timed block), its parts ``import_seconds`` / ``startup_seconds``
    / ``total_seconds`` (timed block incl. garbage collection and result extraction), the secondary
    ``core_seconds`` (sum of the step timers) and ``untracked_seconds`` (total - sum of the steps)."""
    d = _no_controls(df).filter((pl.col("status") == "ok") & (pl.col("profile") == "core"))
    cols = [c for c in ("end_to_end_seconds", "startup_seconds", "import_seconds", "total_seconds", "core_seconds",
                        "untracked_seconds", "process_wall_seconds") if c in d.columns]
    if not d.height or not cols:
        return pl.DataFrame()
    return (d.group_by(["dataset", "tool", "threads"]).agg(pl.len().alias("n"), *[pl.col(c).median().alias(c) for c in cols],
                                                          *([pl.col("end_to_end_seconds").min().alias("end_to_end_min"),
                                                             pl.col("end_to_end_seconds").max().alias("end_to_end_max")]
                                                            if "end_to_end_seconds" in cols else []))
            .sort(["dataset", "tool", "threads"]))


def control_table(df: pl.DataFrame, endpoint: str = "end_to_end") -> pl.DataFrame:
    """Each control (unpinned, default-settings) next to the main runs of the same (dataset, tool, threads):
    medians, and control / main. Controls are never part of the headline."""
    col = ENDPOINT_COLUMNS[endpoint]
    if "control" not in df.columns or col not in df.columns:
        return pl.DataFrame()
    ok = df.filter((pl.col("status") == "ok") & (pl.col("profile") == "core") & pl.col(col).is_not_null())
    ctl, main = ok.filter(pl.col("control").is_not_null()), ok.filter(pl.col("control").is_null())
    rows = []
    for (ds, tool, th, kind), g in ctl.group_by(["dataset", "tool", "threads", "control"], maintain_order=True):
        m = main.filter((pl.col("dataset") == ds) & (pl.col("tool") == tool) & (pl.col("threads") == th))
        c = float(np.median(g[col].to_numpy()))
        mm = float(np.median(m[col].to_numpy())) if m.height else None
        rows.append({"dataset": ds, "tool": tool, "threads": th, "control": kind, "n_control": g.height, "n_main": m.height,
                     "control_s": c, "main_s": mm, "control_over_main": c / mm if mm else None, "endpoint": endpoint})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def memory_table(df: pl.DataFrame) -> pl.DataFrame:
    """Peak memory of the core workflow at the highest thread count of every dataset that has a
    ``scanpy-lean`` (memory-lean profile) run: scanpy, scanpy-lean and CREST side by side, absolute and
    minus the RSS after the imports and warm-up, with the ratio to idiomatic scanpy."""
    d = _no_controls(df).filter((pl.col("status") == "ok") & (pl.col("profile") == "core")
                                & pl.col("core_peak_rss_gb").is_not_null())
    rows = []
    for ds in d.filter(pl.col("tool") == "scanpy-lean")["dataset"].unique(maintain_order=True).to_list():
        g = d.filter(pl.col("dataset") == ds)
        th = g.filter(pl.col("tool") == "scanpy-lean")["threads"].max()
        g = g.filter(pl.col("threads") == th)
        base = g.filter(pl.col("tool") == "scanpy")["core_peak_rss_gb"].to_numpy()
        for tool in ("scanpy", "scanpy-lean", "crest", "crest-ooc"):
            v = g.filter(pl.col("tool") == tool)
            if not v.height:
                continue
            pk = float(np.median(v["core_peak_rss_gb"].to_numpy()))
            rows.append({"dataset": ds, "threads": th, "tool": tool, "n": v.height, "peak_gb": pk,
                         "peak_minus_baseline_gb": float(np.median(v["core_peak_delta_gb"].drop_nulls().to_numpy()))
                         if v["core_peak_delta_gb"].drop_nulls().len() else None,
                         "scanpy_over_this": float(np.median(base)) / pk if len(base) and pk else None})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def quiet_table(attempts: pl.DataFrame) -> pl.DataFrame:
    """Every attempt that violated the quiet-host rule (median foreign busy cores >= the threshold), from
    ``attempts.jsonl`` (in the order the attempts were made), with its outcome: ``re-run``, or ``flagged``
    for the last attempt of a cell that violated the rule again in its re-run."""
    if not attempts.height or "quiet_violation" not in attempts.columns:
        return pl.DataFrame()
    ok = [r for r in attempts.iter_rows(named=True) if r.get("status") == "ok"]
    flagged = set(attempts.filter(pl.col("status") == "flagged")["tag"].to_list())
    last = {r["tag"]: i for i, r in enumerate(ok)}
    rows = []
    for i, r in enumerate(ok):
        if r.get("quiet_violation"):
            is_last = last[r["tag"]] == i
            rows.append({"tag": r["tag"], "started": r.get("started"),
                         "median_foreign_busy_cores": r.get("median_foreign_busy_cores"),
                         "outcome": "flagged (violated in the re-run too; kept with its load)"
                         if is_last and r["tag"] in flagged else "re-run"})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def _dataset_kinds(df: pl.DataFrame) -> dict:
    """{dataset: (kind, parent)}: from the run records, else from the name (``synth_*`` synthetic,
    ``<parent>_sub<N>`` real subsample)."""
    out = {}
    for ds in df["dataset"].unique().to_list():
        kind = parent = None
        if "dataset_kind" in df.columns:
            sub = df.filter((pl.col("dataset") == ds) & pl.col("dataset_kind").is_not_null())
            if sub.height:
                kind = sub["dataset_kind"][0]
                parent = sub["dataset_parent"][0] if "dataset_parent" in sub.columns else None
        if kind is None:
            sp = protocol.parse_subset_name(ds)
            kind, parent = ("synthetic", None) if ds.startswith("synth_") else (("real-subsample", sp[0]) if sp else ("real", None))
        out[ds] = (kind, parent)
    return out


def scaling_series(df: pl.DataFrame) -> dict[str, list[str]]:
    """Datasets forming a scaling series, by name: ``real subsamples of <parent>`` (the subsamples and the
    parent itself when it was run), the primary series, and ``synthetic (supplement)``. A series needs >= 2
    datasets."""
    kinds = _dataset_kinds(df)
    ncell = dict(df.group_by("dataset").agg(pl.col("n_cells").max()).iter_rows())
    series: dict[str, list[str]] = {}
    for ds, (kind, parent) in kinds.items():
        if kind == "real-subsample" and parent:
            series.setdefault(f"real subsamples of {parent}", []).append(ds)
        elif kind == "synthetic":
            series.setdefault("synthetic (supplement)", []).append(ds)
    for name in list(series):
        if name.startswith("real subsamples of "):
            parent = name[len("real subsamples of "):]
            if parent in kinds:
                series[name].append(parent)
    return {k: sorted(v, key=lambda d: ncell.get(d) or 0) for k, v in series.items() if len(v) >= 2}


def step_table(df: pl.DataFrame) -> pl.DataFrame:
    ok = _main_runs(df).filter(pl.col("status") == "ok")
    steps = [c[5:] for c in df.columns if c.startswith("step:")]
    rows = []
    for (ds, tool), g in ok.group_by(["dataset", "tool"], maintain_order=True):
        for s in steps:
            v = g[f"step:{s}"].drop_nulls().to_numpy()
            if len(v):
                # CPU/wall is meaningless for steps shorter than the timer resolution (lazy steps)
                p = g[f"par:{s}"].drop_nulls().to_numpy() if np.median(v) >= 0.05 else np.array([])
                m = g[f"mem:{s}"].drop_nulls().to_numpy()
                rows.append({"dataset": ds, "tool": tool, "step": s, "median_s": float(np.median(v)), "iqr_s": iqr(v),
                             "parallelism": float(np.median(p)) if len(p) else None,
                             "peak_rss_gb": float(np.median(m)) if len(m) else None})
    return pl.DataFrame(rows)


def thread_table(df: pl.DataFrame, physical_cores: int | None = None) -> pl.DataFrame:
    """Strong scaling of the core workflow. Amdahl's law is fitted only on thread counts up
    to the tool's peak speed-up (and up to the physical core count: SMT siblings are not
    independent processors). When the speed-up at the highest thread count falls more than
    3% below the peak, no serial fraction is reported (Amdahl's law cannot describe a
    speed-up that decreases); the row says "regresses beyond N threads" instead. Residuals
    (observed - fitted speed-up) are reported for every point, fitted or not."""
    ok = _no_controls(df).filter((pl.col("status") == "ok") & (pl.col("profile") == "core"))
    rows = []
    for (ds, tool), g in ok.group_by(["dataset", "tool"], maintain_order=True):
        med = g.group_by("threads").agg(pl.col("core_seconds").median().alias("total_seconds")).sort("threads")
        if med.height < 2 or 1 not in med["threads"].to_list():
            continue
        t1 = med.filter(pl.col("threads") == 1)["total_seconds"][0]
        p, t = med["threads"].to_numpy(), med["total_seconds"].to_numpy()
        S = t1 / t
        ipk = int(np.argmax(S))
        p_peak = int(p[ipk])
        cap = min(p_peak, physical_cores) if physical_cores else p_peak
        fit_mask = p <= cap
        regresses = S[-1] < 0.97 * S[ipk] and ipk < len(p) - 1
        f = amdahl(p[fit_mask], S[fit_mask]) if fit_mask.sum() >= 2 else None
        if regresses:
            note = f"regresses beyond {p_peak} threads (peak {S[ipk]:.2f}x)"
        elif ipk < len(p) - 1:
            note = f"plateaus beyond {p_peak} threads"
        else:
            note = "still improving at the highest thread count"
        pred = amdahl_speedup(f, p) if f is not None else np.full(len(p), np.nan)
        res = S - pred
        rms = float(np.sqrt(np.mean(res[fit_mask] ** 2))) if f is not None else None
        for i, (pi, ti) in enumerate(zip(p, t)):
            rows.append({"dataset": ds, "tool": tool, "threads": int(pi), "median_s": float(ti), "speedup": float(S[i]),
                         "efficiency": float(S[i] / pi), "peak_threads": p_peak,
                         "amdahl_serial_fraction": None if regresses else f,
                         "amdahl_fit_max_threads": None if regresses else int(cap),
                         "amdahl_fitted": bool(fit_mask[i]) and not regresses,
                         "amdahl_residual": None if (regresses or f is None) else float(res[i]),
                         "amdahl_rms_residual": None if regresses else rms, "scaling": note})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
    return plt


def _save(fig, figs: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"{name}.{ext}")


def fig_totals(df, figs, time_col: str = "core_seconds"):
    plt = _style()
    df = _no_controls(df)
    ok = df.filter((pl.col("status") == "ok") & pl.col(time_col).is_not_null())
    ok = ok.filter(pl.col("threads") == ok["threads"].max())
    # prefer core-profile runs; fall back to the core steps of full runs
    ok = ok.filter((pl.col("profile") == "core") | ~pl.struct("dataset", "tool").is_in(
        ok.filter(pl.col("profile") == "core").select(pl.struct("dataset", "tool")).to_series().implode()))
    ds = ok.group_by("dataset").agg(pl.col("n_cells").max()).sort("n_cells")["dataset"].to_list()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, lab in ((axes[0], time_col, ("end-to-end time (s, log)" if time_col == "end_to_end_seconds"
                                              else "core workflow time (s, log)")),
                         (axes[1], "core_peak_rss_gb", "core workflow peak memory (GB, log)")):
        w = 0.8 / len(TOOLS)
        for k, t in enumerate(TOOLS):
            for i, d in enumerate(ds):
                v = ok.filter((pl.col("dataset") == d) & (pl.col("tool") == t))[col].drop_nulls().to_numpy()
                if not len(v):
                    continue
                x = i + (k - (len(TOOLS) - 1) / 2) * w
                ax.bar(x, np.median(v), w * 0.9, color=COLORS[t], label=LABELS[t] if i == 0 else None, zorder=2)
                ax.scatter(np.full(len(v), x), v, s=8, color=INK, zorder=3, linewidths=0)
        ax.set_yscale("log")
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels(ds, rotation=30, ha="right")
        ax.set_ylabel(lab)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    _save(fig, figs, "fig1_time_memory")
    plt.close(fig)


def fig_steps(steps: pl.DataFrame, figs, dataset: str):
    plt = _style()
    s = steps.filter(pl.col("dataset") == dataset)
    if not s.height:
        return
    names = s["step"].unique(maintain_order=True).to_list()
    floor = TIMER_FLOOR_S  # fused stages are never below this; a bar is drawn at the floor for anything that is
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(names) + 1))
    h = 0.8 / len(TOOLS)
    for k, t in enumerate(TOOLS):
        for i, n in enumerate(names):
            v = s.filter((pl.col("tool") == t) & (pl.col("step") == n))["median_s"].to_numpy()
            if len(v):
                y = i + (k - 1) * h
                ax.barh(y, max(v[0], floor), h * 0.9, color=COLORS[t], label=LABELS[t] if i == 0 else None, zorder=2)
                ax.text(max(v[0], floor) * 1.08, y, f"{v[0]:.2g} s" if v[0] >= floor else "< 0.01 s",
                        va="center", fontsize=6, color=MUTED)
    ax.set_xlim(left=floor)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([STAGE_LABELS.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(f"median stage time (s, log), lazy steps fused - {dataset}")
    ax.legend(loc="lower right")
    _save(fig, figs, f"fig2_steps_{dataset}")
    plt.close(fig)


def fig_scaling(df, figs, main_sets: dict | None = None, endpoint: str = "core") -> list[dict]:
    """Time vs cells, one panel per series (``scaling_series``): the REAL SUBSAMPLES of a large dataset are
    the primary series; the synthetic sets are a labelled supplement. Each tool at its best main
    thread count per dataset, core-profile runs preferred (as in the main table); a log-log fit with its
    95% interval for every (series, tool) with at least 3 points."""
    plt = _style()
    samp = _core_samples(df, endpoint)
    ncell = dict(df.group_by("dataset").agg(pl.col("n_cells").max()).iter_rows())
    series = scaling_series(df)
    fits = []
    if not series:
        return fits
    fig, axes = plt.subplots(1, len(series), figsize=(4.6 * len(series), 3.6), squeeze=False)
    drawn = False
    for ax, (sname, dss) in zip(axes[0], sorted(series.items(), key=lambda kv: kv[0].startswith("synthetic"))):
        for t in TOOLS:
            pts = []
            for ds in dss:
                best, _ = _best_threads(samp, ds, t, (main_sets or {}).get(t))
                if best is not None and ncell.get(ds):
                    pts.append((ncell[ds], float(np.median(samp[(ds, t, best)]["t"])), best, ds))
            if len(pts) < 2:
                continue
            drawn = True
            n, y = np.array([q[0] for q in pts], float), np.array([q[1] for q in pts])
            ax.plot(n, y, "o-", color=COLORS[t], lw=2, ms=6, label=LABELS[t], zorder=3)
            if len(pts) >= 3:
                f = loglog_fit(n, y)
                fits.append({"series": sname, "tool": t, **f, "datasets": " ".join(q[3] for q in pts),
                             "best_threads": " ".join(str(q[2]) for q in pts)})
                ax.annotate(f"α = {f['alpha']:.2f}", (n[-1], y[-1]), textcoords="offset points", xytext=(6, 0),
                            color=INK, fontsize=8, va="center")
        ax.set_title(sname, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cells (log)")
        ax.set_ylabel("time at best threads (s, log)")
        ax.legend()
    if not drawn:
        plt.close(fig)
        return fits
    _save(fig, figs, "fig3_scaling")
    plt.close(fig)
    return fits


def fig_threads(th: pl.DataFrame, figs):
    if not th.height:
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    pmax = th["threads"].max()
    ax.plot([1, pmax], [1, pmax], color=MUTED, lw=1, ls="--", label="ideal")
    for t in TOOLS:
        g = th.filter(pl.col("tool") == t).sort("threads")
        if g.height:
            f = g["amdahl_serial_fraction"][0]
            lab = (f"{LABELS[t]} (Amdahl f = {f:.2f}, fit to {g['amdahl_fit_max_threads'][0]} threads)"
                   if f is not None else f"{LABELS[t]} ({g['scaling'][0]})")
            ax.plot(g["threads"], g["speedup"], "o-", color=COLORS[t], lw=2, ms=6, label=lab)
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup over 1 thread")
    ax.legend()
    _save(fig, figs, "fig4_thread_scaling")
    plt.close(fig)


def fig_timeline(raw: Path, figs, dataset: str):
    """Memory over time and per-core utilisation heatmaps, repeat 0, all threads."""
    plt = _style()
    tls = {}
    for t in TOOLS:
        c = sorted(raw.glob(f"{t}__{dataset}__*__r0.timeline.csv"), key=lambda p: ("__full__" in p.name, p.name))
        if c:
            tls[t] = pl.read_csv(c[-1], infer_schema_length=None)
    if not tls:
        return
    fig, axes = plt.subplots(1 + len(tls), 1, figsize=(9, 2.4 + 2.0 * len(tls)), sharex=False,
                             gridspec_kw={"height_ratios": [2] + [1.3] * len(tls)})
    ax = axes[0]
    for t, tl in tls.items():
        ax.plot(tl["t"], tl["rss_gb"], color=COLORS[t], lw=2, label=LABELS[t])
    ax.set_ylabel("RSS (GB)")
    ax.set_xlabel("time (s)")
    ax.legend()
    fig.subplots_adjust(hspace=0.6)
    for ax, (t, tl) in zip(axes[1:], tls.items()):
        uc = [c for c in tl.columns if c.startswith("util_")]
        M = tl.select(uc).to_numpy().T
        im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=100, interpolation="nearest",
                       extent=[tl["t"][0], tl["t"][-1], len(uc) - 0.5, -0.5])
        ax.set_ylabel(f"{LABELS[t]}\ncore")
        ax.grid(False)
        # step boundaries; label steps longer than 6% of the run, inside the panel
        steps, ts = tl["step"].to_list(), tl["t"].to_numpy()
        starts = [i for i in range(len(steps)) if steps[i] not in ("between", "setup") and (i == 0 or steps[i] != steps[i - 1])]
        span = ts[-1] - ts[0]
        for j, i in enumerate(starts):
            end = next((q for q in range(i + 1, len(steps)) if steps[q] != steps[i]), len(steps) - 1)
            ax.axvline(ts[i], color="#ffffff", lw=0.8)
            if ts[end] - ts[i] > 0.06 * span:
                ax.text((ts[i] + ts[end]) / 2, len(uc) - 0.5, steps[i], fontsize=6, ha="center", va="bottom", color=INK)
        ax.set_xlabel("time (s)")
    fig.colorbar(im, ax=axes[1:], label="core utilisation (%)", shrink=0.8)
    _save(fig, figs, f"fig5_timeline_{dataset}")
    plt.close(fig)


# --------------------------------------------------------------------------- output
def to_latex(df: pl.DataFrame, path: Path, caption: str):
    cols = df.columns
    lines = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}", "\\begin{tabular}{" + "l" * len(cols) + "}",
             "\\hline", " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\hline"]
    for row in df.iter_rows():
        lines.append(" & ".join("" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v).replace("_", "\\_"))
                                for v in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def compact_main(t: pl.DataFrame, settings_label: str = "default thread settings") -> pl.DataFrame:
    """Readable headline: time per tool, speedups with min-max ranges, memory. ``settings_label`` names the
    thread settings behind the matched columns (default: only the thread count was set; tuned: the pilot's
    per-tool choice)."""
    rows = []
    rng = lambda r, k: f"{r[k]:.2f} ({r[k + '_min']:.2f}-{r[k + '_max']:.2f})"  # noqa: E731
    for r in t.iter_rows(named=True):
        row = {"dataset": r["dataset"], "cells": r["n_cells"]}
        for tool in TOOLS:
            if r.get(f"{tool}_time_median") is not None:
                row[f"{LABELS[tool]} (s)"] = (f"{r[f'{tool}_time_median']:.3g} "
                                              f"[{r[f'{tool}_time_min']:.3g}-{r[f'{tool}_time_max']:.3g}]")
                row[f"{LABELS[tool]} best (s @ threads)"] = (f"{r[f'{tool}_best_time']:.3g} @ {r[f'{tool}_best_threads']}"
                                                             + (" ¶" if r.get(f"{tool}_best_edge") else ""))
            elif r.get(f"{tool}_failed"):
                row[f"{LABELS[tool]} (s)"] = "failed"
        if r.get("speedup") is not None:
            common = [int(x) for x in str(r["matched_threads"]).split()]
            row[f"matched, {settings_label} (threads: speed-up)"] = "; ".join(f"{p}: {rng(r, f'speedup_t{p}')}" for p in common)
            row["best vs best"] = rng(r, "speedup_best")
            row["headline"] = f"**{r['speedup']:.2f}** ({r['speedup_min']:.2f}-{r['speedup_max']:.2f})" + (
                " †" if r.get("single_thread_count") else "")
            row["basis"] = r["speedup_basis"]
            row["Mann-Whitney p"] = None if r["mannwhitney_p"] is None else f"{r['mannwhitney_p']:.2g}"
            row["HL shift (s)"] = f"{r['hodges_lehmann_s']:.3g}"
            row["memory ratio"] = rng(r, "memory_ratio") if r.get("memory_ratio") is not None else "n/a ‡"
            row["memory ratio (peak - baseline)"] = (rng(r, "memory_ratio_minus_baseline")
                                                     if r.get("memory_ratio_minus_baseline") is not None else "n/a ‡")
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def load_attempts(res: Path) -> pl.DataFrame:
    """Every attempt from the append-only attempts.jsonl (empty if the file is missing)."""
    f = res / "attempts.jsonl"
    if not f.exists():
        return pl.DataFrame()
    recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return pl.DataFrame(recs, infer_schema_length=None) if recs else pl.DataFrame()


def transpose_metrics(acc: pl.DataFrame) -> pl.DataFrame:
    """Metric rows x dataset columns."""
    ds = acc["dataset"].to_list()
    cols = [c for c in acc.columns if c != "dataset"]
    return pl.DataFrame({"metric": cols, **{d: [None if acc[c][i] is None else f"{acc[c][i]:.4g}"
                                               if isinstance(acc[c][i], float) else str(acc[c][i]) for c in cols]
                                           for i, d in enumerate(ds)}})


def md(df: pl.DataFrame) -> str:
    if not df.height:
        return "_(no data)_\n"
    s = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    for row in df.iter_rows():
        s += "| " + " | ".join("" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)) for v in row) + " |\n"
    return s


def _machine(raw: Path) -> dict:
    """Logical / physical core counts from the first run JSON that recorded them."""
    for f in sorted(raw.glob("*.json")):
        env = json.loads(f.read_text()).get("env") or {}
        if env.get("logical_cores"):
            return {"logical": env.get("logical_cores"), "physical": env.get("physical_cores")}
    return {"logical": None, "physical": None}


def _topology(raw: Path) -> dict:
    """Affinity and SMT sibling map recorded by the first run that has them, and the CPUs each (tool, threads)
    was pinned to."""
    siblings, pins = None, {}
    for f in sorted(raw.glob("*.json")):
        r = json.loads(f.read_text())
        env = r.get("env") or {}
        siblings = siblings or env.get("smt_siblings")
        if r.get("pin_cpus") and not r.get("control"):
            pins.setdefault((r.get("tool"), r.get("threads")), r["pin_cpus"])
    return {"siblings": siblings, "pins": pins}


def _endpoint_note(endpoint: str) -> str:
    if endpoint == "end_to_end":
        return ("**Primary endpoint: end-to-end process wall time** = from process creation to the end of the last "
                "pipeline step: interpreter start-up, the imports the workflow needs, and the timed pipeline "
                "(every step plus the garbage collections and result extraction between them). The in-process "
                "JIT warm-up and the untimed accuracy extras are not included. Secondary endpoint: the sum of the "
                "step timers (`core_seconds`). `tables/endpoints.csv` lists both, the import and untracked time of "
                "each tool, and `tables/runs.csv` every repeat.")
    return ("**Endpoint: sum of the step timers** (`core_seconds`; interpreter start-up, imports, garbage collection "
            "between steps and result extraction excluded). This is the secondary endpoint; the primary one is "
            "the end-to-end process wall time.")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Tables, figures and report.md from a benchmark results directory")
    ap.add_argument("results")
    ap.add_argument("--endpoint", choices=["end_to_end", "core"], default="end_to_end",
                    help="time compared in the headline (default: end-to-end process wall time)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write a draft report although required result files are missing")
    a = ap.parse_args(argv)
    res = Path(a.results)
    raw, tables, figs = res / "raw", res / "tables", res / "figures"
    missing = protocol.missing_result_files(res)
    if missing:
        res.mkdir(parents=True, exist_ok=True)
        (res / "missing_files.txt").write_text("\n".join(missing) + "\n")
        print(f"INCOMPLETE: {len(missing)} required result file(s) missing (see {res / 'missing_files.txt'}); "
              f"first: {', '.join(missing[:5])}", file=sys.stderr)
        if not a.allow_incomplete:
            sys.exit(1)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = load_runs(raw)
    if not df.height:
        print("no runs found")
        return
    ep = a.endpoint
    if ep == "end_to_end" and ("end_to_end_seconds" not in df.columns or df["end_to_end_seconds"].drop_nulls().len() == 0):
        sys.exit("the runs have no end_to_end_seconds (an older harness wrote them): summarise with --endpoint core")
    mach = _machine(raw)
    df.write_csv(tables / "runs.csv")
    plan_json = json.loads((res / "plan.json").read_text()) if (res / "plan.json").exists() else {}
    tuned = bool(plan_json.get("tuning"))
    settings_label = "tuned thread settings" if tuned else "default thread settings"
    main_sets = main_thread_sets(res, _core_samples(df, "core"), mach["logical"])
    main_t = main_table(df, mach["logical"], main_sets, endpoint=ep)
    main_core = main_table(df, mach["logical"], main_sets, endpoint="core") if ep != "core" else pl.DataFrame()
    steps = step_table(df)
    stages = stage_table(df)
    stage_su = stage_speedup_table(stages) if stages.height else pl.DataFrame()
    th = thread_table(df, mach["physical"])
    reg = thread_regression_table(df, main_sets)
    attempts = load_attempts(res)
    for name, t, cap in (("main", main_t, "Wall time and peak memory, CREST vs scanpy"),
                         ("main_core_steps", main_core, "Secondary endpoint (sum of step timers), CREST vs scanpy"),
                         ("steps", steps, "Per-step median time, parallelism and peak memory (raw steps)"),
                         ("stages", stages, "Per-stage median time, lazy steps fused"),
                         ("threads", th, "Strong scaling")):
        if t.height:
            t.write_csv(tables / f"{name}.csv")
            to_latex(t, tables / f"{name}.tex", cap)
    for name, t in (("endpoints", endpoint_table(df)), ("stage_speedups", stage_su), ("thread_regressions", reg),
                    ("controls", control_table(df, ep)), ("memory_lean", memory_table(df)),
                    ("quiet_host", quiet_table(attempts))):
        if t.height:
            t.write_csv(tables / f"{name}.csv")
    fig_totals(df, figs, ENDPOINT_COLUMNS[ep])
    for ds in df["dataset"].unique().to_list():
        fig_steps(stages, figs, ds)
    fits = fig_scaling(df, figs, main_sets, ep)
    if fits:
        pl.DataFrame(fits).write_csv(tables / "scaling.csv")
    fig_threads(th, figs)
    largest = df.filter(pl.col("status") == "ok").sort("n_cells")["dataset"].to_list()
    for ds in dict.fromkeys(largest[-1:] + ["kang"]):
        fig_timeline(raw, figs, ds)
    acc = json.loads((res / "accuracy.json").read_text()) if (res / "accuracy.json").exists() else []
    acc_df = pl.DataFrame(acc, infer_schema_length=None) if acc else pl.DataFrame()
    if acc_df.height:
        acc_df.write_csv(tables / "accuracy.csv")
    mods = sorted((res / "modules").glob("*.md")) if (res / "modules").exists() else []
    env = (res / "env" / "summary.md").read_text() if (res / "env" / "summary.md").exists() else ""
    mods_t = module_table(df)
    slow = not_faster_table(stages)
    for name, t in (("modules", mods_t), ("not_faster", slow)):
        if t.height:
            t.write_csv(tables / f"{name}.csv")
    topo = _topology(raw)
    reg_flag = reg.filter(pl.col("flagged")) if reg.height else reg
    with open(res / "report.md", "w") as fh:
        fh.write("# CREST benchmark report\n\n")
        if missing:
            fh.write(f"> **INCOMPLETE: {len(missing)} required result file(s) are missing** (`missing_files.txt`). "
                     "This is a draft, not a result.\n\n")
        fh.write(f"{env}\n\n## Core workflow: CREST vs scanpy\n\n")
        fh.write(_endpoint_note(ep) + "\n\n")
        fh.write("Core workflow = read, QC filter, normalize + log1p, HVG, scale + PCA, neighbours, Leiden, UMAP, "
                 "t-test and Wilcoxon marker genes. Headline candidates = the main thread counts "
                 "(the thread grid), identical for every dataset; "
                 "thread-scan counts appear only in the thread-scaling table. Times are medians [min-max] over "
                 "repeats at the highest main thread count; *best* is each tool's fastest main thread count. Speedup = scanpy / CREST, given at every thread count both "
                 "tools were measured at (*matched*; "
                 f"{settings_label}) and *best vs best*, each with its min-max envelope "
                 "[min(scanpy)/max(CREST), max(scanpy)/min(CREST)] (no bootstrap CIs: with 5 repeats they are not "
                 "meaningful). The **headline** is the minimum of all of these; Mann-Whitney p and the "
                 "Hodges-Lehmann shift (scanpy - CREST, seconds) are computed on the two samples behind the "
                 "headline (*basis*). Memory ratio = scanpy / CREST peak RSS of the core workflow on the same basis, "
                 "absolute and minus each process's RSS after the imports and the warm-up's release.\n\n")
        fh.write(md(compact_main(main_t, settings_label)) if main_t.height else "_(no data)_\n")
        if main_t.height and "single_thread_count" in main_t.columns and main_t["single_thread_count"].any():
            fh.write("\n† only one thread count was measured for both tools on this dataset, so *best vs best* "
                     "may equal *matched*; the headline can then overstate CREST's advantage.\n")
        edge_cols = [c for c in main_t.columns if c.endswith("_best_edge")] if main_t.height else []
        edges = [(r["dataset"], c[: -len("_best_edge")], r[c]) for r in main_t.iter_rows(named=True)
                 for c in edge_cols if r.get(c)]
        if edges:
            fh.write("\n¶ best thread count at the edge of the scanned range (the true best may lie outside it):\n\n")
            for ds, tool, why in edges:
                fh.write(f"* {ds}, {LABELS.get(tool, tool)}: {why}\n")
        if main_t.height and "speedup" in main_t.columns and (
                "memory_ratio" not in main_t.columns
                or main_t.filter(pl.col("speedup").is_not_null() & pl.col("memory_ratio").is_null()).height):
            fh.write("\n‡ memory not reported: these times come from full-profile runs, whose peak memory includes "
                     "the optional modules. Clean core-workflow memory needs core-profile runs.\n")
        fh.write("\nFull statistics (per-thread speedups and ranges, IQR, memory, failures) in `tables/main.csv`.\n")
        if main_core.height:
            fh.write("\n### Secondary endpoint: sum of the step timers\n\n")
            fh.write(md(compact_main(main_core, settings_label)))
        et = endpoint_table(df)
        fh.write("\n### Endpoints per tool and thread count\n\nMedians in seconds; `untracked_seconds` = timed block "
                 "minus the step timers (garbage collection, result extraction), `import_seconds` = the imports the "
                 "workflow needs.\n\n" + md(et))
        fig = lambda name, alt: f"\n![{alt}](figures/{name}.png)\n" if (figs / f"{name}.png").exists() else ""  # noqa: E731
        fh.write(fig("fig1_time_memory", "time and memory") + "\n## Where CREST is not clearly faster (< 1.2x)\n\n"
                 "Fused stages: normalize + log1p + HVG are one stage, because CREST applies the normalisation "
                 "inside the next kernel.\n\n")
        fh.write(md(slow) if slow.height else "none\n")
        fh.write("\n### Steps that get slower with more threads (>= 1.5x and >= 10 s)\n\n"
                 "Highest main thread count against the fastest lower one, per step (all rows in "
                 "`tables/thread_regressions.csv`).\n\n")
        fh.write(md(reg_flag) if reg_flag.height else "none\n")
        ctl = control_table(df, ep)
        fh.write("\n## Controls (not part of any statistic above)\n\n"
                 "`unpinned`: the same run without CPU pinning; `default-settings`: the untuned thread settings "
                 "where a tuning pilot chose something else. `control_over_main` > 1 means the control was slower.\n\n")
        fh.write(md(ctl) if ctl.height else "none\n")
        fh.write("\n## Pinning, CPU topology and thread settings\n\n"
                 + (f"Policy: {plan_json['pin_policy']}.\n\n" if plan_json.get("pin_policy") else
                    "Policy: not recorded in plan.json.\n\n"))
        if topo["pins"]:
            fh.write("CPUs each (tool, threads) was pinned to (first run):\n\n")
            for (tool, p), cpus in sorted(topo["pins"].items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
                fh.write(f"* {tool}, {p} threads: {' '.join(map(str, cpus))}\n")
        if topo["siblings"]:
            groups = sorted({tuple(v) for v in topo["siblings"].values()})
            fh.write(f"\nSMT sibling groups (`thread_siblings_list`, {len(groups)} cores): "
                     + "; ".join(",".join(map(str, g)) for g in groups[:8]) + (" ..." if len(groups) > 8 else "") + "\n")
        else:
            fh.write("\nSMT sibling groups: not recorded (no sysfs topology on the host).\n")
        if "numba_threading_layer" in df.columns:
            tset = (_no_controls(df).filter(pl.col("profile") == "core")
                    .group_by("tool", "threads", "variant", "numba_threading_layer", "numba_num_threads", "n_jobs")
                    .agg(pl.len().alias("runs")).sort("tool", "threads"))
            fh.write("\nThread settings the runs actually had (numba threading layer and thread count as the process "
                     "reports them):\n\n" + md(tset))
        qt = quiet_table(attempts)
        flagged = df.filter(pl.col("quiet_flag").is_not_null()) if "quiet_flag" in df.columns else pl.DataFrame()
        fh.write("\n## Quiet host\n\nA run violates the rule when the median over its samples of the busy cores "
                 "not due to the run reaches 1. A violating run is re-run once; a cell that violates twice is kept "
                 "and flagged.\n\n")
        fh.write(md(qt) if qt.height else "no violations\n")
        if flagged.height:
            fh.write("\nFlagged cells:\n\n" + md(flagged.select("tool", "dataset", "threads", "repeat", "quiet_median_foreign_cores", "quiet_flag")))
        mem_lean = memory_table(df)
        if mem_lean.height:
            fh.write("\n## Memory: idiomatic and memory-lean scanpy\n\nPeak RSS of the core workflow, sampled every "
                     "50 ms (see `sample_interval_s` in `tables/runs.csv`).\n\n" + md(mem_lean))
        fh.write("\n## Optional modules\n\nFrom full-profile runs at the highest thread count; scanpy has no pseudobulk "
                 "DESeq2 (see the DESeq2 module benchmark for R and pydeseq2).\n\n")
        fh.write(md(mods_t) if mods_t.height else "_(no full-profile runs)_\n")
        fh.write("\n## Scaling with cells (core workflow)\n\n")
        if fits:
            fh.write("Log-log OLS fit of time on cell count (time ∝ cells^alpha) with its 95% t-interval, per series: "
                     "**random subsamples of a real dataset** are the primary series; the synthetic sets "
                     "(interpolated from 68k real cells with CREST's own PCA and kNN) are a labelled supplement. Each tool "
                     "at its best main thread count per dataset (`best_threads`); `n_points` is the number of datasets. "
                     "With few points the interval is wide: read alpha together with it.\n\n")
            fh.write(md(pl.DataFrame(fits)))
        else:
            fh.write("_(needs >= 3 datasets of one series: real subsamples of a large dataset, or the synthetic supplement)_\n")
        fh.write(fig("fig3_scaling", "scaling") + "\n## Thread scaling (core workflow)\n\n")
        if th.height:
            fh.write("Speedup over 1 thread (sum of the step timers). Amdahl's serial fraction is fitted only up to each tool's peak speedup "
                     "and the physical core count (`amdahl_fit_max_threads`); `amdahl_residual` = observed - fitted "
                     "speedup (also shown for points outside the fit). Where the speedup falls more than 3% below "
                     "its peak at higher thread counts, no serial fraction is given (`scaling` says where it "
                     "regresses).\n\n")
        fh.write(md(th))
        fh.write(fig("fig4_thread_scaling", "threads") + "\n## Per-stage breakdown\n\nStage times with the lazily "
                 "deferred steps fused; the raw steps are in `tables/steps.csv`. A stage below the timer floor, or a "
                 "ratio above 1000, gets no ratio.\n\n" + md(stage_su))
        for ds in df["dataset"].unique().to_list():
            fh.write(fig(f"fig2_steps_{ds}", f"stages {ds}") + fig(f"fig5_timeline_{ds}", f"timeline {ds}"))
        fh.write("\n## Agreement with scanpy\n\n"
                 "`knn_graph_jaccard` compares the kNN graphs the two tools actually built; `pca_knn15_jaccard` "
                 "recomputes 15-NN in each tool's PCA space with the same (CREST) kNN, so it measures the PCA "
                 "embeddings, not the graphs. `knn_recall_*`: recall@k of each tool's graph vs brute-force exact "
                 "neighbours (up to 10,000 query cells against all cells). `same_graph_*`: CREST's and scanpy's "
                 "Leiden on scanpy's graph, modularity scored by igraph.\n\n"
                 + (md(transpose_metrics(acc_df)) if acc_df.height else "_(no data)_\n"))
        for m in mods:
            fh.write(f"\n## Module: {m.stem}\n\n" + m.read_text())
        if "skipped_steps" in df.columns:
            sk = (df.filter(pl.col("skipped_steps").is_not_null())
                  .group_by("dataset", "tool", "skipped_steps").agg(pl.len().alias("runs")).sort("dataset", "tool"))
            fh.write("\n## Skipped steps\n\n" + (md(sk) if sk.height else "none\n"))
        wf = df.filter(pl.col("warmup_status") == "failed") if "warmup_status" in df.columns else pl.DataFrame()
        fh.write("\n## Warm-up failures\n\n" + (md(wf.select("tool", "dataset", "profile", "threads", "repeat",
                                                               "warmup_error")) if wf.height else "none\n"))
        fh.write("\n## Failed runs\n\n")
        if attempts.height:
            real = attempts.filter(~pl.col("status").is_in(["skipped", "flagged"]))
            failed = real.filter(pl.col("status") != "ok")
            ok_tags = set(real.filter(pl.col("status") == "ok")["tag"].to_list())
            fh.write(f"{real.height} attempts in `attempts.jsonl` ({real.height - failed.height} ok, {failed.height} "
                     f"failed; {attempts.height - real.height} skipped on resume because already ok, or flagged).\n\n")
            if failed.height:
                cols = [c for c in ("tool", "dataset", "profile", "threads", "repeat", "started", "reason")
                        if c in failed.columns]
                later = pl.Series("later succeeded", [t in ok_tags for t in failed["tag"].to_list()])
                fh.write(md(failed.select(cols).with_columns(later)))
        else:
            fails = df.filter(pl.col("status") != "ok").select("tool", "dataset", "profile", "threads", "repeat", "status")
            fh.write("(no attempts.jsonl: final status of each run only)\n\n" + (md(fails) if fails.height else "none\n"))
    print(f"report: {res / 'report.md'}")


if __name__ == "__main__":
    main()
