"""Statistics, tables and figures from a benchmark results directory.

    python bench/paper/summarize.py RESULTS_DIR

Reads RESULTS_DIR/raw/*.json (+ timelines, accuracy.json, env/) and writes
RESULTS_DIR/tables/*.csv|.tex, RESULTS_DIR/figures/*.pdf|.png and
RESULTS_DIR/report.md.

Reporting rules (CLAUDE.md, "honest benchmarks"):
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

CORE_STEPS = ["read", "qc_filter", "normalize_log1p", "hvg", "scale_pca", "neighbors", "leiden", "umap",
              "de_ttest", "de_wilcoxon"]
MODULE_STEPS = ["hvg_seurat_v3", "scrublet", "leiden_sweep", "harmony", "pseudobulk_deseq2"]
COLORS = {"crest": "#2a78d6", "scanpy": "#eb6834", "crest-ooc": "#1baf7a"}  # validated categorical slots 1-3
LABELS = {"crest": "CREST", "scanpy": "scanpy", "crest-ooc": "CREST (out-of-core)"}
TOOLS = ["crest", "crest-ooc", "scanpy"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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
    prof = "full" if "full" in df["profile"].drop_nulls().to_list() else "core"
    d = df.filter(pl.col("profile") == prof)
    return d.filter(pl.col("threads") == d["threads"].max())


def _core_samples(df: pl.DataFrame) -> dict:
    """{(dataset, tool, threads): {"t": core times, "mem": peak RSS, "dmem": peak - baseline, "src": profile}}.

    Core-profile runs are used wherever a (dataset, tool) has any; only a (dataset, tool) with
    no core-profile run at all falls back to the core steps of its full-profile runs (time
    only: their memory includes the optional modules, which also run first and can perturb
    the core timings), so headline numbers never mix the two profiles."""
    ok = df.filter((pl.col("status") == "ok") & pl.col("core_seconds").is_not_null())
    has_core = set(ok.filter(pl.col("profile") == "core").select("dataset", "tool").unique().iter_rows())
    ok = ok.filter((pl.col("profile") == "core") | ~pl.struct("dataset", "tool").map_elements(
        lambda r: (r["dataset"], r["tool"]) in has_core, return_dtype=pl.Boolean))
    out = {}
    for (ds, tool, th), g in ok.group_by(["dataset", "tool", "threads"], maintain_order=True):
        core = g.filter(pl.col("profile") == "core")
        use = core if core.height else g
        mem = core["core_peak_rss_gb"].drop_nulls().to_numpy() if core.height else np.array([])
        dmem = core["core_peak_delta_gb"].drop_nulls().to_numpy() if core.height else np.array([])
        out[(ds, tool, int(th))] = {"t": use["core_seconds"].to_numpy(), "mem": mem if len(mem) else None,
                                    "dmem": dmem if len(dmem) else None, "src": "core" if core.height else "full"}
    return out


def main_thread_sets(res: Path, samp: dict, ncpu: int | None) -> dict:
    """{tool: thread counts of the MAIN runs} - the only counts the headline may use.

    Read from plan.json (``args.main_threads`` for CREST / scanpy, ``args.ooc_threads`` for
    out-of-core CREST, defaulting to the main set), clamped to the core count exactly as
    run_all.plan() does. The thread-scan counts (--threads-scan on --thread-dataset) are
    not in it, so every dataset's headline answers the same question. Without plan.json:
    the thread counts present on every dataset the tool was run on."""
    plan = res / "plan.json"
    if plan.exists():
        args = json.loads(plan.read_text()).get("args", {})
        n = ncpu or 1
        main = sorted({min(int(q), n) for q in (args.get("main_threads") or [n])})
        ooc = sorted({min(int(q), n) for q in (args.get("ooc_threads") or main)})
        return {"crest": set(main), "scanpy": set(main), "crest-ooc": set(ooc)}
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


def main_table(df: pl.DataFrame, ncpu: int | None = None, main_sets: dict | None = None) -> pl.DataFrame:
    """Headline: core workflow, CREST vs scanpy.

    Headline candidates = the --main-threads set (``main_sets``), identical for every
    dataset; thread-scan counts appear only in the thread-scaling table (and, for
    information, in the per-thread time columns and ``{tool}_best_threads_any``). The
    speed-up is computed at EVERY main thread count both tools were measured at (matched)
    and best vs best (each tool at its fastest main thread count). The headline is the
    minimum of all of these, and the Mann-Whitney test and Hodges-Lehmann shift are
    computed on the two samples behind the headline."""
    samp = _core_samples(df)
    main_sets = main_sets or {t: None for t in TOOLS}
    rows = []
    datasets = df.group_by("dataset").agg(pl.col("n_cells").max()).sort("n_cells")
    for ds, n in datasets.iter_rows():
        row = {"dataset": ds, "n_cells": int(n or 0)}
        th = {}
        for t in TOOLS:
            fail = df.filter((pl.col("dataset") == ds) & (pl.col("tool") == t) & (pl.col("status") != "ok")).height
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


def not_faster_table(steps: pl.DataFrame, threshold: float = 1.2) -> pl.DataFrame:
    """Every (dataset, step) where CREST is not clearly faster than scanpy."""
    rows = []
    for (ds, st), g in steps.group_by(["dataset", "step"], maintain_order=True):
        c = g.filter(pl.col("tool") == "crest")["median_s"].to_numpy()
        s = g.filter(pl.col("tool") == "scanpy")["median_s"].to_numpy()
        if len(c) and len(s) and c[0] > 0.005 and s[0] / c[0] < threshold:
            rows.append({"dataset": ds, "step": st, "crest_s": float(c[0]), "scanpy_s": float(s[0]),
                         "speedup": float(s[0] / c[0])})
    return pl.DataFrame(rows).sort(["speedup"]) if rows else pl.DataFrame()


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
    ok = df.filter((pl.col("status") == "ok") & (pl.col("profile") == "core"))
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


def fig_totals(df, figs):
    plt = _style()
    ok = df.filter((pl.col("status") == "ok") & pl.col("core_seconds").is_not_null())
    ok = ok.filter(pl.col("threads") == ok["threads"].max())
    # prefer core-profile runs; fall back to the core steps of full runs
    ok = ok.filter((pl.col("profile") == "core") | ~pl.struct("dataset", "tool").is_in(
        ok.filter(pl.col("profile") == "core").select(pl.struct("dataset", "tool")).to_series().implode()))
    ds = ok.group_by("dataset").agg(pl.col("n_cells").max()).sort("n_cells")["dataset"].to_list()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, lab in ((axes[0], "core_seconds", "core workflow time (s, log)"),
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
    floor = 0.01  # lazy steps (CREST normalize/log1p: recorded, applied inside later kernels) take ~0 s
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(names) + 1))
    h = 0.8 / len(TOOLS)
    for k, t in enumerate(TOOLS):
        for i, n in enumerate(names):
            v = s.filter((pl.col("tool") == t) & (pl.col("step") == n))["median_s"].to_numpy()
            if len(v):
                y = i + (k - 1) * h
                ax.barh(y, max(v[0], floor), h * 0.9, color=COLORS[t], label=LABELS[t] if i == 0 else None, zorder=2)
                ax.text(max(v[0], floor) * 1.08, y, f"{v[0]:.2g} s" if v[0] >= floor else "< 0.01 s (lazy)",
                        va="center", fontsize=6, color=MUTED)
    ax.set_xlim(left=floor)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(f"median step time (s, log) - {dataset}")
    ax.legend(loc="lower right")
    _save(fig, figs, f"fig2_steps_{dataset}")
    plt.close(fig)


def fig_scaling(df, figs, main_sets: dict | None = None) -> list[dict]:
    """Core-workflow time vs cells over the synthetic series only (one generator and one
    source dataset, so only the cell count changes), each tool at its best main (--main-threads) thread
    count per dataset, core-profile runs preferred (as in the main table)."""
    plt = _style()
    samp = _core_samples(df)
    ncell = dict(df.group_by("dataset").agg(pl.col("n_cells").max()).iter_rows())
    synth = sorted({k[0] for k in samp if k[0].startswith("synth_")}, key=lambda d: ncell.get(d) or 0)
    fits = []
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for t in TOOLS:
        pts = []
        for ds in synth:
            best, _ = _best_threads(samp, ds, t, (main_sets or {}).get(t))
            if best is not None and ncell.get(ds):
                pts.append((ncell[ds], float(np.median(samp[(ds, t, best)]["t"])), best, ds))
        if len(pts) < 2:
            continue
        n, y = np.array([q[0] for q in pts], float), np.array([q[1] for q in pts])
        ax.plot(n, y, "o-", color=COLORS[t], lw=2, ms=6, label=LABELS[t], zorder=3)
        if len(pts) >= 3:
            f = loglog_fit(n, y)
            fits.append({"tool": t, **f, "datasets": " ".join(q[3] for q in pts),
                         "best_threads": " ".join(str(q[2]) for q in pts)})
            ax.annotate(f"α = {f['alpha']:.2f}", (n[-1], y[-1]), textcoords="offset points", xytext=(6, 0),
                        color=INK, fontsize=8, va="center")
    if not ax.lines:
        plt.close(fig)
        return fits
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells (log), synthetic series")
    ax.set_ylabel("core workflow time at best threads (s, log)")
    ax.legend()
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


def compact_main(t: pl.DataFrame) -> pl.DataFrame:
    """Readable headline: core workflow time per tool, speedups with min-max ranges, memory."""
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
            row["matched (per thread count)"] = "; ".join(f"{p}: {rng(r, f'speedup_t{p}')}" for p in common)
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


def main():
    res = Path(sys.argv[1])
    raw, tables, figs = res / "raw", res / "tables", res / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = load_runs(raw)
    if not df.height:
        print("no runs found")
        return
    mach = _machine(raw)
    df.write_csv(tables / "runs.csv")
    main_sets = main_thread_sets(res, _core_samples(df), mach["logical"])
    main_t = main_table(df, mach["logical"], main_sets)
    steps = step_table(df)
    th = thread_table(df, mach["physical"])
    attempts = load_attempts(res)
    for name, t, cap in (("main", main_t, "Wall time and peak memory, CREST vs scanpy"),
                         ("steps", steps, "Per-step median time, parallelism and peak memory"),
                         ("threads", th, "Strong scaling")):
        if t.height:
            t.write_csv(tables / f"{name}.csv")
            to_latex(t, tables / f"{name}.tex", cap)
    fig_totals(df, figs)
    for ds in df["dataset"].unique().to_list():
        fig_steps(steps, figs, ds)
    fits = fig_scaling(df, figs, main_sets)
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
    slow = not_faster_table(steps)
    for name, t in (("modules", mods_t), ("not_faster", slow)):
        if t.height:
            t.write_csv(tables / f"{name}.csv")
    with open(res / "report.md", "w") as fh:
        fh.write(f"# CREST benchmark report\n\n{env}\n\n## Core workflow: CREST vs scanpy\n\n")
        fh.write("Core workflow = read, QC filter, normalize + log1p, HVG, scale + PCA, neighbours, Leiden, UMAP, "
                 "t-test and Wilcoxon marker genes (the sum of these steps; interpreter start-up and JIT warm-up "
                 "excluded). Headline candidates = the --main-threads set, identical for every dataset; "
                 "thread-scan counts appear only in the thread-scaling table. Times are medians [min-max] over "
                 "repeats at the highest main thread count; *best* is each tool's fastest main thread count. Speedup = scanpy / CREST, given at every thread count both "
                 "tools were measured at (*matched*) and *best vs best*, each with its min-max envelope "
                 "[min(scanpy)/max(CREST), max(scanpy)/min(CREST)] (no bootstrap CIs: with 5 repeats they are not "
                 "meaningful). The **headline** is the minimum of all of these; Mann-Whitney p and the "
                 "Hodges-Lehmann shift (scanpy - CREST, seconds) are computed on the two samples behind the "
                 "headline (*basis*). Memory ratio = scanpy / CREST peak RSS of the core workflow on the same basis, "
                 "absolute and minus each process's RSS at the start of the timed run (interpreter + libraries; "
                 "the warm-up's objects are released before it is taken).\n\n")
        fh.write(md(compact_main(main_t)) if main_t.height else "_(no data)_\n")
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
        fig = lambda name, alt: f"\n![{alt}](figures/{name}.png)\n" if (figs / f"{name}.png").exists() else ""  # noqa: E731
        fh.write(fig("fig1_time_memory", "time and memory") + "\n## Where CREST is not clearly faster (< 1.2x)\n\n")
        fh.write(md(slow) if slow.height else "none\n")
        fh.write("\n## Optional modules\n\nFrom full-profile runs at the highest thread count; scanpy has no pseudobulk "
                 "DESeq2 (see the DESeq2 module benchmark for R and pydeseq2).\n\n")
        fh.write(md(mods_t) if mods_t.height else "_(no full-profile runs)_\n")
        fh.write("\n## Scaling with cells (core workflow)\n\n")
        if fits:
            fh.write("Log-log OLS fit of core-workflow time on cell count over the **synthetic series only** "
                     "(one generator and source, so only the cell count varies), each tool at its best main "
                     "thread count per dataset (`best_threads`). `alpha` is the exponent (time ∝ cells^alpha) with "
                     "its 95% t-interval; `n_points` is the number of datasets in the fit. With few points the "
                     "interval is wide: read alpha together with it.\n\n")
            fh.write(md(pl.DataFrame(fits)))
        else:
            fh.write("_(needs >= 3 datasets of the synthetic series)_\n")
        fh.write(fig("fig3_scaling", "scaling") + "\n## Thread scaling (core workflow)\n\n")
        if th.height:
            fh.write("Speedup over 1 thread. Amdahl's serial fraction is fitted only up to each tool's peak speedup "
                     "and the physical core count (`amdahl_fit_max_threads`); `amdahl_residual` = observed - fitted "
                     "speedup (also shown for points outside the fit). Where the speedup falls more than 3% below "
                     "its peak at higher thread counts, no serial fraction is given (`scaling` says where it "
                     "regresses).\n\n")
        fh.write(md(th))
        fh.write(fig("fig4_thread_scaling", "threads") + "\n## Per-step breakdown\n\n" + md(steps))
        for ds in df["dataset"].unique().to_list():
            fh.write(fig(f"fig2_steps_{ds}", f"steps {ds}") + fig(f"fig5_timeline_{ds}", f"timeline {ds}"))
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
            real = attempts.filter(pl.col("status") != "skipped")
            failed = real.filter(pl.col("status") != "ok")
            ok_tags = set(real.filter(pl.col("status") == "ok")["tag"].to_list())
            fh.write(f"{real.height} attempts in `attempts.jsonl` ({real.height - failed.height} ok, {failed.height} "
                     f"failed; {attempts.height - real.height} skipped on resume because already ok).\n\n")
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
