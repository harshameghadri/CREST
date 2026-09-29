"""Statistics, tables and figures from a benchmark results directory.

    python bench/paper/summarize.py RESULTS_DIR

Reads RESULTS_DIR/raw/*.json (+ timelines, accuracy.json, env/) and writes
RESULTS_DIR/tables/*.csv|.tex, RESULTS_DIR/figures/*.pdf|.png and
RESULTS_DIR/report.md.

Reporting rules (CLAUDE.md, "honest benchmarks"):
  * the headline compares the CORE workflow (read ... Wilcoxon); optional modules
    (seurat_v3 HVG, Scrublet, resolution sweep, Harmony, pseudobulk DESeq2) are a
    separate table;
  * each speedup is given at matched thread counts and at each tool's best measured
    thread count; the headline is the smaller of the two, and a row is flagged when
    only one thread count was measured (the matched number can then be an upper bound);
  * a table lists every step where CREST is not clearly faster (< 1.2x);
  * failed runs are listed.

Statistics (per dataset, tool pair CREST vs scanpy, n = repeats):
  median, IQR and coefficient of variation of wall time and peak RSS;
  speedup = median(scanpy) / median(CREST) with a 95% percentile bootstrap CI
  (10,000 resamples of each tool's repeats); two-sided Mann-Whitney U test and
  the Hodges-Lehmann estimate of the time difference; memory ratio likewise.
  Scaling: OLS fit of log time on log cells over the synthetic series
  (exponent alpha with a 95% t-interval). Thread scaling: speedup T1/Tp,
  parallel efficiency, and the serial fraction f of Amdahl's law
  (least squares on 1/S = f + (1 - f)/p).
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
                                     "n_cells", "total_seconds", "peak_rss_gb")}
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
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# --------------------------------------------------------------------------- statistics
def iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    return q3 - q1


def boot_ratio(a, b, n=10_000, seed=0):
    """CI of median(b) / median(a), resampling each group independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.median(rng.choice(a, (n, len(a))), axis=1)
    rb = np.median(rng.choice(b, (n, len(b))), axis=1)
    r = rb / ra
    return float(np.median(b) / np.median(a)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


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
    return {"alpha": res.slope, "alpha_lo": res.slope - half, "alpha_hi": res.slope + half,
            "intercept": res.intercept, "r2": res.rvalue ** 2}


def amdahl(p, s):
    p, s = np.asarray(p, float), np.asarray(s, float)
    # 1/S = f + (1-f)/p  ->  1/S - 1/p = f (1 - 1/p)
    x, y = 1 - 1 / p, 1 / s - 1 / p
    f = float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")
    return min(max(f, 0.0), 1.0)


def _main_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Runs used for the per-step breakdown: the richest profile, all cores."""
    prof = "full" if "full" in df["profile"].drop_nulls().to_list() else "core"
    d = df.filter(pl.col("profile") == prof)
    return d.filter(pl.col("threads") == d["threads"].max())


def _core_samples(df: pl.DataFrame) -> dict:
    """{(dataset, tool, threads): (core times, core memory or None, source profile)}.

    Core-profile runs are used where they exist; otherwise the core steps of full-profile
    runs (time only: their memory includes the optional modules)."""
    ok = df.filter((pl.col("status") == "ok") & pl.col("core_seconds").is_not_null())
    out = {}
    for (ds, tool, th), g in ok.group_by(["dataset", "tool", "threads"], maintain_order=True):
        core = g.filter(pl.col("profile") == "core")
        use = core if core.height else g
        mem = core["core_peak_rss_gb"].drop_nulls().to_numpy() if core.height else None
        out[(ds, tool, int(th))] = (use["core_seconds"].to_numpy(), mem if mem is not None and len(mem) else None,
                                    "core" if core.height else "full")
    return out


def main_table(df: pl.DataFrame) -> pl.DataFrame:
    """Headline: core workflow, CREST vs scanpy, matched and best thread counts."""
    samp = _core_samples(df)
    rows = []
    datasets = df.group_by("dataset").agg(pl.col("n_cells").max()).sort("n_cells")
    for ds, n in datasets.iter_rows():
        row = {"dataset": ds, "n_cells": int(n or 0)}
        th = {t: sorted(k[2] for k in samp if k[0] == ds and k[1] == t) for t in TOOLS}
        for t in TOOLS:
            fail = df.filter((pl.col("dataset") == ds) & (pl.col("tool") == t) & (pl.col("status") != "ok")).height
            row[f"{t}_failed"] = fail
            if not th[t]:
                continue
            med = {p: float(np.median(samp[(ds, t, p)][0])) for p in th[t]}
            best = min(med, key=med.get)
            top = max(th[t])
            x, mem, src = samp[(ds, t, top)]
            row.update({f"{t}_n": len(x), f"{t}_threads_measured": " ".join(map(str, th[t])),
                        f"{t}_time_median": med[top], f"{t}_time_iqr": iqr(x),
                        f"{t}_best_threads": best, f"{t}_best_time": med[best],
                        f"{t}_mem_median": float(np.median(mem)) if mem is not None else None, f"{t}_source": src})
        common = sorted(set(th["crest"]) & set(th["scanpy"]))
        if not common:
            rows.append(row)
            continue
        pm = max(common)
        a, b = samp[(ds, "crest", pm)][0], samp[(ds, "scanpy", pm)][0]
        s, lo, hi = boot_ratio(a, b)
        row.update({"matched_threads": pm, "speedup_matched": s, "speedup_matched_lo": lo, "speedup_matched_hi": hi,
                    "mannwhitney_p": mann_whitney(a, b), "hodges_lehmann_s": hodges_lehmann(a, b)})
        ba = samp[(ds, "crest", row["crest_best_threads"])][0]
        bb = samp[(ds, "scanpy", row["scanpy_best_threads"])][0]
        s2, lo2, hi2 = boot_ratio(ba, bb)
        row.update({"speedup_best": s2, "speedup_best_lo": lo2, "speedup_best_hi": hi2})
        if s2 <= s:
            row.update({"speedup": s2, "speedup_ci_lo": lo2, "speedup_ci_hi": hi2, "speedup_basis": "best vs best"})
        else:
            row.update({"speedup": s, "speedup_ci_lo": lo, "speedup_ci_hi": hi, "speedup_basis": f"matched ({pm} threads)"})
        row["single_thread_count"] = len(th["crest"]) < 2 or len(th["scanpy"]) < 2
        ma, mb = samp[(ds, "crest", pm)][1], samp[(ds, "scanpy", pm)][1]
        if ma is not None and mb is not None:
            m, mlo, mhi = boot_ratio(ma, mb)
            row.update({"memory_ratio": m, "memory_ratio_ci_lo": mlo, "memory_ratio_ci_hi": mhi})
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


def thread_table(df: pl.DataFrame) -> pl.DataFrame:
    ok = df.filter((pl.col("status") == "ok") & (pl.col("profile") == "core"))
    rows = []
    for (ds, tool), g in ok.group_by(["dataset", "tool"], maintain_order=True):
        med = g.group_by("threads").agg(pl.col("core_seconds").median().alias("total_seconds")).sort("threads")
        if med.height < 2 or 1 not in med["threads"].to_list():
            continue
        t1 = med.filter(pl.col("threads") == 1)["total_seconds"][0]
        p, t = med["threads"].to_numpy(), med["total_seconds"].to_numpy()
        f = amdahl(p, t1 / t)
        for pi, ti in zip(p, t):
            rows.append({"dataset": ds, "tool": tool, "threads": int(pi), "median_s": float(ti), "speedup": t1 / ti,
                         "efficiency": t1 / ti / pi, "amdahl_serial_fraction": f})
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


def fig_scaling(df, figs) -> list[dict]:
    plt = _style()
    ok = df.filter((pl.col("status") == "ok") & pl.col("core_seconds").is_not_null())
    ok = ok.filter(pl.col("threads") == ok["threads"].max())
    fits = []
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for t in TOOLS:
        g = ok.filter(pl.col("tool") == t).group_by("dataset").agg(pl.col("n_cells").max(), pl.col("core_seconds").median())
        g = g.filter(pl.col("dataset").str.starts_with("synth_") | (pl.col("dataset") == "pbmc68k")).sort("n_cells")
        if g.height < 2:
            continue
        n, y = g["n_cells"].to_numpy(), g["core_seconds"].to_numpy()
        ax.plot(n, y, "o-", color=COLORS[t], lw=2, ms=6, label=LABELS[t], zorder=3)
        if g.height >= 3:
            f = loglog_fit(n, y)
            fits.append({"tool": t, **f})
            ax.annotate(f"α = {f['alpha']:.2f}", (n[-1], y[-1]), textcoords="offset points", xytext=(6, 0),
                        color=INK, fontsize=8, va="center")
    if not fits and not ax.lines:
        plt.close(fig)
        return fits
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells (log)")
    ax.set_ylabel("core workflow time (s, log)")
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
            ax.plot(g["threads"], g["speedup"], "o-", color=COLORS[t], lw=2, ms=6,
                    label=f"{LABELS[t]} (serial f = {g['amdahl_serial_fraction'][0]:.2f})")
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
    """Readable headline: core workflow time per tool, speedups with CIs, memory ratio."""
    rows = []
    for r in t.iter_rows(named=True):
        row = {"dataset": r["dataset"], "cells": r["n_cells"]}
        for tool in TOOLS:
            if r.get(f"{tool}_time_median") is not None:
                row[f"{LABELS[tool]} (s)"] = f"{r[f'{tool}_time_median']:.3g} [{r[f'{tool}_time_iqr']:.2g}]"
                row[f"{LABELS[tool]} best (s @ threads)"] = f"{r[f'{tool}_best_time']:.3g} @ {r[f'{tool}_best_threads']}"
            elif r.get(f"{tool}_failed"):
                row[f"{LABELS[tool]} (s)"] = "failed"
        if r.get("speedup") is not None:
            row["matched"] = f"{r['speedup_matched']:.2f} ({r['speedup_matched_lo']:.2f}-{r['speedup_matched_hi']:.2f})"
            row["best vs best"] = f"{r['speedup_best']:.2f} ({r['speedup_best_lo']:.2f}-{r['speedup_best_hi']:.2f})"
            row["headline speedup"] = f"**{r['speedup']:.2f}**" + (" †" if r.get("single_thread_count") else "")
            row["memory ratio"] = (f"{r['memory_ratio']:.2f} ({r['memory_ratio_ci_lo']:.2f}-{r['memory_ratio_ci_hi']:.2f})"
                                   if r.get("memory_ratio") is not None else "n/a ‡")
            row["Mann-Whitney p"] = None if r["mannwhitney_p"] is None else f"{r['mannwhitney_p']:.2g}"
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


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


def main():
    res = Path(sys.argv[1])
    raw, tables, figs = res / "raw", res / "tables", res / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = load_runs(raw)
    if not df.height:
        print("no runs found")
        return
    df.write_csv(tables / "runs.csv")
    main_t = main_table(df)
    steps = step_table(df)
    th = thread_table(df)
    for name, t, cap in (("main", main_t, "Wall time and peak memory, CREST vs scanpy"),
                         ("steps", steps, "Per-step median time, parallelism and peak memory"),
                         ("threads", th, "Strong scaling")):
        if t.height:
            t.write_csv(tables / f"{name}.csv")
            to_latex(t, tables / f"{name}.tex", cap)
    fig_totals(df, figs)
    for ds in df["dataset"].unique().to_list():
        fig_steps(steps, figs, ds)
    fits = fig_scaling(df, figs)
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
                 "excluded). Times are medians [IQR] over repeats at the highest thread count; *best* is each tool's "
                 "fastest measured thread count. Speedup = scanpy / CREST with a 95% bootstrap CI, at matched "
                 "threads and best vs best; the **headline** is the smaller of the two.\n\n")
        fh.write(md(compact_main(main_t)) if main_t.height else "_(no data)_\n")
        if main_t.height and "single_thread_count" in main_t.columns and main_t["single_thread_count"].any():
            fh.write("\n† only one thread count was measured for this dataset, so *best vs best* equals *matched*. "
                     "Where a thread scan exists, scanpy is typically slower at high thread counts than at its "
                     "best, so this headline can overstate CREST's advantage (see *Thread scaling*).\n")
        if main_t.height and "speedup" in main_t.columns and (
                "memory_ratio" not in main_t.columns
                or main_t.filter(pl.col("speedup").is_not_null() & pl.col("memory_ratio").is_null()).height):
            fh.write("\n‡ memory not reported: these times come from full-profile runs, whose peak memory includes "
                     "the optional modules. Clean core-workflow memory needs core-profile runs.\n")
        fh.write("\nFull statistics (Hodges-Lehmann shifts, CV, failures) in `tables/main.csv`.\n")
        fig = lambda name, alt: f"\n![{alt}](figures/{name}.png)\n" if (figs / f"{name}.png").exists() else ""  # noqa: E731
        fh.write(fig("fig1_time_memory", "time and memory") + "\n## Where CREST is not clearly faster (< 1.2x)\n\n")
        fh.write(md(slow) if slow.height else "none\n")
        fh.write("\n## Optional modules\n\nFrom full-profile runs at the highest thread count; scanpy has no pseudobulk "
                 "DESeq2 (see the DESeq2 module benchmark for R and pydeseq2).\n\n")
        fh.write(md(mods_t) if mods_t.height else "_(no full-profile runs)_\n")
        fh.write("\n## Scaling with cells (core workflow)\n\n")
        fh.write(md(pl.DataFrame(fits)) if fits else "_(needs >= 3 dataset sizes of the synthetic series)_\n")
        fh.write(fig("fig3_scaling", "scaling") + "\n## Thread scaling (core workflow)\n\n" + md(th))
        fh.write(fig("fig4_thread_scaling", "threads") + "\n## Per-step breakdown\n\n" + md(steps))
        for ds in df["dataset"].unique().to_list():
            fh.write(fig(f"fig2_steps_{ds}", f"steps {ds}") + fig(f"fig5_timeline_{ds}", f"timeline {ds}"))
        fh.write("\n## Agreement with scanpy\n\n" + (md(transpose_metrics(acc_df)) if acc_df.height else "_(no data)_\n"))
        for m in mods:
            fh.write(f"\n## Module: {m.stem}\n\n" + m.read_text())
        if "skipped_steps" in df.columns:
            sk = (df.filter(pl.col("skipped_steps").is_not_null())
                  .group_by("dataset", "tool", "skipped_steps").agg(pl.len().alias("runs")).sort("dataset", "tool"))
            fh.write("\n## Skipped steps\n\n" + (md(sk) if sk.height else "none\n"))
        fails = df.filter(pl.col("status") != "ok").select("tool", "dataset", "profile", "threads", "repeat", "status")
        fh.write("\n## Failed runs\n\n" + (md(fails) if fails.height else "none\n"))
    print(f"report: {res / 'report.md'}")


if __name__ == "__main__":
    main()
