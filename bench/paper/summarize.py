"""Statistics, tables and figures from a benchmark results directory.

    python bench/paper/summarize.py RESULTS_DIR

Reads RESULTS_DIR/raw/*.json (+ timelines, accuracy.json, env/) and writes
RESULTS_DIR/tables/*.csv|.tex, RESULTS_DIR/figures/*.pdf|.png and
RESULTS_DIR/report.md.

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
        row["energy_j"] = sum(s.get("energy_j", 0) for s in r.get("steps", {}).values()) or None
        for step, s in (r.get("steps") or {}).items():
            row[f"step:{step}"] = s["seconds"]
            row[f"par:{step}"] = s.get("parallelism")
            row[f"mem:{step}"] = s.get("peak_rss_gb")
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
    """Runs of the main comparison: the richest profile, all cores."""
    prof = "full" if "full" in df["profile"].drop_nulls().to_list() else "core"
    d = df.filter(pl.col("profile") == prof)
    return d.filter(pl.col("threads") == d["threads"].max())


def main_table(df: pl.DataFrame) -> pl.DataFrame:
    ok = _main_runs(df).filter(pl.col("status") == "ok")
    df = _main_runs(df)
    all_threads = ok
    out = []
    for ds in all_threads["dataset"].unique(maintain_order=True).to_list():
        sub = all_threads.filter(pl.col("dataset") == ds)
        row = {"dataset": ds, "n_cells": int(sub["n_cells"].drop_nulls().max() or 0)}
        for t in TOOLS:
            x = sub.filter(pl.col("tool") == t)
            fail = df.filter((pl.col("dataset") == ds) & (pl.col("tool") == t) & (pl.col("status") != "ok")).height
            if x.height:
                tt, mm = x["total_seconds"].to_numpy(), x["peak_rss_gb"].to_numpy()
                row.update({f"{t}_n": x.height, f"{t}_time_median": float(np.median(tt)), f"{t}_time_iqr": iqr(tt),
                            f"{t}_time_cv": float(np.std(tt, ddof=1) / np.mean(tt)) if len(tt) > 1 else None,
                            f"{t}_mem_median": float(np.median(mm)), f"{t}_mem_iqr": iqr(mm)})
            row[f"{t}_failed"] = fail
        a = sub.filter(pl.col("tool") == "crest")["total_seconds"].to_numpy()
        b = sub.filter(pl.col("tool") == "scanpy")["total_seconds"].to_numpy()
        if len(a) and len(b):
            s, lo, hi = boot_ratio(a, b)
            row.update({"speedup": s, "speedup_ci_lo": lo, "speedup_ci_hi": hi,
                        "mannwhitney_p": mann_whitney(a, b), "hodges_lehmann_s": hodges_lehmann(a, b)})
            ma = sub.filter(pl.col("tool") == "crest")["peak_rss_gb"].to_numpy()
            mb = sub.filter(pl.col("tool") == "scanpy")["peak_rss_gb"].to_numpy()
            m, mlo, mhi = boot_ratio(ma, mb)
            row.update({"memory_ratio": m, "memory_ratio_ci_lo": mlo, "memory_ratio_ci_hi": mhi})
        out.append(row)
    return pl.DataFrame(out, infer_schema_length=None).sort("n_cells") if out else pl.DataFrame()


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
        med = g.group_by("threads").agg(pl.col("total_seconds").median()).sort("threads")
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
    ok = _main_runs(df).filter(pl.col("status") == "ok")
    ds = ok.group_by("dataset").agg(pl.col("n_cells").max()).sort("n_cells")["dataset"].to_list()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, lab in ((axes[0], "total_seconds", "wall time (s, log)"), (axes[1], "peak_rss_gb", "peak memory (GB, log)")):
        w = 0.8 / len(TOOLS)
        for k, t in enumerate(TOOLS):
            for i, d in enumerate(ds):
                v = ok.filter((pl.col("dataset") == d) & (pl.col("tool") == t))[col].to_numpy()
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
    ok = _main_runs(df).filter(pl.col("status") == "ok")
    fits = []
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for t in TOOLS:
        g = ok.filter(pl.col("tool") == t).group_by("dataset").agg(pl.col("n_cells").max(), pl.col("total_seconds").median())
        g = g.filter(pl.col("dataset").str.starts_with("synth_") | (pl.col("dataset") == "pbmc68k")).sort("n_cells")
        if g.height < 2:
            continue
        n, y = g["n_cells"].to_numpy(), g["total_seconds"].to_numpy()
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
    ax.set_ylabel("wall time (s, log)")
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
    """Readable summary: median [IQR] per tool, speedup and memory ratio with CIs."""
    rows = []
    for r in t.iter_rows(named=True):
        row = {"dataset": r["dataset"], "cells": r["n_cells"]}
        for tool in TOOLS:
            if r.get(f"{tool}_time_median") is not None:
                row[f"{LABELS[tool]} time (s)"] = f"{r[f'{tool}_time_median']:.3g} [{r[f'{tool}_time_iqr']:.2g}]"
                row[f"{LABELS[tool]} memory (GB)"] = f"{r[f'{tool}_mem_median']:.3g}"
            elif r.get(f"{tool}_failed"):
                row[f"{LABELS[tool]} time (s)"] = "failed"
        if r.get("speedup") is not None:
            row["speedup (95% CI)"] = f"{r['speedup']:.2f} ({r['speedup_ci_lo']:.2f}-{r['speedup_ci_hi']:.2f})"
            row["memory ratio (95% CI)"] = f"{r['memory_ratio']:.2f} ({r['memory_ratio_ci_lo']:.2f}-{r['memory_ratio_ci_hi']:.2f})"
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
    with open(res / "report.md", "w") as fh:
        fh.write(f"# CREST benchmark report\n\n{env}\n\n## Wall time and memory\n\n")
        fh.write("Speedup = median scanpy time / median CREST time, 95% bootstrap CI; memory ratio likewise "
                 "(scanpy / CREST). Mann-Whitney U two-sided p-value; Hodges-Lehmann shift in seconds.\n\n")
        fh.write(md(compact_main(main_t)) if main_t.height else "_(no data)_\n")
        fh.write("\nMedian [IQR] over repeats; full statistics in `tables/main.csv`.\n")
        fig = lambda name, alt: f"\n![{alt}](figures/{name}.png)\n" if (figs / f"{name}.png").exists() else ""  # noqa: E731
        fh.write(fig("fig1_time_memory", "time and memory") + "\n## Scaling with cells\n\n")
        fh.write(md(pl.DataFrame(fits)) if fits else "_(needs >= 3 dataset sizes of the synthetic series)_\n")
        fh.write(fig("fig3_scaling", "scaling") + "\n## Thread scaling\n\n" + md(th))
        fh.write(fig("fig4_thread_scaling", "threads") + "\n## Per-step breakdown\n\n" + md(steps))
        for ds in df["dataset"].unique().to_list():
            fh.write(fig(f"fig2_steps_{ds}", f"steps {ds}") + fig(f"fig5_timeline_{ds}", f"timeline {ds}"))
        fh.write("\n## Agreement with scanpy\n\n" + (md(transpose_metrics(acc_df)) if acc_df.height else "_(no data)_\n"))
        for m in mods:
            fh.write(f"\n## Module: {m.stem}\n\n" + m.read_text())
        fails = df.filter(pl.col("status") != "ok").select("tool", "dataset", "threads", "repeat", "status")
        fh.write("\n## Failed runs\n\n" + (md(fails) if fails.height else "none\n"))
    print(f"report: {res / 'report.md'}")


if __name__ == "__main__":
    main()
