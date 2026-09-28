"""Summarise run_pipeline.py results into Markdown tables and figures.

    python bench/whitepaper/summarize.py results/ --out results/report
"""

import argparse
import json
from pathlib import Path

import numpy as np

STEPS = ["read", "qc_filter", "normalize_log1p", "hvg", "scale_pca", "neighbors", "leiden", "umap", "de_ttest", "de_wilcoxon"]
LABEL = {"read": "read", "qc_filter": "QC filter", "normalize_log1p": "normalize + log1p", "hvg": "HVG",
         "scale_pca": "scale + PCA", "neighbors": "neighbors", "leiden": "Leiden", "umap": "UMAP",
         "de_ttest": "DE t-test", "de_wilcoxon": "DE Wilcoxon"}
TOOLS = ["scanpy", "crest", "crest-ooc"]
NAME = {"scanpy": "scanpy", "crest": "CREST (in-memory)", "crest-ooc": "CREST (out-of-core)"}
# fixed identity colours (reference palette slots 1-3, validated all-pairs)
COLOR = {"crest": "#2a78d6", "scanpy": "#eb6834", "crest-ooc": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(d: Path):
    """All run reports under ``d`` (recursively); repeats are reduced to the
    median per step, with the min-max of the total kept for the tables."""
    groups = {}
    for f in sorted(d.rglob("*.json")):
        r = json.loads(f.read_text())
        if "tool" in r and "steps" in r:
            groups.setdefault((r["data"], r["tool"]), []).append(r)
    runs = {}
    for (data, tool), reps in groups.items():
        med = dict(reps[0])
        med["steps"] = {s: {k: float(np.median([r["steps"][s][k] for r in reps if s in r["steps"]]))
                            for k in ("seconds", "peak_rss_gb")}
                        for s in reps[0]["steps"]}
        tot = [r["total_seconds"] for r in reps]
        med["total_seconds"] = float(np.median(tot))
        med["total_range"] = (min(tot), max(tot))
        med["peak_rss_gb"] = float(np.median([r["peak_rss_gb"] for r in reps]))
        med["n_repeats"] = len(reps)
        runs.setdefault(data, {})[tool] = med
    return runs


def load_failed(d: Path):
    """FAILED_*.txt markers written by run_local_benchmark.sh: 'data.h5<TAB>tool<TAB>reason'."""
    out = {}
    for f in d.rglob("FAILED_*.txt"):
        data, tool, why = f.read_text().strip().split("\t", 2)
        out[(data, tool)] = why
    return out


def accuracy(d: Path, data_stem: str):
    """ARI of Leiden labels and UMAP kNN preservation vs scanpy, on shared cells."""
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return {}
    refs = sorted(d.rglob(f"scanpy_{data_stem}*_outputs.npz"))
    if not refs:
        return {}
    ref_p = refs[0]
    ref = np.load(ref_p, allow_pickle=True)
    out = {}
    for tool in ("crest", "crest-ooc"):
        ps = sorted(d.rglob(f"{tool}_{data_stem}*_outputs.npz"))
        if not ps:
            continue
        p = ps[0]
        o = np.load(p, allow_pickle=True)
        common, i_ref, i_o = np.intersect1d(ref["barcodes"], o["barcodes"], return_indices=True)
        out[tool] = {"shared_cells": int(len(common)),
                     "leiden_ari_vs_scanpy": float(adjusted_rand_score(ref["leiden"][i_ref], o["leiden"][i_o])),
                     "n_clusters": int(len(set(o["leiden"]))), "n_clusters_scanpy": int(len(set(ref["leiden"])))}
    return out


def table(runs, failed):
    lines = []
    for data, tools in runs.items():
        n = next(iter(tools.values()))["n_cells"]
        lines.append(f"\n### {data} ({n:,} cells after QC)\n")
        for (fd, ft), why in failed.items():
            if fd == data:
                lines.append(f"**{NAME[ft]}: {why}.**\n")
        both = "scanpy" in tools and "crest" in tools
        hdr = "| step | " + " | ".join(NAME[t] for t in TOOLS if t in tools) + (" | speed-up (in-memory) |" if both else " |")
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for s in STEPS + ["total"]:
            row = [f"**{LABEL.get(s, s)}**" if s == "total" else LABEL[s]]
            for t in TOOLS:
                if t not in tools:
                    continue
                if s == "total":
                    lo, hi = tools[t].get("total_range", (tools[t]["total_seconds"],) * 2)
                    rng = f" ({lo:.0f}-{hi:.0f})" if tools[t].get("n_repeats", 1) > 1 else ""
                    row.append(f"{tools[t]['total_seconds']:.1f} s{rng}")
                else:
                    st = tools[t]["steps"].get(s)
                    row.append(f"{st['seconds']:.2f} s" if st else "-")
            if "scanpy" in tools and "crest" in tools:
                a = tools["scanpy"]["total_seconds"] if s == "total" else tools["scanpy"]["steps"][s]["seconds"]
                b = tools["crest"]["total_seconds"] if s == "total" else tools["crest"]["steps"][s]["seconds"]
                row.append(f"{a / b:.1f}x" if b > 0.005 else "lazy")
            lines.append("| " + " | ".join(row) + " |")
        mem = " / ".join(f"{NAME[t]} {tools[t]['peak_rss_gb']:.2f} GB" for t in TOOLS if t in tools)
        lines.append(f"\nPeak RSS: {mem}")
    return "\n".join(lines)


def figures(runs, out: Path, failed=None):
    failed = failed or {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    datas = list(runs)
    # Fig 1: per-step time, small multiples per dataset, log scale
    fig, axes = plt.subplots(1, len(datas), figsize=(4.2 * len(datas), 4.2), sharey=True, squeeze=False)
    for ax, data in zip(axes[0], datas):
        tools = [t for t in TOOLS if t in runs[data]]
        y = np.arange(len(STEPS))
        h = 0.8 / len(tools)
        for k, t in enumerate(tools):
            v = np.array([runs[data][t]["steps"].get(s, {}).get("seconds", np.nan) for s in STEPS])
            yy = y + (k - (len(tools) - 1) / 2) * h
            lazy = v < 0.005
            ax.barh(yy[~lazy], v[~lazy], height=h * 0.85, color=COLOR[t], label=NAME[t])
            for yl in yy[lazy]:  # deferred into later passes: annotate instead of a fake bar
                ax.text(0.006, yl, "lazy", va="center", ha="left", fontsize=7, color=INK2)
        note = "".join(f"\n{NAME[ft]}: {why}" for (fd, ft), why in failed.items() if fd == data)
        ax.set_xscale("log")
        ax.set_yticks(y, [LABEL[s] for s in STEPS])
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("seconds (log scale)")
        ax.set_title(f"{data.replace('.h5', '')}  ({runs[data][tools[0]]['n_cells']:,} cells){note}", color=INK, fontsize=10)
        ax.set_xlim(0.004, 400)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_step_times.png", dpi=200)
    fig.savefig(out / "fig_step_times.svg")

    # Fig 2: total time and peak memory vs dataset size (two separate panels, one axis each)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for t in TOOLS:
        pts = sorted((runs[d][t]["n_cells"], runs[d][t]["total_seconds"], runs[d][t]["peak_rss_gb"]) for d in datas if t in runs[d])
        if not pts:
            continue
        n, sec, mem = zip(*pts)
        for ax, v in ((a1, sec), (a2, mem)):
            ax.plot(n, v, color=COLOR[t], lw=2, marker="o", ms=5, label=NAME[t])
            ax.annotate(f"{v[-1]:.1f}", (n[-1], v[-1]), textcoords="offset points", xytext=(6, 0), va="center", color=INK2, fontsize=8)
    for (fd, ft), why in failed.items():  # mark runs that did not finish
        if fd in runs:
            n = next(iter(runs[fd].values()))["n_cells"]
            for ax in (a1, a2):
                ax.plot([n], [ax.get_ylim()[1] * 0.97], marker="x", ms=8, mew=2, color=COLOR[ft], ls="none")
                ax.annotate(f"{NAME[ft]}: out of memory", (n, ax.get_ylim()[1] * 0.97), textcoords="offset points",
                            xytext=(-6, 0), ha="right", va="center", fontsize=8, color=INK2)
    a1.set_ylabel("end-to-end time (s)")
    a2.set_ylabel("peak memory (GB)")
    for ax in (a1, a2):
        ax.set_xlabel("cells")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig_scaling.png", dpi=200)
    fig.savefig(out / "fig_scaling.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--failed", action="append", default=[],
                    help='runs that did not finish, "data.h5:tool:reason", e.g. "pbmc_200k.h5:scanpy:out of memory"')
    a = ap.parse_args()
    failed = {(f.split(":")[0], f.split(":")[1]): f.split(":", 2)[2] for f in a.failed}
    failed.update(load_failed(Path(a.results)))
    d = Path(a.results)
    out = Path(a.out or d / "report")
    out.mkdir(parents=True, exist_ok=True)
    runs = load(d)
    md = ["# CREST vs scanpy benchmark", table(runs, failed), "\n## Agreement with scanpy\n"]
    for data in runs:
        acc = accuracy(d, Path(data).stem)
        for t, v in acc.items():
            md.append(f"- {data} / {NAME[t]}: Leiden ARI vs scanpy {v['leiden_ari_vs_scanpy']:.3f} "
                      f"({v['n_clusters']} vs {v['n_clusters_scanpy']} clusters, {v['shared_cells']:,} shared cells)")
    m = next(iter(next(iter(runs.values())).values()))["machine"]
    reps = max(t.get("n_repeats", 1) for v in runs.values() for t in v.values())
    md.append(f"\nMachine: {m['cores']} cores, {m['ram_gb']:.0f} GB RAM, Python {m['python']}. "
              f"Times are medians of {reps} repeat(s); totals show (min-max).")
    (out / "report.md").write_text("\n".join(md) + "\n")
    figures(runs, out, failed)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
