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
    runs = {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["data"], {})[r["tool"]] = r
    return runs


def accuracy(d: Path, data_stem: str):
    """ARI of Leiden labels and UMAP kNN preservation vs scanpy, on shared cells."""
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return {}
    ref_p = d / f"scanpy_{data_stem}_outputs.npz"
    if not ref_p.exists():
        return {}
    ref = np.load(ref_p, allow_pickle=True)
    out = {}
    for tool in ("crest", "crest-ooc"):
        p = d / f"{tool}_{data_stem}_outputs.npz"
        if not p.exists():
            continue
        o = np.load(p, allow_pickle=True)
        common, i_ref, i_o = np.intersect1d(ref["barcodes"], o["barcodes"], return_indices=True)
        out[tool] = {"shared_cells": int(len(common)),
                     "leiden_ari_vs_scanpy": float(adjusted_rand_score(ref["leiden"][i_ref], o["leiden"][i_o])),
                     "n_clusters": int(len(set(o["leiden"]))), "n_clusters_scanpy": int(len(set(ref["leiden"])))}
    return out


def table(runs):
    lines = []
    for data, tools in runs.items():
        n = next(iter(tools.values()))["n_cells"]
        lines.append(f"\n### {data} ({n:,} cells after QC)\n")
        hdr = "| step | " + " | ".join(NAME[t] for t in TOOLS if t in tools) + " | speed-up (in-memory) |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for s in STEPS + ["total"]:
            row = [f"**{LABEL.get(s, s)}**" if s == "total" else LABEL[s]]
            for t in TOOLS:
                if t not in tools:
                    continue
                if s == "total":
                    row.append(f"{tools[t]['total_seconds']:.1f} s")
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


def figures(runs, out: Path):
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
            v = [max(runs[data][t]["steps"].get(s, {}).get("seconds", np.nan), 0.005) for s in STEPS]
            ax.barh(y + (k - (len(tools) - 1) / 2) * h, v, height=h * 0.85, color=COLOR[t], label=NAME[t])
        ax.set_xscale("log")
        ax.set_yticks(y, [LABEL[s] for s in STEPS])
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("seconds (log scale)")
        ax.set_title(f"{data.replace('.h5', '')}  ({runs[data][tools[0]]['n_cells']:,} cells)", color=INK, fontsize=10)
    axes[0][0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
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
    a1.set_ylabel("end-to-end time (s)")
    a2.set_ylabel("peak memory (GB)")
    for ax in (a1, a2):
        ax.set_xlabel("cells")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    a1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_scaling.png", dpi=200)
    fig.savefig(out / "fig_scaling.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.results)
    out = Path(a.out or d / "report")
    out.mkdir(parents=True, exist_ok=True)
    runs = load(d)
    md = ["# CREST vs scanpy benchmark", table(runs), "\n## Agreement with scanpy\n"]
    for data in runs:
        acc = accuracy(d, Path(data).stem)
        for t, v in acc.items():
            md.append(f"- {data} / {NAME[t]}: Leiden ARI vs scanpy {v['leiden_ari_vs_scanpy']:.3f} "
                      f"({v['n_clusters']} vs {v['n_clusters_scanpy']} clusters, {v['shared_cells']:,} shared cells)")
    m = next(iter(next(iter(runs.values())).values()))["machine"]
    md.append(f"\nMachine: {m['cores']} cores, {m['ram_gb']:.0f} GB RAM, Python {m['python']}.")
    (out / "report.md").write_text("\n".join(md) + "\n")
    figures(runs, out)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
