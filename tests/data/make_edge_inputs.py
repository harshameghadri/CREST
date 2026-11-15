"""Inputs for the edge-case DESeq2 / smoother fixtures in this directory.

Run from the repository root:  python tests/data/make_edge_inputs.py [output_dir]
It writes

* deseq2_edge_848_input.csv / _meta.json      three groups of 8, 4 and 8 samples; outlier counts in groups that are
                                              replaced and in the group that is too small to be replaced
* deseq2_edge_sparse_param_*                  6 samples, ``~ batch + condition``, sparse counts; the parametric trend fits
* deseq2_edge_sparse_extreme_*                6 samples, the same design, counts spanning six orders of magnitude; the
                                              parametric trend fails in R
* deseq2_lowess_ties_cases.csv / _input.csv   random (x, y) vectors with tied x for lowess and loess

The reference output is produced by make_edge_fixtures.R (R DESeq2 / R stats); nothing written here is a result.
"""
import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent


def nb(rng, mu, disp):
    size = 1.0 / disp
    return rng.negative_binomial(size, size / (size + mu)).astype(np.int64)


def write_counts(name, cts, obs, design, contrast, **extra):
    """cts: genes x samples; obs: dict of per-sample columns."""
    n = cts.shape[1]
    cols = {"sample": [f"s{i}" for i in range(n)], **obs}
    levels = {k: sorted(set(v), key=lambda x: v.index(x)) for k, v in obs.items() if isinstance(v[0], str)}
    lines = [",".join(f"gene{i}" for i in range(cts.shape[0])) + "," + ",".join(cols)]
    for j in range(n):
        lines.append(",".join(str(int(x)) for x in cts[:, j]) + "," + ",".join(str(cols[k][j]) for k in cols))
    (OUT / f"deseq2_{name}_input.csv").write_text("\n".join(lines) + "\n")
    meta = {"design": design, "contrast": contrast, "levels": levels, **extra}
    (OUT / f"deseq2_{name}_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def edge_848():
    """Groups c0 (8), c1 (4), c2 (8). Genes 0-9 carry outliers: with min_replicates_for_replace = 7 the groups of 8 are
    replaced and the group of 4 is not; with 4 all three groups are; with 9 none is."""
    rng = np.random.default_rng(1848)
    n_genes, n_per = 500, (8, 4, 8)
    m = sum(n_per)
    cond = np.repeat(["c0", "c1", "c2"], n_per)
    mu0 = np.clip(2 ** rng.normal(6, 1.5, n_genes), 5, 1e5)
    sf = 2 ** rng.normal(0, 0.2, m)
    sf /= np.exp(np.mean(np.log(sf)))
    lfc = np.where(rng.random(n_genes) < 0.15, rng.normal(0, 1, n_genes), 0.0)
    mu = mu0[:, None] * sf[None, :] * (2 ** (lfc[:, None] * (cond == "c2")[None, :]))
    cts = nb(rng, mu, 0.05 + 2 / mu0[:, None])
    idx = {"c0": np.flatnonzero(cond == "c0"), "c1": np.flatnonzero(cond == "c1"), "c2": np.flatnonzero(cond == "c2")}
    big = lambda g, s, f: cts.__setitem__((g, s), int(cts[g].mean() * f) + 20)  # noqa: E731
    big(0, idx["c0"][2], 40)   # outlier in a group of 8
    big(1, idx["c2"][5], 40)   # outlier in the other group of 8
    big(2, idx["c1"][1], 40)   # outlier in the group of 4
    big(3, idx["c0"][0], 40); big(3, idx["c1"][3], 40)                     # one replaced, one not
    big(4, idx["c0"][7], 40); big(4, idx["c2"][1], 40)                     # both replaced
    big(5, idx["c0"][4], 40); big(5, idx["c1"][0], 40); big(5, idx["c2"][6], 40)  # all three groups
    big(6, idx["c0"][1], 12); big(7, idx["c1"][2], 12); big(8, idx["c2"][3], 12)  # moderate: near the Cook's cut-off
    big(9, idx["c0"][3], 25); big(9, idx["c0"][6], 25)                    # two outliers in one group
    write_counts("edge_848", cts, {"condition": cond.tolist()}, "~ condition", ["condition", "c2", "c0"],
                 min_replicates_for_replace=[7, 4, 9], extra_contrasts=[["condition", "c1", "c0"]])


def sparse_param():
    rng = np.random.default_rng(2602)
    n_genes, m = 3000, 6
    batch = np.repeat(["a", "b", "c"], 2)
    cond = np.tile(["x", "y"], 3)
    mu0 = 2 ** rng.normal(-0.5, 2.2, n_genes)
    mu = mu0[:, None] * (2 ** rng.normal(0, 0.4, m))[None, :]
    cts = nb(rng, mu, 0.5 / np.maximum(mu0, 0.05)[:, None] + 1.0)
    cts[:200, 0] += rng.negative_binomial(2, 2 / (2 + 300), 200)
    write_counts("edge_sparse_param", cts, {"batch": batch.tolist(), "condition": cond.tolist()}, "~ batch + condition",
                 ["condition", "y", "x"])


def sparse_extreme():
    rng = np.random.default_rng(909)
    n_genes, m = 1500, 6
    batch = np.repeat(["a", "b", "c"], 2)
    cond = np.tile(["x", "y"], 3)
    mu0 = 2 ** rng.normal(3, 2, n_genes)
    mu = mu0[:, None] * (2 ** rng.normal(0, 0.3, m))[None, :]
    cts = nb(rng, mu, 0.2)
    for g in range(100):
        cts[g] = 0
        cts[g, rng.integers(m)] = 10 ** rng.integers(2, 7)
    for g in range(100, 200):
        cts[g] = 0
        cts[g, rng.choice(m, 2, replace=False)] = (10 ** rng.uniform(0, 5, 2)).astype(np.int64)
    for g in range(200, 300):
        cts[g] = (10 ** rng.uniform(0, 6, m)).astype(np.int64) * rng.binomial(1, 0.4, m)
    for g in range(300, 400):
        cts[g] = 0
        cts[g, 4:] = (10 ** rng.uniform(0, 4, 2)).astype(np.int64)
    write_counts("edge_sparse_extreme", cts, {"batch": batch.tolist(), "condition": cond.tolist()}, "~ batch + condition",
                 ["condition", "y", "x"])


def lowess_ties(n_cases=60):
    """Random vectors with tied x (x drawn on a coarse grid), spans 0.1-0.9, 0-4 robustness iterations, default delta
    (0.01 * range) or 0 or a coarse value; outliers in y. x is sorted; ties are ordered by the draw, not by y."""
    rng = np.random.default_rng(31415)
    cases, rows = [], []
    for k in range(n_cases):
        n = int(rng.integers(8, 300))
        n_distinct = int(rng.integers(4, max(5, n // 2)))
        grid = np.sort(rng.uniform(0, 10, n_distinct))
        x = np.sort(rng.choice(grid, n, replace=True))
        y = np.sin(x) + 0.3 * rng.normal(size=n)
        out = rng.random(n) < 0.05
        y[out] += rng.normal(0, 5, out.sum())
        f = float(rng.choice([0.1, 0.2, 1 / 3, 0.5, 2 / 3, 0.9]))
        it = int(rng.choice([0, 1, 3, 4]))
        dmode = str(rng.choice(["default", "zero", "coarse"]))
        delta = {"default": 0.01 * (x[-1] - x[0]), "zero": 0.0, "coarse": 0.05 * (x[-1] - x[0])}[dmode]
        span = float(rng.choice([0.4, 0.5, 0.75, 0.9]))
        cases.append((k, n, int(len(np.unique(x))), f, it, dmode, repr(float(delta)), span))
        rows += [(k, i, repr(float(a)), repr(float(b))) for i, (a, b) in enumerate(zip(x, y))]
    (OUT / "deseq2_lowess_ties_cases.csv").write_text(
        "case,n,n_distinct,f,iter,delta_mode,delta,span\n" + "\n".join(",".join(str(v) for v in c) for c in cases) + "\n")
    (OUT / "deseq2_lowess_ties_input.csv").write_text("case,i,x,y\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")


if __name__ == "__main__":
    edge_848()
    sparse_param()
    sparse_extreme()
    lowess_ties()
