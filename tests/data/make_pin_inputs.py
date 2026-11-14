"""Simulate the count tables behind the R-pinned DESeq2 fixtures in this directory.

Run from the repository root:  python tests/data/make_pin_inputs.py [output_dir]
It writes deseq2_<name>_input.csv (samples x genes, then the sample metadata columns) and
deseq2_<name>_meta.json (design, factor levels, contrast, options). The reference output is
produced from these files by make_pin_fixtures.R (DESeq2 run in R); nothing here is a result.
"""
import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent


def nb(rng, mu, disp):
    size = 1.0 / disp
    return rng.negative_binomial(size, size / (size + mu)).astype(np.int64)


def simulate(n_genes, n_per, seed, mean_log2=(5, 2), disp=lambda mu: 4 / mu + 0.1, sf_sd=0.25, sd_disp=0.0,
             de_frac=0.15, lfc_sd=1.0):
    """Negative-binomial counts (genes x samples) for len(n_per) groups; the last group carries the effect."""
    rng = np.random.default_rng(seed)
    m = sum(n_per)
    cond = np.repeat([f"c{i}" for i in range(len(n_per))], n_per)
    mu0 = np.clip(2 ** rng.normal(mean_log2[0], mean_log2[1], n_genes), 0.05, 1e7)
    lfc = np.where(rng.random(n_genes) < de_frac, rng.normal(0, lfc_sd, n_genes), 0)
    sf = 2 ** rng.normal(0, sf_sd, m)
    sf /= np.exp(np.mean(np.log(sf)))
    mu = mu0[:, None] * sf[None, :] * (2 ** (lfc[:, None] * (cond == cond[-1])[None, :]))
    d = disp(mu0) * np.exp(rng.normal(0, sd_disp, n_genes)) if sd_disp else disp(mu0)
    size = 1 / d[:, None]
    return rng.negative_binomial(size, size / (size + mu)).astype(np.int64), cond


def write(name, cts, obs, design, contrast, **extra):
    """cts: genes x samples; obs: dict of per-sample columns (first key 'sample' is added)."""
    n = cts.shape[1]
    cols = {"sample": [f"s{i}" for i in range(n)], **obs}
    levels = {k: sorted(set(v), key=lambda x: (v.index(x))) for k, v in obs.items() if isinstance(v[0], str)}
    lines = [",".join(f"gene{i}" for i in range(cts.shape[0])) + "," + ",".join(cols)]
    for j in range(n):
        lines.append(",".join(str(int(x)) for x in cts[:, j]) + "," + ",".join(str(cols[k][j]) for k in cols))
    (OUT / f"deseq2_{name}_input.csv").write_text("\n".join(lines) + "\n")
    meta = {"design": design, "contrast": contrast, "levels": levels, **extra}
    (OUT / f"deseq2_{name}_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def two_group(name, n_genes, n_per, seed, **kw):
    cts, cond = simulate(n_genes, n_per, seed, **kw)
    write(name, cts, {"condition": cond.tolist()}, "~ condition", ["condition", f"c{len(n_per) - 1}", "c0"])


_BASE = np.array([2, 1, 3, 2, 2, 1, 3, 2, 2, 3, 1, 2, 2, 2, 3, 1])


def main():
    # prior variance for <= 3 residual degrees of freedom: heterogeneous true dispersions
    het = dict(mean_log2=(6, 2), disp=lambda mu: 4 / mu + 0.1)
    two_group("prior_2v2_sd10", 1000, (2, 2), 21, sd_disp=1.0, mean_log2=(4, 2), disp=het["disp"])
    two_group("prior_2v2_sd15", 1000, (2, 2), 22, sd_disp=1.5, **het)
    two_group("prior_2v3", 1000, (2, 3), 23, sd_disp=1.0, **het)
    two_group("prior_3v2", 1000, (3, 2), 24, sd_disp=1.0, **het)

    # outlier-refit subset: two genes with one extreme count, every other gene NB with dispersion 0.04
    for seed in (41, 42, 43):
        cts, cond = simulate(800, (8, 8), seed, disp=lambda mu: 0.04 + 0 * mu)
        rng = np.random.default_rng(seed * 7 + 1)
        for g in (0, 1):
            row = rng.permutation(_BASE).copy()
            row[rng.integers(0, 16)] = 60
            cts[g] = row
        if seed == 41:
            write("refit_outliers", cts, {"condition": cond.tolist()}, "~ condition", ["condition", "c1", "c0"])

    # counts up to 1e8: the parametric trend must still fit
    cts, cond = simulate(800, (4, 4), 36, mean_log2=(26, 2), disp=lambda mu: 0.05 + 0 * mu)
    assert cts.max() > 2 ** 24
    write("very_large_counts", cts, {"condition": cond.tolist()}, "~ condition", ["condition", "c1", "c0"])

    # independent filtering removes many genes and Cook's distances remove p-values (4 vs 4: no replacement)
    cts, cond = simulate(1000, (4, 4), 51, mean_log2=(2, 2.5), disp=lambda mu: 4 / mu + 0.1, sd_disp=0.5)
    rng = np.random.default_rng(5151)
    hit = rng.random(cts.shape) < 0.04
    cts = np.where(hit, cts * 25, cts)
    write("filtering_cooks", cts, {"condition": cond.tolist()}, "~ condition", ["condition", "c1", "c0"])

    # parametric trend fit fails in R as well (R substitutes a local fit)
    cts, cond = simulate(120, (3, 3), 62, mean_log2=(2, 1.5))
    write("trend_fallback", cts, {"condition": cond.tolist()}, "~ condition", ["condition", "c1", "c0"])

    # every gene has at least one zero: ratio size factors undefined, poscounts needed
    cts, cond = simulate(400, (5, 5), 71, mean_log2=(3, 1.5), disp=lambda mu: 1 / mu + 0.2)
    rng = np.random.default_rng(7171)
    cts[np.arange(cts.shape[0]), rng.integers(0, cts.shape[1], cts.shape[0])] = 0
    assert (cts == 0).any(axis=1).all()
    write("poscounts", cts, {"condition": cond.tolist()}, "~ condition", ["condition", "c1", "c0"], sf_type="poscounts")

    # numeric covariate + factor, tuple contrast, genes that are zero in one condition
    cts, cond = simulate(500, (6, 6), 81, mean_log2=(5, 2), disp=lambda mu: 2 / mu + 0.08)
    rng = np.random.default_rng(8181)
    age = np.round(rng.normal(50, 10, 12), 1)
    scale = np.exp(0.01 * (age - 50))
    cts = rng.binomial(cts, np.clip(scale / scale.max(), 0, 1)[None, :]).astype(np.int64)
    cts[:15, cond == "c1"] = 0
    cts[15:30, cond == "c0"] = 0
    write("covariate", cts, {"condition": cond.tolist(), "age": age.tolist()}, "~ age + condition",
          ["condition", "c1", "c0"], extra_names=["age"])

    # likelihood-ratio test, three groups: genes 0-3 carry one extreme count each, chosen so that the largest
    # Cook's distance is within 5% of the cut-off (ratios to the cut-off 0.967, 1.030, 0.959, 1.039)
    cts, cond = simulate(500, (4, 4, 4), 32, mean_log2=(6, 1.5), disp=lambda mu: 2 / mu + 0.05)
    cts[:4] = np.array([
        [405, 127, 121, 107, 127, 118, 125, 135, 113, 140, 114, 91],
        [107, 118, 139, 86, 116, 1026, 144, 126, 102, 95, 121, 156],
        [114, 103, 110, 141, 130, 102, 97, 98, 136, 2089, 110, 111],
        [134, 134, 1030, 118, 117, 86, 122, 130, 130, 78, 80, 132]])
    write("lrt_cooks", cts, {"condition": cond.tolist()}, "~ condition", ["condition", "c2", "c0"],
          test="LRT", reduced="~ 1")


if __name__ == "__main__":
    main()
