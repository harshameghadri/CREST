"""Inputs for the local dispersion-trend fixtures in this directory.

Run from the repository root:  python tests/data/make_local_fit_inputs.py [output_dir]
It writes

* deseq2_local_fit_cases.csv   (case, mean, disp): mean / gene-wise dispersion pairs for DESeq2's ``localDispersionFit``,
                               built to hit the cases of the locfit evaluation structure: ties in the means, very few
                               genes, extreme weights, a trend that needs many vertices, a gap in the means, outliers,
                               values below ``10 * minDisp`` that the fit leaves out
* deseq2_local_fit_at.csv      (case, at): the means at which the trend is compared (up to 60 of the distinct means of the case, and
                               a grid that reaches beyond the range of the means on both sides)

The reference trend is produced by make_local_fit_fixtures.R; nothing written here is a result.
"""
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent


def trend(m, a=0.8, b=0.12):
    return a / m + b


def noisy(rng, m, sd=0.35, **kw):
    return trend(m, **kw) * np.exp(rng.normal(0, sd, len(m)))


def build():
    rng = np.random.default_rng(20261115)
    cases = {}
    m = np.exp(rng.normal(3.0, 2.0, 800))
    cases["smooth_800"] = (m, noisy(rng, m))
    m = np.exp(rng.normal(1.0, 1.5, 1500))                  # low-count genes dominate, as in sparse strata
    cases["low_counts_1500"] = (m, noisy(rng, m, 0.7, a=2.0, b=0.4))
    levels = np.exp(np.linspace(-1, 9, 40))
    m = rng.choice(levels, 500)                             # ties in log(mean)
    cases["ties_40_levels"] = (m, noisy(rng, m))
    m = np.repeat(np.exp(np.linspace(0, 6, 12)), 20)         # ties, several disps per mean
    cases["ties_12_levels_x20"] = (m, noisy(rng, m, 0.5))
    m = np.full(60, 12.5)                                    # one distinct mean
    cases["single_mean"] = (m, noisy(rng, m, 0.4))
    for n in (2, 3, 4, 5, 6, 8, 10, 15, 30):
        m = np.exp(rng.normal(3, 1.5, n))
        cases[f"few_genes_{n}"] = (m, noisy(rng, m, 0.5))
    m = np.exp(np.concatenate([rng.normal(-2, 0.4, 400), rng.normal(11, 0.3, 30), rng.normal(4, 0.5, 5)]))
    cases["extreme_weights"] = (m, noisy(rng, m, 0.6))      # a few high-mean genes carry nearly all the weight
    m = np.exp(rng.uniform(-6, 13, 1200))
    d = trend(m) * np.exp(0.9 * np.sin(np.log(m) * 1.7) + rng.normal(0, 0.15, len(m)))
    cases["wiggly_trend_1200"] = (m, d)                      # needs many vertices; pseudo-vertices occur
    m = np.exp(np.concatenate([rng.uniform(-1, 1, 250), rng.uniform(8, 10, 250)]))
    cases["gap_in_means"] = (m, noisy(rng, m))
    m = np.exp(rng.normal(3, 1.8, 800))
    d = noisy(rng, m)
    d[rng.choice(len(d), 25, replace=False)] *= np.exp(rng.uniform(2, 5, 25))
    cases["outliers_800"] = (m, d)
    m = np.exp(rng.normal(3, 1.8, 700))
    d = noisy(rng, m, 0.4)
    d[rng.choice(len(d), 100, replace=False)] = 1e-8         # at the dispersion floor: below 10 * minDisp, left out
    d[rng.choice(len(d), 20, replace=False)] = 5e-8
    cases["floor_values_700"] = (m, d)
    m = np.exp(rng.normal(3, 1.8, 400))
    cases["constant_disp"] = (m, np.full(len(m), 0.3))
    m = np.exp(rng.normal(2, 1.2, 400))
    cases["decreasing_steep"] = (m, trend(m, a=40.0, b=0.01) * np.exp(rng.normal(0, 0.2, len(m))))
    m = np.sort(np.exp(rng.normal(3, 1.5, 400)))
    cases["sorted_input"] = (m, noisy(rng, m))
    m = np.exp(rng.normal(3, 2.5, 800)) * 1.0
    cases["rounded_means"] = (np.round(m, 2) + 0.01, noisy(rng, np.round(m, 2) + 0.01))
    return cases


def main():
    cases = build()
    with open(OUT / "deseq2_local_fit_cases.csv", "w") as f, open(OUT / "deseq2_local_fit_at.csv", "w") as g:
        f.write("case,mean,disp\n")
        g.write("case,at\n")
        for name, (m, d) in cases.items():
            for mi, di in zip(m, d):
                f.write(f"{name},{float(mi)!r},{float(di)!r}\n")
            ok = d >= 1e-7
            lm = np.log(m[ok])
            um = np.unique(m[ok])
            um = um[np.linspace(0, len(um) - 1, min(len(um), 60)).astype(int)]
            at = np.unique(np.concatenate([um, np.exp(np.linspace(lm.min() - 2.5, lm.max() + 2.5, 80))]))
            for a in at:
                g.write(f"{name},{float(a)!r}\n")
    print({k: len(v[0]) for k, v in cases.items()})


if __name__ == "__main__":
    main()
