"""Input for the outlier-replacement fixture ``replace_hat`` in this directory.

Run from the repository root:  python tests/data/make_replace_inputs.py [output_dir]
It writes

* deseq2_replace_hat_input.csv  one row per sample (columns ``gene0``..``geneN`` of counts, then ``condition``),
* deseq2_replace_hat_meta.json  design, contrast, levels and the minReplicatesForReplace used.

Two groups: ``a`` has 4 samples (below minReplicatesForReplace = 7: never replaced), ``b`` has 12 (replaceable).
The genes are of two kinds: ordinary negative-binomial genes that carry the dispersion trend, and genes with zero counts in
all of group ``a`` and heavy-tailed counts in group ``b``. For the second kind the coefficient of the group is not finite, the
iteratively reweighted least squares fit diverges, and both tools refit the row (``betaConv`` is FALSE in R); the hat
diagonals behind Cook's distance of these rows then decide which counts are replaced. The reference output is produced by
make_replace_fixtures.R; nothing written here is a result.
"""
import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent
N_A, N_B, N_NORMAL, N_SEP = 4, 12, 300, 450


def nb(rng, mu, disp):
    return rng.poisson(rng.gamma(1.0 / disp, mu * disp))


def build(seed=11):
    rng = np.random.default_rng(seed)
    n = N_A + N_B
    sf = np.exp(rng.normal(0, 0.2, n))
    cols = []
    for _ in range(N_NORMAL):
        mu = np.exp(rng.normal(4.0, 1.5))
        lfc = rng.normal(0, 0.5) * (rng.random() < 0.3)
        m = np.r_[np.full(N_A, mu), np.full(N_B, mu * 2**lfc)] * sf
        cols.append(nb(rng, m, 0.05 + 0.8 / (1 + mu / 5)))
    for _ in range(N_SEP):
        mu = np.exp(rng.normal(5.5, 1.0))
        y = np.r_[np.zeros(N_A), nb(rng, np.full(N_B, mu) * sf[N_A:], rng.uniform(6, 16))]
        cols.append(y.astype(int))
    X = np.array(cols).T
    return X


def write(X, seed):
    n, g = X.shape
    genes = [f"gene{i}" for i in range(g)]
    cond = ["a"] * N_A + ["b"] * N_B
    with open(OUT / "deseq2_replace_hat_input.csv", "w") as f:
        f.write(",".join(genes + ["condition"]) + "\n")
        for i in range(n):
            f.write(",".join(str(int(v)) for v in X[i]) + f",{cond[i]}\n")
    meta = {"design": "~ condition", "contrast": ["condition", "b", "a"], "levels": {"condition": ["a", "b"]},
            "min_replicates_for_replace": [7], "seed": seed, "n_normal": N_NORMAL, "n_separated": N_SEP}
    (OUT / "deseq2_replace_hat_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    write(build(), 11)
