"""Harmony with two batch covariates (``key=[a, b]``) against harmonypy 2.x.

harmonypy 2.x combines the per-covariate diversity penalties of a cell by
multiplying them (it adds their logarithms to the assignment logits); CREST does
the same. R ``harmony`` 2.0 instead sums the penalty factors (``pow(...) * Phi``
with a stacked one-hot Phi), so for several covariates the two references
differ. CREST follows harmonypy, the reference named in CLAUDE.md.

Harmony's initialisation is random and the two tools use different random
streams, so, as in ``test_harmony_mixes_batches_and_keeps_types``, outcomes are
compared, not values. The batch shifts are large enough that the clustering
penalty matters: replacing the product with a sum moves CREST's residual batch
gap (~9.4 vs ~7) and its neighbourhoods (kNN overlap with harmonypy ~0.80 vs
~0.89, seed-to-seed within harmonypy ~0.91) well outside these bounds.
"""

import numpy as np
import polars as pl
import pytest

import crest

sp = pytest.importorskip("scipy.sparse")


def _counts(n_cells, n_genes, n_clusters=5, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, n_genes)
    labels = rng.integers(0, n_clusters, n_cells)
    prog = np.ones((n_clusters, n_genes))
    for c in range(n_clusters):
        up = rng.choice(n_genes, 40, replace=False)
        prog[c, up] *= rng.uniform(3, 10, 40)
    lib = rng.lognormal(7.5, 0.4, n_cells)
    mu = prog[labels] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    names = [f"MT-{i}" if i < 10 else f"G{i}" for i in range(n_genes)]
    return sp.csr_matrix(counts), names, labels


def _two_covariate_data(n=4000):
    X, names, lab = _counts(n, 600, seed=5)
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=300)
    crest.pp.scale(bf)
    crest.tl.pca(bf, n_comps=20)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, n)  # e.g. donor
    b = rng.integers(0, 2, n)  # e.g. chemistry, independent of a
    Z = bf.obsm["X_pca"].astype(np.float64)
    Z[:, 1] += 8.0 * (a == 1) - 6.0 * (a == 2)
    Z[:, 2] += 8.0 * (b == 1)
    return np.ascontiguousarray(Z, dtype=np.float32), a.astype(str), b.astype(str), lab


def _gap(Z, cov, pc, lab):
    """Largest spread of per-level means along the shifted PC, over true types."""
    out = []
    for t in np.unique(lab):
        mu = [Z[(lab == t) & (cov == lv), pc].mean() for lv in np.unique(cov)]
        out.append(max(mu) - min(mu))
    return max(out)


def _knn(Z, k=15):
    idx, _ = crest.crest.knn_graph(np.ascontiguousarray(Z, np.float32), k, exact=True, seed=0)
    return idx


def _overlap(i1, i2):
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(i1, i2)]))


def test_harmony_two_covariates_matches_harmonypy():
    harmonypy = pytest.importorskip("harmonypy")
    pd = pytest.importorskip("pandas")
    Z, a, b, lab = _two_covariate_data()
    meta = pd.DataFrame({"a": a, "b": b})
    seeds = (0, 1)
    ours = [crest.harmony.run_harmony(Z, [a, b], random_state=s)["Z_corr"] for s in seeds]
    ref = [np.asarray(harmonypy.run_harmony(Z, meta, ["a", "b"], random_state=s, verbose=False).Z_corr)
           for s in seeds]
    for Zc in ours + ref:
        assert Zc.shape == Z.shape and np.isfinite(Zc).all()

    # both covariates are corrected, by about as much as harmonypy corrects them
    raw = (_gap(Z, a, 1, lab), _gap(Z, b, 2, lab))
    for cov, pc, r in ((a, 1, raw[0]), (b, 2, raw[1])):
        g_ours = np.mean([_gap(Zc, cov, pc, lab) for Zc in ours])
        g_ref = np.mean([_gap(Zc, cov, pc, lab) for Zc in ref])
        assert g_ours < 0.6 * r and g_ref < 0.6 * r
        assert abs(g_ours - g_ref) < 1.0, (g_ours, g_ref)

    # neighbourhoods agree with harmonypy's about as well as two harmonypy seeds agree
    k_ours, k_ref = [_knn(x) for x in ours], [_knn(x) for x in ref]
    cross = np.mean([_overlap(p, q) for p in k_ours for q in k_ref])
    within = _overlap(k_ref[0], k_ref[1])
    assert cross > 0.85 and cross > within - 0.05, (cross, within)

    # deterministic for a fixed seed; key=[a, b] through the BioFrame API gives the same
    bf = crest.BioFrame.from_scipy(sp.csr_matrix((len(a), 1), dtype=np.float32),
                                   obs=pl.DataFrame({"a": a, "b": b}))
    bf.obsm["X_pca"] = Z
    crest.tl.harmony(bf, ["a", "b"])
    np.testing.assert_array_equal(bf.obsm["X_pca_harmony"], ours[0])
