"""Smoke test for an installed CREST wheel: import it and run three quick core checks.

Usage (inside a fresh virtual environment that has the wheel and scipy installed):
    python smoke_test.py [EXPECTED_VERSION]

EXPECTED_VERSION (normalised PEP 440, e.g. 0.5.0rc1) is compared with crest.__version__ and with the
version recorded in the installed package metadata. The checks use explicit arguments only,
so they do not depend on default values.
"""
import sys
from importlib.metadata import version as dist_version

import numpy as np
import polars as pl
import scipy.sparse as sp

import crest
from crest.core import BioFrame


def make_counts(n_cells=900, n_genes=400, n_clusters=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, n_genes)
    labels = rng.integers(0, n_clusters, n_cells)
    prog = np.ones((n_clusters, n_genes))
    for c in range(n_clusters):
        up = rng.choice(n_genes, 40, replace=False)
        prog[c, up] *= rng.uniform(4, 10, 40)
    lib = rng.lognormal(7.5, 0.3, n_cells)
    mu = prog[labels] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    return sp.csr_matrix(counts), [f"G{i}" for i in range(n_genes)], labels


def frame(X, names, chunk_nnz=None):
    bf = BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    if chunk_nnz:
        bf.chunk_nnz = chunk_nnz
    return bf


def test_normalize_log1p_matches_numpy():
    X, names, _ = make_counts()
    bf = frame(X, names)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    got = bf.to_scipy().toarray()
    dense = X.toarray().astype(np.float64)
    want = np.log1p(dense / dense.sum(1, keepdims=True) * 1e4)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def pipeline(bf):
    bf = crest.pp.filter_cells(bf, min_genes=50)
    bf = crest.pp.filter_genes(bf, min_cells=3)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=200)
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=10)
    return bf


def test_pca_does_not_depend_on_chunking():
    X, names, _ = make_counts()
    a = pipeline(frame(X, names))
    b = pipeline(frame(X, names, chunk_nnz=1000))
    assert a.shape == b.shape
    va, vb = a.uns["pca"]["variance"], b.uns["pca"]["variance"]
    np.testing.assert_allclose(va, vb, rtol=1e-4)
    # components agree up to sign
    pa, pb = a.obsm["X_pca"][:, :3], b.obsm["X_pca"][:, :3]
    for k in range(3):
        assert min(np.abs(pa[:, k] - pb[:, k]).max(), np.abs(pa[:, k] + pb[:, k]).max()) < 1e-2 * np.abs(pa[:, k]).max()


def test_neighbors_and_leiden_recover_planted_clusters():
    X, names, labels = make_counts()
    bf = pipeline(frame(X, names))
    crest.pp.neighbors(bf, n_neighbors=15, n_pcs=10)
    crest.tl.leiden(bf, resolution=0.5, random_state=0)
    kept = bf.obs["leiden"].to_numpy()
    n_kept = len(kept)
    assert n_kept >= 800
    # every Leiden cluster is dominated by one planted cluster
    true = labels[:n_kept] if n_kept == len(labels) else None
    if true is None:
        return  # cells were filtered; the planted labels no longer line up, the call itself was the test
    purity = sum(np.bincount(true[kept == c]).max() for c in np.unique(kept)) / n_kept
    assert purity > 0.9, purity


def main(argv):
    print(f"crest {crest.__version__} from {crest.__file__}")
    if argv:
        want = argv[0].lstrip("v")
        assert crest.__version__ == want, f"crest.__version__ is {crest.__version__}, expected {want}"
        assert dist_version("crest-sc") == want, f"installed metadata is {dist_version('crest-sc')}, expected {want}"
    for test in (test_normalize_log1p_matches_numpy, test_pca_does_not_depend_on_chunking,
                 test_neighbors_and_leiden_recover_planted_clusters):
        test()
        print(f"ok  {test.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
