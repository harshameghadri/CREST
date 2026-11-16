"""The pinned reference stack (the `test` extra) must work, so that the parity tests can run.

Without the pins, scanpy 1.11.5 fails in `sc.pp.neighbors` above roughly 8,000 cells and in
`sc.pp.scrublet`, and `sc.tl.ingest` fails with newer pynndescent / umap-learn releases. These
tests exercise exactly those scanpy calls. The reference packages are imported directly: a
missing package is an error, not a skip.
"""

import numpy as np
import pytest

import anndata as ad
import harmonypy  # noqa: F401
import leidenalg  # noqa: F401
import scanpy as sc
import scipy.sparse as sp
import umap  # noqa: F401

N_LARGE = 11_769  # above the size at which the unpinned stack broke
N_SCRUBLET = 2_850


def pca_anndata(n, seed=0, d=30):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(sp.csr_matrix((n, 5), dtype=np.float32))
    a.obsm["X_pca"] = rng.normal(size=(n, d)).astype(np.float32)
    return a


@pytest.mark.parametrize("n", [6_000, N_LARGE])
def test_scanpy_neighbors_work_on_large_inputs(n):
    a = pca_anndata(n)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    conn = a.obsp["connectivities"]
    assert conn.shape == (n, n) and conn.nnz > n * 14


def test_scanpy_scrublet_runs():
    rng = np.random.default_rng(1)
    counts = rng.poisson(rng.gamma(0.5, 2.0, size=200), size=(N_SCRUBLET, 200)).astype(np.float32)
    a = ad.AnnData(sp.csr_matrix(counts))
    sc.pp.scrublet(a, random_state=0)
    assert "doublet_score" in a.obs and len(a.obs["doublet_score"]) == N_SCRUBLET


def test_scanpy_ingest_runs():
    rng = np.random.default_rng(2)
    centers = rng.normal(scale=6, size=(3, 20))
    labels = rng.integers(0, 3, 700)
    x = centers[labels] + rng.normal(size=(700, 20))
    ref = ad.AnnData(x.astype(np.float32))
    ref.obs["cell_type"] = np.array(["a", "b", "c"])[labels[:700]]
    ref.obs["cell_type"] = ref.obs["cell_type"].astype("category")
    sc.pp.pca(ref, n_comps=10)
    sc.pp.neighbors(ref, n_neighbors=10)
    sc.tl.umap(ref)
    query = ad.AnnData(x[:150].astype(np.float32))
    query.var_names = ref.var_names
    sc.tl.ingest(query, ref, obs="cell_type")
    assert (query.obs["cell_type"].astype(str).to_numpy() == ref.obs["cell_type"].astype(str).to_numpy()[:150]).mean() > 0.9
