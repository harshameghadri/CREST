"""A cell filter after highly_variable_genes must not reuse the HVG step's scale cache.

``highly_variable_genes`` caches per-gene log-scale mean/std (var ``mean_log`` /
``std_log``) that ``tl.pca`` reuses for ``scale`` while the transforms are unchanged.
Those statistics describe the cells HVG ran on; after a cell filter, scanpy's
``sc.pp.scale`` uses the mean/std of the remaining cells, so CREST must too.
"""

import numpy as np
import polars as pl
import pytest

import crest

sp = pytest.importorskip("scipy.sparse")
sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
from scipy.linalg import subspace_angles  # noqa: E402


def make_counts(n_cells=1600, n_genes=500, n_clusters=4, seed=11):
    """Negative-binomial counts whose clusters are unevenly split by the cell filter below."""
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
    names = [f"G{i}" for i in range(n_genes)]
    keep = (labels >= 2) | (np.arange(n_cells) % 4 == 0)  # about half the cells, mostly clusters 2-3
    return sp.csr_matrix(counts), names, keep


@pytest.fixture(scope="module")
def data():
    return make_counts()


def _crest(store, X, names, tmp_path):
    var = pl.DataFrame({"gene_name": names})
    if store == "csr":
        return crest.BioFrame.from_scipy(X, var=var)
    crest.BioFrame.from_scipy(X, var=var).write_parquet(tmp_path / "ds", nnz_per_part=40_000)
    return crest.read_parquet(tmp_path / "ds")


def _max_angle_deg(A, B):
    return float(np.degrees(subspace_angles(np.asarray(A, np.float64), np.asarray(B, np.float64)).max()))


@pytest.mark.parametrize("store", ["csr", "parquet"])
def test_cell_filter_after_hvg_then_scale_pca_matches_scanpy(data, store, tmp_path):
    X, names, keep = data
    a = ad.AnnData(X.copy(), var=pd.DataFrame(index=names))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=200, flavor="seurat")
    a = a[keep].copy()
    a = a[:, a.var["highly_variable"].values].copy()
    log_mean = np.asarray(a.X.mean(0)).ravel()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=20, svd_solver="arpack")

    parent = _crest(store, X, names, tmp_path)
    crest.pp.normalize_total(parent, 1e4)
    crest.pp.log1p(parent)
    crest.pp.highly_variable_genes(parent, n_top_genes=200, flavor="seurat")
    hvg_before = dict(parent.uns["hvg"])
    bf = parent.filter_cells(keep)
    assert bf.n_obs == int(keep.sum())
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=20)
    np.testing.assert_allclose(bf.uns["pca"]["variance"][:10], a.uns["pca"]["variance"][:10], rtol=1e-4)
    assert _max_angle_deg(bf.obsm["X_pca"][:, :10], a.obsm["X_pca"][:, :10]) < 0.01
    # the scale statistics are those of the kept cells
    np.testing.assert_allclose(bf.uns["pca"]["projection"]["mean"], log_mean, rtol=1e-5, atol=1e-6)

    # the parent's cache is untouched and still valid for the parent's cells
    assert parent.uns["hvg"] == hvg_before and "ops" in parent.uns["hvg"]
    full = ad.AnnData(X.copy(), var=pd.DataFrame(index=names))
    sc.pp.normalize_total(full, target_sum=1e4)
    sc.pp.log1p(full)
    sc.pp.highly_variable_genes(full, n_top_genes=200, flavor="seurat", subset=True)
    sc.pp.scale(full, max_value=10)
    sc.tl.pca(full, n_comps=20, svd_solver="arpack")
    crest.pp.scale(parent, max_value=10)
    crest.tl.pca(parent, n_comps=20)
    assert _max_angle_deg(parent.obsm["X_pca"][:, :10], full.obsm["X_pca"][:, :10]) < 0.01


def test_cell_filter_invalidates_only_cell_dependent_state(data):
    X, names, keep = data
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=200)
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=10)
    n, k = bf.n_obs, 5
    bf.uns["ingest"] = {"n_neighbors": k, "obs": [], "indices": np.arange(n * k, dtype=np.uint32).reshape(n, k),
                        "distances": np.ones((n, k), np.float32)}
    sub = bf.filter_cells(keep)
    assert "ops" not in sub.uns["hvg"] and sub.uns["hvg"]["flavor"] == "seurat"
    assert sub.uns["scale"] == bf.uns["scale"]  # a parameter, not a statistic
    assert sub.uns["pca"] is bf.uns["pca"]  # describes the PCs that obsm["X_pca"] rows still hold
    np.testing.assert_array_equal(sub.uns["ingest"]["indices"], bf.uns["ingest"]["indices"][keep])
    assert sub.uns["ingest"]["distances"].shape == (int(keep.sum()), k)
    assert bf.uns["ingest"]["indices"].shape == (n, k) and "ops" in bf.uns["hvg"]
    # a gene filter keeps the cache (same cells, same values)
    assert "ops" in bf.filter_genes(~pl.col("gene_name").is_in(["G0", "G1"])).uns["hvg"]
