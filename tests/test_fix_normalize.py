"""normalize_total fixes each cell's total when it is called, as scanpy does.

A gene filter applied after ``normalize_total`` (dropping MT- genes, or
``highly_variable_genes(subset=True)``) must not renormalise the cells over the
genes that remain: scanpy divides by the totals over the genes present at
``normalize_total`` time and later subsets leave the values unchanged. CREST
computes those totals when ``normalize_total`` is called and stores them.
"""

import numpy as np
import polars as pl
import pytest

import crest
from crest import crest as _native

sp = pytest.importorskip("scipy.sparse")
sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
from scipy.linalg import subspace_angles  # noqa: E402

STORES = ["csr", "csr_small_chunks", "triplets", "parquet"]


def make_counts(n_cells=1200, n_genes=500, n_clusters=5, n_mt=25, seed=3):
    """Negative-binomial counts with cluster programs and a large, cell-varying MT- fraction."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, n_genes)
    base[:n_mt] *= 20  # MT genes: a few % to tens of % of each cell's counts
    labels = rng.integers(0, n_clusters, n_cells)
    prog = np.ones((n_clusters, n_genes))
    for c in range(n_clusters):
        up = rng.choice(np.arange(n_mt, n_genes), 40, replace=False)
        prog[c, up] *= rng.uniform(3, 10, 40)
    mt_load = rng.uniform(0.2, 3.0, n_cells)  # per-cell MT burden, so totals over kept genes vary
    lib = rng.lognormal(7.5, 0.4, n_cells)
    mu = prog[labels] * base[None, :]
    mu[:, :n_mt] *= mt_load[:, None]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    names = [f"MT-{i}" if i < n_mt else f"G{i}" for i in range(n_genes)]
    return sp.csr_matrix(counts), names


@pytest.fixture(scope="module")
def data():
    return make_counts()


def _crest(store, X, names, tmp_path):
    var = pl.DataFrame({"gene_name": names})
    if store == "csr":
        return crest.BioFrame.from_scipy(X, var=var)
    if store == "csr_small_chunks":
        bf = crest.BioFrame.from_scipy(X, var=var)
        bf.chunk_nnz = 5000
        return bf
    if store == "triplets":
        coo = X.tocoo()
        df = pl.DataFrame({"cell_id": coo.row.astype(np.uint32), "gene_id": coo.col.astype(np.uint32),
                           "count": coo.data})
        return crest.BioFrame.from_triplets(df, X.shape[0], X.shape[1], var=var)
    crest.BioFrame.from_scipy(X, var=var).write_parquet(tmp_path / "ds", nnz_per_part=40_000)
    return crest.read_parquet(tmp_path / "ds")


def _scanpy(X, names):
    return ad.AnnData(X.copy(), var=pd.DataFrame(index=names))


def _dense(a):
    return a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)


def _not_mt(bf):
    return bf.filter_genes(~pl.col("gene_name").str.starts_with("MT-"))


def _max_angle_deg(A, B):
    return float(np.degrees(subspace_angles(np.asarray(A, np.float64), np.asarray(B, np.float64)).max()))


@pytest.mark.parametrize("store", STORES)
def test_drop_mt_after_normalize_matches_scanpy(data, store, tmp_path):
    X, names = data
    a = _scanpy(X, names)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, ~a.var_names.str.startswith("MT-")].copy()

    bf = _crest(store, X, names, tmp_path)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    bf = _not_mt(bf)
    assert bf.var_names == list(a.var_names)
    np.testing.assert_allclose(bf.to_scipy().toarray(), _dense(a), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("store", STORES)
def test_two_gene_filters_after_normalize_match_scanpy(data, store, tmp_path):
    """The totals are fixed once; a second gene filter must not recompute them."""
    X, names = data
    a = _scanpy(X, names)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, ~a.var_names.str.startswith("MT-")].copy()
    sc.pp.filter_genes(a, min_cells=400)

    bf = _crest(store, X, names, tmp_path)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    bf = _not_mt(bf)
    bf = crest.pp.filter_genes(bf, min_cells=400)
    assert bf.n_vars < len(names) - 25  # the second filter removed genes
    assert bf.var_names == list(a.var_names)
    np.testing.assert_allclose(bf.to_scipy().toarray(), _dense(a), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("store", STORES)
def test_cell_filter_after_pinned_totals_matches_scanpy(data, store, tmp_path):
    """Totals are keyed by store cell id, so a later cell filter keeps each cell's own total."""
    X, names = data
    keep = np.arange(X.shape[0]) % 3 != 0
    a = _scanpy(X, names)
    sc.pp.normalize_total(a, target_sum=1e4)
    a = a[:, ~a.var_names.str.startswith("MT-")].copy()
    a = a[keep].copy()
    sc.pp.log1p(a)

    bf = _crest(store, X, names, tmp_path)
    crest.pp.normalize_total(bf, 1e4)
    bf = _not_mt(bf)
    bf = bf.filter_cells(keep)
    crest.pp.log1p(bf)
    np.testing.assert_allclose(bf.to_scipy().toarray(), _dense(a), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("store", STORES)
def test_normalize_again_after_gene_filter_matches_scanpy(data, store, tmp_path):
    """Calling normalize_total after a gene filter normalises over the kept genes (as scanpy)."""
    X, names = data
    a = _scanpy(X, names)
    sc.pp.normalize_total(a, target_sum=1e4)
    a = a[:, ~a.var_names.str.startswith("MT-")].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    bf = _crest(store, X, names, tmp_path)
    crest.pp.normalize_total(bf, 1e4)
    first_op = bf.ops[-1]
    bf = _not_mt(bf)
    old = bf.cell_totals
    crest.pp.normalize_total(bf, 1e4)
    assert bf.cell_totals is not old and bf.ops[-1] != first_op  # replaced, with a new fingerprint
    ids = bf.obs["cell_id"].to_numpy()
    kept = X[:, bf.var["gene_id"].to_numpy()]
    np.testing.assert_allclose(bf.cell_totals[ids], np.asarray(kept.sum(1)).ravel(), rtol=1e-6)
    assert np.all(old[ids] > bf.cell_totals[ids])  # the MT counts are no longer in the totals
    assert [op[0] for op in bf.ops] == ["normalize_total"]
    crest.pp.log1p(bf)
    Y = bf.to_scipy().toarray()
    np.testing.assert_allclose(Y, _dense(a), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.expm1(Y.astype(np.float64)).sum(1), 1e4, rtol=1e-4)


@pytest.mark.parametrize("store", ["csr", "parquet"])
def test_normalize_again_discards_hvg_scale_cache(data, store, tmp_path):
    """normalize -> HVG -> gene filter -> normalize again: same ops, new values, so PCA
    must not scale with the mean/std cached by HVG under the old totals."""
    X, names = data
    a = _scanpy(X, names)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.highly_variable_genes(a, n_top_genes=200, flavor="cell_ranger")
    a = a[:, ~a.var_names.str.startswith("MT-")].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    a = a[:, a.var["highly_variable"].values].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=20, svd_solver="arpack")

    bf = _crest(store, X, names, tmp_path)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.highly_variable_genes(bf, n_top_genes=200, flavor="cell_ranger")
    parent = _not_mt(bf)
    child = parent.copy()
    crest.pp.normalize_total(child, 1e4)
    assert child.uns["hvg"]["ops"] != child.ops  # new totals fingerprint: the cache is stale
    assert parent.uns["hvg"]["ops"] == parent.ops  # the parent's totals and cache are unchanged
    crest.pp.scale(child, max_value=10)
    crest.tl.pca(child, n_comps=20)
    assert child.var.filter(pl.col("highly_variable"))["gene_name"].to_list() == list(a.var_names)
    np.testing.assert_allclose(child.uns["pca"]["variance"][:10], a.uns["pca"]["variance"][:10], rtol=1e-4)
    assert _max_angle_deg(child.obsm["X_pca"][:, :10], a.obsm["X_pca"][:, :10]) < 0.01


@pytest.mark.parametrize("store", STORES)
def test_cell_filter_after_normalize_matches_scanpy(data, store, tmp_path):
    """normalize -> filter cells: every kept cell keeps its own total."""
    X, names = data
    keep = np.arange(X.shape[0]) % 5 >= 2
    a = _scanpy(X, names)
    sc.pp.normalize_total(a, target_sum=1e4)
    a = a[keep].copy()
    sc.pp.log1p(a)

    bf = _crest(store, X, names, tmp_path)
    crest.pp.normalize_total(bf, 1e4)
    bf = bf.filter_cells(keep)
    crest.pp.log1p(bf)
    np.testing.assert_allclose(bf.to_scipy().toarray(), _dense(a), rtol=1e-5, atol=1e-5)


def test_same_totals_same_fingerprint(data):
    """Re-normalising over the same genes records the same op, so ops-keyed caches stay valid."""
    X, names = data
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=200)
    op = bf.ops[0]
    assert len(op) == 3 and op[:2] == ("normalize_total", 1e4)
    again = bf.copy()
    again.ops = []
    crest.pp.normalize_total(again, 1e4)
    assert again.ops[0] == op
    sub = bf.filter_cells(np.arange(bf.n_obs) % 2 == 0)
    assert sub.ops == bf.ops and sub.cell_totals is bf.cell_totals  # carried, not recomputed


def test_gene_filter_before_normalize_unchanged(data):
    """The benchmark path (filter genes, then normalize): totals over the kept genes."""
    X, names = data
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    bf = crest.pp.filter_genes(bf, min_cells=3)
    bf = _not_mt(bf)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=200)
    bf = bf.filter_cells(np.arange(bf.n_obs) % 2 == 0)
    D = X.toarray()[::2][:, bf.var["gene_id"].to_numpy()]
    ref = np.log1p(D / D.sum(1, keepdims=True) * 1e4)
    np.testing.assert_allclose(bf.to_scipy().toarray(), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("store", ["csr", "parquet"])
def test_hvg_subset_then_scale_pca_matches_scanpy(data, store, tmp_path):
    X, names = data
    a = _scanpy(X, names)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=200, flavor="seurat", subset=True)
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=20, svd_solver="arpack")

    bf = _crest(store, X, names, tmp_path)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    bf = crest.pp.highly_variable_genes(bf, n_top_genes=200, flavor="seurat", subset=True)
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=20)
    assert bf.var_names == list(a.var_names)
    np.testing.assert_allclose(bf.uns["pca"]["variance"][:10], a.uns["pca"]["variance"][:10], rtol=1e-4)
    assert _max_angle_deg(bf.obsm["X_pca"][:, :10], a.obsm["X_pca"][:, :10]) < 0.01


def test_cell_totals_validated(data):
    X, names = data
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    crest.pp.normalize_total(bf, 1e4)
    bf = _not_mt(bf)
    ctx = list(next(bf.iter_ctx()))
    out = np.zeros((bf.n_vars, 5))
    ctx[6] = np.ones(len(ctx[2]) - 1)  # wrong length
    with pytest.raises(ValueError, match="cell_totals"):
        _native.gene_stats(*ctx, out)
    # negative counts give a negative total: rejected once, when normalize_total stores it
    Y = X.copy()
    Y.data[: Y.indptr[1]] = -1.0
    neg = crest.BioFrame.from_scipy(Y, var=pl.DataFrame({"gene_name": names}))
    with pytest.raises(ValueError, match="finite and >= 0"):
        crest.pp.normalize_total(neg, 1e4)


def test_fingerprinted_op_without_totals_raises(data):
    X, names = data
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    crest.pp.normalize_total(bf, 1e4)
    bf.cell_totals = None
    with pytest.raises(ValueError, match="normalize_total again"):
        crest.pp.gene_stats(bf)
    # a legacy two-element op (set by hand) still falls back to totals over the kept genes
    bf.ops = [("normalize_total", 1e4)]
    D = X.toarray()
    np.testing.assert_allclose(bf.to_scipy().toarray(), D / D.sum(1, keepdims=True) * 1e4, rtol=1e-5)
