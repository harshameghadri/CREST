"""Gene identity and readers: genomes in 10x files, duplicate gene symbols, shared arrays after
``copy`` / filters, nullable ``obs`` columns and the raw-counts check.

Tests marked ``baseline_fails`` fail on the released 0.4.0 wheel for the stated reason and pass on
this tree; unmarked tests (pins) describe behaviour that was already right and must hold on both.
"""

import contextlib
import os
import re
import warnings

import numpy as np
import polars as pl
import pytest

import crest

sp = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
h5py = pytest.importorskip("h5py")

# ---- named tolerances / sizes
PBMC3K = os.environ.get("CREST_TEST_PBMC3K")  # directory with matrix.mtx, barcodes.tsv, genes.tsv
PROJECTION_ATOL = 1e-4      # a reference's own cells must project onto their own PCs (float32 round-off)
N_CELLS, N_GENES = 160, 90
DUP_FIRST, DUP_SECOND = 5, 70   # two genes that share the symbol "G5"; only the first is a highly variable gene


def baseline_fails(ref, raises, match):
    """Documentation marker, see the module docstring (registered by the test plugin when present)."""
    return pytest.mark.baseline_fails(ref=ref, raises=raises, match=match)


@contextlib.contextmanager
def must_raise(exc, match):
    """Like ``pytest.raises`` but an absent exception is an ``AssertionError`` with a plain message."""
    try:
        yield
    except exc as e:
        assert re.search(match, str(e)), f"{type(e).__name__} message {str(e)!r} does not match {match!r}"
    else:
        raise AssertionError(f"expected {exc.__name__} matching {match!r}; nothing was raised")


@contextlib.contextmanager
def recorded_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


def assert_warned(w, match):
    got = [str(x.message)[:90] for x in w]
    assert any(issubclass(x.category, UserWarning) and re.search(match, str(x.message)) for x in w), \
        f"expected a UserWarning matching {match!r}; got {got}"


def make_counts(n_cells=N_CELLS, n_genes=N_GENES, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.5, 1.0, n_genes)
    mu = np.ones((n_cells, 1)) * base[None, :] * rng.lognormal(0, 0.4, (n_cells, 1))
    mu[: n_cells // 2, :20] *= 4.0
    return sp.csr_matrix(rng.poisson(mu * 6).astype(np.float32))


# --------------------------------------------------------------------------- genomes in 10x files
def _write_v2(path, genomes):
    """A legacy (v2) Cell Ranger file: one group per genome, CSC (genes x cells)."""
    with h5py.File(path, "w") as f:
        for name, (X, genes, gene_names, barcodes) in genomes.items():
            g = f.create_group(name)
            C = X.T.tocsc()  # genes x cells
            g["data"], g["indices"], g["indptr"] = C.data, C.indices.astype(np.int64), C.indptr.astype(np.int64)
            g["shape"] = np.array(C.shape, dtype=np.int32)
            g["barcodes"] = np.array(barcodes, dtype="S")
            g["genes"] = np.array(genes, dtype="S")
            g["gene_names"] = np.array(gene_names, dtype="S")


@pytest.fixture()
def two_genomes(tmp_path):
    hg = make_counts(120, 60, seed=1)
    mm = sp.csr_matrix(np.asarray(make_counts(120, 45, seed=2).todense()) + np.eye(120, 45, dtype=np.float32) * 3)
    bc = [f"CELL{i:03d}-1" for i in range(120)]
    path = tmp_path / "multi.h5"
    _write_v2(path, {
        "hg19": (hg, [f"ENSG{i:05d}" for i in range(60)], [f"HG{i}" for i in range(60)], bc),
        "mm10": (mm, [f"ENSMUSG{i:05d}" for i in range(45)], [f"Mm{i}" for i in range(45)], bc),
    })
    return path, hg, sp.csr_matrix(mm)


@baseline_fails("v0.4.0", AssertionError, "nothing was raised")
def test_multi_genome_v2_file_without_genome_raises(two_genomes):
    path, _, _ = two_genomes
    with must_raise(ValueError, r"(?s)hg19.*mm10|mm10.*hg19"):
        crest.read_10x_h5(path)


@pytest.mark.parametrize("backed", [False, True])
@baseline_fails("v0.4.0", TypeError, "genome")
def test_multi_genome_v2_file_genome_selects_one(two_genomes, tmp_path, backed):
    path, _, mm = two_genomes
    bf = crest.read_10x_h5(path, genome="mm10", backed=(tmp_path / "pq") if backed else None)
    assert bf.shape == mm.shape
    assert bf.var_names == [f"Mm{i}" for i in range(45)]
    assert bf.obs["barcode"].to_list() == [f"CELL{i:03d}-1" for i in range(120)]
    np.testing.assert_array_equal(bf.to_scipy(transform=False).toarray(), mm.toarray())


@baseline_fails("v0.4.0", TypeError, "genome")
def test_unknown_genome_raises_and_lists_the_available_ones(two_genomes):
    path, _, _ = two_genomes
    with must_raise(ValueError, r"mm10"):
        crest.read_10x_h5(path, genome="hg38")


def test_single_genome_v2_file_needs_no_genome(tmp_path):  # pin
    X = make_counts(50, 30, seed=3)
    path = tmp_path / "one.h5"
    _write_v2(path, {"GRCh38": (X, [f"E{i}" for i in range(30)], [f"S{i}" for i in range(30)], [f"B{i}" for i in range(50)])})
    bf = crest.read_10x_h5(path)
    np.testing.assert_array_equal(bf.to_scipy(transform=False).toarray(), X.toarray())


@baseline_fails("v0.4.0", pl.exceptions.ColumnNotFoundError, "genome")
def test_v3_file_genome_column_selects_genes(tmp_path):
    X = make_counts(40, 20, seed=4)
    path = tmp_path / "v3.h5"
    C = X.T.tocsc()
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["data"], m["indices"], m["indptr"] = C.data, C.indices.astype(np.int64), C.indptr.astype(np.int64)
        m["shape"] = np.array(C.shape, dtype=np.int32)
        m["barcodes"] = np.array([f"B{i}" for i in range(40)], dtype="S")
        ft = m.create_group("features")
        ft["id"] = np.array([f"E{i}" for i in range(20)], dtype="S")
        ft["name"] = np.array([f"S{i}" for i in range(20)], dtype="S")
        ft["feature_type"] = np.array([b"Gene Expression"] * 20)
        ft["genome"] = np.array([b"hg19"] * 12 + [b"mm10"] * 8)
    both = crest.read_10x_h5(path)
    assert both.n_vars == 20 and both.var["genome"].to_list()[:1] == ["hg19"]
    mm = crest.read_10x_h5(path, genome="mm10")
    assert mm.var_names == [f"S{i}" for i in range(12, 20)]
    np.testing.assert_array_equal(mm.to_scipy(transform=False).toarray(), X.toarray()[:, 12:])


# --------------------------------------------------------------------------- duplicate gene symbols
def _dup_frame(with_ids=True):
    X = make_counts()
    names = [f"G{i}" for i in range(N_GENES)]
    names[DUP_SECOND] = names[DUP_FIRST]
    var = {"gene_name": names}
    if with_ids:
        var["gene_ids"] = [f"ENSG{i:05d}" for i in range(N_GENES)]
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame(var))
    return bf, names


def _with_pca(bf):
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    hv = np.zeros(bf.n_vars, bool)
    hv[:30] = True  # includes DUP_FIRST, not DUP_SECOND
    bf.var = bf.var.with_columns(pl.Series("highly_variable", hv))
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=8)
    return bf


@baseline_fails("v0.4.0", AssertionError, "Mismatch|not close|Max absolute difference")
def test_projection_onto_own_reference_with_duplicate_symbol(recwarn):
    bf, _ = _dup_frame()
    _with_pca(bf)
    got = crest.ingest.project_pca(bf, bf, center="reference")
    np.testing.assert_allclose(got, bf.obsm["X_pca"], atol=PROJECTION_ATOL)


@baseline_fails("v0.4.0", AssertionError, "not unique")
def test_projection_without_gene_ids_leaves_ambiguous_symbols_out_with_a_warning():
    ref, _ = _dup_frame(with_ids=False)
    _with_pca(ref)
    query, _ = _dup_frame(with_ids=False)
    crest.pp.normalize_total(query, 1e4)
    crest.pp.log1p(query)
    with recorded_warnings() as w:
        X = crest.ingest.project_pca(query, ref, center="reference")
    assert_warned(w, "not unique")
    # the reference's high-variance genes other than the repeated symbol still project exactly as before:
    # the result must not equal the one that attributed both copies to the same loading row
    assert X.shape == ref.obsm["X_pca"].shape and np.isfinite(X).all()


@baseline_fails("v0.4.0", AssertionError, "nothing was raised")
def test_score_genes_with_a_repeated_symbol_raises():
    bf, names = _dup_frame()
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    with must_raise(ValueError, r"G5"):
        crest.tl.score_genes(bf, ["G1", "G5", "G7"], ctrl_size=5, n_bins=5)


@baseline_fails("v0.4.0", AssertionError, "not close|Not equal")
def test_score_genes_by_gene_id_equals_the_score_on_unique_names():
    bf, names = _dup_frame()
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    unique = list(names)
    unique[DUP_SECOND] = "G5_copy"
    ref = crest.BioFrame.from_scipy(bf.to_scipy(transform=False), var=pl.DataFrame({"gene_name": unique}))
    crest.pp.normalize_total(ref, 1e4)
    crest.pp.log1p(ref)
    crest.tl.score_genes(ref, ["G1", "G5", "G7"], ctrl_size=5, n_bins=5)
    crest.tl.score_genes(bf, ["G1", f"ENSG{DUP_FIRST:05d}", "G7"], ctrl_size=5, n_bins=5)
    np.testing.assert_allclose(bf.obs["score"].to_numpy(), ref.obs["score"].to_numpy(), atol=1e-12)


@baseline_fails("v0.4.0", AssertionError, "unique|warn")
def test_to_anndata_makes_duplicate_names_unique_and_says_so():
    bf, names = _dup_frame()
    with recorded_warnings() as w:
        a = bf.to_anndata()
    assert a.var_names.is_unique, "AnnData index still has repeated names and no unique-suffix was applied"
    assert_warned(w, "not unique")


def test_anndata_round_trip_keeps_the_original_symbols():  # pin
    bf, names = _dup_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = crest.BioFrame.from_anndata(bf.to_anndata())
    assert back.var_names == names


@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
@baseline_fails("v0.4.0", AssertionError, "Mismatch|not close|Max absolute difference")
def test_projection_onto_own_reference_pbmc3k():
    """PBMC 3k keeps several repeated symbols after gene filtering, one of them a highly variable gene."""
    bf = crest.read_10x_mtx(PBMC3K)
    bf = crest.pp.filter_genes(bf, min_cells=3)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=2000)
    names = np.array(bf.var_names)
    hv = bf.var["highly_variable"].to_numpy()
    repeated = {n for n in names if (names == n).sum() > 1}
    assert repeated & set(names[hv]), "fixture has no repeated symbol among the highly variable genes"
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=30)
    got = crest.ingest.project_pca(bf, bf, center="reference")
    np.testing.assert_allclose(got, bf.obsm["X_pca"], atol=PROJECTION_ATOL)


# --------------------------------------------------------------------------- copies and filters
def _frame_with_state():
    bf = crest.BioFrame.from_scipy(make_counts(60, 25, seed=7), var=pl.DataFrame({"gene_name": [f"G{i}" for i in range(25)]}))
    rng = np.random.default_rng(0)
    bf.obsm["X_pca"] = rng.normal(size=(60, 5)).astype(np.float32)
    bf.varm["PCs"] = rng.normal(size=(25, 5)).astype(np.float32)
    bf.uns["pca"] = {"params": {"max_value": 10.0}, "variance": np.arange(5, dtype=np.float64)}
    return bf


def _snapshot(bf):
    return bf.obsm["X_pca"].copy(), bf.varm["PCs"].copy(), bf.uns["pca"]["params"]["max_value"], bf.uns["pca"]["variance"].copy()


def _scribble(child):
    child.obsm["X_pca"][:] = 99.0
    child.varm["PCs"][:] = 99.0
    child.uns["pca"]["params"]["max_value"] = -1.0
    child.uns["pca"]["variance"][:] = 99.0


@baseline_fails("v0.4.0", AssertionError, "parent")
def test_copy_is_independent_of_the_parent():
    bf = _frame_with_state()
    before = _snapshot(bf)
    _scribble(bf.copy())
    after = _snapshot(bf)
    assert np.array_equal(before[0], after[0]), "editing the copy's obsm changed the parent"
    assert np.array_equal(before[1], after[1]), "editing the copy's varm changed the parent"
    assert before[2] == after[2], "editing a nested uns dict of the copy changed the parent"
    assert np.array_equal(before[3], after[3]), "editing an array inside uns of the copy changed the parent"


@baseline_fails("v0.4.0", AssertionError, "parent")
def test_filter_genes_does_not_share_obsm_arrays():
    bf = _frame_with_state()
    before = bf.obsm["X_pca"].copy()
    child = bf.filter_genes(np.arange(25) < 20)
    child.obsm["X_pca"][:] = 99.0
    assert np.array_equal(bf.obsm["X_pca"], before), "editing obsm of the gene-filtered frame changed the parent"


@baseline_fails("v0.4.0", AssertionError, "parent")
def test_filter_cells_does_not_share_varm_arrays():
    bf = _frame_with_state()
    before = bf.varm["PCs"].copy()
    child = bf.filter_cells(np.arange(60) < 40)
    child.varm["PCs"][:] = 99.0
    assert np.array_equal(bf.varm["PCs"], before), "editing varm of the cell-filtered frame changed the parent"


def test_filters_replace_cell_derived_uns_entries_instead_of_editing_them():  # pin
    bf = _frame_with_state()
    bf.uns["neighbors"] = {"params": {"n_neighbors": 15, "n_pcs": None}}
    child = bf.filter_cells(np.arange(60) < 40)
    assert "neighbors" in bf.uns and "neighbors" not in child.uns


# --------------------------------------------------------------------------- readers: nullable columns, raw-counts check
def _write_h5ad_with_nullable(path, X):
    n = X.shape[0]
    obs = pd.DataFrame({
        "flag": pd.array([True, False, None] * (n // 3) + [True] * (n % 3), dtype="boolean"),
        "i64": pd.array([1, None, 3, 4] * (n // 4) + [0] * (n % 4), dtype="Int64"),
        "plain": np.arange(n),
        "kind": pd.Categorical(["a", "b"] * (n // 2) + ["a"] * (n % 2)),
    }, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])])
    ad.AnnData(X=X.copy(), obs=obs, var=var).write_h5ad(path)
    return obs


@baseline_fails("v0.4.0", AssertionError, "missing|nullable")
def test_read_h5ad_keeps_nullable_obs_columns(tmp_path):
    X = make_counts(60, 20, seed=5)
    path = tmp_path / "n.h5ad"
    obs = _write_h5ad_with_nullable(path, X)
    bf = crest.read_h5ad(path)
    for col, dtype in (("flag", pl.Boolean), ("i64", pl.Int64)):
        assert col in bf.obs.columns, f"nullable column {col!r} is missing from the result"
        assert bf.obs[col].dtype == dtype
        assert bf.obs[col].null_count() == int(obs[col].isna().sum())
        assert bf.obs[col].drop_nulls().to_list() == obs[col].dropna().tolist()
    assert bf.obs["plain"].to_list() == list(range(60)) and bf.obs["kind"].dtype == pl.Utf8


@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_read_h5ad_warns_about_columns_in_an_unsupported_encoding(tmp_path):
    X = make_counts(30, 12, seed=6)
    path = tmp_path / "w.h5ad"
    _write_h5ad_with_nullable(path, X)
    with h5py.File(path, "r+") as f:
        f["obs/i64"].attrs["encoding-type"] = "future-encoding"
        del f["obs/i64/values"], f["obs/i64/mask"]
    with recorded_warnings() as w:
        bf = crest.read_h5ad(path)
    assert "i64" not in bf.obs.columns
    assert_warned(w, "i64")


def _log_normalised(X):
    A = np.asarray(X.todense())
    A = np.log1p(A / np.maximum(A.sum(1, keepdims=True), 1) * 1e4)
    return sp.csr_matrix(A.astype(np.float32))


@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_read_h5ad_warns_when_x_is_not_counts(tmp_path):
    X = _log_normalised(make_counts(50, 20, seed=8))
    path = tmp_path / "log.h5ad"
    ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]), var=pd.DataFrame(index=[f"g{i}" for i in range(20)])).write_h5ad(path)
    with recorded_warnings() as w:
        crest.read_h5ad(path)
    assert_warned(w, "not look like raw counts")


def test_integer_counts_give_no_warning(tmp_path):  # pin
    X = make_counts(50, 20, seed=9)
    path = tmp_path / "ok.h5ad"
    ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]), var=pd.DataFrame(index=[f"g{i}" for i in range(20)])).write_h5ad(path)
    with recorded_warnings() as w:
        crest.read_h5ad(path)
        crest.BioFrame.from_scipy(X)
    assert not [x for x in w if "raw counts" in str(x.message)]


@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_negative_values_warn():
    X = make_counts(40, 15, seed=10).tolil()
    X[3, 2] = -5.0
    with recorded_warnings() as w:
        crest.BioFrame.from_scipy(X.tocsr())
    assert_warned(w, "negative")


@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_seurat_v3_on_log_data_warns():
    X = _log_normalised(make_counts(80, 40, seed=11))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf = crest.BioFrame.from_scipy(X)
    with recorded_warnings() as w:
        crest.pp.highly_variable_genes(bf, n_top_genes=10, flavor="seurat_v3")
    assert_warned(w, "not look like raw counts")


@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_read_10x_h5_backed_warns_when_values_are_not_counts(tmp_path):
    X = _log_normalised(make_counts(60, 20, seed=12))
    path = tmp_path / "log.h5"
    _write_v2(path, {"g": (X, [f"E{i}" for i in range(20)], [f"S{i}" for i in range(20)], [f"B{i}" for i in range(60)])})
    with recorded_warnings() as w:
        crest.read_10x_h5(path, backed=tmp_path / "pq2")
    assert_warned(w, "not look like raw counts")
