"""Defaults and semantics that follow scanpy: ``normalize_total`` / ``scale`` called without arguments,
count filters after a transform, ingest on a filtered reference, and the small documented deviations.

Tests marked ``baseline_fails`` fail on the released 0.4.0 wheel (``ref="rc0"``: on the first 0.5
release candidate) for the stated reason and pass on this tree; unmarked tests (pins) describe
behaviour that was already right and must hold on both. The real-data variants run when
``CREST_TEST_PBMC3K`` points at a directory holding PBMC 3k's 10x MTX files and ``CREST_TEST_PBMC10K``
at the PBMC 10k ``.h5`` file.
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
sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
from scipy.linalg import subspace_angles  # noqa: E402

# ---- named tolerances / sizes
NORMALIZE_RTOL = 2e-6          # values of normalize_total(no arguments) against scanpy (float32 arithmetic)
PCA_ANGLE_DEG = 0.01           # principal angle between the top PCs after scale(no arguments) and scanpy's
N_ANGLE_PCS = 10
PBMC3K = os.environ.get("CREST_TEST_PBMC3K")
PBMC10K = os.environ.get("CREST_TEST_PBMC10K")
N_CELLS, N_GENES = 700, 400
COUNT_FILTER_MIN = 9000.0       # min_counts used by the scanpy workflow after normalize_total(1e4)


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


# --------------------------------------------------------------------------- data
def make_counts(n_cells=N_CELLS, n_genes=N_GENES, n_clusters=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.3, 1.0, n_genes)
    labels = rng.integers(0, n_clusters, n_cells)
    prog = np.ones((n_clusters, n_genes))
    for c in range(n_clusters):
        n_up = min(30, n_genes)
        up = rng.choice(n_genes, n_up, replace=False)
        prog[c, up] *= rng.uniform(4, 12, n_up)
    lib = rng.lognormal(6.3, 0.5, n_cells)  # wide depth range: the median total is not a round number
    mu = prog[labels] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    X = sp.csr_matrix(rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32))
    names = [f"MT-{i}" if i < 8 else f"G{i}" for i in range(n_genes)]
    return X, names, labels


def frame(X, names):
    return crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names, "gene_ids": [f"E{i}" for i in range(len(names))]}))


def scanpy_obj(X, names):
    a = ad.AnnData(X=X.copy().astype(np.float32))
    a.var_names = names
    a.var_names_make_unique()
    return a


def sorted_csr(M):
    M = M.tocsr().astype(np.float32)
    M.sort_indices()
    return M


def assert_same_sparse(A, B, rtol):
    A, B = sorted_csr(A), sorted_csr(B)
    assert A.shape == B.shape
    assert np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices), "different non-zero pattern"
    np.testing.assert_allclose(A.data, B.data, rtol=rtol)


def real_frames():
    """(name, BioFrame) for the real datasets that are available."""
    out = []
    if PBMC3K:
        out.append(("pbmc3k", crest.read_10x_mtx(PBMC3K)))
    if PBMC10K:
        out.append(("pbmc10k", crest.read_10x_h5(PBMC10K)))
    return out


def max_angle_deg(A, B):
    return float(np.degrees(subspace_angles(A, B).max()))


# --------------------------------------------------------------------------- normalize_total / scale defaults
@pytest.mark.parametrize("after_gene_filter", [False, True])
@baseline_fails("v0.4.0", AssertionError, "Mismatched|not close|Max relative|nnz|pattern")
def test_normalize_total_without_arguments_equals_scanpy(after_gene_filter):
    X, names, _ = make_counts()
    bf, a = frame(X, names), scanpy_obj(X, names)
    if after_gene_filter:
        bf = crest.pp.filter_genes(bf, min_cells=20)
        sc.pp.filter_genes(a, min_cells=20)
    crest.pp.normalize_total(bf)
    sc.pp.normalize_total(a)
    assert_same_sparse(bf.to_scipy(), a.X, NORMALIZE_RTOL)


@pytest.mark.skipif(not (PBMC3K or PBMC10K), reason="CREST_TEST_PBMC3K / CREST_TEST_PBMC10K not set")
@baseline_fails("v0.4.0", AssertionError, "Mismatched|not close|Max relative|nnz|pattern")
def test_normalize_total_without_arguments_equals_scanpy_real_data():
    for name, bf in real_frames():
        a = scanpy_obj(bf.to_scipy(transform=False), bf.var_names)
        for filtered in (False, True):
            b = crest.pp.filter_genes(bf, min_cells=3) if filtered else bf.copy()
            aa = a[:, np.asarray(b.var["gene_id"].to_numpy())].copy() if filtered else a.copy()
            crest.pp.normalize_total(b)
            sc.pp.normalize_total(aa)
            assert_same_sparse(b.to_scipy(), aa.X, NORMALIZE_RTOL), name


@baseline_fails("v0.4.0", AssertionError, "median")
def test_default_target_is_the_median_total_and_is_recorded_as_a_number():
    X, names, _ = make_counts()
    bf = frame(X, names)
    totals = np.asarray(X.sum(1)).ravel()
    crest.pp.normalize_total(bf)
    assert bf.ops[0][0] == "normalize_total"
    assert bf.ops[0][1] == pytest.approx(float(np.median(totals[totals > 0])), rel=1e-9), "target is not the median total"
    kept = crest.pp.filter_cells(bf, min_genes=50)
    assert kept.ops[0][1] == bf.ops[0][1], "a later cell filter must not change the recorded target"


def test_explicit_target_sum_still_works_and_matches_scanpy():  # pin
    X, names, _ = make_counts()
    bf, a = frame(X, names), scanpy_obj(X, names)
    crest.pp.normalize_total(bf, 1e4)
    sc.pp.normalize_total(a, target_sum=1e4)
    assert_same_sparse(bf.to_scipy(), a.X, NORMALIZE_RTOL)


def test_target_sum_must_be_positive():  # pin
    X, names, _ = make_counts(20, 30)
    with must_raise(ValueError, "positive"):
        crest.pp.normalize_total(frame(X, names), 0)


@baseline_fails("v0.4.0", AssertionError, "nothing was raised")
def test_a_matrix_of_zeros_has_no_median_total():
    zero = frame(sp.csr_matrix((20, 30), dtype=np.float32), [f"G{i}" for i in range(30)])
    with must_raise(ValueError, "median|target_sum"):
        crest.pp.normalize_total(zero)


def _pipeline(bf_or_a, kind, hv=None, max_value=None):
    """normalize (no arguments) -> log1p -> HVG mask -> scale (no arguments) -> PCA, for either tool."""
    if kind == "crest":
        bf = bf_or_a
        crest.pp.normalize_total(bf)
        crest.pp.log1p(bf)
        crest.pp.highly_variable_genes(bf, n_top_genes=150, flavor="seurat")
        crest.pp.scale(bf)
        crest.tl.pca(bf, n_comps=N_ANGLE_PCS)
        return bf.obsm["X_pca"], bf.var["highly_variable"].to_numpy()
    a = bf_or_a
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a = a[:, hv].copy()
    sc.pp.scale(a)
    sc.tl.pca(a, n_comps=N_ANGLE_PCS, svd_solver="arpack")
    return a.obsm["X_pca"], a.X


@baseline_fails("v0.4.0", AssertionError, "angle|differ|Mismatch")
def test_scale_without_arguments_does_not_clip_and_pca_equals_scanpy():
    X, names, _ = make_counts()
    bf, a = frame(X, names), scanpy_obj(X, names)
    pcs, hv = _pipeline(bf, "crest")
    ref_pcs, scaled = _pipeline(a, "scanpy", hv=hv)
    assert np.abs(np.asarray(scaled)).max() > 10, "fixture never exceeds the old clipping value"
    angle = max_angle_deg(pcs, ref_pcs)
    assert angle < PCA_ANGLE_DEG, f"principal angle between CREST's and scanpy's top PCs is {angle:.4f} degrees (angle > {PCA_ANGLE_DEG})"


@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
@baseline_fails("v0.4.0", AssertionError, "angle")
def test_scale_without_arguments_pca_equals_scanpy_pbmc3k():
    bf = crest.pp.filter_genes(crest.read_10x_mtx(PBMC3K), min_cells=3)
    a = scanpy_obj(bf.to_scipy(transform=False), bf.var_names)
    pcs, hv = _pipeline(bf, "crest")
    ref_pcs, _ = _pipeline(a, "scanpy", hv=hv)
    angle = max_angle_deg(pcs, ref_pcs)
    assert angle < PCA_ANGLE_DEG, f"principal angle {angle:.4f} degrees (angle > {PCA_ANGLE_DEG})"


def test_explicit_scale_max_value_still_clips():  # pin
    X, names, _ = make_counts()
    bf, a = frame(X, names), scanpy_obj(X, names)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    crest.pp.highly_variable_genes(bf, n_top_genes=150, flavor="seurat")
    hv = bf.var["highly_variable"].to_numpy()
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=N_ANGLE_PCS)
    a = a[:, hv].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=N_ANGLE_PCS, svd_solver="arpack")
    assert max_angle_deg(bf.obsm["X_pca"], a.obsm["X_pca"]) < PCA_ANGLE_DEG


@baseline_fails("v0.4.0", AssertionError, "None|assert")
def test_docstrings_state_the_scanpy_defaults():
    inspect_sig = __import__("inspect").signature
    assert inspect_sig(crest.pp.normalize_total).parameters["target_sum"].default is None
    assert inspect_sig(crest.pp.scale).parameters["max_value"].default is None
    assert "median" in crest.pp.normalize_total.__doc__ and "clip" in crest.pp.scale.__doc__


# --------------------------------------------------------------------------- count filters after a transform
@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_filter_cells_min_counts_after_normalize_total_warns():
    X, names, _ = make_counts()
    bf = frame(X, names)
    crest.pp.normalize_total(bf, 1e4)
    with recorded_warnings() as w:
        crest.pp.filter_cells(bf, min_counts=COUNT_FILTER_MIN)
    assert_warned(w, "raw counts")


@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_filter_genes_min_counts_after_log1p_warns():
    X, names, _ = make_counts()
    bf = frame(X, names)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    with recorded_warnings() as w:
        crest.pp.filter_genes(bf, min_counts=5.0)
    assert_warned(w, "raw counts")


@baseline_fails("v0.4.0", TypeError, "transformed")
def test_transformed_count_filters_equal_scanpy():
    X, names, _ = make_counts()
    bf, a = frame(X, names), scanpy_obj(X, names)
    crest.pp.normalize_total(bf, 1e4)
    sc.pp.normalize_total(a, target_sum=1e4)
    got = crest.pp.filter_cells(bf, min_counts=COUNT_FILTER_MIN, transformed=True)
    keep, _ = sc.pp.filter_cells(a, min_counts=COUNT_FILTER_MIN, inplace=False)
    assert got.n_obs == int(keep.sum()) == N_CELLS and got.n_obs > 0
    crest.pp.log1p(bf)
    sc.pp.log1p(a)
    thr = float(np.median(np.asarray(a.X.sum(0)).ravel()))
    got = crest.pp.filter_genes(bf, min_counts=thr, transformed=True)
    keep, _ = sc.pp.filter_genes(a, min_counts=thr, inplace=False)
    assert got.n_vars == int(keep.sum()) and 0 < got.n_vars < N_GENES
    assert got.var["gene_id"].to_list() == np.flatnonzero(keep).tolist()


@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
@baseline_fails("v0.4.0", TypeError, "transformed")
def test_transformed_count_filters_equal_scanpy_pbmc3k():
    bf = crest.read_10x_mtx(PBMC3K)
    a = scanpy_obj(bf.to_scipy(transform=False), bf.var_names)
    crest.pp.normalize_total(bf, 1e4)
    sc.pp.normalize_total(a, target_sum=1e4)
    assert crest.pp.filter_cells(bf, min_counts=COUNT_FILTER_MIN, transformed=True).n_obs == a.n_obs
    crest.pp.log1p(bf)
    sc.pp.log1p(a)
    keep, _ = sc.pp.filter_genes(a, min_counts=5000, inplace=False)
    assert crest.pp.filter_genes(bf, min_counts=5000, transformed=True).n_vars == int(keep.sum())


def test_count_filters_on_raw_counts_are_unchanged_and_silent():  # pin
    X, names, _ = make_counts()
    bf, a = frame(X, names), scanpy_obj(X, names)
    with recorded_warnings() as w:
        got = crest.pp.filter_cells(bf, min_counts=300, min_genes=40)
        genes = crest.pp.filter_genes(bf, min_counts=20, min_cells=3)
    assert not [x for x in w if "raw counts" in str(x.message)]
    keep, _ = sc.pp.filter_cells(a, min_counts=300, inplace=False)
    keep2, _ = sc.pp.filter_cells(a, min_genes=40, inplace=False)
    assert got.n_obs == int((keep & keep2).sum())
    gk, _ = sc.pp.filter_genes(a, min_counts=20, inplace=False)
    gk2, _ = sc.pp.filter_genes(a, min_cells=3, inplace=False)
    assert genes.n_vars == int((gk & gk2).sum())


def test_n_genes_and_n_cells_filters_do_not_warn_after_a_transform():  # pin
    X, names, _ = make_counts()
    bf = frame(X, names)
    crest.pp.normalize_total(bf, 1e4)
    with recorded_warnings() as w:
        crest.pp.filter_cells(bf, min_genes=40)
        crest.pp.filter_genes(bf, min_cells=3)
    assert not [x for x in w if "raw counts" in str(x.message)]


# --------------------------------------------------------------------------- ingest after a cell filter of the reference
def _reference_and_query():
    X, names, labels = make_counts(500, 300, seed=4)
    Y, _, _ = make_counts(150, 300, seed=5)
    ref, qry = frame(X, names), frame(Y, names)
    for b in (ref, qry):
        crest.pp.normalize_total(b, 1e4)
        crest.pp.log1p(b)
    crest.pp.highly_variable_genes(ref, n_top_genes=120, flavor="seurat")
    crest.pp.scale(ref, max_value=10)
    crest.tl.pca(ref, n_comps=30)
    ref.obs = ref.obs.with_columns(pl.Series("label", labels.astype(str)))
    crest.pp.neighbors(ref, n_neighbors=30, n_pcs=10)
    return ref, qry


@baseline_fails("rc0", AssertionError, "n_neighbors")
def test_ingest_after_cell_filter_keeps_neighbour_settings():
    ref, qry = _reference_and_query()
    filtered = crest.pp.filter_cells(ref, min_genes=60)
    assert 0 < filtered.n_obs < ref.n_obs
    fresh = filtered.copy()
    crest.pp.neighbors(fresh, n_neighbors=30, n_pcs=10)  # the settings spelled out on the filtered reference
    a = crest.tl.ingest(qry.copy(), filtered, obs="label")
    b = crest.tl.ingest(qry.copy(), fresh, obs="label")
    assert a.uns["ingest"]["n_neighbors"] == 30, f"ingest used n_neighbors={a.uns['ingest']['n_neighbors']} instead of the reference's 30"
    np.testing.assert_array_equal(a.uns["ingest"]["indices"], b.uns["ingest"]["indices"])
    assert a.obs["label"].to_list() == b.obs["label"].to_list()


@baseline_fails("rc0", AssertionError, "nothing was raised")
def test_ingest_without_any_neighbour_settings_raises_instead_of_guessing():
    ref, qry = _reference_and_query()
    ref.uns.pop("neighbors")
    with must_raise(ValueError, "neighbo"):
        crest.tl.ingest(qry, ref)
    out = crest.tl.ingest(qry.copy(), ref, n_neighbors=12)  # explicit value is honoured
    assert out.uns["ingest"]["n_neighbors"] == 12


def test_ingest_on_an_unfiltered_reference_is_unchanged():  # pin
    ref, qry = _reference_and_query()
    out = crest.tl.ingest(qry, ref, obs="label")
    assert out.uns["ingest"]["n_neighbors"] == 30 and out.uns["ingest"]["indices"].max() < ref.n_obs


# --------------------------------------------------------------------------- small deviations from scanpy, documented
@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_second_log1p_warns_and_records_one_transform():
    X, names, _ = make_counts(50, 30)
    bf = frame(X, names)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    with recorded_warnings() as w:
        crest.pp.log1p(bf)
    assert [o[0] for o in bf.ops] == ["normalize_total", "log1p"]
    assert_warned(w, "already applied")


def test_normalize_total_after_log1p_still_raises():  # pin
    X, names, _ = make_counts(50, 30)
    bf = frame(X, names)
    crest.pp.log1p(bf)
    with must_raise(ValueError, "before log1p"):
        crest.pp.normalize_total(bf, 1e4)


@baseline_fails("v0.4.0", AssertionError, "NotImplementedError")
def test_hvg_batch_key_for_the_dispersion_flavours_is_documented():
    X, names, labels = make_counts(200, 80)
    bf = frame(X, names)
    bf.obs = bf.obs.with_columns(pl.Series("batch", (labels % 2).astype(str)))
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    with must_raise(NotImplementedError, "seurat_v3"):
        crest.pp.highly_variable_genes(bf, n_top_genes=20, flavor="seurat", batch_key="batch")
    assert "NotImplementedError" in crest.pp.highly_variable_genes.__doc__


@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_pca_with_more_components_than_genes_warns_and_returns_the_available_ones():
    X, names, _ = make_counts(100, 20)
    bf = frame(X, names)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    with recorded_warnings() as w:
        crest.tl.pca(bf, n_comps=50, use_highly_variable=False)
    assert bf.obsm["X_pca"].shape == (100, 20)
    assert_warned(w, "n_comps=50")


def _frame_with_an_empty_cell():
    X, names, _ = make_counts(40, 30, seed=6)
    X = X.tolil()
    X[7, :] = 0
    return frame(X.tocsr(), names)


@baseline_fails("v0.4.0", AssertionError, "NaN|nan")
def test_pct_counts_mt_is_nan_for_a_cell_without_counts_as_in_scanpy():
    bf = _frame_with_an_empty_cell()
    obs, _ = crest.pp.calculate_qc_metrics(bf, inplace=False)
    pct = obs["pct_counts_mt"].to_numpy()
    assert np.isnan(pct[7]), f"pct_counts_mt of the empty cell is {pct[7]}, scanpy gives NaN"
    assert np.isfinite(np.delete(pct, 7)).all()
    a = scanpy_obj(bf.to_scipy(transform=False), bf.var_names)
    a.var["mt"] = [n.startswith("MT-") for n in bf.var_names]
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, inplace=True)
    np.testing.assert_allclose(pct, a.obs["pct_counts_mt"].to_numpy(), equal_nan=True, rtol=1e-6)


@baseline_fails("v0.4.0", AssertionError, "empty cell")
def test_max_pct_mt_removes_a_cell_without_counts_like_scanpy():
    bf = _frame_with_an_empty_cell()
    kept = crest.pp.filter_cells(bf, max_pct_mt=100.0)
    assert 7 not in kept.obs["cell_id"].to_list(), "the empty cell survived max_pct_mt"
