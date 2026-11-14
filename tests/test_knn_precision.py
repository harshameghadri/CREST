"""k-nearest-neighbour search: precision of the exact path, non-finite input, queries far
from the reference, and the approximate index above 20k cells.

Every ``baseline_fails`` test fails on the named released build and passes on this one;
tests without the marker pin behaviour that was already right.
"""

import re
import warnings

import numpy as np
import pytest

import crest
from crest import crest as _native

baseline_fails = pytest.mark.baseline_fails

# ---- named tolerances / sizes ---------------------------------------------------------
K = 14                           # neighbours per cell (n_neighbors = 15)
RECALL_VS_TRUTH = 0.999          # exact paths against float64 truth on the float32 array passed
OFFSETS = (100.0, 1.0e3, 1.0e4)  # coordinate offsets added to the embedding
N_EXACT = 3000                   # cells in the offset tests
N_SCORED = 600                   # cells whose neighbour sets are scored in the offset tests
QUERY_RECALL_IN_DISTRIBUTION = 0.998
QUERY_RECALL_OFF_MANIFOLD = 0.98
N_REFERENCE_IVF = 60_000         # reference cells of the IVF query tests (117 inverted lists)
N_QUERY = 1500
OFF_MANIFOLD_FRACTION_BELOW_08 = 0.05
SHIFT_SD = 0.7                   # shifted queries: every column moved by this many column sds
N_LARGE = 120_000                # cells of the large approximate-search test
N_LARGE_SCORED = 800
LARGE_MEAN_RECALL = 0.99         # tier 1 for the approximate search above 100k cells
LARGE_FRACTION_BELOW_HALF = 0.005
EXACT_LIMIT = 20_000


# ---- helpers --------------------------------------------------------------------------
def embedding(n, d=50, seed=0, n_clusters=25, norm=15.0):
    """Clustered embedding with PCA-like decaying column scales and a median row norm of ``norm``."""
    rng = np.random.default_rng(seed)
    sd = 1.0 / np.sqrt(1.0 + np.arange(d))
    centres = rng.normal(size=(n_clusters, d)) * sd * 3.0
    lab = rng.integers(0, n_clusters, n)
    X = centres[lab] + rng.normal(size=(n, d)) * sd
    X -= X.mean(0)
    X *= norm / np.median(np.linalg.norm(X, axis=1))
    return np.ascontiguousarray(X, dtype=np.float32)


def truth_sets(ref, query, k, rows=None, self_ids=None):
    """float64 brute-force neighbour ids of ``query`` rows among ``ref`` rows, computed on the
    float32 arrays as passed; ``self_ids`` gives each query's own row in ``ref`` (excluded)."""
    R = ref.astype(np.float64)
    mu = R.mean(0)
    R -= mu
    rn = (R ** 2).sum(1)
    Q = query.astype(np.float64) - mu
    out = np.empty((len(Q), k), dtype=np.int64)
    step = max(8, int(2e7 // len(R)))
    for a in range(0, len(Q), step):
        q = Q[a:a + step]
        D = (q ** 2).sum(1)[:, None] + rn[None, :] - 2.0 * q @ R.T
        if self_ids is not None:
            D[np.arange(len(q)), self_ids[a:a + step]] = np.inf
        out[a:a + step] = np.argpartition(D, k - 1, axis=1)[:, :k]
    return out


def row_recall(found, truth):
    return np.array([len(set(f.tolist()) & set(t.tolist())) for f, t in zip(found, truth)]) / truth.shape[1]


def graph_recall(X, exact, rows):
    idx, _ = _native.knn_graph(X, K, exact=exact)
    return row_recall(idx[rows], truth_sets(X, X[rows], K, self_ids=rows))


def raises_value_error(fn, match):
    try:
        out = fn()
    except ValueError as e:
        assert re.search(match, str(e)), f"ValueError message {str(e)!r} does not match {match!r}"
        return
    raise AssertionError(f"expected ValueError matching {match!r}, the call returned {type(out).__name__}")


class _Frame:
    """Minimal stand-in for the part of a BioFrame that ``pp.neighbors`` touches."""

    def __init__(self, X, key="X_pca"):
        self.obsm = {key: X}
        self.uns = {}


# ---- precision of the exact path ------------------------------------------------------
@pytest.fixture(scope="module")
def pca_like():
    return embedding(N_EXACT, seed=1)


@pytest.fixture(scope="module")
def scored_rows():
    return np.random.default_rng(0).choice(N_EXACT, N_SCORED, replace=False)


@pytest.mark.parametrize("exact", [True, False])
def test_centred_embedding_matches_float64_truth(pca_like, scored_rows, exact):
    r = graph_recall(pca_like, exact, scored_rows)
    assert r.mean() >= RECALL_VS_TRUTH, f"recall {r.mean():.4f}"


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("offset", [
    pytest.param(c, marks=baseline_fails(ref="v0.4.0", raises=AssertionError, match="recall"), id=f"offset{c:g}")
    for c in OFFSETS])
def test_neighbours_do_not_depend_on_the_offset_of_the_embedding(pca_like, scored_rows, exact, offset):
    X = np.ascontiguousarray(pca_like + np.float32(offset))
    r = graph_recall(X, exact, scored_rows)
    assert r.mean() >= RECALL_VS_TRUTH, f"recall {r.mean():.4f} at offset {offset:g} (exact={exact})"


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="recall")
def test_small_local_scale_next_to_a_large_offset_is_exact(pca_like, scored_rows):
    X = np.ascontiguousarray(pca_like * np.float32(0.01) + np.float32(1.0e3))
    r = graph_recall(X, True, scored_rows)
    assert r.mean() >= RECALL_VS_TRUTH, f"recall {r.mean():.4f}"


def test_exact_distances_are_sorted_and_exclude_the_cell(pca_like):
    idx, dist = _native.knn_graph(pca_like, K, exact=True)
    assert not (idx == np.arange(len(idx))[:, None]).any()
    assert (np.diff(dist, axis=1) >= 0).all()
    ref = np.linalg.norm(pca_like[:50, None, :].astype(np.float64) - pca_like[idx[:50]].astype(np.float64), axis=2)
    np.testing.assert_allclose(dist[:50], ref, rtol=1e-5)


@pytest.mark.parametrize("offset", [
    pytest.param(c, marks=baseline_fails(ref="v0.4.0", raises=AssertionError, match="recall"), id=f"offset{c:g}")
    for c in OFFSETS])
def test_knn_query_exact_does_not_depend_on_the_offset(offset):
    R = embedding(2500, seed=3)
    Q = embedding(300, seed=4)
    R2, Q2 = np.ascontiguousarray(R + np.float32(offset)), np.ascontiguousarray(Q + np.float32(offset))
    idx, _ = _native.knn_query(R2, Q2, K, exact=True)
    r = row_recall(idx, truth_sets(R2, Q2, K))
    assert r.mean() >= RECALL_VS_TRUTH, f"recall {r.mean():.4f} at offset {offset:g}"


# ---- NaN / Inf ------------------------------------------------------------------------
BAD_VALUES = [np.nan, np.inf, -np.inf]
_NOT_RAISED = dict(ref="v0.4.0", raises=AssertionError, match="expected ValueError")


@pytest.mark.parametrize("bad", [pytest.param(b, marks=baseline_fails(**_NOT_RAISED), id=str(b)) for b in BAD_VALUES])
@pytest.mark.parametrize("exact", [None, True, False])
def test_knn_graph_rejects_non_finite_rows(bad, exact):
    X = embedding(600, seed=5)
    X[17, 3] = bad
    X[40, 0] = np.nan
    raises_value_error(lambda: _native.knn_graph(X, 10, exact=exact), r"NaN or Inf.*\b17\b")


@pytest.mark.parametrize("bad", [pytest.param(b, marks=baseline_fails(**_NOT_RAISED), id=str(b)) for b in BAD_VALUES])
def test_knn_query_rejects_non_finite_rows(bad):
    R, Q = embedding(500, seed=6), embedding(60, seed=7)
    Rbad, Qbad = R.copy(), Q.copy()
    Rbad[321, 1] = bad
    Qbad[12, 2] = bad
    raises_value_error(lambda: _native.knn_query(Rbad, Q, 5), r"reference.*NaN or Inf.*\b321\b")
    raises_value_error(lambda: _native.knn_query(R, Qbad, 5), r"query.*NaN or Inf.*\b12\b")
    raises_value_error(lambda: _native.knn_query(Rbad, Qbad, 5, exact=False), r"NaN or Inf")


@pytest.mark.parametrize("bad", [pytest.param(b, marks=baseline_fails(**_NOT_RAISED), id=str(b)) for b in BAD_VALUES])
def test_neighbors_rejects_non_finite_embedding(bad):
    X = embedding(400, seed=8)
    X[9, 4] = bad
    bf = _Frame(X)
    raises_value_error(lambda: crest.pp.neighbors(bf), r"X_pca.*NaN or Inf.*first bad row: 9")
    assert "neighbors" not in bf.uns


def test_non_finite_values_beyond_n_pcs_are_not_checked():
    X = embedding(400, d=20, seed=9)
    X[9, 15] = np.nan
    crest.pp.neighbors(_Frame(X), n_pcs=10)


def test_clean_input_gives_the_same_neighbours_as_float64_truth():
    X = embedding(800, seed=10)
    bf = crest.pp.neighbors(_Frame(X), n_neighbors=11, exact=True)
    idx = bf.uns["neighbors"]["indices"]
    assert idx.shape == (800, 10)
    assert row_recall(idx, truth_sets(X, X, 10, self_ids=np.arange(800))).mean() >= RECALL_VS_TRUTH


# ---- queries far from the reference ----------------------------------------------------
@pytest.fixture(scope="module")
def ivf_query_sets():
    X = embedding(N_REFERENCE_IVF + N_QUERY, d=30, seed=11, norm=12.0)
    R, Q_in = np.ascontiguousarray(X[:N_REFERENCE_IVF]), np.ascontiguousarray(X[N_REFERENCE_IVF:])
    sd = R.std(0)
    rng = np.random.default_rng(12)
    sets = {
        "in_distribution": Q_in,
        "shifted": np.ascontiguousarray(Q_in + (SHIFT_SD * sd).astype(np.float32)),
        "off_manifold": (rng.normal(size=Q_in.shape) * sd).astype(np.float32),
    }
    truth = {name: truth_sets(R, Q, K) for name, Q in sets.items()}
    return R, sets, truth


def query_recall(ivf_query_sets, name, **kw):
    R, sets, truth = ivf_query_sets
    idx, _ = _native.knn_query(R, sets[name], K, exact=False, **kw)
    return row_recall(idx, truth[name])


def test_query_in_distribution_recall(ivf_query_sets):
    assert query_recall(ivf_query_sets, "in_distribution").mean() >= QUERY_RECALL_IN_DISTRIBUTION


def test_shifted_query_recall(ivf_query_sets):
    assert query_recall(ivf_query_sets, "shifted").mean() >= QUERY_RECALL_OFF_MANIFOLD


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="recall")
def test_off_manifold_query_recall(ivf_query_sets):
    r = query_recall(ivf_query_sets, "off_manifold")
    assert r.mean() >= QUERY_RECALL_OFF_MANIFOLD, f"recall {r.mean():.4f} for off-manifold queries"
    assert (r < 0.8).mean() <= OFF_MANIFOLD_FRACTION_BELOW_08, f"{(r < 0.8).mean():.3f} of queries below 0.8"


def test_query_ivf_is_deterministic_for_a_fixed_seed(ivf_query_sets):
    R, sets, _ = ivf_query_sets
    Q = sets["off_manifold"][:400]
    a = _native.knn_query(R, Q, K, exact=False, seed=3)
    b = _native.knn_query(R, Q, K, exact=False, seed=3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_query_exact_path_is_unchanged_for_off_manifold_queries(ivf_query_sets):
    R, sets, truth = ivf_query_sets
    idx, _ = _native.knn_query(R[:15000], sets["off_manifold"][:200], K, exact=True)
    assert row_recall(idx, truth_sets(R[:15000], sets["off_manifold"][:200], K)).mean() >= RECALL_VS_TRUTH


# ---- the approximate index above 20k cells --------------------------------------------
@pytest.fixture(scope="module")
def large_embedding():
    return embedding(N_LARGE, d=30, seed=13, norm=12.0)


def _run_neighbors(X, **kw):
    bf = _Frame(X)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crest.pp.neighbors(bf, **kw)
    return bf, [w for w in caught if issubclass(w.category, UserWarning) and "approximate" in str(w.message)]


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="warning")
def test_implicit_approximate_search_warns_once_and_is_recorded():
    X = embedding(EXACT_LIMIT + 2000, d=10, seed=14)
    bf, warned = _run_neighbors(X)
    assert len(warned) == 1, f"expected one approximate-search warning, got {len(warned)}"
    p = bf.uns["neighbors"]["params"]
    assert p["knn_method"] == "approximate"
    assert p["nprobe"] >= 20 and p["n_lists"] >= p["nprobe"]
    assert p["n_neighbors"] == 15 and p["method"] == "umap"


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="knn_method")
def test_explicit_choices_do_not_warn_and_are_recorded():
    X = embedding(EXACT_LIMIT + 2000, d=10, seed=14)
    for exact, method in ((True, "exact"), (False, "approximate")):
        bf, warned = _run_neighbors(X, exact=exact)
        assert not warned
        assert bf.uns["neighbors"]["params"].get("knn_method") == method, "knn_method not recorded"
    bf, warned = _run_neighbors(X[:2000])
    assert not warned
    p = bf.uns["neighbors"]["params"]
    assert (p.get("knn_method"), p.get("nprobe"), p.get("n_lists")) == ("exact", None, None)


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="nprobe")
def test_nprobe_grows_with_the_number_of_lists():
    small, _ = _run_neighbors(embedding(21_000, d=5, seed=16), exact=False)
    large, _ = _run_neighbors(embedding(N_LARGE + 20_000, d=5, seed=17), exact=False)
    p_small, p_large = (b.uns["neighbors"]["params"] for b in (small, large))
    assert p_small.get("nprobe") == 20, f"nprobe {p_small.get('nprobe')} at {p_small.get('n_lists')} lists"
    assert p_large.get("nprobe", 0) > 20, f"nprobe {p_large.get('nprobe')} at {p_large.get('n_lists')} lists"


def test_plan_matches_the_search_that_runs():
    """``knn_plan_info`` (new binding, no released-build counterpart) reports the search ``knn_graph`` runs."""
    plans = {n: _native.knn_plan_info(n, None) for n in (30_000, 120_000, 300_000, 1_000_000)}
    assert all(not exact for exact, _, _ in plans.values())
    probes = [v[1] for v in plans.values()]
    assert probes == sorted(probes) and probes[-1] > probes[0] >= 20
    assert all(v[1] <= v[2] for v in plans.values())
    assert _native.knn_plan_info(EXACT_LIMIT, None)[0] is True
    assert _native.knn_plan_info(10 * EXACT_LIMIT, True)[0] is True
    assert _native.knn_plan_info(5, False)[0] is True


def test_large_approximate_search_recall_and_determinism(large_embedding):
    X = large_embedding
    rows = np.random.default_rng(15).choice(len(X), N_LARGE_SCORED, replace=False)
    idx, dist = _native.knn_graph(X, K, seed=2)
    r = row_recall(idx[rows], truth_sets(X, X[rows], K, self_ids=rows))
    assert r.mean() >= LARGE_MEAN_RECALL, f"recall {r.mean():.4f}"
    assert (r < 0.5).mean() <= LARGE_FRACTION_BELOW_HALF
    assert (np.diff(dist, axis=1) >= 0).all()
    assert not (idx == np.arange(len(idx))[:, None]).any()
    idx2, dist2 = _native.knn_graph(X, K, seed=2)
    np.testing.assert_array_equal(idx, idx2)
    np.testing.assert_array_equal(dist, dist2)
