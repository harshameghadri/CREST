"""Input validation in the native bindings: memory layout, edge weights and
parameters of the graph kernels, and the public-binding edge cases (rank-sum
counts, ``k`` limits, ``t_pvalues`` degrees of freedom, panics)."""

import re
import warnings

import numpy as np
import pytest

from crest import crest as _native

sp = pytest.importorskip("scipy.sparse")
stats = pytest.importorskip("scipy.stats")

# a defect test fails on the named baseline with the named exception
baseline_fails = pytest.mark.baseline_fails

# ---- named tolerances / limits ------------------------------------------------------
T_PVALUE_RTOL = 1e-6        # t_pvalues vs scipy over the whole df range (measured worst case ~1e-8)
T_PVALUE_ATOL = 1e-300
LEIDEN_WEIGHT_FLOOR = 1e-6  # the bindings warn when the largest edge weight is below this
WEIGHT_SCALE_BELOW_FLOOR = 1e-9
WEIGHT_SCALE_ABOVE_FLOOR = 1e-3
N_FUZZ_SEEDS = 2


# ---- helpers --------------------------------------------------------------------------
def raises_value_error(fn, match):
    """The call must raise ValueError whose message matches ``match``."""
    try:
        out = fn()
    except ValueError as e:
        assert re.search(match, str(e)), f"ValueError message {str(e)!r} does not match {match!r}"
        return
    raise AssertionError(f"expected ValueError matching {match!r}, the call returned {type(out).__name__}")


def flat(obj):
    if isinstance(obj, dict):
        return [np.asarray(obj[k]) for k in sorted(obj) if isinstance(obj[k], np.ndarray)]
    if isinstance(obj, (tuple, list)):
        return [a for o in obj for a in flat(o)]
    return [np.asarray(obj)]


def c_or_identical(call, args_c, args_f):
    """Fortran-ordered inputs must raise 'C-contiguous' or give the C-ordered result exactly."""
    ref = flat(call(*args_c()))
    try:
        got = flat(call(*args_f()))
    except ValueError as e:
        if "C-contiguous" not in str(e):
            raise AssertionError(f"Fortran-ordered input was accepted and misread (ValueError: {e})")
        return
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            raise AssertionError("Fortran-ordered input was accepted and changed the result")


def blobs(n=300, d=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(k, d))
    lab = rng.integers(0, k, n)
    return (centers[lab] + rng.normal(size=(n, d))).astype(np.float32)


@pytest.fixture(scope="module")
def graph():
    X = blobs(400)
    idx, dist = _native.knn_graph(X, 14)
    rows, cols, w = _native.connectivities(idx, dist, 15)
    return len(X), rows, cols, w


def chunk_ctx(n_cells=60, n_genes=20, seed=1):
    import crest
    m = sp.random(n_cells, n_genes, density=0.4, format="csr", dtype=np.float32, random_state=seed)
    m.data = np.ceil(m.data * 9).astype(np.float32)
    bf = crest.BioFrame.from_scipy(m)
    return list(next(bf.iter_ctx(transform=False))), n_cells, n_genes


# ======================================================================================
# memory layout
# ======================================================================================
F_ORDER_MSG = "Fortran-ordered input was accepted"
f_marker = baseline_fails(ref="v0.4.0", raises=AssertionError, match=F_ORDER_MSG)


@f_marker
def test_knn_graph_fortran_order():
    X = blobs(200)
    c_or_identical(lambda x: _native.knn_graph(x, 10), lambda: (X,), lambda: (np.asfortranarray(X),))


@f_marker
def test_knn_query_fortran_order():
    R, Q = blobs(200), blobs(50, seed=2)
    c_or_identical(lambda r, q: _native.knn_query(r, q, 10), lambda: (R, Q),
                   lambda: (np.asfortranarray(R), np.asfortranarray(Q)))


@f_marker
def test_connectivities_fortran_order():
    idx, dist = _native.knn_graph(blobs(200), 14)
    c_or_identical(lambda i, d: _native.connectivities(i, d, 15), lambda: (idx, dist),
                   lambda: (np.asfortranarray(idx), np.asfortranarray(dist)))


@f_marker
def test_harmony_fortran_order():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(300, 5)).astype(np.float32)
    lab = rng.integers(0, 2, 300)
    batch = np.vstack([lab, rng.integers(0, 3, 300) + 2]).astype(np.uint32)

    def call(z, b):
        return _native.harmony(z, b, [2, 3], 6, [2.0] * 5, None, 0.1, 0.2, 0.05, 3, 4, 1e-3, 1e-2, 1e-5, 0)

    c_or_identical(call, lambda: (z, batch), lambda: (np.asfortranarray(z), np.asfortranarray(batch)))
    c_or_identical(call, lambda: (z, batch), lambda: (z, np.asfortranarray(batch)))


@f_marker
def test_deseq2_fit_fortran_order():
    rng = np.random.default_rng(4)
    counts = rng.negative_binomial(5, 0.3, size=(40, 8)).astype(np.float64)
    design = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])

    def call(c, d):
        return _native.deseq2_fit(c, d)

    c_or_identical(call, lambda: (counts, design), lambda: (np.asfortranarray(counts), design))
    c_or_identical(call, lambda: (counts, design), lambda: (counts, np.asfortranarray(design)))
    reduced = np.ones((8, 1))
    c_or_identical(lambda c, d, r: _native.deseq2_fit(c, d, None, "ratio", "parametric", 7.0, 0.5, r),
                   lambda: (counts, design, reduced), lambda: (counts, design, np.asfortranarray(design[:, :1] * 1.0)))


@f_marker
def test_chunk_kernels_fortran_order_outputs_and_matrices():
    ctx, n_cells, n_genes = chunk_ctx()
    G = 3
    grp = (np.arange(n_cells) % G).astype(np.uint32)

    def totals(s):
        _native.group_gene_totals(*ctx, grp, s)
        return np.asarray(s)

    c_or_identical(totals, lambda: (np.zeros((G, n_genes)),), lambda: (np.asfortranarray(np.zeros((G, n_genes))),))

    d = 6
    sub = np.where(np.arange(n_genes) < d, np.arange(n_genes), -1).astype(np.int32)

    def gram(g):
        cs = np.zeros(d)
        _native.gram_accumulate(*ctx, sub, g, cs, False, None, None, 0.0)
        return np.asarray(g), cs

    c_or_identical(gram, lambda: (np.zeros((d, d)),), lambda: (np.asfortranarray(np.zeros((d, d))),))

    rng = np.random.default_rng(5)
    load = rng.normal(size=(d, 4))

    def project(L, out):
        _native.project(*ctx, sub, L, out, False, None, None, 0.0)
        return np.asarray(out)

    c_or_identical(project, lambda: (load, np.zeros((n_cells, 4), np.float32)),
                   lambda: (np.asfortranarray(load), np.zeros((n_cells, 4), np.float32)))
    c_or_identical(project, lambda: (load, np.zeros((n_cells, 4), np.float32)),
                   lambda: (load, np.asfortranarray(np.zeros((n_cells, 4), np.float32))))


def test_strided_and_one_dimensional_arrays_keep_working():
    X = blobs(100, d=8)
    raises_value_error(lambda: _native.knn_graph(np.ascontiguousarray(X)[:, ::2], 5), "C-contiguous")
    # a single row or column is both C- and Fortran-contiguous and is accepted
    one_col = np.ascontiguousarray(blobs(50)[:, :1])
    assert np.asfortranarray(one_col).flags.c_contiguous
    a = _native.knn_graph(one_col, 3)
    b = _native.knn_graph(np.asfortranarray(one_col), 3)
    assert np.array_equal(a[0], b[0])


# ======================================================================================
# edge weights and parameters of the graph kernels
# ======================================================================================
W_MSG = "expected ValueError"
w_marker = baseline_fails(ref="v0.4.0", raises=AssertionError, match=W_MSG)
BAD_WEIGHTS = {"nan": np.nan, "inf": np.inf, "minus_inf": -np.inf, "negative": -0.5}


def _with_weight(graph, value, pos=3):
    n, rows, cols, w = graph
    w = w.copy()
    w[pos] = value
    return n, rows, cols, w


@w_marker
@pytest.mark.parametrize("kind", sorted(BAD_WEIGHTS))
def test_leiden_rejects_invalid_weights(graph, kind):
    n, rows, cols, w = _with_weight(graph, BAD_WEIGHTS[kind])
    raises_value_error(lambda: _native.leiden(n, rows, cols, w), "weight")
    raises_value_error(lambda: _native.leiden_sweep(n, rows, cols, w, [1.0], [0]), "weight")


@w_marker
@pytest.mark.parametrize("kind", sorted(BAD_WEIGHTS))
def test_umap_rejects_invalid_weights(graph, kind):
    n, rows, cols, w = _with_weight(graph, BAD_WEIGHTS[kind])
    raises_value_error(lambda: _native.umap(n, rows, cols, w), "weight")


@w_marker
def test_invalid_weight_on_a_dropped_diagonal_entry_is_rejected_too(graph):
    n, rows, cols, w = graph
    rows, cols = np.append(rows, np.uint32(0)), np.append(cols, np.uint32(0))
    w = np.append(w, np.float32(np.nan))
    raises_value_error(lambda: _native.leiden(n, rows, cols, w), "weight")


@w_marker
def test_umap_rejects_inconsistent_parameters(graph):
    n, rows, cols, w = graph
    raises_value_error(lambda: _native.umap(n, rows, cols, w, 2, 2.0, 1.0), "min_dist")
    raises_value_error(lambda: _native.umap(n, rows, cols, w, 2, 0.5, np.inf), "spread")
    raises_value_error(lambda: _native.umap(n, rows, cols, w, 2, np.inf, np.inf), "min_dist|spread")


def test_valid_weights_still_accepted(graph):
    n, rows, cols, w = graph
    w0 = w.copy()
    w0[:5] = 0.0  # zero weights are legal (an absent edge)
    lab = _native.leiden(n, rows, cols, w0)
    assert lab.shape == (n,) and 1 < len(np.unique(lab)) < n // 10
    emb = _native.umap(n, rows, cols, w0, 2, 0.5, 1.0, 20)
    assert emb.shape == (n, 2) and np.isfinite(emb).all()
    assert _native.leiden_sweep(n, rows, cols, w0, [0.5, 1.0], [0, 1]).shape == (2, n)
    # parameters that were valid stay valid (scanpy defaults, min_dist == spread)
    assert np.isfinite(_native.umap(n, rows, cols, w0, 2, 1.0, 1.0, 5)).all()
    _native.umap(n, rows, cols, w0, 2, 0.0, 1.0, 5)
    _native.umap(n, rows, cols, w0, 3, 0.5, 1.0, 5)


def _warns_small_weights(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return [str(r.message) for r in rec if re.search("edge weight", str(r.message))]


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="no warning")
def test_leiden_warns_when_the_largest_weight_is_below_the_floor(graph):
    n, rows, cols, w = graph
    small = (w / w.max() * np.float32(WEIGHT_SCALE_BELOW_FLOOR)).astype(np.float32)
    assert small.max() < LEIDEN_WEIGHT_FLOOR
    for fn in (lambda: _native.leiden(n, rows, cols, small), lambda: _native.leiden_sweep(n, rows, cols, small, [1.0], [0])):
        msgs = _warns_small_weights(fn)
        assert msgs, "no warning for a graph whose largest weight is below the floor"
        assert "rescale" in msgs[0].lower()


def test_leiden_does_not_warn_at_ordinary_weight_scales(graph):
    n, rows, cols, w = graph
    for scale in (1.0, 10.0, WEIGHT_SCALE_ABOVE_FLOOR):
        ws = (w / w.max() * np.float32(scale)).astype(np.float32)
        assert not _warns_small_weights(lambda: _native.leiden(n, rows, cols, ws))
    # an empty graph has no weights to judge
    assert not _warns_small_weights(lambda: _native.leiden(5, np.zeros(0, np.uint32), np.zeros(0, np.uint32), np.zeros(0, np.float32)))


# ======================================================================================
# public-binding hygiene
# ======================================================================================
@baseline_fails(ref="v0.4.0", raises=AssertionError, match="expected ValueError")
def test_wilcoxon_rejects_more_entries_than_cells():
    genes = np.zeros(3, np.uint32)
    vals = np.array([1.0, 2.0, 3.0], np.float32)
    groups = np.zeros(3, np.uint32)
    raises_value_error(lambda: _native.wilcoxon_rank_sums(genes, vals, groups, 1, np.array([2], np.uint64)), "more .*entries")
    # per group: group 1 holds one cell but two non-zero entries
    genes, vals, groups = np.zeros(4, np.uint32), np.arange(1, 5, dtype=np.float32), np.array([0, 0, 1, 1], np.uint32)
    raises_value_error(lambda: _native.wilcoxon_rank_sums(genes, vals, groups, 1, np.array([3, 1], np.uint64)), "more .*entries")


def test_wilcoxon_rank_sums_match_scipy_on_valid_input():
    rng = np.random.default_rng(7)
    n_cells, n_genes = 40, 6
    dense = rng.poisson(0.7, size=(n_cells, n_genes)).astype(np.float32)
    grp = rng.integers(0, 3, n_cells)
    r, g = np.nonzero(dense)
    rs, ties = _native.wilcoxon_rank_sums(g.astype(np.uint32), dense[r, g], grp[r].astype(np.uint32), n_genes,
                                          np.bincount(grp, minlength=3).astype(np.uint64))
    for j in range(n_genes):
        ranks = stats.rankdata(dense[:, j])
        for k in range(3):
            assert rs[k, j] == pytest.approx(ranks[grp == k].sum())
        _, counts = np.unique(dense[:, j], return_counts=True)
        assert ties[j] == pytest.approx((counts.astype(float) ** 3 - counts).sum())
    # an all-zero gene and a gene present in every cell are fine
    full = np.arange(1, n_cells + 1, dtype=np.float32)
    rs, _ = _native.wilcoxon_rank_sums(np.zeros(n_cells, np.uint32), full, np.zeros(n_cells, np.uint32), 2,
                                       np.array([n_cells], np.uint64))
    assert rs[0, 0] == n_cells * (n_cells + 1) / 2 and rs[0, 1] == n_cells * (n_cells + 1) / 2


K_MSG = "expected ValueError"


@baseline_fails(ref="v0.4.0", raises=AssertionError, match=K_MSG)
def test_knn_k_zero_is_rejected():
    X = blobs(50)
    raises_value_error(lambda: _native.knn_graph(X, 0), "k must be")
    raises_value_error(lambda: _native.knn_query(X, X[:5], 0), "k must be")


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="no warning")
def test_knn_k_above_the_limit_warns_and_clamps():
    X = blobs(30)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        idx, _ = _native.knn_graph(X, 100)
    assert idx.shape == (30, 29)
    assert any("clamp" in str(r.message) and "29" in str(r.message) for r in rec), "no warning when k is clamped"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        idx, _ = _native.knn_query(X, X[:4], 50)
    assert idx.shape == (4, 30)
    assert any("clamp" in str(r.message) for r in rec), "no warning when k is clamped"


def test_knn_valid_k_is_silent():
    X = blobs(30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idx, _ = _native.knn_graph(X, 29)  # k == n - 1 is not a clamp
        _native.knn_graph(X, 5)
        _native.knn_query(X, X[:4], 30)    # k == n_reference is not a clamp
    assert idx.shape == (30, 29)


@baseline_fails(ref="v0.4.0", raises=BaseException, match="")  # a PanicException on v0.4.0; fixed earlier on the integration branch
def test_approximate_knn_of_two_points_is_the_exact_result():
    two = blobs(2)
    a = _native.knn_graph(two, 1, exact=False)
    b = _native.knn_graph(two, 1, exact=True)
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="differs from scipy")
def test_t_pvalues_follow_scipy_over_the_whole_df_range():
    df = np.array([1e-300, 1e-20, 1e-12, 1e-10, 1e-3, 0.5, 1.0, 2.5, 30.0, 1e4, 1e8, 1e9, 1e11, 1e15, 1e300, np.inf])
    for x in (0.0, 0.5, 2.0, 5.0, -3.0):
        t = np.full(len(df), x)
        got = _native.t_pvalues(t, df)
        ref = 2 * stats.t.sf(np.abs(t), df)
        bad = ~np.isclose(got, ref, rtol=T_PVALUE_RTOL, atol=T_PVALUE_ATOL)
        assert not bad.any(), f"t_pvalues differs from scipy at t={x}, df={df[bad]}: {got[bad]} vs {ref[bad]}"
    # invalid df / t stay NaN
    out = _native.t_pvalues(np.array([1.0, 1.0, 1.0, np.nan, np.inf]), np.array([0.0, -1.0, np.nan, 5.0, 5.0]))
    assert np.isnan(out).all()


def _fuzz_cases(seed):
    rng = np.random.default_rng(seed)

    def x(n, d):
        return rng.normal(size=(n, d)).astype(np.float32)

    cases = []
    for n in (0, 1, 2, 3):
        for d in (0, 1, 3):
            for k in (0, 1, 5):
                for ex in (None, True, False):
                    cases.append((f"knn_graph n={n} d={d} k={k} exact={ex}", lambda n=n, d=d, k=k, ex=ex: _native.knn_graph(x(n, d), k, ex, 0)))
    for nr in (0, 1, 2, 5):
        for nq in (0, 1, 3):
            for d in (0, 1, 3):
                for k in (0, 1, 4, 10):
                    for ex in (None, True, False):
                        cases.append((f"knn_query nr={nr} nq={nq} d={d} k={k} exact={ex}",
                                      lambda nr=nr, nq=nq, d=d, k=k, ex=ex: _native.knn_query(x(nr, d), x(nq, d), k, ex, 0)))
    for n in (0, 1, 3):
        for k in (0, 1, 2):
            idx = rng.integers(0, max(n, 1), size=(n, k)).astype(np.uint32)
            dist = np.abs(x(n, k))
            for nn in (0, 1, 15):
                cases.append((f"connectivities n={n} k={k} nn={nn}", lambda idx=idx, dist=dist, nn=nn: _native.connectivities(idx, dist, nn)))
                cases.append((f"connectivities out-of-range ids n={n} k={k} nn={nn}",
                              lambda idx=idx + 7, dist=dist, nn=nn: _native.connectivities(idx, dist, nn)))
    for ng in (0, 1, 3):
        for ns in (0, 1, 4):
            for k in (0, 1, 9):
                g = np.abs(rng.normal(size=(ng, ng)))
                cs = rng.normal(size=ng)
                cases.append((f"pca_from_gram d={ng} n={ns} k={k}", lambda g=g, cs=cs, ns=ns, k=k: _native.pca_from_gram(g, cs, ns, k)))
    for n in (0, 1, 10):
        for f in (0.0, np.nan, 10.0):
            cases.append((f"lowess n={n} f={f}", lambda n=n, f=f: _native.lowess(np.arange(n, dtype=float), np.arange(n, dtype=float), f, 3, None)))
    for g in (0, 1, 3):
        for m in (0, 1, 2, 6):
            for p in (0, 1, 2):
                c = rng.poisson(10, size=(g, m)).astype(float)
                cases.append((f"deseq2_fit g={g} m={m} p={p}", lambda c=c, m=m, p=p: _native.deseq2_fit(c, np.ones((m, p)))))
    return cases


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="BaseException")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("seed", range(N_FUZZ_SEEDS))
def test_no_panic_escapes_as_base_exception(seed):
    """Odd inputs either work or raise an ordinary Exception, never a PanicException."""
    escaped = []
    for name, fn in _fuzz_cases(seed):
        try:
            fn()
        except Exception:
            pass
        except BaseException as e:  # noqa: BLE001
            escaped.append(f"{name}: {type(e).__name__}")
    assert not escaped, f"{len(escaped)} calls raised a BaseException, first: {escaped[0]}"
