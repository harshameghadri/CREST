"""Scrublet neighbour convention, ingest parity statements, the counts warning in ``normalize_total`` and
the loess error on degenerate predictors.

Tests marked ``baseline_fails`` fail on the released 0.4.0 wheel for the stated reason and pass on this
tree; unmarked tests (pins) describe behaviour that was already right and must hold on both. The real-data
tests run when ``CREST_TEST_PBMC3K`` points at a directory with PBMC 3k's 10x MTX files.
"""

import contextlib
import os
import re
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import crest
from crest import doublets as _doublets

sp = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata")
stats = pytest.importorskip("scipy.stats")

# ---- named tolerances / sizes
PBMC3K = os.environ.get("CREST_TEST_PBMC3K")
SEEDS = (0, 1, 2, 3)
N_SPIKED = 150
MEAN_SCORE_RTOL = 0.02       # mean observed score over the seeds against scanpy's (measured 0.4% with the self neighbour, 6-11% without)
SPEARMAN_FLOOR = 0.94        # per-seed rank agreement; the two tools agree with themselves across seeds at 0.957-0.963 only
BRUTE_FORCE_ATOL = 1e-9      # scores against a brute-force neighbour search over the same embedding
PROJECTION_ATOL = 1e-4       # ingest projection against scanpy's own code (measured 1.5e-5 on PBMC 3k)
LOESS_ATOL = 1e-8            # non-singular fits against skmisc.loess


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


def make_counts(n_cells=300, n_genes=200, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.3, 1.0, n_genes)
    labels = rng.integers(0, 3, n_cells)
    prog = np.ones((3, n_genes))
    for c in range(3):
        up = rng.choice(n_genes, 25, replace=False)
        prog[c, up] *= rng.uniform(4, 12, 25)
    lib = rng.lognormal(6.3, 0.4, n_cells)
    mu = prog[labels] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    return sp.csr_matrix(rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32))


def frame(X):
    return crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": [f"G{i}" for i in range(X.shape[1])]}))


# --------------------------------------------------------------------------- Scrublet neighbour convention
def _capture_knn(monkeypatch):
    """Record every embedding and k that Scrublet hands to the neighbour search."""
    calls = []
    real = _doublets._native.knn_graph

    def spy(M, k, **kw):
        calls.append((np.array(M), int(k)))
        return real(M, k, **kw)

    monkeypatch.setattr(_doublets._native, "knn_graph", spy)
    return calls


def _expected_scores(M, n, k_search, include_self, rho=0.05, r=2.0):
    """Doublet scores from a brute-force neighbour search over the embedding ``M`` (first ``n`` rows observed)."""
    d2 = ((M[:, None, :].astype(np.float64) - M[None, :, :].astype(np.float64)) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)  # the cell itself is never one of the searched neighbours
    idx = np.argsort(d2, axis=1, kind="stable")[:, :k_search]
    nd = (idx >= n).sum(1).astype(np.float64)
    kk = float(k_search)
    if include_self:
        nd[n:] += 1.0  # a simulated cell is its own doublet neighbour
        kk += 1.0
    q = (nd + 1) / (kk + 2)
    return (q * rho / r / (1 - rho - q * (1 - rho - rho / r)))[:n]


@pytest.mark.parametrize("include_self", [True, False])
@baseline_fails("v0.4.0", TypeError, "include_self_neighbor|AssertionError")
def test_scrublet_scores_follow_the_selected_neighbour_convention(monkeypatch, include_self):
    X = make_counts(260, 220, seed=3)
    calls = _capture_knn(monkeypatch)
    bf = frame(X)
    crest.pp.scrublet(bf, random_state=1, include_self_neighbor=include_self)
    M, k = calls[-1]
    n = X.shape[0]
    n_neighbors = int(round(0.5 * np.sqrt(n)))
    k_adj = int(round(n_neighbors * (1 + int(n * 2.0) / float(n))))
    assert k == (k_adj - 1 if include_self else k_adj), f"searched {k} neighbours for k_adj={k_adj}"
    got = bf.obs["doublet_score"].to_numpy()
    want = _expected_scores(M, n, k, include_self)
    np.testing.assert_allclose(got, want, atol=BRUTE_FORCE_ATOL)
    assert bf.uns["scrublet"]["parameters"]["include_self_neighbor"] is include_self


@baseline_fails("v0.4.0", AssertionError, "mean|k_adj")
def test_scrublet_default_is_the_scanpy_convention(monkeypatch):
    """Default call: k_adj - 1 other cells are searched (the cell itself is the k_adj-th neighbour)."""
    X = make_counts(260, 220, seed=4)
    calls = _capture_knn(monkeypatch)
    bf = frame(X)
    crest.pp.scrublet(bf, random_state=0)
    M, k = calls[-1]
    n = X.shape[0]
    k_adj = int(round(int(round(0.5 * np.sqrt(n))) * 3.0))
    assert k == k_adj - 1, f"default searched {k} neighbours; the scanpy convention is k_adj - 1 = {k_adj - 1}"


def _scanpy_scrublet_works():
    sc = pytest.importorskip("scanpy")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ad.AnnData(make_counts(250, 150, seed=9))
            sc.pp.scrublet(a, random_state=0)
    except Exception as e:  # an environment whose pynndescent / scipy break scanpy's neighbour search
        pytest.skip(f"sc.pp.scrublet does not run here: {type(e).__name__}")
    return sc


@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
@baseline_fails("v0.4.0", AssertionError, "mean observed")
def test_scrublet_scores_on_pbmc3k_with_spiked_doublets_agree_with_scanpy():
    sc = _scanpy_scrublet_works()
    bf0 = crest.read_10x_mtx(PBMC3K)
    X = bf0.to_scipy(transform=False).tocsr()
    n = X.shape[0]
    crest_means, scanpy_means, rhos = [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        i = rng.integers(0, n, N_SPIKED)
        j = (i + rng.integers(1, n, N_SPIKED)) % n
        Y = sp.vstack([X, X[i] + X[j]]).tocsr()
        bf = crest.BioFrame.from_scipy(Y, var=bf0.var.select(["gene_ids", "gene_name"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crest.pp.scrublet(bf, random_state=seed)
            a = ad.AnnData(Y.astype(np.float32))
            a.var_names = bf.var_names
            a.var_names_make_unique()
            sc.pp.scrublet(a, random_state=seed)
        c = bf.obs["doublet_score"].to_numpy()
        s = a.obs["doublet_score"].to_numpy()
        ok = ~np.isnan(c)
        crest_means.append(c[ok].mean())
        scanpy_means.append(s[ok].mean())
        rhos.append(stats.spearmanr(c[ok], s[ok])[0])
    ratio = np.mean(crest_means) / np.mean(scanpy_means)
    assert abs(ratio - 1) < MEAN_SCORE_RTOL, \
        f"mean observed score is {ratio:.3f} times scanpy's over {len(SEEDS)} seeds (per-seed Spearman {np.round(rhos, 3).tolist()})"
    assert min(rhos) > SPEARMAN_FLOOR, f"Spearman {np.round(rhos, 3).tolist()}"


# --------------------------------------------------------------------------- ingest statements and projection parity
def _text(rel):
    root = Path(__file__).resolve().parents[1]
    f = root / rel
    if not f.exists():
        pytest.skip(f"{rel} is not next to the tests")
    return f.read_text(encoding="utf8")


@baseline_fails("v0.4.0", AssertionError, "scanpy 1.11|1.12.3")
def test_ingest_docstrings_name_the_scanpy_versions():
    for doc in (crest.ingest.__doc__, crest.ingest.project_pca.__doc__, crest.tl.ingest.__doc__):
        assert "1.12.3" in doc and "1.11" in doc, f"a parity statement without scanpy versions: {doc[:120]!r}"


def test_ingest_parity_statements_in_the_text_name_the_scanpy_version():
    down = _text("docs/downstream.md")
    section = down[down.index("## Reference mapping"):]
    assert "1.11.5" in section and "1.12.3" in section and "own mean" in section
    for rel in ("README.md", "docs/benchmarks.md"):
        for line in _text(rel).splitlines():
            if "ingest" in line and re.search(r"\b\d+(\.\d+)?%? vs \d+(\.\d+)?%", line):
                assert "scanpy 1.1" in line or "scanpy 1.1" in _text(rel).split(line)[0][-400:], f"{rel}: {line[:100]}"


def _scanpy_projection(pcs, query_scaled, mean_ref):
    """scanpy's own ``Ingest._pca`` with the neighbour and UMAP initialisers left out."""
    from scanpy.tools._ingest import Ingest

    ing = Ingest.__new__(Ingest)
    ing._pca_centered, ing._pca_use_hvg, ing._pca_basis = True, False, pcs
    ing._adata_new = query_scaled
    if hasattr(Ingest, "_init_pca") and "_pca_mean" in Ingest._init_pca.__code__.co_names:  # scanpy >= 1.12.3
        ing._pca_mean = mean_ref
        return ing._pca(None), "reference"
    return ing._pca(None), "query"


def test_ingest_projection_equals_scanpy_when_the_query_is_scaled_with_the_reference_statistics():  # pin
    pytest.importorskip("scanpy")
    X = make_counts(500, 160, seed=5)
    ref, qry = frame(X[:350]), frame(X[350:])
    for b in (ref, qry):
        crest.pp.normalize_total(b, 1e4)
        crest.pp.log1p(b)
    crest.pp.scale(ref, max_value=10)
    crest.tl.pca(ref, n_comps=20, use_highly_variable=False)
    P = ref.uns["pca"]["projection"]
    mu, sd = np.asarray(P["mean"]), np.asarray(P["std"])
    Q = qry.to_scipy().toarray().astype(np.float64)
    Qs = np.clip((Q - mu) / sd, -10, 10)  # what CREST applies internally, done by the caller for scanpy
    a_new = ad.AnnData(Qs.astype(np.float32))
    pcs = np.asarray(P["loadings"], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got_sc, which = _scanpy_projection(pcs, a_new, mean_ref=np.zeros(pcs.shape[0]))
    got = crest.ingest.project_pca(qry, ref, center=which)
    np.testing.assert_allclose(got, got_sc, atol=PROJECTION_ATOL)


# --------------------------------------------------------------------------- counts warning in normalize_total
@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_normalize_total_warns_for_non_integer_input():
    X = make_counts(60, 40, seed=6)
    L = X.copy()
    L.data = np.log1p(L.data) + 0.37
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf = frame(L)
    with recorded_warnings() as w:
        crest.pp.normalize_total(bf, 1e4)
    assert_warned(w, "not look like raw counts")


@baseline_fails("v0.4.0", AssertionError, "UserWarning")
def test_normalize_total_warns_for_negative_entries_with_non_negative_totals():
    X = make_counts(40, 30, seed=7).tolil()
    X[2, 5] = X[2, 5] + 400.0
    X[2, 6] = -3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf = frame(X.tocsr())
    with recorded_warnings() as w:
        crest.pp.normalize_total(bf, 1e4)
    assert_warned(w, "negative")


def test_normalize_total_of_integer_counts_is_silent():  # pin
    bf = frame(make_counts(60, 40, seed=8))
    with recorded_warnings() as w:
        crest.pp.normalize_total(bf, 1e4)
    assert not [x for x in w if "raw counts" in str(x.message)]


# --------------------------------------------------------------------------- loess on degenerate predictors
loess_mod = pytest.importorskip("skmisc.loess")


def _skmisc_raises(x, y, span, degree):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = loess_mod.loess(x, y, span=span, degree=degree)
            m.fit()
        return None, m.outputs.fitted_values
    except Exception as e:  # noqa: BLE001 - skmisc raises a plain ValueError with a message from the C code
        return e, None


@baseline_fails("v0.4.0", AssertionError, "nothing was raised")
def test_loess_raises_on_heavily_tied_predictor_like_skmisc():
    from crest._loess import loess_fit

    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 3000).astype(float)
    y = rng.normal(size=3000)
    assert _skmisc_raises(x, y, 0.3, 2)[0] is not None  # the reference stops with an error on this input
    with must_raise(ValueError, "singular"):
        loess_fit(x, y, 0.3, 2)


@baseline_fails("v0.4.0", AssertionError, "disagree")
def test_loess_raises_exactly_where_skmisc_does_and_agrees_elsewhere():
    from crest._loess import loess_fit

    rng = np.random.default_rng(5)
    disagree, compared = [], 0
    for _ in range(120):
        n = int(rng.choice([100, 400, 1000]))
        nd = int(rng.integers(1, 12))
        span = float(rng.choice([0.2, 0.3, 0.5, 0.75]))
        degree = int(rng.choice([1, 2]))
        x = (rng.integers(0, nd, n) * rng.uniform(0.5, 2)).astype(float) if rng.random() < 0.6 \
            else np.round(rng.normal(size=n), int(rng.integers(0, 3)))
        y = rng.normal(size=n)
        err, ref = _skmisc_raises(x, y, span, degree)
        try:
            fit = loess_fit(x, y, span, degree)
        except ValueError:
            fit = None
        if (err is None) != (fit is not None):
            disagree.append((n, len(np.unique(x)), span, degree))
        elif fit is not None:
            compared += 1
            np.testing.assert_allclose(fit, ref, atol=LOESS_ATOL)
    assert not disagree, f"{len(disagree)} of 120 cases disagree with skmisc about raising: {disagree[:5]}"
    assert compared > 20


def test_loess_matches_skmisc_on_continuous_predictors():  # pin
    from crest._loess import loess_fit

    rng = np.random.default_rng(2)
    for _ in range(40):
        n = int(rng.choice([200, 800]))
        x = rng.normal(size=n)
        y = np.sin(x) + rng.normal(scale=0.3, size=n)
        err, ref = _skmisc_raises(x, y, 0.3, 2)
        assert err is None
        np.testing.assert_allclose(loess_fit(x, y, 0.3, 2), ref, atol=LOESS_ATOL)
