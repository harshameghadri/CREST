"""Scrublet threshold: parity of ``threshold_minimum`` with skimage (including the
cases where skimage raises), and the warnings / uns diagnostics for failed or
degenerate automatic thresholds (bench/doublets: Kang 2018, where both CREST and
scanpy put some batches' threshold in the tail of the simulated-score histogram)."""

import warnings

import numpy as np
import polars as pl
import pytest

import crest
import crest.doublets as D
from crest.doublets import threshold_minimum

sp = pytest.importorskip("scipy.sparse")


def ski_or_none(v, **kw):
    """skimage's threshold, or None where it raises RuntimeError (as scanpy catches it)."""
    ski = pytest.importorskip("skimage.filters")
    try:
        return float(ski.threshold_minimum(v, **kw))
    except RuntimeError:
        return None


def quantized_scores(seed, n=20000, k=177, n_sat=2):
    """Scrublet-shaped simulated doublet scores: the Bayesian score of
    Binomial(k, p) simulated neighbours (so ~k distinct values), one broad mode,
    and ``n_sat`` scores saturated at the maximum, as on Kang 2018."""
    rng = np.random.default_rng(seed)
    nd = rng.binomial(k, rng.beta(2.0, 1.5, n)).astype(np.float64)
    nd[:n_sat] = k
    rho, r = 0.05, 2.0
    q = (nd + 1) / (k + 2)
    return q * rho / r / (1 - rho - q * (1 - rho - rho / r))


def n_iterations(v, nbins=256):
    """Smoothing iterations skimage needs before the histogram has < 3 maxima."""
    ski = pytest.importorskip("skimage.filters")
    lo, hi = 1, 10000
    while lo < hi:  # the smallest max_num_iter that skimage accepts, minus one
        mid = (lo + hi) // 2
        try:
            ski.threshold_minimum(v, nbins=nbins, max_num_iter=mid)
            hi = mid
        except RuntimeError:
            lo = mid + 1
    return lo - 1


@pytest.mark.parametrize("seed", range(10))
def test_threshold_minimum_parity_quantized_tail(seed):
    # the Kang failure mode: skimage itself splits between the main mode and the saturated tail
    v = quantized_scores(seed)
    ref = ski_or_none(v)
    assert ref is not None and (v > ref).mean() < 0.01
    assert threshold_minimum(v) == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_threshold_minimum_parity_bimodal(seed):  # the clean case: two modes
    rng = np.random.default_rng(seed)
    v = np.r_[rng.normal(0.1, 0.03, 4000), rng.normal(0.6, 0.1, 1500)]
    ref = ski_or_none(v)
    assert ref is not None
    assert threshold_minimum(v) == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("v", [
    np.linspace(0.0, 1.0, 5000),                                # flat: no maximum survives
    np.random.default_rng(66).exponential(1.0, 5000),           # one maximum left after smoothing
    np.r_[np.zeros(10), np.ones(30)],                           # two end spikes
    np.full(100, 0.3),                                          # constant
], ids=["flat", "exponential", "end_spikes", "constant"])
def test_threshold_minimum_none_where_skimage_raises(v):
    assert ski_or_none(v) is None, "fixture must make skimage raise RuntimeError"
    assert threshold_minimum(v) is None


@pytest.mark.parametrize("seed", range(3))
def test_threshold_minimum_max_iteration_matches_skimage(seed):
    # skimage raises "Maximum iteration reached" when two maxima appear only on the last
    # allowed iteration; one more iteration and it returns the threshold
    v = quantized_scores(seed)
    it = n_iterations(v)
    assert it > 1
    assert ski_or_none(v, max_num_iter=it) is None
    assert threshold_minimum(v, max_num_iter=it) is None
    ref = ski_or_none(v, max_num_iter=it + 1)
    assert ref is not None
    assert threshold_minimum(v, max_num_iter=it + 1) == pytest.approx(ref, rel=1e-12)


def two_batches(n=800, g=300, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, g)
    lab = rng.integers(0, 4, n)
    prog = np.ones((4, g))
    for c in range(4):
        prog[c, rng.choice(g, 40, replace=False)] *= rng.uniform(3, 10, 40)
    mu = prog[lab] * base
    mu = mu / mu.sum(1, keepdims=True) * rng.lognormal(7.5, 0.4, n)[:, None]
    X = sp.csr_matrix(rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32))
    obs = pl.DataFrame({"batch": np.where(np.arange(n) < n // 2, "a", "b")})
    return crest.BioFrame.from_scipy(X, obs=obs)


def test_scrublet_warns_and_records_failed_threshold(monkeypatch):
    real = D.threshold_minimum
    calls = []

    def fail_first(v, *a, **kw):  # batch "a": threshold_minimum finds no two maxima
        calls.append(1)
        return None if len(calls) == 1 else real(v, *a, **kw)

    monkeypatch.setattr(D, "threshold_minimum", fail_first)
    bf = two_batches()
    with pytest.warns(UserWarning, match="failed to automatically identify.*batch 'a'"):
        crest.pp.scrublet(bf, batch_key="batch")
    u = bf.uns["scrublet"]
    assert u["thresholds"]["a"] is None and u["threshold_status"]["a"] == "failed"
    assert "threshold" not in u["batches"]["a"]
    assert u["threshold_status"]["b"] == "auto" and u["thresholds"]["b"] == u["batches"]["b"]["threshold"]
    pred = bf.obs["predicted_doublet"].to_numpy()
    a = (bf.obs["batch"] == "a").to_numpy()
    assert not pred[a].any()  # as scanpy: predicted_doublet = False where the threshold failed


def test_scrublet_warns_on_degenerate_threshold(monkeypatch):
    # an automatic threshold above which < 1% of simulated doublets score (the Kang tail case)
    monkeypatch.setattr(D, "threshold_minimum", lambda v, *a, **kw: float(np.quantile(v, 0.999)))
    bf = two_batches()
    with pytest.warns(UserWarning, match="lies in the tail") as rec:
        crest.pp.scrublet(bf, batch_key="batch")
    assert sum("lies in the tail" in str(w.message) for w in rec) == 2
    u = bf.uns["scrublet"]
    for b in ("a", "b"):
        assert u["threshold_status"][b] == "degenerate"
        assert u["batches"][b]["detectable_doublet_fraction"] < 0.01
        assert u["thresholds"][b] == u["batches"][b]["threshold"]


def test_scrublet_no_warning_on_healthy_or_user_threshold():
    bf = two_batches()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="scrublet")
        crest.pp.scrublet(bf, batch_key="batch")
        assert set(bf.uns["scrublet"]["threshold_status"].values()) == {"auto"}
        # a user threshold is never second-guessed, even one that calls nothing
        crest.pp.scrublet(bf, batch_key="batch", threshold=10.0)
    assert set(bf.uns["scrublet"]["threshold_status"].values()) == {"user"}
    assert bf.uns["scrublet"]["thresholds"] == {"a": 10.0, "b": 10.0}
