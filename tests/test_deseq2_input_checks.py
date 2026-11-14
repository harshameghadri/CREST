"""Input checks of ``crest.tl.DESeq2`` that mirror R's, and the variants it does not support.

Tests marked ``baseline_fails`` fail on the stated released build with the declared exception and pass on
this tree. Unmarked tests pin behaviour that is already correct.
"""

import numpy as np
import polars as pl
import pytest

import crest


def _must_raise(exc, fn, match):
    import re
    try:
        fn()
    except exc as e:
        assert re.search(match, str(e)), f"raised {type(e).__name__}({e}) which does not match {match!r}"
        return e
    raise AssertionError(f"did not raise {exc.__name__}")


def _table(n_genes=60, n=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(10, 0.2, (n, n_genes)).astype(float), pl.DataFrame({"condition": ["a"] * (n // 2) + ["b"] * (n // 2)})


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="did not raise")
@pytest.mark.parametrize("value", [0, 1, 2, 2.9, float("nan")])
def test_replacement_threshold_below_three_raises(value):
    """R stops: a count outlier cannot be identified with fewer than three replicates."""
    cts, obs = _table()
    _must_raise(ValueError, lambda: crest.tl.DESeq2(cts, obs, quiet=True, min_replicates_for_replace=value), "at least 3")


@pytest.mark.parametrize("value", [3, 7, float("inf")])
def test_replacement_threshold_of_three_or_more_runs(value):
    cts, obs = _table()
    ds = crest.tl.DESeq2(cts, obs, quiet=True, min_replicates_for_replace=value)
    assert ds.fit["dispersion"].shape == (cts.shape[1],)


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AttributeError, match="height")
def test_pandas_obs_raises_type_error():
    pd = pytest.importorskip("pandas")
    cts, obs = _table()
    _must_raise(TypeError, lambda: crest.tl.DESeq2(cts, obs.to_pandas(), quiet=True), "polars")
    assert isinstance(obs, pl.DataFrame) and pd is not None


def test_size_factor_type_iterate_is_not_available():
    """R also offers sfType = "iterate"; here only "ratio" and "poscounts" exist, and the call says so."""
    cts, obs = _table()
    _must_raise(ValueError, lambda: crest.tl.DESeq2(cts, obs, quiet=True, sf_type="iterate"), "sf_type")


def test_identical_counts_in_every_sample_run_at_the_dispersion_floor():
    """When every gene-wise dispersion is at the floor R's trend fit stops; here the fit continues with the
    mean trend and dispersion 10 x the floor (documented in docs/deseq2.md)."""
    cts = np.tile(np.array([10.0, 12.0, 11.0, 10.0, 12.0, 11.0]), (30, 1)) * (1 + np.arange(30)[:, None] % 3)
    obs = pl.DataFrame({"condition": ["a"] * 3 + ["b"] * 3})
    ds = crest.tl.DESeq2(cts.T, obs, quiet=True)
    assert ds.fit["fitType"] == "mean" and np.allclose(ds.fit["dispersion"], 1e-7)
