"""Harmony's fixed ridge penalty ``lamb``: invalid values raise, valid ones match harmonypy.

With a ridge of zero the per-cluster design (intercept plus batch indicators) is
collinear, so the system is singular; a negative or non-finite penalty has no
meaning either. ``lamb=-1`` is the "estimate it" sentinel in CREST and in
harmonypy and is not an error.
"""

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

import crest

# parity with harmonypy on the 2-batch design below (relative L2 of the correction)
PARITY_REL_L2_TINY_LAMB = 0.01     # lamb = 1e-6 (measured 0.0000)
PARITY_REL_L2_UNIT_LAMB = 0.05     # lamb = 1.0  (measured 0.0200)
N_CELLS, N_DIM = 2000, 20

_DID_NOT_RAISE = dict(ref="v0.4.0", raises=pytest.fail.Exception, match="DID NOT RAISE")


def _design(seed=0):
    """Three cell types, two batches with a batch shift of about 5 along the first axis."""
    rng = np.random.default_rng(seed)
    types = rng.integers(0, 3, N_CELLS)
    batch = rng.integers(0, 2, N_CELLS)
    centres = rng.normal(0, 3.0, (3, N_DIM))
    Z = centres[types] + rng.normal(0, 1.0, (N_CELLS, N_DIM))
    Z[:, 0] += 5.0 * batch
    return np.ascontiguousarray(Z, dtype=np.float32), batch.astype(str)


@pytest.mark.parametrize(
    "lamb",
    [
        pytest.param(0.0, marks=pytest.mark.baseline_fails(**_DID_NOT_RAISE), id="zero"),
        pytest.param(-2.0, marks=pytest.mark.baseline_fails(**_DID_NOT_RAISE), id="negative"),
        pytest.param(float("inf"), marks=pytest.mark.baseline_fails(**_DID_NOT_RAISE), id="inf"),
        pytest.param(
            float("nan"),
            marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="did not match"),
            id="nan",
        ),
    ],
)
def test_invalid_lamb_raises(lamb):
    Z, b = _design()
    with pytest.raises(ValueError, match="lamb"):
        crest.harmony.run_harmony(Z, [b], lamb=lamb)


@pytest.mark.baseline_fails(**_DID_NOT_RAISE)
def test_zero_lamb_with_duplicated_covariate_raises():
    Z, b = _design()
    with pytest.raises(ValueError, match="lamb"):
        crest.harmony.run_harmony(Z, [b, b], lamb=0.0)


@pytest.mark.baseline_fails(**_DID_NOT_RAISE)
def test_zero_lamb_with_three_batches_raises():
    Z, b = _design()
    b3 = np.where(np.arange(len(b)) % 3 == 0, "x", b)
    with pytest.raises(ValueError, match="lamb"):
        crest.harmony.run_harmony(Z, [b3], lamb=0.0)


@pytest.mark.baseline_fails(**_DID_NOT_RAISE)
def test_negative_per_level_lamb_raises():
    Z, b = _design()
    with pytest.raises(ValueError, match="lamb"):
        crest.harmony.run_harmony(Z, [b], lamb=[1.0, -2.0])


@pytest.mark.baseline_fails(**_DID_NOT_RAISE)
def test_zero_lamb_raises_through_the_bioframe_api():
    Z, b = _design()
    bf = crest.BioFrame.from_scipy(sp.csr_matrix((len(b), 1), dtype=np.float32), obs=pl.DataFrame({"batch": b}))
    bf.obsm["X_pca"] = Z
    with pytest.raises(ValueError, match="lamb"):
        crest.tl.harmony(bf, "batch", lamb=0.0)
    assert "X_pca_harmony" not in bf.obsm


def test_minus_one_is_the_estimate_sentinel():
    Z, b = _design()
    est = crest.harmony.run_harmony(Z, [b], lamb=None)["Z_corr"]
    sentinel = crest.harmony.run_harmony(Z, [b], lamb=-1)["Z_corr"]
    np.testing.assert_array_equal(est, sentinel)
    assert np.abs(est - Z).max() > 0.1  # the default path really corrects


@pytest.mark.parametrize("lamb, bound", [(1e-6, PARITY_REL_L2_TINY_LAMB), (1.0, PARITY_REL_L2_UNIT_LAMB)])
def test_fixed_lamb_matches_harmonypy(lamb, bound):
    harmonypy = pytest.importorskip("harmonypy")
    pd = pytest.importorskip("pandas")
    Z, b = _design()
    ours = crest.harmony.run_harmony(Z, [b], lamb=lamb, random_state=0)["Z_corr"].astype(np.float64)
    ref = np.asarray(harmonypy.run_harmony(Z, pd.DataFrame({"b": b}), ["b"], lamb=lamb, random_state=0,
                                           verbose=False).Z_corr, dtype=np.float64)
    if ref.shape != Z.shape:
        ref = ref.T
    d_ours, d_ref = ours - Z, ref - Z
    rel = np.linalg.norm(d_ours - d_ref) / np.linalg.norm(d_ref)
    assert np.linalg.norm(d_ref) > 1.0, "the design must need a correction"
    assert rel < bound, rel
