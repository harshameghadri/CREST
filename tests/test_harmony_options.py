"""Harmony options: the native default of ``max_iter_kmeans``, a per-cluster ``sigma`` and the
harmonypy version named in the documentation.

``sigma`` is the width of the soft cluster assignment. harmonypy 2.1.0 takes one value or one value
per cluster; the native binding and ``run_harmony`` must do the same, and ``max_iter_kmeans`` must
default to 4 (harmonypy's value) in the binding as well as in the Python function.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import crest
from crest import crest as _native

# parity with harmonypy 2.1.0 on the 2-batch design below, relative L2 of the correction
# (per-cluster sigma: measured 0.0006; harmonypy seed-to-seed 0.0158 on a related design)
PARITY_REL_L2_SIGMA_ARRAY = 0.05
OBJECTIVE_RTOL = 1e-12  # parallel reductions may differ in the last bit
N_CELLS, N_DIM, N_CLUSTERS = 2000, 20, 66
REPO = Path(__file__).resolve().parents[1]
DOC_FILES = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]


def _design(seed=0):
    """Three cell types, two batches with a batch shift of about 5 along the first axis."""
    rng = np.random.default_rng(seed)
    types = rng.integers(0, 3, N_CELLS)
    batch = rng.integers(0, 2, N_CELLS)
    centres = rng.normal(0, 3.0, (3, N_DIM))
    Z = centres[types] + rng.normal(0, 1.0, (N_CELLS, N_DIM))
    Z[:, 0] += 5.0 * batch
    return np.ascontiguousarray(Z, dtype=np.float32), batch.astype(str)


def _native_call(Z, b, **kw):
    codes = np.ascontiguousarray((b == "1").astype(np.uint32)[None, :])
    return _native.harmony(Z, codes, [2], N_CLUSTERS, [2.0, 2.0], **kw)


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="native default")
def test_native_default_max_iter_kmeans_is_four():
    Z, b = _design()
    default = _native_call(Z, b)
    explicit = _native_call(Z, b, max_iter_kmeans=4)
    assert default["kmeans_rounds"] == explicit["kmeans_rounds"], (
        f"native default max_iter_kmeans differs from 4: rounds {default['kmeans_rounds']} vs {explicit['kmeans_rounds']}")
    np.testing.assert_array_equal(default["Z_corr"], explicit["Z_corr"])


def test_wrapper_and_native_defaults_agree():
    Z, b = _design()
    via_wrapper = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS)
    assert via_wrapper["kmeans_rounds"] == _native_call(Z, b, max_iter_kmeans=4)["kmeans_rounds"]


@pytest.mark.baseline_fails(ref="v0.4.0", raises=TypeError, match="0-dimensional")
def test_per_cluster_sigma_array_is_accepted():
    Z, b = _design()
    out = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS, sigma=np.linspace(0.05, 0.3, N_CLUSTERS))
    assert out["Z_corr"].shape == Z.shape and np.isfinite(out["Z_corr"]).all()
    assert out["R"].shape == (N_CELLS, N_CLUSTERS)


@pytest.mark.baseline_fails(ref="v0.4.0", raises=TypeError, match="real number")
def test_constant_sigma_array_equals_scalar():
    Z, b = _design()
    scalar = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS, sigma=0.15)
    as_list = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS, sigma=[0.15] * N_CLUSTERS)
    np.testing.assert_array_equal(scalar["Z_corr"], as_list["Z_corr"])
    np.testing.assert_allclose(scalar["objective_harmony"], as_list["objective_harmony"], rtol=OBJECTIVE_RTOL, atol=0)


@pytest.mark.baseline_fails(ref="v0.4.0", raises=TypeError, match="0-dimensional")
def test_varying_sigma_changes_the_result():
    Z, b = _design()
    scalar = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS, sigma=0.1)["Z_corr"]
    varied = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS, sigma=np.linspace(0.05, 0.4, N_CLUSTERS))["Z_corr"]
    assert np.abs(scalar - varied).max() > 1e-3


@pytest.mark.baseline_fails(ref="v0.4.0", raises=TypeError, match="0-dimensional")
def test_per_cluster_sigma_matches_harmonypy():
    harmonypy = pytest.importorskip("harmonypy")
    pd = pytest.importorskip("pandas")
    Z, b = _design()
    sigma = np.linspace(0.05, 0.3, N_CLUSTERS)
    ours = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS, sigma=sigma, random_state=0)["Z_corr"].astype(np.float64)
    ref = np.asarray(harmonypy.run_harmony(Z, pd.DataFrame({"b": b}), ["b"], nclust=N_CLUSTERS, sigma=sigma,
                                           random_state=0, verbose=False).Z_corr, dtype=np.float64)
    if ref.shape != Z.shape:
        ref = ref.T
    d_ours, d_ref = ours - Z, ref - Z
    assert np.linalg.norm(d_ref) > 1.0, "the design must need a correction"
    rel = np.linalg.norm(d_ours - d_ref) / np.linalg.norm(d_ref)
    print(f"sigma-array parity rel L2 {rel:.4f}")
    assert rel < PARITY_REL_L2_SIGMA_ARRAY, rel


_ARRAY_NOT_ACCEPTED = pytest.mark.baseline_fails(ref="v0.4.0", raises=TypeError, match="0-dimensional")
_ONE = np.full(N_CLUSTERS, 0.1)


@pytest.mark.parametrize("sigma", [
    pytest.param(_ONE[:-1], marks=_ARRAY_NOT_ACCEPTED, id="too-short"),
    pytest.param(np.r_[_ONE, 0.1], marks=_ARRAY_NOT_ACCEPTED, id="too-long"),
    pytest.param(np.stack([_ONE, _ONE]), marks=_ARRAY_NOT_ACCEPTED, id="two-dimensional"),
    pytest.param(np.r_[0.0, _ONE[1:]], marks=_ARRAY_NOT_ACCEPTED, id="array-with-zero"),
    pytest.param(np.r_[-0.1, _ONE[1:]], marks=_ARRAY_NOT_ACCEPTED, id="array-with-negative"),
    pytest.param(np.r_[np.nan, _ONE[1:]], marks=_ARRAY_NOT_ACCEPTED, id="array-with-nan"),
    pytest.param(np.r_[np.inf, _ONE[1:]], marks=_ARRAY_NOT_ACCEPTED, id="array-with-inf"),
    pytest.param(0.0, id="scalar-zero"),
    pytest.param(-1.0, id="scalar-negative"),
    pytest.param(float("nan"), id="scalar-nan"),
])
def test_invalid_sigma_raises(sigma):
    Z, b = _design()
    with pytest.raises(ValueError, match="sigma"):
        crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS, sigma=sigma)


@pytest.mark.baseline_fails(ref="v0.4.0", raises=TypeError, match="real number")
def test_native_binding_validates_sigma_length():
    Z, b = _design()
    with pytest.raises(ValueError, match="sigma"):
        _native_call(Z, b, sigma=[0.1] * (N_CLUSTERS - 1))


def test_scalar_sigma_unchanged():
    Z, b = _design()
    out = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS, sigma=0.1)
    again = crest.harmony.run_harmony(Z, [b], nclust=N_CLUSTERS)
    np.testing.assert_array_equal(out["Z_corr"], again["Z_corr"])


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="harmonypy version")
def test_docs_name_the_compared_harmonypy_release():
    assert all(p.exists() for p in DOC_FILES[:2]), DOC_FILES
    stale = []
    for p in DOC_FILES:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"harmonypy\s+2\.0(\.\d+)?\b", line):
                stale.append(f"{p.name}:{i}")
    assert not stale, f"harmonypy version in the documentation is not the compared release 2.1.0: {stale}"
