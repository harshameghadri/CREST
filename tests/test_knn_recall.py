"""Recall of the approximate neighbour search: cells near the edge of their inverted list.

The approximate search (above 20,000 cells, or ``exact=False``) groups the cells into inverted lists and
searches each cell against the lists that are nearest to it. Searching every cell of a list against the lists
around the centroid of *that list* loses the neighbours of cells that lie near the edge of their list: on a
continuum of cell states, roughly half of a cell's neighbours are in other lists, and those lists are not
always among the nearest ones to the list centroid. Every ``baseline_fails`` test fails on the released builds
and passes here; the others pin behaviour that was already right.

``CREST_TEST_KNN_EMBEDDING`` may name a real embedding (``.npy`` array, or ``.npz`` with a ``pca`` /
``X_pca`` array) for the last test, which applies the registered recall thresholds to it.
"""

import os

import numpy as np
import pytest

import crest
from crest import crest as _native

baseline_fails = pytest.mark.baseline_fails

# ---- named tolerances / sizes ---------------------------------------------------------
K = 14
N_CELLS = 60_000                 # 117 inverted lists of about 512 cells
N_COLUMNS = 50
N_SCORED = 1500
MEAN_RECALL = 0.995              # target with margin over the registered 0.99
FRACTION_BELOW_08 = 0.005        # target with margin over the registered 1.0%
FRACTION_BELOW_05 = 0.005        # registered
REGISTERED_MEAN_RECALL = 0.99
REGISTERED_FRACTION_BELOW_08 = 0.01
REAL_N_SCORED = 2000
WIDE_NPROBE = 40


# ---- helpers --------------------------------------------------------------------------
def continuum(n, d=N_COLUMNS, seed=1, kind="gaussian"):
    """A continuum of states with PCA-like column scales and a median row norm of 12 (no clusters)."""
    rng = np.random.default_rng(seed)
    sd = 1.0 / np.sqrt(1.0 + np.arange(d))
    X = (rng.normal(size=(n, d)) if kind == "gaussian" else rng.standard_t(5, size=(n, d))) * sd
    X -= X.mean(0)
    X *= 12.0 / np.median(np.linalg.norm(X, axis=1))
    return np.ascontiguousarray(X, dtype=np.float32)


def truth(X, rows, k=K):
    """float64 brute-force neighbours of ``rows`` (self excluded), on the float32 array as passed."""
    Xd = X.astype(np.float64)
    sq = (Xd ** 2).sum(1)
    out = np.empty((len(rows), k), dtype=np.int64)
    kth = np.empty(len(rows))
    for a in range(0, len(rows), 100):
        r = rows[a:a + 100]
        D = sq[r][:, None] + sq[None, :] - 2.0 * Xd[r] @ Xd.T
        D[np.arange(len(r)), r] = np.inf
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        out[a:a + 100] = part
        kth[a:a + 100] = np.take_along_axis(D, part, 1).max(1)
    return out, kth


def recall_per_cell(X, idx, rows):
    """Share of each cell's exact k neighbours in its list (distance ties at the k-th place count as hits)."""
    Xd = X.astype(np.float64)
    _, kth = truth(X, rows)
    rec = np.empty(len(rows))
    for i, r in enumerate(rows):
        d = ((Xd[idx[r]] - Xd[r]) ** 2).sum(1)
        rec[i] = min(int((d <= kth[i] * (1 + 1e-9) + 1e-12).sum()), idx.shape[1]) / idx.shape[1]
    return rec


@pytest.fixture(scope="module")
def cells():
    X = continuum(N_CELLS)
    rows = np.sort(np.random.default_rng(3).choice(N_CELLS, N_SCORED, replace=False))
    return X, rows


# ---- the defect -----------------------------------------------------------------------
@baseline_fails(ref="v0.4.0", raises=AssertionError, match="recall")
@baseline_fails(ref="rc0", raises=AssertionError, match="recall")
def test_neighbours_in_other_lists_are_found(cells):
    X, rows = cells
    idx, _ = _native.knn_graph(X, K, seed=0)
    r = recall_per_cell(X, idx, rows)
    assert r.mean() >= MEAN_RECALL, f"recall {r.mean():.4f}"
    assert (r < 0.8).mean() <= FRACTION_BELOW_08, f"recall: {100 * (r < 0.8).mean():.2f}% of cells below 0.8"
    assert (r < 0.5).mean() <= FRACTION_BELOW_05


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="recall")
@baseline_fails(ref="rc0", raises=AssertionError, match="recall")
def test_heavy_tailed_continuum_recall():
    X = continuum(N_CELLS, kind="heavy", seed=4)
    rows = np.sort(np.random.default_rng(5).choice(N_CELLS, N_SCORED, replace=False))
    idx, _ = _native.knn_graph(X, K, seed=0)
    r = recall_per_cell(X, idx, rows)
    assert r.mean() >= MEAN_RECALL, f"recall {r.mean():.4f}"
    assert (r < 0.8).mean() <= FRACTION_BELOW_08


def test_exact_search_is_unchanged_by_the_index(cells):
    X, rows = cells
    small = np.ascontiguousarray(X[:6000])
    idx, _ = _native.knn_graph(small, K, exact=True)
    r = recall_per_cell(small, idx, np.arange(0, 6000, 7))
    assert r.mean() == 1.0


# ---- tuning and determinism ------------------------------------------------------------
def test_more_probes_do_not_lose_recall_and_the_search_is_deterministic(cells):
    X, rows = cells
    a, _ = _native.knn_graph(X, K, seed=0, nprobe=8)
    b, _ = _native.knn_graph(X, K, seed=0, nprobe=WIDE_NPROBE)
    c, _ = _native.knn_graph(X, K, seed=0, nprobe=WIDE_NPROBE)
    np.testing.assert_array_equal(b, c)
    ra, rb = recall_per_cell(X, a, rows), recall_per_cell(X, b, rows)
    assert rb.mean() >= ra.mean() and rb.mean() >= 0.999
    assert ra.mean() < rb.mean(), "nprobe=8 should search less than nprobe=40"


def test_nprobe_is_validated_ignored_by_exact_search_and_recorded():
    X = continuum(2500, d=10, seed=2)
    with pytest.raises(ValueError, match="nprobe must be at least 1"):
        _native.knn_graph(X, K, nprobe=0)
    exact_default, _ = _native.knn_graph(X, K, exact=True)
    exact_nprobe, _ = _native.knn_graph(X, K, exact=True, nprobe=3)
    np.testing.assert_array_equal(exact_default, exact_nprobe)

    class Frame:
        obsm = {"X_pca": continuum(21_000, d=8, seed=6)}
        uns = {}

    bf = crest.pp.neighbors(Frame(), exact=False, nprobe=7)
    p = bf.uns["neighbors"]["params"]
    assert (p["knn_method"], p["nprobe"], p["n_lists"]) == ("approximate", 7, 21_000 // 512)
    bf = crest.pp.neighbors(Frame(), exact=True, nprobe=7)
    assert bf.uns["neighbors"]["params"]["nprobe"] is None
    with pytest.raises(ValueError, match="nprobe"):
        crest.pp.neighbors(Frame(), exact=False, nprobe=0)


# ---- a real embedding -------------------------------------------------------------------
REAL = os.environ.get("CREST_TEST_KNN_EMBEDDING")


@pytest.mark.skipif(not REAL, reason="CREST_TEST_KNN_EMBEDDING not set")
def test_real_embedding_meets_the_registered_recall_thresholds():
    z = np.load(REAL)
    if hasattr(z, "files"):
        key = "pca" if "pca" in z.files else "X_pca"
        z = z[key]
    X = np.ascontiguousarray(z, dtype=np.float32)
    assert len(X) > 20_000, "the default search is exact up to 20,000 cells"
    rows = np.sort(np.random.default_rng(0).choice(len(X), min(REAL_N_SCORED, len(X)), replace=False))
    idx, _ = _native.knn_graph(X, K, seed=0)
    r = recall_per_cell(X, idx, rows)
    assert r.mean() >= REGISTERED_MEAN_RECALL, f"recall {r.mean():.4f}"
    assert (r < 0.5).mean() <= FRACTION_BELOW_05
    assert (r < 0.8).mean() <= REGISTERED_FRACTION_BELOW_08
