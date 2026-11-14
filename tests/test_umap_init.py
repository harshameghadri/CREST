"""UMAP spectral initialisation: convergence and the layout of disconnected graphs.

umap-learn takes the initial layout from the smallest non-trivial eigenvectors of
the normalised graph Laplacian (ARPACK ``eigsh``) and lays out connected
components one by one. These tests build kNN graphs from synthetic data (no
reference file is needed), compute the eigenvectors with scipy and compare.

``n_epochs=0`` returns the initial layout itself (rescaled like the layout that
enters the optimiser), so the initialisation is tested without the SGD that
follows it. Tests that run the full optimiser pin ``RAYON_NUM_THREADS=1`` in a
subprocess: the parallel snapshot SGD changes global coherence with the number
of threads and is not what these tests measure.
"""

import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import crest
from crest import crest as _native

# initial layout vs the converged eigenvectors: worst R2 over the two axes
INIT_R2_MIN = 0.999
# disconnected 2-D blobs: min over component pairs of centroid distance / larger within-std
BLOB_SEPARATION_MIN = 2.0
BLOB_WITHIN_STD_MAX = 3.0       # about 1.5 x the within-component std umap-learn reaches (1.9-2.0)
# initial layout: fraction of points whose 10 nearest layout neighbours all lie in their own component
COMPONENT_PURITY_MIN = 0.99
# final layout, one thread: mean pairwise Spearman of cluster-centroid distances across seeds
CENTROID_SPEARMAN_MIN = 0.65
SEEDS = (0, 1, 2)


# ----------------------------------------------------------------------------- helpers
def _graph(X, k=15):
    X = np.ascontiguousarray(X, dtype=np.float32)
    idx, dist = _native.knn_graph(X, k - 1, exact=True, seed=0)
    rows, cols, w = _native.connectivities(idx, dist, k)
    n = len(X)
    C = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    return rows, cols, w, (C + C.T).tocsr()


def _chain_of_blobs(n, d, seed, spread, n_clusters=8):
    """Gaussian clusters whose centres follow a random walk: a connected graph whose
    leading Laplacian eigenvalues are close together."""
    rng = np.random.default_rng(seed)
    centres = np.cumsum(rng.normal(0, spread, (n_clusters, d)), 0)
    lab = rng.integers(0, n_clusters, n)
    return centres[lab] + rng.normal(0, 1, (n, d)), lab


def _eigsh_reference(C, dim=2):
    """Eigenvalues (descending, dim + 2 of them) and the eigenvectors 2 .. dim + 1."""
    n = C.shape[0]
    deg = np.asarray(C.sum(0)).ravel()
    D = sp.diags(1.0 / np.sqrt(deg))
    vals, vecs = sla.eigsh(D @ C @ D, dim + 2, which="LA", ncv=max(2 * (dim + 1) + 1, int(np.sqrt(n))),
                           tol=1e-10, v0=np.ones(n), maxiter=50 * n)
    order = np.argsort(-vals)
    return vals[order], vecs[:, order[1:dim + 1]]


def _min_axis_r2(emb, ref):
    A = np.c_[ref, np.ones(len(ref))]
    out = []
    for j in range(emb.shape[1]):
        beta, *_ = np.linalg.lstsq(A, emb[:, j], rcond=None)
        out.append(1.0 - (emb[:, j] - A @ beta).var() / emb[:, j].var())
    return min(out)


def _cluster_stats(emb, lab):
    """Per-axis within-cluster std and min over pairs of centroid distance / larger std."""
    ks = np.unique(lab)
    cen = np.array([emb[lab == k].mean(0) for k in ks])
    sd = np.array([np.sqrt(((emb[lab == k] - cen[i]) ** 2).sum(1).mean() / emb.shape[1]) for i, k in enumerate(ks)])
    sep = min(np.linalg.norm(cen[i] - cen[j]) / max(sd[i], sd[j])
              for i in range(len(ks)) for j in range(i + 1, len(ks)))
    return sd, sep


def _component_purity(emb, lab, k=10):
    idx, _ = _native.knn_graph(np.ascontiguousarray(emb, dtype=np.float32), k, exact=True, seed=0)
    return float((lab[idx] == lab[:, None]).all(1).mean())


_ONE_THREAD_SCRIPT = textwrap.dedent("""
    import sys, numpy as np
    from crest import crest as native
    z = np.load(sys.argv[1])
    out = {f"s{s}": native.umap(int(z["n"]), z["rows"], z["cols"], z["w"], seed=s) for s in z["seeds"]}
    np.savez(sys.argv[2], **out)
""")


def _layouts_one_thread(tmp_path, n, rows, cols, w, seeds=SEEDS):
    src, dst = tmp_path / "graph.npz", tmp_path / "layouts.npz"
    np.savez(src, n=n, rows=rows, cols=cols, w=w, seeds=np.array(seeds))
    script = tmp_path / "run.py"
    script.write_text(_ONE_THREAD_SCRIPT)
    env = {**os.environ, "RAYON_NUM_THREADS": "1"}
    subprocess.run([sys.executable, str(script), str(src), str(dst)], check=True, env=env)
    z = np.load(dst)
    return [z[f"s{s}"] for s in seeds]


@pytest.fixture(scope="module")
def chain_graph():
    X, lab = _chain_of_blobs(10_000, 10, seed=1, spread=1.5)
    rows, cols, w, C = _graph(X)
    vals, ref = _eigsh_reference(C, 2)
    # the wanted eigenvalues are distinct enough for the vectors to be well defined
    assert (vals[1] - vals[2]) > 1e-3 and (vals[2] - vals[3]) > 1e-3
    return dict(n=len(X), rows=rows, cols=cols, w=w, C=C, lab=lab, ref=ref)


# ----------------------------------------------------------------------------- convergence
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="init R2")
@pytest.mark.parametrize("seed", SEEDS)
def test_spectral_init_matches_eigsh(chain_graph, seed):
    g = chain_graph
    emb = _native.umap(g["n"], g["rows"], g["cols"], g["w"], n_epochs=0, seed=seed)
    r2 = _min_axis_r2(emb, g["ref"])
    assert r2 >= INIT_R2_MIN, f"init R2 {r2:.4f} < {INIT_R2_MIN}"


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="seed-to-seed")
def test_spectral_init_does_not_depend_on_the_seed(chain_graph):
    g = chain_graph
    a = _native.umap(g["n"], g["rows"], g["cols"], g["w"], n_epochs=0, seed=0)
    b = _native.umap(g["n"], g["rows"], g["cols"], g["w"], n_epochs=0, seed=7)
    r2 = _min_axis_r2(a, b)
    assert r2 >= INIT_R2_MIN, f"seed-to-seed R2 {r2:.4f} < {INIT_R2_MIN}"


def test_spectral_init_is_deterministic_for_a_seed(chain_graph):
    g = chain_graph
    a = _native.umap(g["n"], g["rows"], g["cols"], g["w"], n_epochs=0, seed=3)
    b = _native.umap(g["n"], g["rows"], g["cols"], g["w"], n_epochs=0, seed=3)
    np.testing.assert_array_equal(a, b)


def test_final_layout_global_structure_is_reproducible_across_seeds(chain_graph, tmp_path):
    g = chain_graph
    embs = _layouts_one_thread(tmp_path, g["n"], g["rows"], g["cols"], g["w"])
    cd = [pdist(np.array([e[g["lab"] == k].mean(0) for k in range(8)])) for e in embs]
    rho = np.mean([spearmanr(cd[i], cd[j])[0] for i, j in ((0, 1), (0, 2), (1, 2))])
    assert rho >= CENTROID_SPEARMAN_MIN, rho


@pytest.mark.skipif("CREST_TEST_PBMC_CONNECTIVITIES" not in os.environ,
                    reason="set CREST_TEST_PBMC_CONNECTIVITIES to an .npz of a scanpy connectivities matrix")
def test_spectral_init_matches_eigsh_on_a_real_graph():
    C = sp.load_npz(os.environ["CREST_TEST_PBMC_CONNECTIVITIES"]).tocsr()
    T = sp.triu(C, 1).tocoo()
    _, ref = _eigsh_reference(C, 2)
    for seed in SEEDS:
        emb = _native.umap(C.shape[0], T.row.astype(np.uint32), T.col.astype(np.uint32),
                           T.data.astype(np.float32), n_epochs=0, seed=seed)
        r2 = _min_axis_r2(emb, ref)
        assert r2 >= INIT_R2_MIN, f"init R2 {r2:.4f} < {INIT_R2_MIN}"


def test_iterative_spectral_solver_leaves_the_floating_point_mode_alone(chain_graph):
    # the eigensolver runs a dense eigendecomposition per restart; subnormal floats must
    # still be read and produced normally afterwards (n > 256 takes the iterative path)
    g = chain_graph
    tiny = np.array([1e-40, 3e-39], np.float32)
    emb = _native.umap(g["n"], g["rows"], g["cols"], g["w"], n_epochs=0, seed=0)
    assert np.isfinite(emb).all()
    assert (tiny * np.float32(1.0) != 0).all()
    assert np.float32(1e-30) * np.float32(1e-10) != 0


# ----------------------------------------------------------------------------- cap and API
@pytest.mark.baseline_fails(ref="v0.4.0", raises=TypeError, match="spectral_max_matvec")
def test_cap_on_matrix_vector_products_warns(chain_graph):
    g = chain_graph
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emb = _native.umap(g["n"], g["rows"], g["cols"], g["w"], n_epochs=0, spectral_max_matvec=30, seed=0)
    assert any(issubclass(c.category, RuntimeWarning) and "spectral initialisation" in str(c.message) for c in caught)
    assert emb.shape == (g["n"], 2) and np.isfinite(emb).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _native.umap(g["n"], g["rows"], g["cols"], g["w"], n_epochs=0, seed=0)


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="spectral_max_matvec")
def test_tl_umap_exposes_the_cap():
    import inspect
    assert "spectral_max_matvec" in inspect.signature(crest.tl.umap).parameters


# ----------------------------------------------------------------------------- connected components
def _blob_graph():
    """5 disconnected 2-D Gaussian blobs of 400 points, centres spread over about 17 sd."""
    rng = np.random.RandomState(0)
    centres = rng.normal(0, 12 * np.sqrt(2), (5, 2))
    X = np.vstack([centres[i] + rng.normal(0, 1, (400, 2)) for i in range(5)])
    lab = np.repeat(np.arange(5), 400)
    rows, cols, w, C = _graph(X)
    assert sp.csgraph.connected_components(C)[0] == 5
    return len(X), rows, cols, w, lab


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="separation")
def test_disconnected_blobs_are_separated_in_the_initial_layout():
    n, rows, cols, w, lab = _blob_graph()
    emb = _native.umap(n, rows, cols, w, n_epochs=0, seed=0)
    _, sep = _cluster_stats(emb, lab)
    assert sep >= BLOB_SEPARATION_MIN, f"separation {sep:.2f} < {BLOB_SEPARATION_MIN}"
    purity = _component_purity(emb, lab)
    assert purity >= COMPONENT_PURITY_MIN, f"purity {purity:.3f} < {COMPONENT_PURITY_MIN}"


def test_disconnected_blobs_final_layout(tmp_path):
    n, rows, cols, w, lab = _blob_graph()
    for emb in _layouts_one_thread(tmp_path, n, rows, cols, w):
        sd, sep = _cluster_stats(emb, lab)
        assert sd.max() <= BLOB_WITHIN_STD_MAX, sd
        assert sep >= BLOB_SEPARATION_MIN, sep


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="purity")
def test_two_large_components_are_separated_in_the_initial_layout():
    # two 2-D Gaussian components of different size, far apart (two components: anchors at +-e1)
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 1, (2000, 2)), rng.normal(0, 1, (700, 2)) + 100.0])
    lab = np.r_[np.zeros(2000, int), np.ones(700, int)]
    rows, cols, w, C = _graph(X)
    assert sp.csgraph.connected_components(C)[0] == 2
    emb = _native.umap(len(X), rows, cols, w, n_epochs=0, seed=0)
    purity = _component_purity(emb, lab)
    assert purity >= COMPONENT_PURITY_MIN, f"purity {purity:.3f} < {COMPONENT_PURITY_MIN}"
    _, sep = _cluster_stats(emb, lab)
    assert sep >= BLOB_SEPARATION_MIN, f"separation {sep:.2f} < {BLOB_SEPARATION_MIN}"


def test_singleton_and_tiny_components_are_placed():
    # a connected graph plus an isolated pair and an isolated triple of vertices
    X, _ = _chain_of_blobs(600, 5, seed=2, spread=1.0)
    rows, cols, w, _ = _graph(X)
    extra = np.array([600, 601, 602, 603, 604], dtype=np.uint32)
    rows = np.r_[rows, [600, 602, 603]].astype(np.uint32)
    cols = np.r_[cols, [601, 603, 604]].astype(np.uint32)
    w = np.r_[w, [1.0, 1.0, 1.0]].astype(np.float32)
    emb = _native.umap(605, rows, cols, w, n_epochs=0, seed=0)
    assert emb.shape == (605, 2) and np.isfinite(emb).all()
    big = emb[:600].mean(0)
    assert np.linalg.norm(emb[extra].mean(0) - big) > 0.5
