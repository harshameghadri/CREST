"""Parity pins for the graph step: connectivities, Leiden quality and UMAP local quality.

These tests state what the documentation claims about the graph algorithms and check it against
the reference implementations (umap-learn, leidenalg). They pass on released versions and on the
development tree; they exist so that a later change cannot break the claims unnoticed.
(The fit of the UMAP curve parameters ``a`` and ``b`` is pinned by a unit test in ``src/umap/utils.rs``,
because the fit is not exposed to Python.)

Fixture ``tests/data/pbmc3k_pca30.npz`` (``X_pca``: 2638 cells x 30 principal components, float32,
about 0.3 MB) was made from the 10x PBMC 3k filtered matrix (hg19) with scanpy 1.11:
``filter_cells(min_genes=200)``, ``filter_genes(min_cells=3)``, keep cells with 200 < n_genes < 2500 and
mitochondrial fraction < 5 %, ``normalize_total(1e4)``, ``log1p``, ``highly_variable_genes(n_top_genes=2000,
flavor="seurat")``, ``scale(max_value=10)``, ``pca(n_comps=30, svd_solver="arpack", random_state=0)``.
The neighbour graph is built here with the native exact search, so the pins do not depend on scanpy's
neighbour code.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from crest import crest as _native

DATA = Path(__file__).resolve().parent / "data" / "pbmc3k_pca30.npz"

N_NEIGHBORS = 15            # counts the cell itself, as in scanpy
SEEDS = tuple(range(10))
CONNECTIVITY_MAX_ABS_DIFF = 1e-6      # measured 1.2e-7 (float32 weights)
LEIDEN_Q_MARGIN = 0.003               # mean modularity may trail leidenalg by this much (measured: ahead by 0.0024)
LEIDEN_MAX_MEAN_CLUSTER_DIFF = 1.0    # |mean cluster count - leidenalg's| (measured 0.0)
TRUSTWORTHINESS_K = 15
TRUSTWORTHINESS_SEEDS = (0, 1, 2)
TRUSTWORTHINESS_MAX_GAP = 0.005       # |mean umap-learn - mean CREST| (measured 0.0001 at 1 thread)


def _pca() -> np.ndarray:
    return np.ascontiguousarray(np.load(DATA)["X_pca"], dtype=np.float32)


def _knn(x: np.ndarray):
    return _native.knn_graph(np.ascontiguousarray(x, dtype=np.float32), N_NEIGHBORS - 1, exact=True)


def _umap_learn_graph(idx, dist):
    """umap-learn's fuzzy simplicial set from the same neighbour lists (self in column 0, as scanpy passes them)."""
    uu = pytest.importorskip("umap.umap_")
    n = len(idx)
    knn_idx = np.hstack([np.arange(n)[:, None], idx]).astype(np.int64)
    knn_dist = np.hstack([np.zeros((n, 1)), dist]).astype(np.float32)
    out = uu.fuzzy_simplicial_set(np.zeros((n, 1)), N_NEIGHBORS, np.random.RandomState(0), "euclidean",
                                  knn_indices=knn_idx, knn_dists=knn_dist)
    return (out[0] if isinstance(out, tuple) else out).tocsr()


def _crest_graph(idx, dist):
    sp = pytest.importorskip("scipy.sparse")
    n = len(idx)
    rows, cols, w = _native.connectivities(idx, dist, N_NEIGHBORS)
    c = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    return (c + c.T).tocsr(), (rows, cols, w)


@pytest.mark.parametrize("duplicates", [False, True], ids=["distinct-points", "duplicate-points"])
def test_connectivities_match_umap_learn_fuzzy_simplicial_set(duplicates):
    x = _pca()[np.random.default_rng(0).choice(2638, 300, replace=False)]
    if duplicates:  # 15 cells become exact copies of 15 others: zero distances between different cells
        pick = np.random.default_rng(1).choice(300, 30, replace=False)
        x[pick[:15]] = x[pick[15:]]
    idx, dist = _knn(x)
    if duplicates:
        assert (dist == 0).sum() >= 30, "the duplicated cells must show up as zero-distance neighbours"
    ref = _umap_learn_graph(idx, dist)
    ours, _ = _crest_graph(idx, dist)
    assert ours.nnz == ref.nnz
    diff = abs(ours - ref)
    assert diff.max() < CONNECTIVITY_MAX_ABS_DIFF, diff.max()


def _modularity_and_k(graph, labels):
    return graph.modularity(list(map(int, labels)), weights="weight"), len(set(map(int, labels)))


def test_leiden_modularity_matches_leidenalg():
    """Reference is leidenalg (RBConfigurationVertexPartition, resolution 1, run until stable); scanpy's igraph
    flavour is a weaker reference and is not used here."""
    ig = pytest.importorskip("igraph")
    la = pytest.importorskip("leidenalg")
    idx, dist = _knn(_pca())
    n = len(idx)
    _, (rows, cols, w) = _crest_graph(idx, dist)
    g = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
    g.es["weight"] = w.astype(float).tolist()
    ours, theirs = [], []
    for s in SEEDS:
        ours.append(_modularity_and_k(g, _native.leiden(n, rows, cols, w, 1.0, 2, s)))
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0,
                                 n_iterations=-1, seed=s)
        theirs.append(_modularity_and_k(g, part.membership))
    (q_c, k_c), (q_l, k_l) = np.mean(ours, axis=0), np.mean(theirs, axis=0)
    print(f"Q crest {q_c:.5f} leidenalg {q_l:.5f} (diff {q_c - q_l:+.5f}); clusters {k_c:.1f} vs {k_l:.1f}")
    assert q_c >= q_l - LEIDEN_Q_MARGIN, (q_c, q_l)
    assert abs(k_c - k_l) <= LEIDEN_MAX_MEAN_CLUSTER_DIFF, (k_c, k_l)


_CREST_UMAP_SCRIPT = textwrap.dedent("""
    import sys
    import numpy as np
    import scipy.sparse as sp
    import crest
    x = np.ascontiguousarray(np.load(sys.argv[1])["X_pca"], dtype=np.float32)
    bf = crest.BioFrame.from_scipy(sp.csr_matrix((len(x), 1), dtype=np.float32))
    bf.obsm["X_pca"] = x
    crest.pp.neighbors(bf, n_neighbors=int(sys.argv[2]), exact=True)
    for s in map(int, sys.argv[4:]):
        crest.tl.umap(bf, min_dist=0.5, spread=1.0, random_state=s)
        np.save(f"{sys.argv[3]}_{s}.npy", np.asarray(bf.obsm["X_umap"]))
""")


def test_umap_trustworthiness_matches_umap_learn(tmp_path):
    """Local structure of the layout: sklearn trustworthiness (k=15) against umap-learn on the same neighbour lists.
    CREST runs single-threaded (its parallel SGD trades layout coherence for speed; see the UMAP documentation)."""
    umap = pytest.importorskip("umap")
    from sklearn.manifold import trustworthiness
    x = _pca()
    idx, dist = _knn(x)
    knn = (np.hstack([np.arange(len(x))[:, None], idx]).astype(np.int64),
           np.hstack([np.zeros((len(x), 1)), dist]).astype(np.float32), None)
    env = dict(os.environ, RAYON_NUM_THREADS="1", PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    subprocess.run([sys.executable, "-c", _CREST_UMAP_SCRIPT, str(DATA), str(N_NEIGHBORS), str(tmp_path / "crest"),
                    *map(str, TRUSTWORTHINESS_SEEDS)], check=True, env=env)
    t_crest, t_ref = [], []
    for s in TRUSTWORTHINESS_SEEDS:
        t_crest.append(trustworthiness(x, np.load(tmp_path / f"crest_{s}.npy"), n_neighbors=TRUSTWORTHINESS_K))
        model = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=0.5, spread=1.0, random_state=s, n_jobs=1,
                          precomputed_knn=knn).fit(x)
        t_ref.append(trustworthiness(x, model.embedding_, n_neighbors=TRUSTWORTHINESS_K))
    gap = abs(np.mean(t_ref) - np.mean(t_crest))
    print(f"trustworthiness crest {np.round(t_crest, 4)} umap-learn {np.round(t_ref, 4)} gap {gap:.5f}")
    assert gap <= TRUSTWORTHINESS_MAX_GAP, (t_crest, t_ref)
