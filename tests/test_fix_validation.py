"""Input validation and edge cases: bad ids raise, tiny approximate kNN, Leiden
``n_iterations=-1``, Scrublet ignores unfilled neighbour slots."""

import subprocess
import sys
import textwrap
import types
import warnings

import numpy as np
import polars as pl
import pytest

import crest
from crest import crest as _native

sp = pytest.importorskip("scipy.sparse")

U32_MAX = np.iinfo(np.uint32).max


def _counts(n_cells, n_genes, n_clusters=5, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, n_genes)
    labels = rng.integers(0, n_clusters, n_cells)
    prog = np.ones((n_clusters, n_genes))
    for c in range(n_clusters):
        up = rng.choice(n_genes, 40, replace=False)
        prog[c, up] *= rng.uniform(3, 10, 40)
    lib = rng.lognormal(7.5, 0.4, n_cells)
    mu = prog[labels] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    names = [f"MT-{i}" if i < 10 else f"G{i}" for i in range(n_genes)]
    return sp.csr_matrix(counts), names, labels


# --------------------------------------------------------------------------- (1) ids
def test_out_of_range_gene_ids_raise():
    indptr = np.array([0, 2, 3], np.int64)
    with pytest.raises(ValueError, match="gene index 5 out of range"):
        crest.BioFrame.from_csr(indptr, np.array([0, 5, 1], np.uint32), np.ones(3, np.float32), n_genes=4)
    with pytest.raises(ValueError, match="indptr"):
        crest.BioFrame.from_csr(np.array([0, 2, 4], np.int64), np.array([0, 1, 1], np.uint32),
                                np.ones(3, np.float32), n_genes=4)
    df = pl.DataFrame({"cell_id": [0, 1, 1], "gene_id": [0, 3, 7], "count": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="gene index 7 out of range"):
        crest.BioFrame.from_triplets(df, n_cells=2, n_genes=4)
    with pytest.raises(ValueError, match="cell index 1 out of range"):
        crest.BioFrame.from_triplets(df, n_cells=1, n_genes=8)
    # valid input still works, and ids inferred from the data are in range
    bf = crest.BioFrame.from_triplets(df)
    assert bf.shape == (2, 8)


def test_out_of_range_cell_ids_raise_in_kernels():
    bf = crest.BioFrame.from_scipy(sp.random(20, 10, density=0.3, format="csr", dtype=np.float32, random_state=0))
    cmap, gmap = bf._cell_map(), bf._gene_map()
    # CSR form: the chunk spans cells 0..19 but cell_map only covers 15 of them
    ctx = list(next(bf.iter_ctx()))
    ctx[2] = cmap[:15].copy()
    with pytest.raises(ValueError, match="cell index out of range"):
        _native.materialize(*ctx)
    # triplet form: cell 25 >= len(cell_map) == 20
    genes, vals = np.array([0, 1, 2], np.uint32), np.ones(3, np.float32)
    with pytest.raises(ValueError, match="cell index out of range"):
        _native.materialize(genes, vals, cmap, gmap, 0.0, False, np.array([0, 3, 25], np.uint32), None, 0)
    rows, _, _ = _native.materialize(genes, vals, cmap, gmap, 0.0, False, np.array([0, 3, 19], np.uint32), None, 0)
    assert len(rows) == 3


# --------------------------------------------------------------------------- (2) knn
@pytest.mark.parametrize("n", [2, 3, 5, 7, 8, 12])
def test_approximate_knn_tiny_inputs(n):
    rng = np.random.default_rng(n)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    idx, dist = _native.knn_graph(X, 3, exact=False, seed=0)
    eidx, edist = _native.knn_graph(X, 3, exact=True, seed=0)
    assert idx.shape == (n, min(3, n - 1))
    assert (idx < n).all() and np.isfinite(dist).all()
    assert all(i not in row for i, row in enumerate(idx))
    np.testing.assert_allclose(dist, edist, rtol=1e-6)


# --------------------------------------------------------------------------- (3) leiden
@pytest.fixture(scope="module")
def graph_bf():
    X, names, lab = _counts(2000, 400, seed=3)
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=200)
    crest.pp.scale(bf)
    crest.tl.pca(bf, n_comps=15)
    crest.pp.neighbors(bf)
    return bf


def test_leiden_negative_iterations_runs_until_unchanged(graph_bf):
    bf = graph_bf
    crest.tl.leiden(bf, key_added="default")
    crest.tl.leiden(bf, n_iterations=2, key_added="two")
    crest.tl.leiden(bf, n_iterations=-1, key_added="until")
    crest.tl.leiden(bf, n_iterations=1000, key_added="many")
    assert (bf.obs["default"] == bf.obs["two"]).all()  # default stays 2
    assert (bf.obs["until"] == bf.obs["many"]).all()
    assert bf.uns["until"]["params"]["n_iterations"] == -1
    tab = crest.tl.leiden_sweep(bf, [0.5, 1.0], n_iterations=-1, key_prefix="sw_")
    assert tab.height == 2
    crest.tl.leiden(bf, resolution=0.5, n_iterations=-1, key_added="r05")
    assert (bf.obs["sw_0.5"] == bf.obs["r05"]).all()


def _ring_of_cliques(n_cliques=40, size=5):
    """Identical cliques joined in a ring by single edges: every clique is
    interchangeable, so Leiden sees many exact ties."""
    rows, cols = [], []
    for c in range(n_cliques):
        base = c * size
        for i in range(size):
            for j in range(i + 1, size):
                rows.append(base + i)
                cols.append(base + j)
        rows.append(base + size - 1)
        cols.append(((c + 1) % n_cliques) * size)
    return n_cliques * size, np.asarray(rows, np.uint32), np.asarray(cols, np.uint32), np.ones(len(rows), np.float32)


def test_leiden_until_stable_terminates_on_ties():
    """n_iterations=-1 releases the GIL, so a non-terminating loop would hang
    uninterruptibly: run it in a subprocess with a hard timeout."""
    code = textwrap.dedent("""
        import warnings, numpy as np, sys
        sys.path.insert(0, %r)
        from test_fix_validation import _ring_of_cliques
        from crest import crest as N
        warnings.simplefilter("error")  # the iteration cap must not be hit
        n, r, c, w = _ring_of_cliques()
        for res in (0.5, 1.0, 2.0):
            for seed in range(6):
                a = N.leiden(n, r, c, w, res, -1, seed)
                b = N.leiden(n, r, c, w, res, 10_000, seed)
                assert (a == b).all()
        N.leiden_sweep(n, r, c, w, [0.5, 1.0, 2.0] * 2, list(range(6)), -1)
        print("ok")
    """) % str(__import__("pathlib").Path(__file__).parent)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_leiden_iteration_cap_warns():
    n, r, c, w = _ring_of_cliques()
    with pytest.warns(RuntimeWarning, match="cap of 1 iterations"):
        _native.leiden(n, r, c, w, 1.0, -1, 0, max_iterations=1)  # 1st iteration always changes it
    with pytest.warns(RuntimeWarning, match="cap of 1 iterations"):
        _native.leiden_sweep(n, r, c, w, [1.0], [0], -1, max_iterations=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _native.leiden(n, r, c, w, 1.0, 1, 0)  # an explicit count never warns
        _native.leiden(n, r, c, w, 1.0, -1, 0)


# --------------------------------------------------------------------------- (4) scrublet
def test_scrublet_ignores_unfilled_neighbour_slots(monkeypatch):
    """The approximate kNN (> 150k points) can leave a slot unfilled (u32::MAX);
    it must not count as a simulated-doublet neighbour."""
    import crest.doublets as dbl

    X, names, _ = _counts(800, 400, seed=11)

    def run(fill):
        def knn_graph(M, k, exact=None, seed=0):
            idx, dist = _native.knn_graph(M, k, exact=exact, seed=seed)
            idx = idx.copy()
            idx[:, -1] = fill  # pretend the last slot is unfilled / an observed cell
            return idx, dist

        shim = {k: getattr(_native, k) for k in dir(_native) if not k.startswith("__")}
        shim["knn_graph"] = knn_graph
        monkeypatch.setattr(dbl, "_native", types.SimpleNamespace(**shim))
        bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
        crest.pp.scrublet(bf, threshold=0.25)
        return bf.obs["doublet_score"].to_numpy()

    s_placeholder = run(U32_MAX)
    s_observed = run(0)  # cell 0 is an observed cell: never a simulated doublet
    np.testing.assert_array_equal(s_placeholder, s_observed)
