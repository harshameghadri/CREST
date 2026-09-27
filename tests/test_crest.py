"""Correctness tests for crest. Parity checks against scanpy run when it is installed."""

import numpy as np
import polars as pl
import pytest

import crest

sp = pytest.importorskip("scipy.sparse")


# --------------------------------------------------------------------------- data
def make_counts(n_cells=1500, n_genes=600, n_clusters=5, seed=0):
    """Negative-binomial counts with cluster-specific programs and MT- genes."""
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


@pytest.fixture(scope="module")
def data():
    return make_counts()


def frame(X, names, chunk_nnz=None):
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    if chunk_nnz:
        bf.chunk_nnz = chunk_nnz
    return bf


def pipeline(bf):
    bf = crest.pp.filter_cells(bf, min_genes=50)
    bf = crest.pp.filter_genes(bf, min_cells=3)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=300)
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=20)
    return bf


# --------------------------------------------------------------------------- invariants
def test_chunking_invariance(data):
    """Results must not depend on how the matrix is chunked (out-of-core correctness)."""
    X, names, _ = data
    a = pipeline(frame(X, names))
    b = pipeline(frame(X, names, chunk_nnz=1000))
    assert a.shape == b.shape
    assert (a.var["highly_variable"] == b.var["highly_variable"]).all()
    np.testing.assert_allclose(np.abs(a.obsm["X_pca"]), np.abs(b.obsm["X_pca"]), rtol=1e-4, atol=1e-3)


def test_stores_agree(data, tmp_path):
    """CSR, triplet DataFrame and Parquet (out-of-core) stores give the same PCA."""
    X, names, _ = data
    a = pipeline(frame(X, names))
    coo = X.tocoo()
    df = pl.DataFrame({"cell_id": coo.row.astype(np.uint32), "gene_id": coo.col.astype(np.uint32), "count": coo.data})
    b = pipeline(crest.BioFrame.from_triplets(df, X.shape[0], X.shape[1], var=pl.DataFrame({"gene_name": names})))
    frame(X, names).write_parquet(tmp_path / "ds", nnz_per_part=50_000)
    c = pipeline(crest.read_parquet(tmp_path / "ds"))
    for other in (b, c):
        np.testing.assert_allclose(np.abs(a.obsm["X_pca"]), np.abs(other.obsm["X_pca"]), rtol=1e-4, atol=1e-3)


def test_lazy_normalisation_matches_numpy(data):
    X, names, _ = data
    bf = frame(X, names)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    Y = bf.to_scipy().toarray()
    D = X.toarray()
    ref = np.log1p(D / D.sum(1, keepdims=True) * 1e4)
    np.testing.assert_allclose(Y, ref, rtol=1e-5, atol=1e-5)


def test_scaled_pca_matches_dense(data):
    """Implicit scale(max_value) + Gram PCA equals dense scale + SVD."""
    X, names, _ = data
    bf = frame(X, names)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.scale(bf, max_value=2.0)  # small max_value exercises clipping of zeros too
    crest.tl.pca(bf, n_comps=10, use_highly_variable=False)
    D = bf.to_scipy().toarray().astype(np.float64)
    mu, sd = D.mean(0), D.std(0, ddof=1)
    sd[sd == 0] = 1
    Z = np.clip((D - mu) / sd, -2, 2)
    Z -= Z.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    ref = (S[:10] ** 2) / (len(Z) - 1)
    np.testing.assert_allclose(bf.uns["pca"]["variance"], ref, rtol=1e-6)
    ref_scores = U[:, :10] * S[:10]
    got = bf.obsm["X_pca"].astype(np.float64)
    cos = np.abs((got * ref_scores).sum(0)) / np.linalg.norm(got, axis=0) / np.linalg.norm(ref_scores, axis=0)
    assert cos.min() > 0.9999


def test_knn_exact_and_ivf():
    rng = np.random.default_rng(1)
    X = (rng.normal(size=(30000, 20)) + rng.integers(0, 15, (30000, 1)) * 3).astype(np.float32)
    idx_e, _ = crest.crest.knn_graph(X, 10, exact=True)
    idx_a, dist_a = crest.crest.knn_graph(X, 10, exact=False)
    recall = (idx_a[:, :, None] == idx_e[:, None, :]).any(-1).mean()
    assert recall > 0.97
    assert (np.diff(dist_a, axis=1) >= 0).all()
    assert not (idx_a == np.arange(len(X))[:, None]).any()


def test_leiden_recovers_clusters(data):
    X, names, labels = data
    bf = pipeline(frame(X, names))
    crest.pp.neighbors(bf)
    crest.tl.leiden(bf, resolution=0.3)
    from sklearn.metrics import adjusted_rand_score
    kept = bf.obs["cell_id"].to_numpy()
    assert adjusted_rand_score(labels[kept], bf.obs["leiden"].to_numpy()) > 0.9


def test_umap_deterministic_and_finite(data):
    X, names, _ = data
    bf = pipeline(frame(X, names))
    crest.pp.neighbors(bf)
    crest.tl.umap(bf)
    u1 = bf.obsm["X_umap"].copy()
    crest.tl.umap(bf)
    assert np.isfinite(u1).all()
    np.testing.assert_array_equal(u1, bf.obsm["X_umap"])


def test_io_roundtrips(data, tmp_path):
    h5py = pytest.importorskip("h5py")
    X, names, _ = data
    Xc = X.tocsr()
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as f:
        m = f.create_group("matrix")
        m["barcodes"] = np.array([f"c{i}".encode() for i in range(X.shape[0])])
        m["data"], m["indices"], m["indptr"] = Xc.data.astype(np.int32), Xc.indices, Xc.indptr
        m["shape"] = np.array([X.shape[1], X.shape[0]])
        ft = m.create_group("features")
        ft["id"] = np.array([n.encode() for n in names])
        ft["name"] = np.array([n.encode() for n in names])
        ft["feature_type"] = np.array([b"Gene Expression"] * len(names))
    for backed in (None, tmp_path / "pq"):
        bf = crest.read_10x_h5(p, backed=backed)
        assert bf.shape == X.shape
        assert abs(bf.to_scipy(transform=False) - X).max() == 0


# --------------------------------------------------------------------------- scanpy parity
sc = None
try:
    import scanpy as sc  # noqa: F811
except Exception:
    pass
needs_scanpy = pytest.mark.skipif(sc is None, reason="scanpy not installed")


@pytest.fixture(scope="module")
def both(data):
    X, names, _ = data
    import anndata as ad
    a = ad.AnnData(X.copy(), var=__import__("pandas").DataFrame(index=names))
    sc.pp.filter_cells(a, min_genes=50)
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    b = frame(X, names)
    b = crest.pp.filter_cells(b, min_genes=50)
    b = crest.pp.filter_genes(b, min_cells=3)
    crest.pp.normalize_total(b, 1e4)
    crest.pp.log1p(b)
    return a, b


@needs_scanpy
def test_hvg_matches_scanpy(both):
    a, b = both
    sc.pp.highly_variable_genes(a, n_top_genes=300, flavor="seurat")
    crest.pp.highly_variable_genes(b, n_top_genes=300, flavor="seurat")
    assert (b.var["highly_variable"].to_numpy() == a.var["highly_variable"].values).all()
    dn, ref = b.var["dispersions_norm"].to_numpy(), a.var["dispersions_norm"].values
    m = np.isfinite(ref)
    np.testing.assert_allclose(dn[m], ref[m], atol=1e-4)


@needs_scanpy
@pytest.mark.parametrize("method", ["t-test", "wilcoxon"])
def test_rank_genes_groups_matches_scanpy(both, method):
    a, b = both
    rng = np.random.default_rng(3)
    g = rng.integers(0, 4, a.n_obs).astype(str)
    # identical input values for both tools
    a2 = sc.AnnData(b.to_scipy(), var=a.var.copy())
    a2.obs["g"] = g
    a2.uns["log1p"] = {"base": None}
    b.obs = b.obs.with_columns(pl.Series("g", g))
    sc.tl.rank_genes_groups(a2, "g", method=method, n_genes=a2.n_vars)
    res = crest.tl.rank_genes_groups(b, "g", method=method, n_genes=None)
    rg = a2.uns["rank_genes_groups"]
    for grp in rg["names"].dtype.names:
        ref = dict(zip(rg["names"][grp], zip(rg["scores"][grp], rg["logfoldchanges"][grp])))
        sub = res.filter(pl.col("group") == grp)
        s_ref = np.array([ref[n][0] for n in sub["names"]])
        f_ref = np.array([ref[n][1] for n in sub["names"]])
        np.testing.assert_allclose(sub["scores"].to_numpy(), s_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(sub["logfoldchanges"].to_numpy(), f_ref, rtol=1e-4, atol=1e-4)


@needs_scanpy
def test_score_genes_matches_scanpy(both):
    a, b = both
    genes = [f"G{i}" for i in range(20, 35)]
    sc.tl.score_genes(a, genes, random_state=0)
    crest.tl.score_genes(b, genes, random_state=0)
    np.testing.assert_allclose(b.obs["score"].to_numpy(), a.obs["score"].values, atol=1e-5)


# --------------------------------------------------------------------------- Polars plugin
def test_bio_normalize_is_streaming_safe(tmp_path):
    """Per-cell sums must be right even when Polars batches the input (row groups)."""
    rng = np.random.default_rng(0)
    n = 200_000
    df = pl.DataFrame({"cell_id": np.sort(rng.integers(0, 2000, n)).astype(np.uint32),
                       "count": rng.integers(1, 20, n).astype(np.float32)})
    path = tmp_path / "t.parquet"
    df.write_parquet(path, row_group_size=10_000)
    out = pl.scan_parquet(path).with_columns(pl.col("count").bio.normalize_cpm(pl.col("cell_id")).alias("n")).collect()
    sums = out.group_by("cell_id").agg(pl.col("n").sum())["n"].to_numpy()
    np.testing.assert_allclose(sums, 1e4, rtol=1e-4)
