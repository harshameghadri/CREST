"""Round trips and write paths: AnnData / h5ad / Parquet in and out of a BioFrame.

Tests marked ``baseline_fails`` fail on the released 0.4.0 wheel for the stated reason and pass
on this tree; unmarked tests (pins) describe behaviour that was already right and must hold on both.
The real-data variants run when ``CREST_TEST_PBMC3K`` points at a directory holding PBMC 3k's
``matrix.mtx``, ``barcodes.tsv`` and ``genes.tsv`` (10x MTX layout).
"""

import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import crest

sp = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

# ---- named tolerances / sizes
UMAP_EXACT = 0.0            # obsm['X_umap'] must survive a write/read bit for bit (float32 in, float32 out)
VARIANCE_RATIO_RTOL = 1e-12  # float64 arrays written and read back unchanged
N_CELLS, N_GENES, N_CLUSTERS = 900, 500, 4
MIN_GENES_FILTER = 150      # removes roughly the low-depth tail, so kept cell ids are no longer 0..n-1
SMALL_PART_NNZ = 20_000     # forces several Parquet parts for the synthetic data


def baseline_fails(ref, raises, match):
    """Documentation marker, see the module docstring (registered by the test plugin when present)."""
    return pytest.mark.baseline_fails(ref=ref, raises=raises, match=match)


# --------------------------------------------------------------------------- data
def make_counts(n_cells=N_CELLS, n_genes=N_GENES, n_clusters=N_CLUSTERS, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, n_genes)
    labels = rng.integers(0, n_clusters, n_cells)
    prog = np.ones((n_clusters, n_genes))
    for c in range(n_clusters):
        up = rng.choice(n_genes, 40, replace=False)
        prog[c, up] *= rng.uniform(3, 10, 40)
    lib = rng.lognormal(6.5, 0.6, n_cells)  # wide depth range: min_genes removes a real fraction
    mu = prog[labels] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    X = sp.csr_matrix(rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32))
    names = [f"MT-{i}" if i < 10 else f"G{i}" for i in range(n_genes)]
    return X, names


@pytest.fixture(scope="module")
def counts():
    return make_counts()


def frame(counts, **kw):
    X, names = counts
    return crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}), **kw)


STORES = ("csr", "frame", "parquet")


def frame_in_store(X, names, kind, tmp_path):
    """The same counts held by each of the three stores."""
    var = pl.DataFrame({"gene_name": names})
    if kind == "csr":
        return crest.BioFrame.from_scipy(X, var=var)
    if kind == "frame":
        coo = X.tocoo()
        df = pl.DataFrame({"cell_id": coo.row.astype(np.uint32), "gene_id": coo.col.astype(np.uint32),
                           "count": coo.data}).sort(["cell_id", "gene_id"])
        return crest.BioFrame.from_triplets(df, X.shape[0], X.shape[1], var=var)
    crest.BioFrame.from_scipy(X, var=var).write_parquet(tmp_path / "pq_store", nnz_per_part=SMALL_PART_NNZ)
    return crest.read_parquet(tmp_path / "pq_store")


def filtered(counts):
    bf = crest.pp.filter_cells(frame(counts), min_genes=MIN_GENES_FILTER)
    ids = bf.obs["cell_id"].to_numpy()
    assert 0 < len(ids) < N_CELLS and not np.array_equal(ids, np.arange(len(ids))), "filter must leave id gaps"
    return bf


def sha256_of_parts(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(path).glob("*.parquet"))}


# --------------------------------------------------------------------------- write_parquet onto its own directory
@pytest.fixture
def live_store(counts, tmp_path):
    """A multi-part Parquet dataset in ``tmp_path/ds`` and a filtered frame reading from it."""
    d = tmp_path / "ds"
    frame(counts).write_parquet(d, nnz_per_part=SMALL_PART_NNZ)
    assert len(list(d.glob("part-*.parquet"))) > 2
    bf = crest.pp.filter_cells(crest.read_parquet(d), min_genes=MIN_GENES_FILTER)
    return d, bf


@baseline_fails("v0.4.0", FileNotFoundError, "part-")
def test_write_parquet_onto_own_directory_is_refused(live_store):
    d, bf = live_store
    before = sha256_of_parts(d)
    with pytest.raises(ValueError, match="open BioFrame"):
        bf.write_parquet(d)
    assert sha256_of_parts(d) == before
    assert not [p for p in d.iterdir() if p.name.startswith(".")], "no staging directory left behind"
    # the frame still reads its data
    assert crest.read_parquet(d).shape == (N_CELLS, N_GENES)
    assert bf.to_scipy(transform=False).shape == bf.shape


@baseline_fails("v0.4.0", FileNotFoundError, "part-")
def test_write_parquet_onto_own_directory_unfiltered_and_other_spelling(counts, tmp_path):
    d = tmp_path / "ds"
    frame(counts).write_parquet(d, nnz_per_part=SMALL_PART_NNZ)
    bf = crest.read_parquet(d)
    before = sha256_of_parts(d)
    for spelling in (d, d / ".." / "ds", str(d) + os.sep):
        with pytest.raises(ValueError, match="open BioFrame"):
            bf.write_parquet(spelling)
    assert sha256_of_parts(d) == before


@baseline_fails("v0.4.0", AssertionError, "old.*part")
def test_failed_write_keeps_the_previous_dataset(counts, tmp_path):
    """Old parts in a different target directory are replaced only after the new dataset is complete."""
    d = tmp_path / "target"
    frame(counts).write_parquet(d, nnz_per_part=SMALL_PART_NNZ)
    before = sha256_of_parts(d)
    bf = filtered(counts)
    real = bf.iter_chunks

    def failing(transform=True):
        for k, chunk in enumerate(real(transform)):
            if k == 1:
                raise RuntimeError("simulated failure mid-write")
            yield chunk

    bf.iter_chunks = failing
    bf.chunk_nnz = SMALL_PART_NNZ
    with pytest.raises(RuntimeError, match="simulated"):
        bf.write_parquet(d, nnz_per_part=SMALL_PART_NNZ // 4)
    assert sha256_of_parts(d) == before, "old parts must all still be present and unchanged"
    assert not [p for p in d.iterdir() if p.name.startswith(".")]


def test_write_parquet_to_other_directory_replaces_old_parts(counts, tmp_path):
    d = tmp_path / "target"
    frame(counts).write_parquet(d, nnz_per_part=SMALL_PART_NNZ)
    n_old = len(list(d.glob("part-*.parquet")))
    bf = filtered(counts)
    bf.write_parquet(d)  # one part: far fewer than before
    assert n_old > 1 and len(list(d.glob("part-*.parquet"))) == 1
    assert sorted(p.name for p in d.iterdir()) == ["obs.parquet", "part-00000.parquet", "var.parquet"]
    back = crest.read_parquet(d)
    assert back.shape == bf.shape
    assert abs(back.to_scipy(transform=False) - bf.to_scipy(transform=False)).max() == 0


def test_write_parquet_allowed_after_the_frame_is_released(counts, tmp_path):
    d = tmp_path / "ds"
    frame(counts).write_parquet(d, nnz_per_part=SMALL_PART_NNZ)
    bf = crest.read_parquet(d)
    del bf
    gc.collect()
    filtered(counts).write_parquet(d)
    assert crest.read_parquet(d).n_obs == filtered(counts).n_obs


# ---- read_10x_h5(backed=...)
def write_10x_h5(path, X, names):
    h5py = pytest.importorskip("h5py")
    X = X.tocsr()
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["barcodes"] = np.array([f"BC{i}".encode() for i in range(X.shape[0])])
        m["indptr"] = X.indptr.astype(np.int64)
        m["indices"] = X.indices.astype(np.int64)
        m["data"] = X.data.astype(np.int32)
        m["shape"] = np.array([X.shape[1], X.shape[0]], dtype=np.int32)
        feats = m.create_group("features")
        feats["id"] = np.array([f"ENSG{i:05d}".encode() for i in range(X.shape[1])])
        feats["name"] = np.array([n.encode() for n in names])
        feats["feature_type"] = np.array([b"Gene Expression"] * X.shape[1])


@baseline_fails("v0.4.0", AssertionError, "open frame")
def test_read_10x_h5_backed_refuses_directory_of_an_open_frame(counts, tmp_path):
    X, names = counts
    h5 = tmp_path / "m.h5"
    write_10x_h5(h5, X, names)
    d = tmp_path / "backed"
    bf = crest.read_10x_h5(h5, backed=d, nnz_per_part=SMALL_PART_NNZ)
    before = sha256_of_parts(d)
    try:
        crest.read_10x_h5(h5, backed=d, nnz_per_part=SMALL_PART_NNZ // 2)
        refused = False
    except ValueError:
        refused = True
    assert refused, "read_10x_h5(backed=) replaced the part files of an open frame"
    assert sha256_of_parts(d) == before
    assert bf.to_scipy(transform=False).shape == X.shape
    # once nothing reads from the directory it is regenerated
    del bf
    gc.collect()
    bf2 = crest.read_10x_h5(h5, backed=d, nnz_per_part=SMALL_PART_NNZ // 2)
    assert abs(bf2.to_scipy(transform=False) - X).max() == 0


def test_read_10x_h5_backed_matches_in_memory(counts, tmp_path):
    X, names = counts
    h5 = tmp_path / "m.h5"
    write_10x_h5(h5, X, names)
    bf = crest.read_10x_h5(h5, backed=tmp_path / "b", nnz_per_part=SMALL_PART_NNZ)
    assert abs(bf.to_scipy(transform=False) - crest.read_10x_h5(h5).to_scipy(transform=False)).max() == 0
    assert not [p for p in (tmp_path / "b").iterdir() if p.name.startswith(".")]


# --------------------------------------------------------------------------- store ids stay inside the store
def assert_same_matrix(b, a_X):
    assert (b.to_scipy(transform=False) != a_X).nnz == 0


def filtered_in_store(counts, kind, tmp_path):
    X, names = counts
    bf = crest.pp.filter_cells(frame_in_store(X, names, kind, tmp_path), min_genes=MIN_GENES_FILTER)
    ids = bf.obs["cell_id"].to_numpy()
    assert 0 < len(ids) < N_CELLS and not np.array_equal(ids, np.arange(len(ids)))
    return bf


@pytest.mark.parametrize("kind", STORES)
@baseline_fails("v0.4.0", IndexError, "out of bounds")
def test_anndata_roundtrip_after_cell_filter(counts, tmp_path, kind):
    bf = filtered_in_store(counts, kind, tmp_path)
    a = bf.to_anndata()
    b = crest.BioFrame.from_anndata(a)
    assert_same_matrix(b, a.X)
    assert "cell_id" not in a.obs.columns and "gene_id" not in a.var.columns
    assert b.obs["cell_id"].to_list() == list(range(b.n_obs))
    assert b.var["gene_id"].to_list() == list(range(b.n_vars))
    crest.pp.calculate_qc_metrics(b)
    np.testing.assert_array_equal(b.obs["total_counts"].to_numpy(), np.asarray(a.X.sum(1)).ravel())
    assert b.obs["barcode"].to_list() == list(a.obs_names)  # cells keep their names


@pytest.mark.parametrize("kind", STORES)
@baseline_fails("v0.4.0", IndexError, "out of bounds")
def test_h5ad_roundtrip_after_cell_filter(counts, tmp_path, kind):
    bf = filtered_in_store(counts, kind, tmp_path)
    a = bf.to_anndata()
    a.write_h5ad(tmp_path / "f.h5ad")
    b = crest.read_h5ad(tmp_path / "f.h5ad")
    assert_same_matrix(b, a.X)
    assert b.obs["cell_id"].to_list() == list(range(b.n_obs))
    crest.pp.calculate_qc_metrics(b)  # first computation on the re-imported frame
    np.testing.assert_array_equal(b.obs["total_counts"].to_numpy(), np.asarray(a.X.sum(1)).ravel())


@baseline_fails("v0.4.0", IndexError, "out of bounds")
def test_h5ad_written_by_the_released_version_is_readable(counts, tmp_path):
    """Files written before this fix carry the stale ids as obs['cell_id'] / var['gene_id']."""
    bf = filtered(counts)
    a = bf.to_anndata()
    a.obs["cell_id"] = bf.obs["cell_id"].to_numpy()
    a.var["gene_id"] = bf.var["gene_id"].to_numpy()
    a.write_h5ad(tmp_path / "old.h5ad")
    b = crest.read_h5ad(tmp_path / "old.h5ad")
    crest.pp.calculate_qc_metrics(b)
    assert b.obs["cell_id"].to_list() == list(range(b.n_obs))
    assert b.obs["cell_id_orig"].to_list() == bf.obs["cell_id"].to_list()


def test_anndata_roundtrip_without_filter_and_names_survive(counts):
    X, names = counts
    bf = frame(counts)
    a = bf.to_anndata()
    b = crest.BioFrame.from_anndata(a)
    assert_same_matrix(b, X)
    assert b.var_names == names


@baseline_fails("v0.4.0", ValueError, "invalid literal")
def test_user_cell_id_and_gene_id_columns_are_kept_not_used_as_ids(counts):
    X, names = counts
    cells = [f"cell{i}" for i in range(X.shape[0])]
    genes = [f"ENSG{i}" for i in range(X.shape[1])]
    bf = crest.BioFrame.from_scipy(X, obs=pl.DataFrame({"cell_id": cells}), var=pl.DataFrame({"gene_id": genes}))
    crest.pp.calculate_qc_metrics(bf, mito_prefix=None)
    assert bf.obs["cell_id_orig"].to_list() == cells
    assert bf.var["gene_id_orig"].to_list() == genes
    assert bf.obs["cell_id"].to_list() == list(range(X.shape[0]))
    assert bf.var["gene_id"].to_list() == list(range(X.shape[1]))
    assert bf.var_names == genes  # gene names given as a 'gene_id' column still name the genes
    np.testing.assert_array_equal(bf.obs["total_counts"].to_numpy(), np.asarray(X.sum(1)).ravel())


@baseline_fails("v0.4.0", AssertionError, "Arrays are not equal")
def test_integer_cell_id_permutation_does_not_reorder_cells(counts):
    X, _ = counts
    n = X.shape[0]
    perm = np.arange(n)[::-1].copy()
    bf = crest.BioFrame.from_scipy(X, obs=pl.DataFrame({"cell_id": perm}))
    crest.pp.calculate_qc_metrics(bf, mito_prefix=None)
    np.testing.assert_array_equal(bf.obs["total_counts"].to_numpy(), np.asarray(X.sum(1)).ravel())
    assert bf.obs["cell_id_orig"].to_list() == list(perm)


@baseline_fails("v0.4.0", AssertionError, "")
def test_input_with_both_id_and_renamed_id_column_is_rejected(counts):
    X, _ = counts
    obs = pl.DataFrame({"cell_id": np.arange(X.shape[0]), "cell_id_orig": np.arange(X.shape[0])})
    try:
        crest.BioFrame.from_scipy(X, obs=obs)
        raised = False
    except ValueError as e:
        raised = "cell_id_orig" in str(e)
    assert raised


@baseline_fails("v0.4.0", AssertionError, "")
def test_from_triplets_assigns_ids(counts):
    X, _ = counts
    coo = X.tocoo()
    df = pl.DataFrame({"cell_id": coo.row.astype(np.uint32), "gene_id": coo.col.astype(np.uint32),
                       "count": coo.data}).sort("cell_id")
    obs = pl.DataFrame({"cell_id": [f"c{i}" for i in range(X.shape[0])]})
    bf = crest.BioFrame.from_triplets(df, X.shape[0], X.shape[1], obs=obs)
    assert bf.obs["cell_id"].to_list() == list(range(X.shape[0]))
    assert bf.obs["cell_id_orig"][0] == "c0"


def test_parquet_roundtrip_ids_unchanged(counts, tmp_path):
    bf = filtered(counts)
    bf.write_parquet(tmp_path / "p", nnz_per_part=SMALL_PART_NNZ)
    back = crest.read_parquet(tmp_path / "p")
    assert set(back.obs.columns) == set(bf.obs.columns)
    assert "cell_id_orig" not in back.obs.columns and "gene_id_orig" not in back.var.columns
    assert back.obs["cell_id"].to_list() == list(range(bf.n_obs))
    assert abs(back.to_scipy(transform=False) - bf.to_scipy(transform=False)).max() == 0


# --------------------------------------------------------------------------- to_anndata().write_h5ad() after the workflow
def run_workflow(bf, harmony=True):
    """The tutorial workflow plus the optional steps that store results in ``uns``."""
    bf = crest.pp.filter_cells(bf, min_genes=MIN_GENES_FILTER)
    bf = crest.pp.filter_genes(bf, min_cells=3)
    crest.pp.scrublet(bf)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    bf = crest.pp.highly_variable_genes(bf, n_top_genes=200)
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=15)
    crest.pp.neighbors(bf, n_neighbors=15)
    crest.tl.leiden(bf, resolution=0.5)
    crest.tl.umap(bf)
    crest.tl.rank_genes_groups(bf, "leiden", method="wilcoxon")
    crest.tl.score_genes(bf, list(bf.var_names[10:25]))
    crest.tl.leiden_sweep(bf, resolutions=[0.3, 0.6], n_seeds=2)
    if harmony:
        bf.obs = bf.obs.with_columns(pl.Series("batch", [f"b{i % 2}" for i in range(bf.n_obs)]))
        crest.tl.harmony(bf, "batch")
    return bf


def check_written(bf, path):
    """Write the AnnData and check what the file holds against the BioFrame."""
    a = bf.to_anndata()
    a.write_h5ad(path)
    b = ad.read_h5ad(path)
    assert b.shape == bf.shape
    np.testing.assert_array_equal(b.obsm["X_umap"], bf.obsm["X_umap"])  # exact, tolerance UMAP_EXACT == 0
    assert b.obs["leiden"].astype(str).to_list() == bf.obs["leiden"].to_list()
    np.testing.assert_allclose(b.uns["pca"]["variance_ratio"], bf.uns["pca"]["variance_ratio"], rtol=VARIANCE_RATIO_RTOL)
    t = bf.uns["rank_genes_groups"]["table"]
    bt = b.uns["rank_genes_groups"]["table"]
    assert list(bt.columns) == t.columns and len(bt) == t.height
    np.testing.assert_allclose(bt["scores"].to_numpy(), t["scores"].to_numpy(), rtol=VARIANCE_RATIO_RTOL)
    assert list(b.uns["leiden_sweep"]["table"].columns) == bf.uns["leiden_sweep"]["table"].columns
    assert all(isinstance(x, str) for x in b.uns["hvg"]["ops"])
    assert b.uns["hvg"]["ops"][-1] == "log1p"
    return a, b


@baseline_fails("v0.4.0", ValueError, "inhomogeneous")
def test_workflow_to_anndata_write_h5ad(counts, tmp_path):
    pytest.importorskip("harmonypy")
    bf = run_workflow(frame(counts))
    before = {k: type(v).__name__ for k, v in bf.uns.items()}
    a, b = check_written(bf, tmp_path / "wf.h5ad")
    assert {k: type(v).__name__ for k, v in bf.uns.items()} == before, "to_anndata must not alter bf.uns"
    assert isinstance(bf.uns["leiden_sweep"]["table"], pl.DataFrame)
    assert isinstance(bf.uns["hvg"]["ops"][0], tuple)
    # and back into a BioFrame, on the same file
    back = crest.read_h5ad(tmp_path / "wf.h5ad")
    assert back.shape == bf.shape and back.obs["cell_id"].to_list() == list(range(bf.n_obs))
    assert_same_matrix(back, a.X)


@baseline_fails("v0.4.0", ValueError, "inhomogeneous")
def test_workflow_without_harmony_unscaled_pca(counts, tmp_path):
    """Unscaled PCA stores None for the projection's mean/std."""
    bf = crest.pp.filter_cells(frame(counts), min_genes=MIN_GENES_FILTER)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.tl.pca(bf, n_comps=10, scale=False)
    assert bf.uns["pca"]["projection"]["mean"] is None
    bf.to_anndata().write_h5ad(tmp_path / "u.h5ad")
    b = ad.read_h5ad(tmp_path / "u.h5ad")
    assert "mean" not in b.uns["pca"]["projection"]


@baseline_fails("v0.4.0", Exception, "No method registered for writing")
def test_leiden_sweep_table_alone_is_writable(counts, tmp_path):
    bf = crest.pp.filter_cells(frame(counts), min_genes=MIN_GENES_FILTER)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=100)
    crest.tl.pca(bf, n_comps=10, scale=False)
    crest.pp.neighbors(bf)
    crest.tl.leiden_sweep(bf, resolutions=[0.3, 0.6], n_seeds=1)
    bf.uns = {"leiden_sweep": bf.uns["leiden_sweep"]}
    bf.to_anndata().write_h5ad(tmp_path / "s.h5ad")
    t = ad.read_h5ad(tmp_path / "s.h5ad").uns["leiden_sweep"]["table"]
    assert list(t.columns) == bf.uns["leiden_sweep"]["table"].columns


@baseline_fails("v0.4.0", ValueError, "inhomogeneous")
def test_uns_value_kinds_are_written(counts, tmp_path):
    """Entries of every shape CREST stores, plus the tuple lengths of old and new ``ops``."""
    bf = frame(counts)
    bf.uns = {
        "ops_two_and_three": {"ops": [("normalize_total", 1e4), ("normalize_total", 1e4, "fp"), ("log1p",)]},
        "none_entries": {"a": None, "b": 1.5, "nested": {"c": None}},
        "table": pl.DataFrame({"g": ["x", None, "z"], "v": [1.0, float("nan"), 3.0], "k": [1, 2, 3],
                               "lst": [[1, 2], [3], []]}),
        "mixed_list": [1, "a", None],
        "empty": [],
        "int_keys": {1: "one", 2: np.arange(3)},
        "ragged_arrays": [np.arange(2), np.arange(5)],
        "path": Path("/tmp/x"),
        "boolean": True,
    }
    bf.to_anndata().write_h5ad(tmp_path / "k.h5ad")
    u = ad.read_h5ad(tmp_path / "k.h5ad").uns
    assert list(u["ops_two_and_three"]["ops"]) == ["normalize_total|10000.0", "normalize_total|10000.0|fp", "log1p"]
    assert u["none_entries"] == {"b": 1.5, "nested": {}}
    assert u["table"]["g"].to_list() == ["x", "", "z"]
    assert np.isnan(u["table"]["v"].iloc[1])
    assert u["table"]["lst"].to_list() == ["1|2", "3", ""]
    assert sorted(u["int_keys"]) == ["1", "2"]
    assert list(u["mixed_list"]) == ["1", "a"]


def test_to_anndata_keeps_neighbors_as_obsp_and_user_columns(counts):
    bf = crest.pp.filter_cells(frame(counts), min_genes=MIN_GENES_FILTER)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=100)
    crest.tl.pca(bf, n_comps=10, scale=False)
    crest.pp.neighbors(bf)
    bf.obs = bf.obs.with_columns(pl.Series("group", ["a", "b"] * (bf.n_obs // 2) + ["a"] * (bf.n_obs % 2)))
    a = bf.to_anndata()
    assert a.obsp["connectivities"].shape == (bf.n_obs, bf.n_obs)
    assert "neighbors" not in a.uns
    assert a.obs["group"].to_list() == bf.obs["group"].to_list()


# --------------------------------------------------------------------------- real data (PBMC 3k)
PBMC3K = os.environ.get("CREST_TEST_PBMC3K")
PBMC_MIN_GENES = 600  # removes 385 of the 2700 cells, so the kept cell ids have gaps


@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
@baseline_fails("v0.4.0", ValueError, "inhomogeneous")
def test_pbmc3k_workflow_write_and_roundtrips(tmp_path):
    pytest.importorskip("harmonypy")
    bf = crest.read_10x_mtx(PBMC3K)
    bf = crest.pp.filter_cells(bf, min_genes=200)
    bf = crest.pp.filter_genes(bf, min_cells=3)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    bf = crest.pp.highly_variable_genes(bf, n_top_genes=2000)
    crest.pp.scale(bf, max_value=10)
    crest.tl.pca(bf, n_comps=50)
    crest.pp.neighbors(bf, n_neighbors=15)
    crest.tl.leiden(bf)
    crest.tl.umap(bf)
    crest.tl.rank_genes_groups(bf, "leiden", method="wilcoxon")
    crest.tl.leiden_sweep(bf, resolutions=[0.5, 1.0], n_seeds=2)
    a, b = check_written(bf, tmp_path / "pbmc.h5ad")
    back = crest.read_h5ad(tmp_path / "pbmc.h5ad")
    assert_same_matrix(back, a.X)
    again = crest.BioFrame.from_anndata(b)
    assert_same_matrix(again, a.X)
    assert again.obs["cell_id"].to_list() == list(range(again.n_obs))


@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
@pytest.mark.parametrize("kind", STORES)
@baseline_fails("v0.4.0", IndexError, "out of bounds")
def test_pbmc3k_filtered_roundtrip_per_store(tmp_path, kind):
    full = crest.read_10x_mtx(PBMC3K)
    X = full.to_scipy(transform=False)
    names = full.var_names
    bf = crest.pp.filter_cells(frame_in_store(X, names, kind, tmp_path), min_genes=PBMC_MIN_GENES)
    assert bf.n_obs < X.shape[0]
    a = bf.to_anndata()
    b = crest.BioFrame.from_anndata(a)
    assert_same_matrix(b, a.X)
    crest.pp.calculate_qc_metrics(b)
    a.write_h5ad(tmp_path / "f.h5ad")
    c = crest.read_h5ad(tmp_path / "f.h5ad")
    assert_same_matrix(c, a.X)
    crest.pp.calculate_qc_metrics(c)
    if kind == "parquet":  # filtered Parquet-backed frame written to a new directory and reopened
        bf.write_parquet(tmp_path / "out")
        assert crest.read_parquet(tmp_path / "out").shape == bf.shape
