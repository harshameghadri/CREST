"""Sparse input that is not in canonical form: duplicate (cell, gene) entries, stored explicit
zeros, and hand-made Parquet directories.

Every store (in-memory CSR, triplet frame, Parquet parts) must give the statistics of the
canonical matrix: duplicates summed, zeros not counted as non-zeros. Tests marked ``baseline_fails``
fail on the stated release (``v0.4.0`` wheel, or ``rc0``: the first 0.5 release candidate) for the
reason given and pass on this tree; unmarked tests (pins) must hold on both.
"""

import contextlib
import re
import warnings

import numpy as np
import polars as pl
import pytest

import crest

sp = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")
h5py = pytest.importorskip("h5py")
ad = pytest.importorskip("anndata")

# ---- named tolerances / sizes
STATS_ATOL = 1e-6          # gene_stats of a split / zero-padded matrix against the canonical one
EIGENVALUE_ATOL = 1e-10    # PCA variances of aligned Parquet parts against the dense float64 reference
N_CELLS, N_GENES = 240, 60
SMALL_PART_NNZ = 1500      # several Parquet parts for the synthetic data
STORES = ["csr", "triplets", "parquet"]


def baseline_fails(ref, raises, match):
    """Documentation marker, see the module docstring (registered by the test plugin when present)."""
    return pytest.mark.baseline_fails(ref=ref, raises=raises, match=match)


@contextlib.contextmanager
def must_raise(exc, match):
    """Like ``pytest.raises`` but an absent exception is an ``AssertionError`` with a plain message."""
    try:
        yield
    except exc as e:
        assert re.search(match, str(e)), f"{type(e).__name__} message {str(e)!r} does not match {match!r}"
    else:
        raise AssertionError(f"expected {exc.__name__} matching {match!r}; nothing was raised")


# --------------------------------------------------------------------------- data
def make_counts(n_cells=N_CELLS, n_genes=N_GENES, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.gamma(0.6, 1.0, n_genes)[None, :] * rng.lognormal(0.5, 0.4, (n_cells, 1)) * 3
    mu[: n_cells // 2, :10] *= 4
    return sp.csr_matrix(rng.poisson(mu).astype(np.float32))


def split_in_two(X, shuffle_rows=False, seed=1):
    """The same matrix as a non-canonical CSR: every entry >= 2 stored twice (parts a and x - a)."""
    rng = np.random.default_rng(seed)
    indptr, indices, data = [0], [], []
    for i in range(X.shape[0]):
        lo, hi = X.indptr[i], X.indptr[i + 1]
        g, v = X.indices[lo:hi], X.data[lo:hi]
        ents = []
        for gg, vv in zip(g, v):
            if vv >= 2:
                a = float(rng.integers(1, int(vv)))
                ents += [(gg, a), (gg, vv - a)]
            else:
                ents.append((gg, vv))
        if shuffle_rows:
            rng.shuffle(ents)
        indices += [e[0] for e in ents]
        data += [e[1] for e in ents]
        indptr.append(len(indices))
    return (np.array(data, np.float32), np.array(indices, np.int32), np.array(indptr, np.int64))


def with_explicit_zeros(X, seed=2):
    """The same matrix with a stored 0.0 added next to ~30% of the entries (gene ids stay sorted)."""
    rng = np.random.default_rng(seed)
    coo = X.tocoo()
    zr = rng.integers(0, X.shape[0], 4 * X.shape[0])
    zc = rng.integers(0, X.shape[1], 4 * X.shape[0])
    have = set(zip(coo.row.tolist(), coo.col.tolist()))
    keep = [(r, c) for r, c in set(zip(zr.tolist(), zc.tolist())) if (r, c) not in have]
    rows = np.concatenate([coo.row, [r for r, _ in keep]])
    cols = np.concatenate([coo.col, [c for _, c in keep]])
    vals = np.concatenate([coo.data, np.zeros(len(keep), np.float32)])
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=X.shape[0]))])
    Z = sp.csr_matrix((vals, cols.astype(np.int32), indptr.astype(np.int64)), shape=X.shape)
    assert Z.nnz > X.nnz and (Z.data == 0).any() and not Z.has_canonical_format is None
    return Z


def names(n=N_GENES):
    return pl.DataFrame({"gene_name": [f"MT-{i}" if i < 5 else f"G{i}" for i in range(n)],
                         "gene_ids": [f"E{i}" for i in range(n)]})


def build(kind, X, tmp_path, as_scipy=True):
    """A BioFrame holding ``X`` (a scipy matrix, possibly with duplicates / explicit zeros) in ``kind``."""
    var = names(X.shape[1])
    if kind == "csr":
        return crest.BioFrame.from_scipy(X, var=var) if as_scipy else crest.BioFrame.from_csr(X.indptr, X.indices, X.data, X.shape[1], var=var)
    coo_rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    df = pl.DataFrame({"cell_id": coo_rows.astype(np.uint32), "gene_id": X.indices.astype(np.uint32), "count": X.data})
    bf = crest.BioFrame.from_triplets(df, X.shape[0], X.shape[1], var=var)
    if kind == "triplets":
        return bf
    d = tmp_path / "pq"
    bf.write_parquet(d, nnz_per_part=SMALL_PART_NNZ)
    return crest.read_parquet(d)


def assert_same_statistics(bf, canon):
    """QC, gene_stats, HVG flags and filter outcomes of ``bf`` equal those of the canonical frame."""
    o1, v1 = crest.pp.calculate_qc_metrics(bf, inplace=False)
    o2, v2 = crest.pp.calculate_qc_metrics(canon, inplace=False)
    for c in ("n_genes_by_counts", "total_counts", "pct_counts_mt"):
        np.testing.assert_allclose(o1[c].to_numpy(), o2[c].to_numpy(), atol=STATS_ATOL, err_msg=f"cell {c}")
    for c in ("n_cells_by_counts", "total_counts"):
        np.testing.assert_allclose(v1[c].to_numpy(), v2[c].to_numpy(), atol=STATS_ATOL, err_msg=f"gene {c}")
    np.testing.assert_allclose(crest.pp.gene_stats(bf), crest.pp.gene_stats(canon), atol=STATS_ATOL, err_msg="gene_stats")
    assert crest.pp.filter_genes(bf, min_cells=5).n_vars == crest.pp.filter_genes(canon, min_cells=5).n_vars


def lognorm(bf):
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    return bf


# --------------------------------------------------------------------------- duplicate (cell, gene) entries
@pytest.fixture(scope="module")
def canon_X():
    return make_counts()


@pytest.mark.parametrize("shuffle", [False, True])
@baseline_fails("v0.4.0", AssertionError, "Not equal|Mismatch|not close")
def test_from_scipy_sums_duplicate_entries_of_a_csr(canon_X, tmp_path, shuffle):
    data, idx, indptr = split_in_two(canon_X, shuffle_rows=shuffle)
    X = sp.csr_matrix((data, idx, indptr), shape=canon_X.shape)
    assert X.nnz > canon_X.nnz
    bf = build("csr", X, tmp_path)
    canon = build("csr", canon_X, tmp_path)
    assert_same_statistics(bf, canon)
    np.testing.assert_allclose(bf.to_scipy(transform=False).toarray(), canon_X.toarray())


@baseline_fails("v0.4.0", AssertionError, "in place|flag")
def test_from_scipy_leaves_the_callers_matrix_untouched(canon_X):
    data, idx, indptr = split_in_two(canon_X, shuffle_rows=True)
    X = sp.csr_matrix((data.copy(), idx.copy(), indptr.copy()), shape=canon_X.shape)
    X.has_sorted_indices = False
    before = (X.indices.copy(), X.data.copy(), X.indptr.copy())
    crest.BioFrame.from_scipy(X)
    assert np.array_equal(X.indices, before[0]), "from_scipy reordered the caller's indices in place"
    assert np.array_equal(X.data, before[1]) and np.array_equal(X.indptr, before[2])
    assert X.has_sorted_indices is False, "from_scipy changed the caller's has_sorted_indices flag"


def test_from_scipy_of_a_canonical_matrix_shares_no_state_and_matches(canon_X, tmp_path):  # pin
    before = canon_X.indices.copy()
    bf = build("csr", canon_X, tmp_path)
    assert np.array_equal(canon_X.indices, before)
    np.testing.assert_allclose(bf.to_scipy(transform=False).toarray(), canon_X.toarray())


@baseline_fails("v0.4.0", AssertionError, "Not equal|Mismatch|not close")
def test_from_scipy_sums_duplicates_of_a_csc_matrix(canon_X, tmp_path):
    data, idx, indptr = split_in_two(canon_X)
    coo_like = sp.csr_matrix((data, idx, indptr), shape=canon_X.shape)
    csc = sp.csc_matrix(coo_like.T.tocsr()).T.tocsc()  # still holds the duplicates, in CSC layout
    bf = crest.BioFrame.from_scipy(csc, var=names())
    assert_same_statistics(bf, build("csr", canon_X, tmp_path))


@pytest.mark.parametrize("shuffle", [False, True])
@baseline_fails("v0.4.0", AssertionError, "Not equal|Mismatch|not close")
def test_from_csr_sums_repeated_entries(canon_X, tmp_path, shuffle):
    data, idx, indptr = split_in_two(canon_X, shuffle_rows=shuffle)  # shuffled: the repeats are not adjacent
    before = (data.copy(), idx.copy(), indptr.copy())
    bf = crest.BioFrame.from_csr(indptr, idx, data, canon_X.shape[1], var=names())
    assert_same_statistics(bf, build("csr", canon_X, tmp_path))
    assert all(np.array_equal(a, b) for a, b in zip(before, (data, idx, indptr))), "the caller's arrays were modified"


@baseline_fails("v0.4.0", AssertionError, "nothing was raised")
def test_csr_store_refuses_repeated_entries_directly(canon_X):
    data, idx, indptr = split_in_two(canon_X)
    with must_raise(ValueError, "stored twice"):
        crest.CSRStore(indptr, idx, data, canon_X.shape[1])


@pytest.mark.parametrize("order", ["descending", "shuffled"])
def test_from_csr_with_unsorted_rows_without_repeats_is_accepted(canon_X, tmp_path, order):  # pin (Cell Ranger lists genes in descending order)
    rng = np.random.default_rng(3)
    ind, dat = canon_X.indices.copy(), canon_X.data.copy()
    for i in range(canon_X.shape[0]):
        lo, hi = canon_X.indptr[i], canon_X.indptr[i + 1]
        o = np.arange(hi - lo)[::-1] if order == "descending" else rng.permutation(hi - lo)
        ind[lo:hi], dat[lo:hi] = ind[lo:hi][o], dat[lo:hi][o]
    bf = crest.BioFrame.from_csr(canon_X.indptr, ind, dat, canon_X.shape[1], var=names())
    assert_same_statistics(bf, build("csr", canon_X, tmp_path))
    np.testing.assert_allclose(bf.to_scipy(transform=False).toarray(), canon_X.toarray())


def test_from_csr_with_canonical_rows_and_empty_rows_is_accepted():  # pin
    X = make_counts(30, 12, seed=4).tolil()
    X[3, :] = 0
    X[29, :] = 0
    X = X.tocsr()
    bf = crest.BioFrame.from_csr(X.indptr, X.indices, X.data, 12)
    np.testing.assert_allclose(bf.to_scipy(transform=False).toarray(), X.toarray())


@baseline_fails("v0.4.0", AssertionError, "Not equal|Mismatch|not close")
def test_triplets_with_repeated_cell_gene_pairs_equal_the_summed_matrix(canon_X, tmp_path):
    data, idx, indptr = split_in_two(canon_X)
    X = sp.csr_matrix((data, idx, indptr), shape=canon_X.shape)
    bf = build("triplets", X, tmp_path)
    canon = build("triplets", canon_X, tmp_path)
    assert_same_statistics(bf, canon)
    for b in (bf, canon):
        lognorm(b)
        crest.pp.highly_variable_genes(b, n_top_genes=20, flavor="seurat")
        crest.pp.scale(b, max_value=10)
        crest.tl.pca(b, n_comps=5)
    assert bf.var["highly_variable"].to_list() == canon.var["highly_variable"].to_list()
    np.testing.assert_allclose(bf.uns["pca"]["variance"], canon.uns["pca"]["variance"], atol=STATS_ATOL)


@baseline_fails("v0.4.0", AssertionError, "Not equal|Mismatch|not close")
def test_h5ad_with_duplicate_entries_reads_as_the_summed_matrix(canon_X, tmp_path):
    data, idx, indptr = split_in_two(canon_X, shuffle_rows=True)
    path = tmp_path / "dup.h5ad"
    a = ad.AnnData(X=canon_X.copy(), var=pd.DataFrame(index=names()["gene_name"].to_list()))
    a.write_h5ad(path)
    with h5py.File(path, "r+") as f:
        for k in ("data", "indices", "indptr"):
            del f["X"][k]
        f["X"]["data"], f["X"]["indices"], f["X"]["indptr"] = data, idx, indptr
    bf = crest.read_h5ad(path)
    assert_same_statistics(bf, build("csr", canon_X, tmp_path))


# --------------------------------------------------------------------------- explicit zeros
@pytest.mark.parametrize("kind", STORES)
@baseline_fails("v0.4.0", AssertionError, "Not equal|Mismatch|not close")
def test_stored_zeros_are_not_counted_as_non_zeros(canon_X, tmp_path, kind):
    Z = with_explicit_zeros(canon_X)
    bf = build(kind, Z, tmp_path, as_scipy=False) if kind == "csr" else build(kind, Z, tmp_path)
    canon = build(kind, canon_X, tmp_path / "c") if kind != "parquet" else build("csr", canon_X, tmp_path)
    assert_same_statistics(bf, canon)
    st = crest.pp.gene_stats(bf)
    np.testing.assert_array_equal(st[:, 4], (canon_X != 0).sum(0).A1)


@pytest.mark.parametrize("kind", STORES)
@baseline_fails("v0.4.0", AssertionError, "Not equal|Mismatch|not close|differ")
def test_stored_zeros_do_not_change_hvg_or_markers(canon_X, tmp_path, kind):
    Z = with_explicit_zeros(canon_X)
    bf = build(kind, Z, tmp_path / "z") if kind != "parquet" else build(kind, Z, tmp_path)
    canon = build("csr", canon_X, tmp_path)
    labels = np.where(np.arange(N_CELLS) < N_CELLS // 2, "a", "b")
    for b in (bf, canon):
        b.obs = b.obs.with_columns(pl.Series("grp", labels))
        lognorm(b)
        crest.pp.highly_variable_genes(b, n_top_genes=20, flavor="seurat")
    np.testing.assert_array_equal(bf.var["n_cells_expr"].to_numpy(), canon.var["n_cells_expr"].to_numpy())
    assert bf.var["highly_variable"].to_list() == canon.var["highly_variable"].to_list()
    for method in ("t-test", "wilcoxon"):
        r1 = crest.tl.rank_genes_groups(bf, "grp", method=method)
        r2 = crest.tl.rank_genes_groups(canon, "grp", method=method)
        np.testing.assert_allclose(r1["scores"].to_numpy(), r2["scores"].to_numpy(), atol=1e-4, err_msg=method)


# --------------------------------------------------------------------------- hand-made Parquet directories
def _write_parts(path, df, bounds, n_cells, n_genes):
    """Write ``df`` (cell_id, gene_id, count) as Parquet parts cut at the given row positions."""
    path.mkdir()
    cuts = [0, *bounds, df.height]
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        df[a:b].write_parquet(path / f"part-{i:05d}.parquet")
    pl.DataFrame({"cell_id": np.arange(n_cells, dtype=np.uint32), "barcode": [f"c{i}" for i in range(n_cells)]}).write_parquet(path / "obs.parquet")
    pl.DataFrame({"gene_id": np.arange(n_genes, dtype=np.uint32), "gene_name": names(n_genes)["gene_name"].to_list()}).write_parquet(path / "var.parquet")
    return path


def _triplets(X):
    c = X.tocoo()
    o = np.lexsort((c.col, c.row))
    return pl.DataFrame({"cell_id": c.row[o].astype(np.uint32), "gene_id": c.col[o].astype(np.uint32), "count": c.data[o]})


def _cell_boundary(df, n_parts=3):
    cells = df["cell_id"].to_numpy()
    return [int(np.searchsorted(cells, cells[len(cells) * k // n_parts], side="left")) for k in range(1, n_parts)]


def _consume(bf):
    return crest.pp.calculate_qc_metrics(bf, inplace=False)


@baseline_fails("rc0", AssertionError, "nothing was raised")
def test_parquet_parts_that_split_a_cell_are_rejected(canon_X, tmp_path):
    df = _triplets(canon_X)
    mid = _cell_boundary(df)[0] + 1  # one entry into the next cell: that cell now lives in two parts
    assert df["cell_id"][mid - 1] == df["cell_id"][mid]
    bf = crest.read_parquet(_write_parts(tmp_path / "split", df, [mid], N_CELLS, N_GENES))
    with must_raise(ValueError, "split|several parts|more than one part|increase"):
        _consume(bf)


@baseline_fails("rc0", AssertionError, "nothing was raised")
def test_parquet_gene_id_out_of_range_is_rejected(canon_X, tmp_path):
    df = _triplets(canon_X)
    g = df["gene_id"].to_numpy().copy()
    g[3] = 999
    df = df.with_columns(pl.Series("gene_id", g))
    bf = crest.read_parquet(_write_parts(tmp_path / "gene", df, _cell_boundary(df), N_CELLS, N_GENES))
    with must_raise(ValueError, "gene"):
        _consume(bf)


@baseline_fails("v0.4.0", AssertionError, "nothing was raised")
def test_parquet_cell_id_out_of_range_is_rejected(canon_X, tmp_path):
    df = _triplets(canon_X)
    c = df["cell_id"].to_numpy().copy()
    c[-1] = 5000
    df = df.with_columns(pl.Series("cell_id", c))
    bf = crest.read_parquet(_write_parts(tmp_path / "cell", df, _cell_boundary(df), N_CELLS, N_GENES))
    with must_raise(ValueError, "cell"):
        _consume(bf)


@baseline_fails("rc0", AssertionError, "nothing was raised")
def test_parquet_parts_out_of_order_are_rejected(canon_X, tmp_path):
    df = _triplets(canon_X)
    b = _cell_boundary(df)
    parts = [df[a:c] for a, c in zip([0, *b], [*b, df.height])]
    shuffled = pl.concat([parts[1], parts[0], parts[2]])
    bf = crest.read_parquet(_write_parts(tmp_path / "order", shuffled, [parts[1].height, parts[1].height + parts[0].height], N_CELLS, N_GENES))
    with must_raise(ValueError, "split|several parts|increase|sorted"):
        _consume(bf)


def test_parquet_unsorted_cells_within_a_part_are_rejected(canon_X, tmp_path):  # pin
    df = _triplets(canon_X)
    c = df["cell_id"].to_numpy().copy()
    i, j = (int(np.searchsorted(c, k)) for k in (1, 5))  # first entries of cells 1 and 5
    c[i], c[j] = c[j], c[i]  # two entries trade places: the cell ids are no longer in order
    df = df.with_columns(pl.Series("cell_id", c))
    bf = crest.read_parquet(_write_parts(tmp_path / "unsorted", df, [], N_CELLS, N_GENES))
    with must_raise(ValueError, "sorted"):
        _consume(bf)


def test_aligned_parquet_parts_pca_equals_the_dense_reference(canon_X, tmp_path):  # pin
    df = _triplets(canon_X)
    bf = crest.read_parquet(_write_parts(tmp_path / "ok", df, _cell_boundary(df), N_CELLS, N_GENES))
    crest.tl.pca(bf, n_comps=8, use_highly_variable=False, scale=False)
    D = canon_X.toarray().astype(np.float64)
    D -= D.mean(0)
    ev = np.linalg.svd(D, compute_uv=False) ** 2 / (N_CELLS - 1)
    np.testing.assert_allclose(bf.uns["pca"]["variance"][:8], ev[:8], atol=EIGENVALUE_ATOL * max(1.0, ev[0]))


@baseline_fails("rc0", AssertionError, "Not equal|Mismatch|not close")
def test_parquet_parts_with_repeated_cell_gene_pairs_equal_the_summed_matrix(canon_X, tmp_path):
    data, idx, indptr = split_in_two(canon_X)
    X = sp.csr_matrix((data, idx, indptr), shape=canon_X.shape)
    rows = np.repeat(np.arange(N_CELLS), np.diff(indptr)).astype(np.uint32)
    df = pl.DataFrame({"cell_id": rows, "gene_id": idx.astype(np.uint32), "count": data})
    bf = crest.read_parquet(_write_parts(tmp_path / "dup", df, _cell_boundary(df), N_CELLS, N_GENES))
    assert_same_statistics(bf, build("csr", canon_X, tmp_path))


@pytest.mark.parametrize("kind", STORES)
@baseline_fails("v0.4.0", AssertionError, "stored zeros|nnz")
def test_to_scipy_does_not_carry_stored_zeros(canon_X, tmp_path, kind):
    Z = with_explicit_zeros(canon_X)
    bf = build(kind, Z, tmp_path / "z") if kind != "parquet" else build(kind, Z, tmp_path)
    M = bf.to_scipy(transform=False)
    assert M.nnz == canon_X.nnz, f"to_scipy holds {M.nnz - canon_X.nnz} stored zeros (nnz {M.nnz} vs {canon_X.nnz})"
    np.testing.assert_array_equal(M.toarray(), canon_X.toarray())
