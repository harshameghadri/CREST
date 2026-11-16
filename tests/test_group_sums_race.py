"""Group sums: the group-id array is read once, under the GIL, into an owned copy, so a
second Python thread rewriting it during a call can no longer steer the kernel's writes out
of bounds; explicit zeros stored in a sparse input are not counted as non-zeros."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from crest import crest as _native

sp = pytest.importorskip("scipy.sparse")

baseline_fails = pytest.mark.baseline_fails

# ---- named parameters / tolerances ----------------------------------------------------
RACE_CELLS = 100_000      # fewer cells make the unfixed build crash less often (4/6 at 50 000)
RACE_NNZ_PER_CELL = 20
RACE_CALLS = 100
RACE_RUNS = 20            # subprocess runs per binding
CONTROL_RUNS = 3
RACE_TIMEOUT_S = 600
SUM_RTOL = 1e-12          # per-group sums against a float64 numpy reduction of the same float32 values
U32_MAX = np.iinfo(np.uint32).max

RACE_SCRIPT = textwrap.dedent('''
    import sys, threading
    import numpy as np
    from crest import crest as nat

    fn_name, mode = sys.argv[1], sys.argv[2]          # mode: "race" (invalid ids) or "control" (valid ids)
    n, per, V, G, calls = {n}, {per}, 200, 4, {calls}
    rng = np.random.default_rng(0)
    indptr = np.arange(0, n * per + 1, per, dtype=np.int64)
    genes = rng.integers(0, V, n * per).astype(np.uint32)
    vals = np.ones(n * per, np.float32)
    import crest
    bf = crest.BioFrame.from_csr(indptr, genes, vals, V)
    bf.chunk_nnz = n * per + 1            # one chunk
    ctx = list(next(bf.iter_ctx(transform=False)))
    grp = (np.arange(n) % G).astype(np.uint32)
    s, q, z = np.zeros((G, V)), np.zeros((G, V)), np.zeros((G, V))
    other = 1 if mode == "control" else 4_000_000_000
    stop = threading.Event()

    def flip():
        while not stop.is_set():
            grp[:] = other
            grp[:] = 0

    t = threading.Thread(target=flip)
    t.start()
    try:
        for _ in range(calls):
            try:
                if fn_name == "group_gene_sums":
                    nat.group_gene_sums(*ctx, grp, s, q, z, None)
                else:
                    nat.group_gene_totals(*ctx, grp, s)
            except ValueError:
                pass   # an out-of-range id was seen: the intended outcome
    finally:
        stop.set()
        t.join()
    print("done")
''').format(n=RACE_CELLS, per=RACE_NNZ_PER_CELL, calls=RACE_CALLS)


def _run(fn_name, mode):
    return subprocess.run([sys.executable, "-c", RACE_SCRIPT, fn_name, mode], capture_output=True, text=True,
                          timeout=RACE_TIMEOUT_S, env={**os.environ})


def _assert_no_crash(fn_name, mode, runs):
    crashed = []
    for i in range(runs):
        p = _run(fn_name, mode)
        if p.returncode != 0:
            crashed.append((i, p.returncode, p.stderr[-300:]))
            if p.returncode < 0:
                break  # killed by a signal: no need to repeat
    if crashed:
        i, rc, err = crashed[0]
        what = f"negative return code {rc} (killed by signal {-rc})" if rc < 0 else f"return code {rc}"
        raise AssertionError(f"{fn_name} race run {i}: {what}; {len(crashed)} failing run(s); stderr tail: {err!r}")


# ======================================================================================
# memory safety
# ======================================================================================
@baseline_fails(ref="v0.4.0", raises=AssertionError, match="negative return code")
@pytest.mark.parametrize("fn_name", ["group_gene_sums", "group_gene_totals"])
def test_rewriting_group_ids_during_a_call_cannot_crash(fn_name):
    _assert_no_crash(fn_name, "race", RACE_RUNS)


@pytest.mark.parametrize("fn_name", ["group_gene_sums", "group_gene_totals"])
def test_control_rewriting_with_valid_group_ids_exits_cleanly(fn_name):
    _assert_no_crash(fn_name, "control", CONTROL_RUNS)


# ======================================================================================
# results and validation (pins: behaviour that is already right)
# ======================================================================================
def _chunk(n_cells=300, n_genes=40, seed=0, zeros=0.0):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.8, size=(n_cells, n_genes)).astype(np.float32)
    m = sp.csr_matrix(dense)
    if zeros:
        # store explicit zeros: take a random subset of the structural zeros and add them as entries
        r, c = np.nonzero(dense == 0)
        pick = rng.random(len(r)) < zeros
        extra = sp.csr_matrix((np.ones(pick.sum()), (r[pick], c[pick])), shape=dense.shape)
        m = (m + extra).tocsr()           # entries 1.0 at the new positions
        m.sort_indices()
        rr = np.repeat(np.arange(n_cells), np.diff(m.indptr))
        pos = dense[rr, m.indices] == 0
        m.data[pos] = 0.0                 # now stored as explicit zeros
        assert (m.data == 0).sum() == pick.sum() > 0
    import crest
    bf = crest.BioFrame.from_csr(m.indptr.astype(np.int64), m.indices.astype(np.uint32), m.data.astype(np.float32), n_genes)
    ctx = list(next(bf.iter_ctx(transform=False)))   # the argument tuple of the installed build
    assert len(list(bf.iter_ctx(transform=False))) == 1
    return dense, ctx


def test_group_sums_match_numpy_and_ignore_cells_without_a_group():
    G = 5
    dense, ctx = _chunk()
    rng = np.random.default_rng(1)
    grp = rng.integers(0, G, len(dense)).astype(np.uint32)
    grp[::7] = U32_MAX
    s, q, z = (np.zeros((G, dense.shape[1])) for _ in range(3))
    _native.group_gene_sums(*ctx, grp, s, q, z, None)
    tot = np.zeros((G, dense.shape[1]))
    _native.group_gene_totals(*ctx, grp, tot)
    for g in range(G):
        x = dense[grp == g].astype(np.float64)
        np.testing.assert_allclose(s[g], x.sum(0), rtol=SUM_RTOL)
        np.testing.assert_allclose(q[g], (x ** 2).sum(0), rtol=SUM_RTOL)
        np.testing.assert_array_equal(z[g], (x != 0).sum(0))
        np.testing.assert_allclose(tot[g], s[g], rtol=SUM_RTOL)


def test_group_sums_with_a_clip_limit():
    G = 3
    dense, ctx = _chunk(seed=2)
    grp = (np.arange(len(dense)) % G).astype(np.uint32)
    clip = np.full((G, dense.shape[1]), 2.0)
    s, q, z = (np.zeros((G, dense.shape[1])) for _ in range(3))
    _native.group_gene_sums(*ctx, grp, s, q, z, clip)
    for g in range(G):
        x = np.minimum(dense[grp == g].astype(np.float64), 2.0)
        np.testing.assert_allclose(s[g], x.sum(0), rtol=SUM_RTOL)
        np.testing.assert_allclose(q[g], (x ** 2).sum(0), rtol=SUM_RTOL)


@pytest.mark.parametrize("bad", [3, 4, 10_000, U32_MAX - 1])
def test_group_ids_out_of_range_raise_value_error(bad):
    G = 3
    dense, ctx = _chunk(seed=3)
    grp = (np.arange(len(dense)) % G).astype(np.uint32)
    grp[17] = bad
    s, q, z = (np.zeros((G, dense.shape[1])) for _ in range(3))
    with pytest.raises(ValueError, match="group id out of range"):
        _native.group_gene_sums(*ctx, grp, s, q, z, None)
    with pytest.raises(ValueError, match="group id out of range"):
        _native.group_gene_totals(*ctx, grp, s)
    assert not s.any() and not q.any() and not z.any()   # nothing was written


# ======================================================================================
# explicit zeros are not non-zeros (kernel half)
# ======================================================================================
@baseline_fails(ref="v0.4.0", raises=AssertionError, match="non-zero count")
def test_gene_stats_count_only_non_zero_entries():
    dense, ctx = _chunk(zeros=0.3)
    V = dense.shape[1]
    out = np.zeros((V, 5))
    _native.gene_stats(*ctx, out)
    got, want = out[:, 4], (dense != 0).sum(0)
    assert np.array_equal(got, want), f"non-zero count differs for {(got != want).sum()} of {V} genes"
    # sums are unaffected by stored zeros
    np.testing.assert_allclose(out[:, 0], dense.sum(0, dtype=np.float64), rtol=SUM_RTOL)
    np.testing.assert_allclose(out[:, 1], (dense.astype(np.float64) ** 2).sum(0), rtol=SUM_RTOL)


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="non-zero count")
def test_group_gene_sums_count_only_non_zero_entries():
    G = 4
    dense, ctx = _chunk(zeros=0.3, seed=4)
    grp = (np.arange(len(dense)) % G).astype(np.uint32)
    s, q, z = (np.zeros((G, dense.shape[1])) for _ in range(3))
    _native.group_gene_sums(*ctx, grp, s, q, z, None)
    for g in range(G):
        want = (dense[grp == g] != 0).sum(0)
        assert np.array_equal(z[g], want), f"non-zero count differs in group {g}"
        np.testing.assert_allclose(s[g], dense[grp == g].sum(0, dtype=np.float64), rtol=SUM_RTOL)


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="non-zero count")
def test_group_gene_sums_with_clip_count_only_non_zero_entries():
    G = 2
    dense, ctx = _chunk(zeros=0.3, seed=5)
    grp = (np.arange(len(dense)) % G).astype(np.uint32)
    clip = np.full((G, dense.shape[1]), 1.0)
    s, q, z = (np.zeros((G, dense.shape[1])) for _ in range(3))
    _native.group_gene_sums(*ctx, grp, s, q, z, clip)
    for g in range(G):
        assert np.array_equal(z[g], (dense[grp == g] != 0).sum(0)), "non-zero count differs"


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="n_cells_expr")
def test_highly_variable_genes_n_cells_expr_ignores_stored_zeros():
    import crest
    dense, _ = _chunk(n_cells=200, n_genes=60, seed=6, zeros=0.3)
    m = sp.csr_matrix(dense)
    r, c = np.nonzero(dense == 0)
    keep = np.random.default_rng(7).random(len(r)) < 0.3
    with_zeros = sp.csr_matrix((np.concatenate([m.data, np.zeros(keep.sum(), np.float32)]),
                                (np.concatenate([np.nonzero(dense)[0], r[keep]]), np.concatenate([np.nonzero(dense)[1], c[keep]]))),
                               shape=dense.shape)
    assert (with_zeros.data == 0).sum() == keep.sum() > 0
    bf = crest.BioFrame.from_scipy(with_zeros)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=20)
    got = bf.var["n_cells_expr"].to_numpy()
    want = (dense != 0).sum(0)
    assert np.array_equal(got, want), f"n_cells_expr differs for {(got != want).sum()} of {len(want)} genes"
