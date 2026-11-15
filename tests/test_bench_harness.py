"""Tests of the benchmark harness: bench/paper (summarize, accuracy, run_one, run_all),
bench/deseq2/kang_pseudobulk.py and scripts/crest_paper_bench.sh.

They run without any benchmark data: hand-built result tables, a synthetic kNN graph, a
mocked Rscript, and the shell functions of the driver script run in a scratch directory.
A test marked ``baseline_fails`` fails on the baseline build with the stated exception and
passes on this one; the others pin behaviour that already held.
"""

import json
import os
import re
import subprocess
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "bench" / "paper"), str(REPO / "bench" / "deseq2")]
SCRIPT = (REPO / "scripts" / "crest_paper_bench.sh").read_text()

# --------------------------------------------------------------------------- tolerances
EXACT = 1e-12                  # closed-form results on noise-free inputs
RECALL_EXACT = 0.999           # recall of an exact kNN graph against brute force
N_FAKE_CELLS = 300
K_NEIGHBORS = 15               # n_neighbors; the graphs have K_NEIGHBORS - 1 columns
N_GENES_BP = 3000              # genes in the BiocParallel chunking tests
N_WORKERS_BP = 64              # BiocParallel workers of the matched benchmark at the highest k


def _bash(code: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", code], capture_output=True, text=True, env={**os.environ, **env})


def _shell_block(start: str, end: str) -> str:
    """Text of the driver script from the regex ``start`` to the first match of ``end`` after it."""
    m = re.search(start + r".*?" + end, SCRIPT, re.S)
    assert m, f"block matching {start!r} not found in scripts/crest_paper_bench.sh"
    return m.group(0)


@pytest.fixture(scope="module")
def summarize():
    import summarize as mod
    return mod


@pytest.fixture(scope="module")
def accuracy():
    import accuracy as mod
    return mod


@pytest.fixture(scope="module")
def run_one():
    """bench/paper/run_one.py. Its monitor module imports psutil at import time only; the
    functions tested here do not use it, so a stub stands in when psutil is not installed."""
    try:
        import psutil  # noqa: F401
    except ImportError:
        sys.modules["psutil"] = types.ModuleType("psutil")
    import run_one as mod
    return mod


@pytest.fixture(scope="module")
def run_all():
    import run_all as mod
    return mod


@pytest.fixture(scope="module")
def kp():
    pytest.importorskip("crest")
    import kang_pseudobulk as mod
    return mod


# --------------------------------------------------------------------------- summarize: statistics
def test_ratio_range_median_and_envelope(summarize):
    a, b = [1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]
    med, lo, hi = summarize.ratio_range(a, b)
    assert med == pytest.approx(10.0, abs=EXACT)
    assert lo == pytest.approx(10.0 / 5.0, abs=EXACT)     # min(b) / max(a)
    assert hi == pytest.approx(50.0 / 1.0, abs=EXACT)     # max(b) / min(a)


def test_iqr_hodges_lehmann(summarize):
    assert summarize.iqr([1, 2, 3, 4, 5]) == pytest.approx(2.0, abs=EXACT)
    assert summarize.hodges_lehmann([1, 2, 3], [11, 12, 13]) == pytest.approx(10.0, abs=EXACT)


def test_mann_whitney_complete_separation_is_the_five_by_five_minimum(summarize):
    p = summarize.mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert p == pytest.approx(2 / 252, abs=1e-9)           # 0.0079: no information beyond separation
    assert summarize.mann_whitney([1.0], [2.0]) is None     # fewer than 2 repeats: no test


def test_amdahl_recovers_a_known_serial_fraction(summarize):
    p = np.array([1, 2, 4, 8, 16])
    f = 0.2
    s = summarize.amdahl_speedup(f, p)
    assert summarize.amdahl(p, s) == pytest.approx(f, abs=1e-9)
    assert summarize.amdahl_speedup(f, 1) == pytest.approx(1.0, abs=EXACT)


def test_best_threads_and_edge_flag(summarize):
    samp = {("d", "crest", p): {"t": np.array([t])} for p, t in ((8, 5.0), (16, 4.0), (32, 6.0))}
    best, th = summarize._best_threads(samp, "d", "crest")
    assert (best, th) == (16, [8, 16, 32])
    assert summarize._best_threads(samp, "d", "crest", allowed={8, 32})[0] == 8
    assert summarize._best_threads(samp, "d", "scanpy") == (None, [])
    assert summarize._edge(16, [8, 16, 32], 64) is None
    assert "lowest scanned" in summarize._edge(8, [8, 16, 32], 64)
    assert "more threads untested" in summarize._edge(32, [8, 16, 32], 64)
    assert summarize._edge(64, [8, 64], 64) == "highest scanned (64)"      # all logical cores scanned
    assert summarize._edge(1, [1, 8], 64) is None                          # 1 thread cannot be improved on


def test_main_thread_sets_follow_plan_and_clamp_to_cores(summarize, tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"args": {"main_threads": [8, 16, 32, 64], "ooc_threads": [8, 64]}}))
    sets = summarize.main_thread_sets(tmp_path, {}, ncpu=16)
    assert sets["crest"] == sets["scanpy"] == {8, 16}
    assert sets["crest-ooc"] == {8, 16}


# --------------------------------------------------------------------------- summarize: headline
def _runs(cells):
    """cells: {(tool, threads): [core seconds...]} for one dataset, core profile; memory 1 / 2 GB."""
    rows = []
    for (tool, th), ts in cells.items():
        for r, t in enumerate(ts):
            gb, base = (1.0, 0.4) if tool == "crest" else (2.0, 0.8)
            rows.append({"tool": tool, "dataset": "ds", "profile": "core", "threads": th, "repeat": r,
                         "status": "ok", "n_cells": 1000, "core_seconds": t, "core_peak_rss_gb": gb,
                         "core_peak_delta_gb": gb - base})
    return pl.DataFrame(rows)


HEADLINE_CELLS = {("crest", 8): [10.0] * 5, ("crest", 64): [6.0] * 5,
                  ("scanpy", 8): [30.0] * 5, ("scanpy", 64): [60.0] * 5}


def test_headline_is_the_minimum_over_matched_counts_and_best_vs_best(summarize):
    t = summarize.main_table(_runs(HEADLINE_CELLS), ncpu=64,
                             main_sets={"crest": {8, 64}, "scanpy": {8, 64}, "crest-ooc": set()})
    r = t.row(0, named=True)
    assert r["speedup_t8"] == pytest.approx(3.0, abs=EXACT)
    assert r["speedup_t64"] == pytest.approx(10.0, abs=EXACT)
    assert r["speedup_best"] == pytest.approx(5.0, abs=EXACT)             # crest 64 threads vs scanpy 8 threads
    assert r["speedup"] == pytest.approx(3.0, abs=EXACT)                  # not matched@64 only (10) nor best (5)
    assert r["speedup_basis"] == "matched @ 8"
    assert (r["crest_best_threads"], r["scanpy_best_threads"]) == (64, 8)
    assert r["single_thread_count"] is False
    assert r["memory_ratio"] == pytest.approx(2.0, abs=EXACT)
    assert r["memory_ratio_minus_baseline"] == pytest.approx(1.2 / 0.6, abs=EXACT)


def test_headline_flags_a_single_common_thread_count(summarize):
    cells = {("crest", 8): [10.0] * 3, ("scanpy", 8): [30.0] * 3}
    r = summarize.main_table(_runs(cells), ncpu=64).row(0, named=True)
    assert r["single_thread_count"] is True and r["speedup"] == pytest.approx(3.0, abs=EXACT)


def test_headline_ignores_thread_counts_outside_the_main_set(summarize):
    cells = {**HEADLINE_CELLS, ("crest", 1): [100.0] * 5, ("scanpy", 1): [90.0] * 5}   # thread-scan only
    t = summarize.main_table(_runs(cells), ncpu=64, main_sets={"crest": {8, 64}, "scanpy": {8, 64}, "crest-ooc": set()})
    r = t.row(0, named=True)
    assert r["speedup"] == pytest.approx(3.0, abs=EXACT) and "speedup_t1" not in t.columns
    assert r["crest_time_t1"] == pytest.approx(100.0, abs=EXACT)           # kept as information


def test_headline_reports_min_max_envelope_not_bootstrap_intervals(summarize):
    """Speed-up spread is the min-max envelope of repeats; there are no bootstrap interval columns."""
    cells = {("crest", 8): [9.0, 10.0, 11.0, 10.0, 10.0], ("scanpy", 8): [27.0, 30.0, 33.0, 30.0, 30.0]}
    t = summarize.main_table(_runs(cells), ncpu=64)
    r = t.row(0, named=True)
    assert r["speedup_min"] == pytest.approx(27.0 / 11.0, abs=EXACT) and r["speedup_max"] == pytest.approx(33.0 / 9.0, abs=EXACT)
    assert not [c for c in t.columns if re.search(r"(_lo|_hi|boot)", c)]
    assert r["n_crest"] == 5 and r["n_scanpy"] == 5


def test_best_thread_edge_flag_in_the_headline_table(summarize):
    cells = {("crest", 8): [10.0] * 3, ("crest", 16): [8.0] * 3, ("crest", 32): [6.0] * 3,
             ("scanpy", 8): [30.0] * 3, ("scanpy", 16): [40.0] * 3, ("scanpy", 32): [50.0] * 3}
    sets = {"crest": {8, 16, 32}, "scanpy": {8, 16, 32}, "crest-ooc": set()}
    r = summarize.main_table(_runs(cells), ncpu=64, main_sets=sets).row(0, named=True)
    assert "more threads untested" in r["crest_best_edge"]                 # best = 32, scan stopped below 64
    assert "lowest scanned" in r["scanpy_best_edge"]
    r_all = summarize.main_table(_runs(cells), ncpu=32, main_sets=sets).row(0, named=True)
    assert r_all["crest_best_edge"] == "highest scanned (32)"              # all logical cores were scanned


def test_not_faster_table_lists_steps_below_threshold(summarize):
    steps = pl.DataFrame({"dataset": ["d"] * 4, "step": ["a", "a", "b", "b"], "tool": ["crest", "scanpy"] * 2,
                          "median_s": [1.0, 1.1, 1.0, 5.0]})
    t = summarize.not_faster_table(steps)
    assert t["step"].to_list() == ["a"] and t["speedup"][0] == pytest.approx(1.1, abs=EXACT)


def test_thread_regression_rule_needs_both_ratio_and_absolute_increase(summarize):
    # the module has no such rule on the baseline build: the attribute lookup raises AttributeError
    f = summarize.is_thread_regression
    assert f(34.7, 177.3)                  # 5.1x and +142.6 s
    assert not f(1.0, 4.0)                 # 4x but only +3 s
    assert not f(100.0, 140.0)             # +40 s but only 1.4x
    assert f(10.0, 20.0)                   # both boundaries are inclusive
    assert not f(10.0, 19.99)


test_thread_regression_rule_needs_both_ratio_and_absolute_increase = pytest.mark.baseline_fails(
    ref="rc0", raises=AttributeError, match="is_thread_regression")(
    test_thread_regression_rule_needs_both_ratio_and_absolute_increase)


# --------------------------------------------------------------------------- memory baseline, thread scan
def _write_run(raw: Path, tag, tool, threads, core_seconds, peak, baseline):
    steps = {s: {"seconds": core_seconds / 10, "peak_rss_gb": peak, "parallelism": 1.0} for s in
             ["read", "qc_filter", "normalize_log1p", "hvg", "scale_pca", "neighbors", "leiden", "umap",
              "de_ttest", "de_wilcoxon"]}
    (raw / f"{tag}.json").write_text(json.dumps({
        "tool": tool, "dataset": "ds", "profile": "core", "threads": threads, "repeat": 0, "status": "ok",
        "n_cells": 100, "total_seconds": core_seconds, "peak_rss_gb": peak, "baseline_rss_gb": baseline,
        "peak_rss_minus_baseline_gb": peak - baseline, "steps": steps}))


def test_memory_is_reported_absolute_and_minus_the_starting_rss(summarize, tmp_path):
    _write_run(tmp_path, "a", "crest", 8, 10.0, 2.0, 0.5)
    _write_run(tmp_path, "b", "scanpy", 8, 30.0, 4.0, 0.75)
    df = summarize.load_runs(tmp_path)
    c = df.filter(pl.col("tool") == "crest").row(0, named=True)
    assert c["core_peak_rss_gb"] == pytest.approx(2.0) and c["core_peak_delta_gb"] == pytest.approx(1.5)
    r = summarize.main_table(df, ncpu=8).row(0, named=True)
    assert r["memory_ratio"] == pytest.approx(4.0 / 2.0)
    assert r["memory_ratio_minus_baseline"] == pytest.approx(3.25 / 1.5)


def test_release_memory_returns_the_method_used(run_one):
    assert run_one._release_memory() in (None, "malloc_trim", "malloc_zone_pressure_relief")


def test_monitor_baseline_is_the_rss_when_the_monitor_starts(run_one):
    psutil = pytest.importorskip("psutil")
    if not hasattr(psutil, "Process"):
        pytest.skip("psutil is not installed")
    from monitor import Monitor
    before = psutil.Process().memory_info().rss
    try:
        mon = Monitor()
    except PermissionError:                       # process listing denied (restricted sandbox)
        pytest.skip("psutil cannot list processes here")
    try:
        keep = np.ones(100_000_000 // 8)            # 100 MB allocated after the monitor started
        keep[:] = 1.0
        assert abs(mon.baseline_rss - before) < 50e6      # not the later, larger RSS
        assert psutil.Process().memory_info().rss - mon.baseline_rss > 50e6
    finally:
        mon.stop()


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"NCPU=64")
def test_driver_default_main_threads_are_the_full_grid_up_to_the_core_count():
    block = _shell_block(r"clip\(\) \{", r"\n  esac\nfi\n")
    for ncpu, want in ((64, "1 2 4 8 16 32 64"), (16, "1 2 4 8 16"), (8, "1 2 4 8"), (10, "1 2 4 8 10")):
        out = _bash(f'NCPU={ncpu}; TIER=standard; MAIN_THREADS=""; {block}\necho "$MAIN_THREADS"')
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == want, f"NCPU={ncpu}: {out.stdout.strip()!r} != {want!r}"


def test_run_plan_runs_every_main_thread_count_for_both_tools(run_all, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    a = Namespace(main_threads=[8, 16, 32, 64], ooc_threads=None, profile="core", datasets=["pbmc3k"],
                  tools=["crest", "scanpy"], scanpy_max_cells=0, no_pin=False, thread_dataset=None,
                  threads_scan=None, module_repeats=None, repeats=2, seed=0)
    jobs = run_all.plan(a)
    for tool in ("crest", "scanpy"):
        assert sorted({j["threads"] for j in jobs if j["tool"] == tool}) == [8, 16, 32, 64]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"pin")
def test_every_run_is_pinned_the_same_way_at_every_thread_count(run_all, monkeypatch):
    """One pinning policy for 8 and 64 threads, module runs included; --no-pin turns it off for all alike."""
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    a = Namespace(main_threads=[8, 64], ooc_threads=None, profile="full", datasets=["pbmc3k"],
                  tools=["crest", "scanpy"], scanpy_max_cells=0, no_pin=False, thread_dataset=None,
                  threads_scan=None, module_repeats=None, repeats=1, seed=0)
    jobs = run_all.plan(a)
    assert {j["profile"] for j in jobs} == {"core", "full"}
    assert all(j["pin"] is True for j in jobs), "runs pinned differently: " + str(
        [(j["threads"], j["profile"], j["pin"]) for j in jobs if j["pin"] is not True])
    a.no_pin = True
    assert all(j["pin"] is False for j in run_all.plan(a))


# --------------------------------------------------------------------------- accuracy helpers
def test_jaccard_rows_ignores_padding(accuracy):
    A = np.array([[1, 2, 3, -1], [4, 5, -1, -1]])
    B = np.array([[3, 2, 9, -1], [4, 5, 6, -1]])
    assert accuracy._jaccard_rows(A, B) == pytest.approx((2 / 4 + 2 / 3) / 2)
    assert accuracy._sets_jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_jaccard_common_counts_neighbours_outside_the_common_cells(accuracy):
    A = np.array([[1, 2, -2]])          # -2: neighbour outside the common cells, counted, never shared
    B = np.array([[1, 2, 3]])
    assert accuracy._jaccard_common(A, B) == pytest.approx(2 / 4)
    assert accuracy._jaccard_common(B, B) == pytest.approx(1.0)


def test_remap_pair_maps_to_the_common_order(accuracy):
    ia, ib = np.array([0, 2]), np.array([1, 0])        # common cells: own indices of each tool
    ka = np.array([[2, 1], [0, -1], [0, 1]])           # tool A: cell 1 is not common
    kb = np.array([[1, 2], [0, 2]])
    ra, rb = accuracy._remap_pair(ka, kb, ia, ib)
    assert ra.shape[0] == rb.shape[0] == 2
    assert ra[0].tolist() == [1, -2] and ra[1].tolist() == [0, -2]


def _points_and_exact_knn(n=N_FAKE_CELLS, d=10, k=K_NEIGHBORS - 1, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, d))
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return X, np.argsort(D, axis=1)[:, :k], D


def test_knn_recall_of_the_exact_graph_is_one_and_of_a_degraded_graph_is_lower(accuracy):
    X, K, _ = _points_and_exact_knn()
    r = accuracy.knn_recall(X, K.astype(np.int32), n_query=N_FAKE_CELLS)
    assert r["recall"] == pytest.approx(1.0, abs=EXACT) and r["k"] == K_NEIGHBORS - 1
    worse = K.copy()
    worse[:, -1] = (worse[:, -1] + 1) % N_FAKE_CELLS      # replace the farthest neighbour by an arbitrary cell
    rw = accuracy.knn_recall(X, np.where(worse == np.arange(N_FAKE_CELLS)[:, None], K, worse), n_query=N_FAKE_CELLS)
    assert rw["recall"] < 1.0 - 0.5 / (K_NEIGHBORS - 1)


def _fake_scanpy_distances(X, D, n_neighbors=K_NEIGHBORS, with_self=True):
    """obsp['distances'] as scanpy writes it: the cell itself (explicit 0) plus n_neighbors - 1 neighbours
    when ``with_self`` (n_neighbors entries per row), otherwise n_neighbors - 1 entries."""
    import scipy.sparse as sp
    n = len(X)
    kk = n_neighbors - 1
    nn = np.argsort(D, axis=1)[:, :kk]
    rows, cols, vals = [], [], []
    for i in range(n):
        idx = list(nn[i]) + ([i] if with_self else [])
        rows += [i] * len(idx)
        cols += idx
        vals += [float(np.sqrt(D[i, j])) for j in nn[i]] + ([0.0] if with_self else [])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


@pytest.fixture
def fake_scanpy_run(run_one):
    X, K, D = _points_and_exact_knn()
    ad = types.SimpleNamespace(obsp={"distances": _fake_scanpy_distances(X, D)},
                               uns={"neighbors": {"params": {"n_neighbors": K_NEIGHBORS}}})
    run_one.LAST.clear()
    run_one.LAST.update(tool="scanpy", obj=ad)
    yield X, K
    run_one.LAST.clear()


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"shape")
def test_scanpy_knn_indices_are_n_minus_1_wide_and_exclude_self(run_one, fake_scanpy_run):
    X, K = fake_scanpy_run
    idx = run_one.knn_indices()
    assert idx.shape == (N_FAKE_CELLS, K_NEIGHBORS - 1), f"shape {idx.shape}"
    assert not (idx == np.arange(N_FAKE_CELLS)[:, None]).any()
    assert (idx == K).all()                                   # nearest first, same neighbours as brute force


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"recall")
def test_scanpy_recall_of_an_exact_graph_is_not_capped_by_the_self_column(run_one, accuracy, fake_scanpy_run):
    X, _ = fake_scanpy_run
    recall = accuracy.knn_recall(X, run_one.knn_indices(), n_query=N_FAKE_CELLS)["recall"]
    assert recall >= RECALL_EXACT, f"recall {recall:.4f} (a self column caps it at {(K_NEIGHBORS - 1) / K_NEIGHBORS:.4f})"


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"jaccard")
def test_knn_graph_jaccard_of_equal_graphs_is_one_across_tools(run_one, accuracy, fake_scanpy_run):
    X, K = fake_scanpy_run
    scanpy_idx = run_one.knn_indices()
    ident = np.arange(N_FAKE_CELLS)
    j = accuracy._jaccard_common(*accuracy._remap_pair(K.astype(np.int32), scanpy_idx, ident, ident))
    assert j == pytest.approx(1.0, abs=EXACT), f"jaccard {j:.4f}"


def test_scanpy_graph_without_a_self_column_gives_the_same_indices(run_one):
    """The brute-force path (small n) stores n_neighbors - 1 entries without self: unchanged result."""
    X, K, D = _points_and_exact_knn()
    ad = types.SimpleNamespace(obsp={"distances": _fake_scanpy_distances(X, D, with_self=False)},
                               uns={"neighbors": {"params": {"n_neighbors": K_NEIGHBORS}}})
    run_one.LAST.clear()
    run_one.LAST.update(tool="scanpy", obj=ad)
    try:
        assert (run_one.knn_indices() == K).all()
    finally:
        run_one.LAST.clear()


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="assert")
def test_scanpy_rows_with_fewer_neighbours_are_padded(run_one):
    import scipy.sparse as sp
    # row 0: self + 1 neighbour; row 1: self + 2 neighbours, listed farthest first; row 2: empty
    ad = types.SimpleNamespace(obsp={"distances": sp.csr_matrix((np.array([0.0, 2.0, 5.0, 0.0, 1.0]),
                                                                np.array([0, 1, 0, 1, 2]), np.array([0, 2, 5, 5])),
                                                               shape=(3, 3))},
                               uns={"neighbors": {"params": {"n_neighbors": 3}}})
    run_one.LAST.clear()
    run_one.LAST.update(tool="scanpy", obj=ad)
    try:
        idx = run_one.knn_indices()
    finally:
        run_one.LAST.clear()
    assert idx.tolist() == [[1, -1], [2, 0], [-1, -1]]


def test_knn_recall_rejects_a_graph_that_contains_the_cell_itself(accuracy):
    X, K, _ = _points_and_exact_knn()
    with_self = np.concatenate([np.arange(N_FAKE_CELLS)[:, None], K], axis=1).astype(np.int32)
    try:
        out = accuracy.knn_recall(X, with_self, n_query=N_FAKE_CELLS)
    except ValueError as e:
        assert "own neighbour" in str(e)
        return
    pytest.fail(f"a self column was accepted (recall {out['recall']:.4f})")


test_knn_recall_rejects_a_graph_that_contains_the_cell_itself = pytest.mark.baseline_fails(
    ref="rc0", raises=pytest.fail.Exception, match="self column was accepted")(
    test_knn_recall_rejects_a_graph_that_contains_the_cell_itself)


def test_knn_recall_checks_the_expected_width(accuracy):
    X, K, _ = _points_and_exact_knn()
    assert accuracy.knn_recall(X, K, n_query=50)["k"] == K_NEIGHBORS - 1
    assert accuracy.knn_recall(X, K, n_query=50, k_expected=K_NEIGHBORS - 1)["k"] == K_NEIGHBORS - 1
    with pytest.raises(ValueError, match="expected"):
        accuracy.knn_recall(X, K, n_query=50, k_expected=K_NEIGHBORS)


# the argument does not exist on the baseline build: the call raises TypeError (unexpected keyword)
test_knn_recall_checks_the_expected_width = pytest.mark.baseline_fails(
    ref="rc0", raises=TypeError, match="k_expected")(test_knn_recall_checks_the_expected_width)


def test_compare_smoke_on_two_identical_outputs(accuracy, tmp_path):
    pytest.importorskip("sklearn")
    X, K, _ = _points_and_exact_knn(d=12)
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(3), N_FAKE_CELLS // 3)

    def out(extra=None):
        d = {"barcodes": np.array([f"c{i}" for i in range(N_FAKE_CELLS)]), "leiden": labels, "pca": X.astype(np.float32),
             "umap": X[:, :2].astype(np.float32), "hvg": np.array(["g1", "g2", "g3"]),
             "de_groups": np.array(["0", "1", "2"]), "de_top": np.array([[f"g{j}" for j in range(50)]] * 3),
             "knn_indices": K.astype(np.int32)}
        return d

    np.savez(tmp_path / "a.npz", **out())
    np.savez(tmp_path / "b.npz", **out())
    m = accuracy.compare(tmp_path / "a.npz", tmp_path / "b.npz")
    assert m["n_common_cells"] == N_FAKE_CELLS and m["leiden_ari"] == pytest.approx(1.0)
    assert m["knn_graph_jaccard"] == pytest.approx(1.0) and m["knn_graph_k"] == K_NEIGHBORS - 1
    assert m["knn_recall_crest"] >= RECALL_EXACT and m["knn_recall_scanpy"] >= RECALL_EXACT
    assert m["hvg_jaccard"] == pytest.approx(1.0) and m["de_top50_overlap"] == pytest.approx(1.0)
    assert m["pca_subspace_cos_min"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- kang_pseudobulk: R runs
def _counts_with_trailing_zeros(n_samples=16, n_genes=N_GENES_BP, n_zero=600, seed=0):
    c = np.random.default_rng(seed).poisson(20.0, size=(n_samples, n_genes)).astype(np.int64)
    c[:, n_genes - n_zero:] = 0
    return c


def _obs(n):
    return pl.DataFrame({"pseudobulk": [f"s{i}" for i in range(n)], "ind": [str(i % 8) for i in range(n)],
                         "stim": ["ctrl"] * (n // 2) + ["stim"] * (n - n // 2)})


class _FakeRscript:
    """Stands in for ``subprocess.run(["Rscript", ...])``: reads the counts R would read, records the
    gene order it was handed, and writes r.csv / diag.json (or fails with R's error text)."""

    def __init__(self, fail_workers=(), message="Error in solve: no right-hand side in b"):
        self.fail_workers, self.message, self.calls = set(fail_workers), message, []

    def __call__(self, cmd, check=False, capture_output=False, text=False, env=None, **kw):
        d, workers = Path(cmd[2]), int(cmd[3])
        counts = pl.read_csv(d / "counts.csv")
        self.calls.append({"workers": workers, "genes": counts["gene"].to_list(),
                           "mat": counts.drop("gene").to_numpy()})
        if workers in self.fail_workers:
            if check:
                raise subprocess.CalledProcessError(1, cmd, stderr=self.message)
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=self.message)
        mat = counts.drop("gene").to_numpy()
        base = mat.mean(axis=1)
        pl.DataFrame({"log2FoldChange": base, "pvalue": 1.0 / (1.0 + base), "padj": 1.0 / (2.0 + base),
                      "baseMean": base}).write_csv(d / "r.csv")
        (d / "diag.json").write_text(json.dumps({"elapsed": 1.0 + workers, "workers": workers}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="no chunk")
def test_every_biocparallel_chunk_holds_expressed_genes_at_64_workers(kp, monkeypatch):
    """All-zero genes stored at the end of the matrix must not fill whole chunks of rows."""
    fake = _FakeRscript()
    monkeypatch.setattr(kp.subprocess, "run", fake)
    counts = _counts_with_trailing_zeros()
    genes = [f"g{i}" for i in range(counts.shape[1])]
    kp.run_r(counts, _obs(counts.shape[0]), genes, workers=N_WORKERS_BP)
    mat = fake.calls[0]["mat"]
    expressed = mat.sum(axis=1) > 0
    chunks = _bp_chunks(len(expressed), N_WORKERS_BP)
    assert all(expressed[c].any() for c in chunks), "no chunk may consist only of all-zero genes"
    shares = [expressed[c].mean() for c in chunks]
    assert min(shares) > 0.5 * expressed.mean()


def _bp_chunks(n, workers):
    """Consecutive row blocks of DESeq2's parallelisation: ``sort(rep(seq_len(workers), length = n))``."""
    base, extra = divmod(n, workers)
    out, lo = [], 0
    for i in range(workers):
        hi = lo + base + (1 if i < extra else 0)
        out.append(slice(lo, hi))
        lo = hi
    return out


@pytest.mark.parametrize("workers", [1, 8, N_WORKERS_BP])
def test_run_r_returns_results_in_the_input_gene_order_for_every_worker_count(kp, monkeypatch, workers):
    monkeypatch.setattr(kp.subprocess, "run", _FakeRscript())
    counts = _counts_with_trailing_zeros(seed=3)
    genes = [f"g{i}" for i in range(counts.shape[1])]
    res, diag = kp.run_r(counts, _obs(counts.shape[0]), genes, workers=workers)
    assert res.height == len(genes) and diag["workers"] == workers
    np.testing.assert_allclose(res["baseMean"].to_numpy(), counts.mean(axis=0), rtol=1e-12)


def test_run_r_does_not_change_the_gene_set_or_values_seen_by_r(kp, monkeypatch):
    """The reordering drops nothing and changes no count: the multiset of genes R sees is the input's."""
    fake = _FakeRscript()
    monkeypatch.setattr(kp.subprocess, "run", fake)
    counts = _counts_with_trailing_zeros(seed=4)
    genes = [f"g{i}" for i in range(counts.shape[1])]
    kp.run_r(counts, _obs(counts.shape[0]), genes, workers=N_WORKERS_BP)
    seen = fake.calls[0]
    assert sorted(seen["genes"]) == sorted(genes)
    by_gene = {g: seen["mat"][i] for i, g in enumerate(seen["genes"])}
    for j in (0, 17, counts.shape[1] - 1):
        np.testing.assert_array_equal(by_gene[genes[j]], counts[:, j])
    np.testing.assert_allclose(np.sort(seen["mat"].mean(axis=1)), np.sort(counts.mean(axis=0)))   # same baseMean quantiles


def test_gene_order_is_the_identity_for_one_worker_and_for_nothing_to_spread(kp):
    counts = _counts_with_trailing_zeros()
    assert kp.gene_order_for_workers(counts, 1).tolist() == list(range(counts.shape[1]))
    expressed_only = np.ones((4, 10), int)
    assert kp.gene_order_for_workers(expressed_only, 8).tolist() == list(range(10))


test_gene_order_is_the_identity_for_one_worker_and_for_nothing_to_spread = pytest.mark.baseline_fails(
    ref="rc0", raises=AttributeError, match="gene_order_for_workers")(
    test_gene_order_is_the_identity_for_one_worker_and_for_nothing_to_spread)


def test_gene_order_is_a_permutation_that_spreads_zero_genes(kp):
    counts = _counts_with_trailing_zeros(n_zero=1000)
    order = kp.gene_order_for_workers(counts, N_WORKERS_BP)
    assert sorted(order.tolist()) == list(range(counts.shape[1]))
    zero = (counts.sum(axis=0) == 0)[order]
    per_chunk = [zero[c].sum() for c in _bp_chunks(len(zero), N_WORKERS_BP)]
    assert max(per_chunk) - min(per_chunk) <= 2                     # evenly spread: 1000 / 64 ~ 15.6 per chunk
    assert kp.worker_chunks(10, 3) == [range(0, 4), range(4, 7), range(7, 10)]


test_gene_order_is_a_permutation_that_spreads_zero_genes = pytest.mark.baseline_fails(
    ref="rc0", raises=AttributeError, match="gene_order_for_workers")(test_gene_order_is_a_permutation_that_spreads_zero_genes)


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="Rscript stderr")
def test_a_failed_rscript_reports_its_stderr(kp, monkeypatch):
    monkeypatch.setattr(kp.subprocess, "run", _FakeRscript(fail_workers={8}, message="no right-hand side in b"))
    counts = _counts_with_trailing_zeros(n_genes=50, n_zero=5)
    with pytest.raises(Exception) as e:
        kp.run_r(counts, _obs(counts.shape[0]), [f"g{i}" for i in range(50)], workers=8)
    assert "no right-hand side in b" in str(e.value), "Rscript stderr is missing from the error"


def _r_with_deseq2() -> bool:
    import shutil
    if not shutil.which("Rscript"):
        return False
    return subprocess.run(["Rscript", "-e", "suppressMessages(library(DESeq2))"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _r_with_deseq2(), reason="Rscript with DESeq2 is not installed")
def test_interleaving_all_zero_genes_leaves_r_results_unchanged(kp, monkeypatch):
    """R DESeq2 on the interleaved gene order (serial run) gives the same padj for every gene as on the
    input order, so the significant sets are unchanged (dropping the all-zero genes would not). Skipped
    where R is missing; it tests a new function, so it has no baseline marker."""
    counts = _counts_with_trailing_zeros(n_genes=1500, n_zero=400, seed=5)
    counts[8:, :300] = counts[8:, :300] * 3                       # some differential genes
    genes = [f"g{i}" for i in range(counts.shape[1])]
    obs = _obs(counts.shape[0])
    plain, _ = kp.run_r(counts, obs, genes, workers=1)
    spread = kp.gene_order_for_workers
    monkeypatch.setattr(kp, "gene_order_for_workers", lambda c, w: spread(c, N_WORKERS_BP))
    reordered, _ = kp.run_r(counts, obs, genes, workers=1)
    assert not (kp.gene_order_for_workers(counts, 1) == np.arange(counts.shape[1])).all()   # really reordered
    assert kp.padj_ndiff(plain, reordered) == 0 and kp.sig_ndiff(plain, reordered) == 0
    assert (plain["padj"].fill_null(1) < 0.1).sum() > 0


R_CHUNK_SCRIPT = r"""
suppressMessages(library(DESeq2))
d <- commandArgs(TRUE)[1]; workers <- as.integer(commandArgs(TRUE)[2])
cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
ord <- read.csv(file.path(d, "order.csv"))$order + 1L
cd <- data.frame(ind = factor(rep(1:8, 2)), stim = factor(rep(c("ctrl", "stim"), each = 8), levels = c("ctrl", "stim")),
                 row.names = colnames(cts))
chunk_status <- function(m) {
  dds <- estimateSizeFactors(DESeqDataSetFromMatrix(m, cd, ~ ind + stim))
  mm <- model.matrix(design(dds), colData(dds))
  idx <- factor(sort(rep(seq_len(workers), length.out = nrow(dds))))   # as in DESeq2's DESeqParallel
  sapply(levels(idx), function(l) inherits(try(
    estimateDispersionsGeneEst(dds[idx == l, ], quiet = TRUE, modelMatrix = mm), silent = TRUE), "try-error"))
}
cat("INPUT_ORDER_FAILED_CHUNKS", sum(suppressMessages(chunk_status(cts))), "\n")
cat("INTERLEAVED_FAILED_CHUNKS", sum(suppressMessages(chunk_status(cts[ord, ]))), "\n")
"""


@pytest.mark.skipif(not _r_with_deseq2(), reason="Rscript with DESeq2 is not installed")
def test_r_fails_on_an_all_zero_chunk_and_not_on_interleaved_chunks(kp, tmp_path):
    """The failure that stopped the 64-worker run, at chunk level and without BiocParallel: DESeq2 hands each
    worker a consecutive block of rows, and a block holding only all-zero genes cannot be fitted."""
    workers = 8
    counts = _counts_with_trailing_zeros(n_genes=800, n_zero=200, seed=6)
    order = kp.gene_order_for_workers(counts, workers)
    cols = [f"s{i}" for i in range(counts.shape[0])]
    pl.DataFrame({"gene": [f"g{i}" for i in range(counts.shape[1])]}).hstack(pl.DataFrame(counts.T, schema=cols)).write_csv(
        tmp_path / "counts.csv")
    pl.DataFrame({"order": order}).write_csv(tmp_path / "order.csv")
    (tmp_path / "chunk.R").write_text(R_CHUNK_SCRIPT)
    out = subprocess.run(["Rscript", str(tmp_path / "chunk.R"), str(tmp_path), str(workers)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    got = dict(re.findall(r"(INPUT_ORDER_FAILED_CHUNKS|INTERLEAVED_FAILED_CHUNKS) (\d+)", out.stdout))
    assert int(got["INPUT_ORDER_FAILED_CHUNKS"]) >= 1          # the last blocks are all-zero genes
    assert int(got["INTERLEAVED_FAILED_CHUNKS"]) == 0


def test_r_script_records_the_r_and_bioconductor_versions(kp):
    for needle in ("R.version.string", 'packageVersion("DESeq2")', 'packageVersion("BiocParallel")'):
        assert needle in kp.R_SCRIPT, needle


test_r_script_records_the_r_and_bioconductor_versions = pytest.mark.baseline_fails(
    ref="rc0", raises=AssertionError, match="R.version.string")(test_r_script_records_the_r_and_bioconductor_versions)


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="matched @ 8")
def test_speedups_use_a_thread_count_only_if_every_cell_type_has_a_time(kp):
    df = pl.DataFrame({"time_CREST_t1": [2.0, 2.0], "time_R_t1": [10.0, 10.0],
                       "time_CREST_t8": [1.0, 1.0], "time_R_t8": [4.0, None]}, strict=False)
    rows = kp._speedups(df, [1, 8], "R")
    bases = [r["basis"] for r in rows]
    assert "matched @ 1" in bases and "matched @ 8" not in bases, bases     # a failed run is not summed over
    assert next(r for r in rows if r["basis"] == "matched @ 1")["speedup"] == pytest.approx(5.0)


@pytest.mark.baseline_fails(ref="rc0", raises=AttributeError, match="padj_ndiff")
def test_padj_and_significance_difference_counts(kp):
    a = pl.DataFrame({"padj": [0.01, 0.2, None, 0.05]})
    b = pl.DataFrame({"padj": [0.01, 0.2, None, 0.06]})
    c = pl.DataFrame({"padj": [0.01, 0.05, 0.3, 0.05]})
    assert kp.padj_ndiff(a, a) == 0 and kp.padj_ndiff(a, b) == 1
    assert kp.padj_ndiff(a, c) == 2                                  # 0.2 -> 0.05, missing -> 0.3
    assert kp.sig_ndiff(a, b) == 0 and kp.sig_ndiff(a, c) == 1       # only 0.2 -> 0.05 crosses alpha = 0.1


# --------------------------------------------------------------------------- kang_pseudobulk: the k loop
@pytest.fixture
def kang_run(kp, monkeypatch, tmp_path):
    """kp.main() on a small synthetic BioFrame with a mocked R, no pydeseq2 and fixed CREST timings."""
    import crest
    rng = np.random.default_rng(0)
    n_genes, per = 150, 12
    obs = [(f"{ct}_{d}_{s}_{i}", str(d), s, ct) for ct in ("A", "B") for d in range(4) for s in ("ctrl", "stim")
           for i in range(per)]
    mu = rng.gamma(2.0, 5.0, n_genes)
    X = rng.poisson(mu, size=(len(obs), n_genes)).astype(np.float32)
    import scipy.sparse as sp
    bf = crest.BioFrame.from_scipy(sp.csr_matrix(X), obs=pl.DataFrame({
        "barcode": [o[0] for o in obs], "ind": [o[1] for o in obs], "stim": [o[2] for o in obs],
        "cell": [o[3] for o in obs], "multiplets": ["singlet"] * len(obs)}),
        var=pl.DataFrame({"gene_ids": [f"g{i}" for i in range(n_genes)], "gene_name": [f"n{i}" for i in range(n_genes)]}))
    monkeypatch.setattr(kp, "fetch", lambda d: None)
    monkeypatch.setattr(kp, "load", lambda d, singlets_only=True: bf)
    monkeypatch.setattr(kp, "crest_times", lambda subs, threads, repeats: {ct: [0.01] for ct, _ in subs})

    def run(fake_r, threads=("1", "2", "4")):
        monkeypatch.setattr(kp, "run_r", fake_r)
        monkeypatch.setattr(sys, "argv", ["kang_pseudobulk.py", "--data", str(tmp_path / "d"), "--out", str(tmp_path / "out"),
                                          "--skip-pydeseq2", "--repeats", "1", "--threads", *threads])
        code = kp.main()
        return code, pl.read_csv(tmp_path / "out" / "kang_pseudobulk.csv", null_values=[""]), tmp_path / "out"
    return run


def _r_table(n, rng_seed=0, bump=None):
    rng = np.random.default_rng(rng_seed)
    p = rng.uniform(size=n)
    padj = np.minimum(1.0, p * 3)
    if bump is not None:
        padj[bump] = 0.001
    return pl.DataFrame({"log2FoldChange": rng.normal(size=n), "pvalue": p, "padj": padj, "baseMean": rng.gamma(2, 5, n),
                         "dispersion": np.full(n, 0.1), "dispGeneEst": np.full(n, 0.1), "dispGeneIter": np.full(n, 5.0),
                         "dispFit": np.full(n, 0.1), "dispMAP": np.full(n, 0.1), "dispIter": np.full(n, 5.0),
                         "dispOutlier": np.zeros(n, bool), "betaConv": np.ones(n, bool), "maxCooks": np.zeros(n),
                         "replace": np.zeros(n, bool)})


def _fake_r(fail_at=(), differ_at=()):
    diag = {"fitType": "parametric", "n_beta_nonconverged": 0, "n_grid_gene": 0, "n_grid_map": 0, "n_disp_outlier": 0,
            "n_replaced": 0, "deseq2": "1.50.2", "biocparallel": "1.44.0", "r_version": "R version 4.5.2"}

    def run_r(counts, obs, genes, workers=1):
        if workers in fail_at:
            raise subprocess.CalledProcessError(1, ["Rscript"], stderr="no right-hand side in b")
        return _r_table(len(genes), bump=0 if workers in differ_at else None), {**diag, "elapsed": 10.0 / workers}
    return run_r


def _max_k(kp_threads=(1, 2, 4)):
    return max(min(t, os.cpu_count() or 1) for t in kp_threads)


@pytest.mark.baseline_fails(ref="rc0", raises=subprocess.CalledProcessError, match="")
def test_a_failed_r_run_at_the_highest_k_does_not_lose_the_other_k(kang_run):
    if _max_k() < 2:
        pytest.skip("needs at least two logical cores")
    k = _max_k()
    code, df, out = kang_run(_fake_r(fail_at={k}))
    assert code == 2                                              # failure is reported, after writing the results
    assert df[f"time_R_t{k}"].null_count() == df.height
    assert df["time_R_t1"].null_count() == 0 and df["sig_R"].null_count() == 0
    assert "failed run" in (out / "kang_pseudobulk.md").read_text()


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="sig_R_t")
def test_r_results_are_kept_for_every_k_and_compared_with_the_first(kang_run):
    if _max_k() < 2:
        pytest.skip("needs at least two logical cores")
    k = _max_k()
    code, df, _ = kang_run(_fake_r(differ_at={k}))
    assert f"sig_R_t{k}" in df.columns, f"no sig_R_t{k} column: R results are kept for the first k only"
    assert df["R_padj_ndiff_t1"].sum() == 0
    assert (df[f"R_padj_ndiff_t{k}"] == 1).all() and (df[f"sig_R_diff_t{k}"] == 1).all()
    assert code == 0


def test_the_run_is_clean_and_records_r_versions_when_every_k_succeeds(kang_run):
    code, df, out = kang_run(_fake_r())
    md = (out / "kang_pseudobulk.md").read_text()
    assert "DESeq2 1.50.2" in md and "BiocParallel 1.44.0" in md
    assert df.height == 2 and df["time_R_t1"].null_count() == 0
    assert code == 0


test_the_run_is_clean_and_records_r_versions_when_every_k_succeeds = pytest.mark.baseline_fails(
    ref="rc0", raises=AssertionError, match="DESeq2 1.50.2")(test_the_run_is_clean_and_records_r_versions_when_every_k_succeeds)


# --------------------------------------------------------------------------- driver script
def _run_mod_harness(tmp_path, calls):
    block = _shell_block(r"run_mod\(\) \{", r"\n  \}\n")
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    code = f'set -euo pipefail; OUT="{out}"; {block}\n{calls}\necho reached-end'
    return _bash(code), out


def test_run_mod_keeps_going_after_a_failing_module(tmp_path):
    r, out = _run_mod_harness(tmp_path, "run_mod bad false\nrun_mod good true\n")
    assert r.returncode == 0 and "reached-end" in r.stdout          # later modules and statistics still run
    assert (out / "logs" / "module_good.log").exists()


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="module_failures")
def test_run_mod_records_failures_and_the_script_ends_non_zero(tmp_path):
    r, out = _run_mod_harness(tmp_path, "run_mod bad sh -c 'exit 3'\nrun_mod good true\n")
    failures = out / "module_failures.txt"
    assert failures.exists(), "module_failures.txt was not written"
    assert failures.read_text().strip() == "bad (exit status 3)"
    closing = _shell_block(r"# -+ module failures", r"\nfi\n")
    end = _bash(f'OUT="{out}"; {closing}')
    assert end.returncode == 1 and "bad" in end.stderr
    ok = tmp_path / "ok"
    ok.mkdir()
    assert _bash(f'OUT="{ok}"; {closing}').returncode == 0


def test_driver_scans_r_workers_from_one_to_the_core_count():
    m = re.search(r'DE_THREADS="\$\(clip ((?:\d+ )+)', SCRIPT)
    assert m, "DE_THREADS definition not found"
    assert m.group(1).split() == ["1", "2", "4", "8", "16", "32", "64"], m.group(1)


test_driver_scans_r_workers_from_one_to_the_core_count = pytest.mark.baseline_fails(
    ref="rc0", raises=AssertionError, match="")(test_driver_scans_r_workers_from_one_to_the_core_count)


def _fake_rscript(tmp_path, line):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    f = d / "Rscript"
    f.write_text(f"#!/bin/sh\nprintf '%s' '{line}'\n")
    f.chmod(0o755)
    return d


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="record_r_versions")
def test_environment_record_contains_r_bioconductor_and_deseq2_versions(tmp_path):
    func = _shell_block(r"record_r_versions\(\) \{", r"\n\}\n")
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    bindir = _fake_rscript(tmp_path, "R 4.5.2, DESeq2 1.50.2, BiocParallel 1.44.0")
    r = _bash(f'{func}\nrecord_r_versions "{env_dir}"', PATH=f"{bindir}:{os.environ['PATH']}")
    assert r.returncode == 0, r.stderr
    assert (env_dir / "R_versions.txt").read_text().strip() == "R 4.5.2, DESeq2 1.50.2, BiocParallel 1.44.0"
    assert json.loads((env_dir / "R_versions.json").read_text()) == {"R": "4.5.2", "DESeq2": "1.50.2", "BiocParallel": "1.44.0"}


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="record_r_versions")
def test_environment_record_without_r_says_so(tmp_path):
    func = _shell_block(r"record_r_versions\(\) \{", r"\n\}\n")
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    empty = tmp_path / "nobin"
    empty.mkdir()
    # a failing Rscript first on PATH hides an installed R, so the test means the same on hosts with R
    stub = empty / "Rscript"
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    r = _bash(f'{func}\nrecord_r_versions "{env_dir}"', PATH=f"{empty}:/usr/bin:/bin")
    assert r.returncode == 0, r.stderr
    assert json.loads((env_dir / "R_versions.json").read_text())["DESeq2"] is None
    assert "not available" in (env_dir / "R_versions.txt").read_text()


def test_driver_script_passes_bash_syntax_check():
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "crest_paper_bench.sh")]).returncode == 0


# =========================================================================== protocol v2
# Tolerances and constants of the protocol tests (named, never inline).
FOREIGN_LOAD_BELOW = 0.5          # median foreign busy cores of a quiet run in the fake samples
FOREIGN_LOAD_ABOVE = 2.0          # ... of a violating one
THREAD_GRID_DEFAULT = [1, 2, 4, 8, 16, 32, 64]
TUNING_WIN = 1.5                  # a variant this many times faster than the default must be chosen
TUNING_NOISE = 1.02               # a variant this much faster than the default must not be chosen (margin 5%)
STEP_RATIO_CEILING = 1000.0       # no per-stage ratio above this is ever printed
LAZY_STEP_SECONDS = 1.5e-5        # CREST's recorded normalize_log1p step: applied lazily, takes ~0 s
RTOL = 1e-9
CORE_STEPS = ["read", "qc_filter", "normalize_log1p", "hvg", "scale_pca", "neighbors", "leiden", "umap",
              "de_ttest", "de_wilcoxon"]
FORBIDDEN_PUBLIC = re.compile(r"\b(?:F|PY|WP|G|D|B|SB|BL)-\d+[ab]?\b|\bB\d{1,2}[ab]?\b")   # internal item identifiers (shape only)


def _need(name: str):
    """Import a bench module; a missing one is a failed assertion (not an import error) in the test."""
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise AssertionError(f"bench/paper/{name}.py is missing: {e}") from None


def _has(mod, attr: str):
    assert hasattr(mod, attr), f"{mod.__name__}.{attr} is missing"
    return getattr(mod, attr)


# --------------------------------------------------------------------------- thread grid and pinning
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"thread counts")
def test_thread_grid_default_is_one_to_sixty_four(run_all, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    a = Namespace(main_threads=[], ooc_threads=[], profile="core", datasets=["pbmc3k"], tools=["crest", "scanpy"],
                  scanpy_max_cells=0, no_pin=False, thread_dataset=None, threads_scan=None, module_repeats=None,
                  repeats=1, seed=0)
    for tool in ("crest", "scanpy"):
        got = sorted({j["threads"] for j in run_all.plan(a) if j["tool"] == tool})
        assert got == THREAD_GRID_DEFAULT, f"{tool} thread counts {got}"


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"thread counts")
def test_thread_grid_is_configurable_and_clamped_to_the_cpus(run_all, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    a = Namespace(main_threads=[], ooc_threads=[], thread_grid=[1, 3, 64], profile="core", datasets=["pbmc3k"],
                  tools=["crest"], scanpy_max_cells=0, no_pin=False, thread_dataset=None, threads_scan=None,
                  module_repeats=None, repeats=1, seed=0)
    got = sorted({j["threads"] for j in run_all.plan(a)})
    assert got == [1, 3, 10], f"thread counts {got}"


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_clamp_grid_is_sorted_distinct_and_at_least_one():
    protocol = _need("protocol")
    assert protocol.clamp_grid([64, 8, 16, 8, 0], 12) == [1, 8, 12]
    assert protocol.clamp_grid(protocol.THREAD_GRID, 64) == THREAD_GRID_DEFAULT


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_cpu_list_syntax():
    protocol = _need("protocol")
    assert protocol.parse_cpu_list("0,32") == [0, 32]
    assert protocol.parse_cpu_list("0-3\n") == [0, 1, 2, 3]
    assert protocol.parse_cpu_list("0-1,8-9") == [0, 1, 8, 9]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_pin_order_takes_one_context_per_core_before_any_sibling():
    protocol = _need("protocol")
    smt_adjacent = {c: (c - c % 2, c - c % 2 + 1) for c in range(8)}       # siblings 0,1 / 2,3 / ...
    assert protocol.pin_cpus(4, range(8), smt_adjacent) == [0, 2, 4, 6]
    assert protocol.pin_cpus(6, range(8), smt_adjacent) == [0, 2, 4, 6, 1, 3]
    smt_far = {c: (c % 4, c % 4 + 4) for c in range(8)}                    # siblings i, i + 4 (the Threadripper layout)
    assert protocol.pin_cpus(4, range(8), smt_far) == [0, 1, 2, 3]
    assert protocol.pin_cpus(8, range(8), smt_far) == list(range(8))
    assert protocol.pin_cpus(3, [5, 2, 9], None) == [2, 5, 9]              # no topology: numerical order
    assert protocol.pin_cpus(99, range(4), smt_far) == [0, 1, 2, 3]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_smt_siblings_are_read_from_a_sysfs_tree(tmp_path):
    protocol = _need("protocol")
    for cpu, sib in ((0, "0,2"), (1, "1,3"), (2, "0,2"), (3, "1,3")):
        d = tmp_path / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(sib + "\n")
    (tmp_path / "cpufreq").mkdir()
    assert protocol.read_smt_siblings(str(tmp_path)) == {0: (0, 2), 1: (1, 3), 2: (0, 2), 3: (1, 3)}
    assert protocol.read_smt_siblings(str(tmp_path / "nothing")) == {}


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"threading layer")
def test_environment_record_has_the_numba_threading_layer_and_thread_count(run_one, monkeypatch):
    """The numba threading layer the process actually used, its thread count and the requested layer are
    recorded in every run, and the SMT sibling map next to them."""
    psutil = pytest.importorskip("psutil")
    if not hasattr(psutil, "cpu_count"):
        pytest.skip("psutil is not installed")
    fake = types.ModuleType("numba")
    fake.config = types.SimpleNamespace(NUMBA_NUM_THREADS=8)
    fake.threading_layer = lambda: "omp"
    fake.get_num_threads = lambda: 8
    monkeypatch.setitem(sys.modules, "numba", fake)
    monkeypatch.setenv("NUMBA_THREADING_LAYER", "omp")
    env = run_one.environment()
    assert env.get("numba_threading_layer") == "omp", "threading layer not recorded"
    assert env.get("numba_num_threads") == 8 and env.get("numba_get_num_threads") == 8
    assert env["thread_env"].get("NUMBA_THREADING_LAYER") == "omp"
    assert "smt_siblings" in env


# --------------------------------------------------------------------------- tuning pilot
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_thread_settings_of_the_default_variant_set_every_variable_to_p():
    protocol = _need("protocol")
    for tool in ("crest", "crest-ooc", "scanpy", "scanpy-lean"):
        s = protocol.variant_settings(tool, "default", 16)
        env = protocol.settings_env(s)
        assert {env[v] for v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                 "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")} == {"16"}
        assert s["n_jobs"] == 16 and "NUMBA_THREADING_LAYER" not in env


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_thread_settings_variants_change_only_what_they_name():
    protocol = _need("protocol")
    s = protocol.variant_settings("scanpy", "numba_cap8", 64)
    assert (s["numba_threads"], s["blas_threads"], s["omp_threads"], s["n_jobs"]) == (8, 64, 64, 64)
    assert protocol.variant_settings("scanpy", "numba_cap8", 4)["numba_threads"] == 4     # a cap never raises
    s = protocol.variant_settings("scanpy", "blas1", 64)
    assert (s["numba_threads"], s["blas_threads"], s["omp_threads"]) == (64, 1, 1)
    assert protocol.settings_env(protocol.variant_settings("scanpy", "workqueue", 8))["NUMBA_THREADING_LAYER"] == "workqueue"
    s = protocol.variant_settings("crest", "rayon_half", 64)
    assert s["rayon_threads"] == 32 and protocol.variant_settings("crest", "rayon_half", 1)["rayon_threads"] == 1
    assert protocol.variant_settings("crest", "rayon_cap8", 64)["rayon_threads"] == 8
    with pytest.raises(KeyError):
        protocol.variant_settings("crest", "workqueue", 8)


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_both_tools_get_the_same_tuning_budget():
    protocol = _need("protocol")
    assert protocol.tuning_budget_is_equal()
    assert len(protocol.VARIANTS["scanpy"]) == len(protocol.VARIANTS["crest"])
    assert all(protocol.DEFAULT_VARIANT in v for v in protocol.VARIANTS.values())


def _pilot(tool, ds, p, variant, seconds):
    return {"tool": tool, "dataset": ds, "threads": p, "variant": variant, "seconds": seconds}


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_tuning_choice_picks_a_clearly_faster_variant_and_keeps_the_default_otherwise():
    protocol = _need("protocol")
    rec = []
    for ds in ("a", "b"):
        rec += [_pilot("scanpy", ds, 64, "default", 100.0), _pilot("scanpy", ds, 64, "numba_cap8", 100.0 / TUNING_WIN),
                _pilot("scanpy", ds, 8, "default", 50.0), _pilot("scanpy", ds, 8, "numba_cap8", 50.0 / TUNING_NOISE),
                _pilot("crest", ds, 8, "default", 10.0), _pilot("crest", ds, 8, "rayon_half", 12.0)]
    t = protocol.choose_tuning(rec)
    assert t["variant"]["scanpy"]["64"] == "numba_cap8"
    assert t["variant"]["scanpy"]["8"] == "default"          # 2% faster is inside the margin
    assert t["variant"]["crest"]["8"] == "default"
    assert protocol.tuned_variant(t, "scanpy", 64) == "numba_cap8"
    assert protocol.tuned_variant(t, "scanpy-lean", 64) == "numba_cap8"   # a profile of the same tool
    assert protocol.tuned_variant(t, "scanpy", 2) == "default"             # never piloted: the default
    assert protocol.tuned_variant(None, "crest", 8) == "default"
    assert t["pilot_runs_per_tool"] == {"scanpy": 8, "crest": 4}


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_tuning_choice_ignores_a_variant_that_failed_on_a_dataset_and_finds_the_best_thread_count():
    protocol = _need("protocol")
    rec = [_pilot("scanpy", "a", 8, "default", 40.0), _pilot("scanpy", "b", 8, "default", 80.0),
           _pilot("scanpy", "a", 8, "workqueue", 10.0), _pilot("scanpy", "b", 8, "workqueue", None),   # failed on b
           _pilot("scanpy", "a", 64, "default", 90.0), _pilot("scanpy", "b", 64, "default", 160.0)]
    t = protocol.choose_tuning(rec)
    assert t["variant"]["scanpy"]["8"] == "default"          # workqueue is not eligible: it has no run on b
    assert t["best_threads"]["scanpy"] == {"a": 8, "b": 8}


def _plan_args(**kw):
    base = dict(main_threads=[8, 64], ooc_threads=[], profile="core", datasets=["pbmc3k"], tools=["crest", "scanpy"],
                scanpy_max_cells=0, no_pin=False, thread_dataset=None, threads_scan=None, module_repeats=None,
                repeats=1, seed=0)
    base.update(kw)
    return Namespace(**base)


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"settings")
def test_run_plan_carries_the_tuned_settings_of_every_run(run_all, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    tuning = {"variant": {"scanpy": {"64": "numba_cap8"}}, "best_threads": {}}
    jobs = run_all.plan(_plan_args(tuning_data=tuning))
    sc64 = next(j for j in jobs if j["tool"] == "scanpy" and j["threads"] == 64)
    assert sc64.get("variant") == "numba_cap8" and sc64.get("settings", {}).get("numba_threads") == 8, "no settings in the run"
    sc8 = next(j for j in jobs if j["tool"] == "scanpy" and j["threads"] == 8)
    assert sc8["variant"] == "default" and sc8["settings"]["numba_threads"] == 8
    cr = next(j for j in jobs if j["tool"] == "crest" and j["threads"] == 64)
    assert cr["variant"] == "default" and cr["settings"]["rayon_threads"] == 64


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_pilot_plan_runs_every_variant_at_every_thread_count_for_both_tools(run_all, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    protocol = _need("protocol")
    a = _plan_args(pilot_datasets=["pbmc3k", "kang"], main_threads=[], thread_grid=[1, 8, 64])
    jobs = _has(run_all, "plan_pilot")(a)
    per_tool = {t: len([j for j in jobs if j["tool"] == t]) for t in ("crest", "scanpy")}
    # a variant with the same resolved settings as the default is not run (it would only add noise):
    # crest runs 1 / 3 / 4 variants and scanpy 2 / 3 / 4 at 1 / 8 / 64 threads, on two datasets
    assert per_tool == {"crest": 2 * (1 + 3 + 4), "scanpy": 2 * (2 + 3 + 4)}, per_tool
    assert {(j["tool"], j["variant"]) for j in jobs if j["threads"] == 1} == {
        ("crest", "default"), ("scanpy", "default"), ("scanpy", "workqueue")}
    assert {j["variant"] for j in jobs if j["tool"] == "crest" and j["threads"] == 8} == {"default", "blas1", "rayon_half"}
    assert len({protocol.job_tag(j) for j in jobs}) == len(jobs)             # every pilot run has its own files
    assert all(j["repeat"] == 0 and j["profile"] == "core" and not j["save"] for j in jobs)


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_a_run_gets_exactly_the_environment_of_its_settings(run_all, tmp_path, monkeypatch):
    protocol = _need("protocol")
    seen = {}

    class FakeProc:
        pid = 1

        def __init__(self, cmd, env=None, **kw):
            seen["cmd"], seen["env"] = cmd, env

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(run_all.subprocess, "Popen", FakeProc)
    monkeypatch.setenv("NUMBA_THREADING_LAYER", "inherited-from-the-shell")
    raw, logs = tmp_path / "raw", tmp_path / "logs"
    raw.mkdir()
    logs.mkdir()
    job = {"tool": "scanpy", "dataset": "pbmc3k", "profile": "core", "threads": 64, "repeat": 0, "save": False,
           "pin": True, "variant": "workqueue", "control": None,
           "settings": protocol.variant_settings("scanpy", "workqueue", 64)}
    a = Namespace(out=str(tmp_path), data_dir=str(tmp_path), timeout=5, scrublet_max_cells=0, sample_interval=0.05)
    run_all.run(job, a, logs)
    env = seen["env"]
    assert env.get("NUMBA_THREADING_LAYER") == "workqueue", "threading layer not set"
    assert abs(float(env["CREST_BENCH_SPAWN_TIME"]) - time.time()) < 60.0     # the spawn timestamp run_one measures from
    assert env["NUMBA_NUM_THREADS"] == "64" and env["OPENBLAS_NUM_THREADS"] == "64"
    assert "--pin" in seen["cmd"] and seen["cmd"][seen["cmd"].index("--n-jobs") + 1] == "64"
    job2 = dict(job, variant="default", settings=protocol.variant_settings("scanpy", "default", 64))
    run_all.run(job2, a, logs)
    assert "NUMBA_THREADING_LAYER" not in seen["env"], "a layer set in the shell leaked into a default run"


# --------------------------------------------------------------------------- controls and tags
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_run_tags_of_controls_and_pilot_runs_never_collide_with_main_runs():
    protocol = _need("protocol")
    main = protocol.run_tag("crest", "d", "core", 8, 2)
    assert main == "crest__d__core__t8__r2"
    tags = {main, protocol.run_tag("crest", "d", "core", 8, 2, "unpinned"),
            protocol.run_tag("crest", "d", "core", 8, 2, "default-settings"),
            protocol.run_tag("crest", "d", "core", 8, 2, None, "blas1")}
    assert len(tags) == 4


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"control")
def test_controls_are_planned_next_to_the_main_runs_and_never_replace_them(run_all, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    tuning = {"variant": {"scanpy": {"64": "numba_cap8"}}, "best_threads": {}}
    a = _plan_args(tuning_data=tuning, thread_dataset="pbmc3k", controls=["unpinned", "default-settings"],
                   control_datasets=[], control_threads=[8, 64])
    jobs = run_all.plan(a)
    ctl = [j for j in jobs if j.get("control")]
    assert {(j["tool"], j["threads"], j["control"], j["pin"]) for j in ctl} == {
        ("crest", 8, "unpinned", False), ("scanpy", 8, "unpinned", False),
        ("scanpy", 64, "default-settings", True)}, "control runs"
    assert all(j["variant"] == "default" for j in ctl if j["control"] == "default-settings")
    main = [j for j in jobs if not j.get("control")]
    assert {(j["tool"], j["threads"]) for j in main} == {(t, p) for t in ("crest", "scanpy") for p in (8, 64)}
    assert [j["variant"] for j in main if j["tool"] == "scanpy" and j["threads"] == 64] == ["numba_cap8"]


# --------------------------------------------------------------------------- quiet host
def _samples(foreign_cores, proc_cores=4.0, n_logical=16, n=40):
    """Monitor samples in which the machine is busy with ``proc_cores`` of this process and ``foreign_cores`` of others."""
    total = (proc_cores + foreign_cores) / n_logical * 100.0
    return [{"core_util": [total] * n_logical, "proc_cores": proc_cores} for _ in range(n)]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_foreign_load_is_the_machine_load_minus_the_runs_own_cores(run_one):
    stats = _has(_need("monitor"), "foreign_busy_stats")(_samples(FOREIGN_LOAD_BELOW))
    assert stats["median_foreign_busy_cores"] == pytest.approx(FOREIGN_LOAD_BELOW, rel=RTOL) and not stats["violated"]
    stats = _need("monitor").foreign_busy_stats(_samples(FOREIGN_LOAD_ABOVE))
    assert stats["median_foreign_busy_cores"] == pytest.approx(FOREIGN_LOAD_ABOVE, rel=RTOL) and stats["violated"]
    assert stats["threshold"] == 1.0 and stats["n_samples"] == 40
    # a process using more cores than the machine reports busy (sampling skew) never gives negative load
    assert _need("monitor").foreign_busy_stats(_samples(0.0, proc_cores=6.0)[:3] + [
        {"core_util": [10.0] * 16, "proc_cores": 6.0}])["median_foreign_busy_cores"] >= 0.0
    assert _need("monitor").foreign_busy_stats([])["violated"] is False


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_the_quiet_rule_judges_the_median_not_a_spike(run_one):
    monitor = _need("monitor")
    _has(monitor, "foreign_busy_stats")
    quiet = _samples(0.2, n=39) + _samples(14.0, n=1)          # one burst of foreign load in 40 samples
    assert not monitor.foreign_busy_stats(quiet)["violated"]
    busy = _samples(0.2, n=19) + _samples(FOREIGN_LOAD_ABOVE, n=21)
    assert monitor.foreign_busy_stats(busy)["violated"]
    assert monitor.foreign_busy_stats(_samples(1.0))["violated"]  # the threshold itself violates: "below 1" is the rule


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_quiet_decision_reruns_once_then_flags():
    protocol = _need("protocol")
    assert protocol.quiet_decision(False, 1) == "accept" and protocol.quiet_decision(False, 2) == "accept"
    assert protocol.quiet_decision(True, 1) == "rerun"
    assert protocol.quiet_decision(True, 2) == "flag"
    assert protocol.quiet_decision(True, 2, max_reruns=2) == "rerun"


def _quiet_cell(tmp_path, a):
    """A finished run that violated the quiet-host rule: its JSON, one ok attempt, and a job record."""
    raw, logs = tmp_path / "raw", tmp_path / "logs"
    raw.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    job = {"tool": "crest", "dataset": "d", "profile": "core", "threads": 8, "repeat": 0, "save": False, "pin": True,
           "variant": "default", "control": None}
    tag = "crest__d__core__t8__r0"
    return raw, logs, job, tag


def _fake_run_factory(run_all, tmp_path, tag, outcomes):
    """A stand-in for run(): writes a result JSON with the next quiet outcome and logs an ok attempt."""
    calls = []

    def fake(job, a, log_dir, raw=None, force=False):
        violated = outcomes[len(calls)]
        calls.append(force)
        raw = raw or Path(a.out) / "raw"
        (raw / f"{tag}.json").write_text(json.dumps({"status": "ok", "quiet": {
            "violated": violated, "median_foreign_busy_cores": 3.0 if violated else 0.1, "threshold": 1.0}}))
        (raw / f"{tag}.timeline.csv").write_text("t\n0\n")
        run_all._log_attempt(a, {"tag": tag, "status": "ok", "quiet_violation": violated,
                                 "median_foreign_busy_cores": 3.0 if violated else 0.1})
        return "ok"

    return fake, calls


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_a_violating_run_is_rerun_once_and_a_second_violation_is_flagged_not_dropped(run_all, tmp_path, monkeypatch):
    a = Namespace(out=str(tmp_path))
    raw, logs, job, tag = _quiet_cell(tmp_path, a)
    fake, calls = _fake_run_factory(run_all, tmp_path, tag, [True, True, True])
    monkeypatch.setattr(run_all, "run", fake)
    fake(job, a, logs, raw)                                     # the first attempt violates
    notes = _has(run_all, "quiet_pass")([job], a, logs, raw)
    assert calls == [False, True], "exactly one re-run"
    rec = json.loads((raw / f"{tag}.json").read_text())
    assert "flag" in rec["quiet"] and "2 attempts" in rec["quiet"]["flag"]
    assert (raw / "_quiet" / f"{tag}.attempt1.json").exists()   # the first attempt stays on disk, outside the loader's glob
    assert any("flagged" in n for n in notes)
    again = run_all.quiet_pass([job], a, logs, raw)             # a second pass never re-runs it again
    assert calls == [False, True] and again == [n for n in again if "flagged" in n]
    attempts = [json.loads(x) for x in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [x["status"] for x in attempts] == ["ok", "ok", "flagged"]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_a_rerun_that_is_quiet_is_accepted_and_a_quiet_run_is_left_alone(run_all, tmp_path, monkeypatch):
    a = Namespace(out=str(tmp_path))
    raw, logs, job, tag = _quiet_cell(tmp_path, a)
    fake, calls = _fake_run_factory(run_all, tmp_path, tag, [True, False])
    monkeypatch.setattr(run_all, "run", fake)
    fake(job, a, logs, raw)
    _has(run_all, "quiet_pass")([job], a, logs, raw)
    assert calls == [False, True]
    assert "flag" not in json.loads((raw / f"{tag}.json").read_text())["quiet"]
    fake2, calls2 = _fake_run_factory(run_all, tmp_path, tag, [False])
    monkeypatch.setattr(run_all, "run", fake2)
    fake2(job, a, logs, raw)
    run_all.quiet_pass([job], a, logs, raw)
    assert calls2 == [False]                                   # no re-run of a quiet run


# --------------------------------------------------------------------------- required result files
def _plan_dir(tmp_path, jobs, files=(), statuses=None):
    res = tmp_path / "res"
    (res / "raw").mkdir(parents=True)
    (res / "plan.json").write_text(json.dumps({"args": {"main_threads": [8]}, "runs": jobs}))
    for i, j in enumerate(jobs):
        tag = f"{j['tool']}__{j['dataset']}__{j['profile']}__t{j['threads']}__r{j['repeat']}"
        for f in files:
            if f == "json":
                (res / "raw" / f"{tag}.json").write_text(json.dumps({"status": (statuses or {}).get(i, "ok")}))
            else:
                (res / "raw" / f"{tag}.{f}").write_text("x")
    return res


def _job(tool="crest", dataset="d", threads=8, repeat=0, save=False, profile="core"):
    return {"tool": tool, "dataset": dataset, "profile": profile, "threads": threads, "repeat": repeat, "save": save,
            "control": None}


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_required_files_per_run_depend_on_its_outcome():
    protocol = _need("protocol")
    plain, saving = _job(), _job(save=True)
    assert protocol.required_files(plain, "ok") == ["crest__d__core__t8__r0.json", "crest__d__core__t8__r0.timeline.csv"]
    assert protocol.required_files(saving, "ok")[-1] == "crest__d__core__t8__r0.outputs.npz"
    assert protocol.required_files(dict(saving, repeat=1), "ok")[-1].endswith(".timeline.csv")   # only repeat 0 saves
    assert protocol.required_files(saving, "failed") == ["crest__d__core__t8__r0.json"]           # a failure leaves its JSON
    assert protocol.required_files(dict(plain, control="unpinned"))[0] == "crest__d__core__t8__r0__ctl-unpinned.json"


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_missing_result_files_lists_every_gap(tmp_path):
    protocol = _need("protocol")
    jobs = [_job(), _job(repeat=1), _job(tool="scanpy"), _job(tool="scanpy", dataset="e", repeat=1, save=True)]
    res = _plan_dir(tmp_path / "a", jobs, files=("json", "timeline.csv"))
    assert protocol.missing_result_files(res) == []             # the saving job is repeat 1: it saves nothing
    jobs[3]["repeat"] = 0                                       # now it saves: the per-cell outputs are required
    res = _plan_dir(tmp_path / "b", jobs, files=("json", "timeline.csv"))
    assert protocol.missing_result_files(res) == ["scanpy__e__core__t8__r0.outputs.npz"]
    (res / "raw" / "crest__d__core__t8__r1.json").unlink()
    (res / "raw" / "scanpy__d__core__t8__r0.timeline.csv").unlink()
    assert sorted(protocol.missing_result_files(res)) == sorted([
        "crest__d__core__t8__r1.json",                      # no JSON: only the JSON is demanded (its status is unknown)
        "scanpy__d__core__t8__r0.timeline.csv", "scanpy__e__core__t8__r0.outputs.npz"])


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_a_failed_run_needs_only_its_json_and_no_plan_is_one_missing_entry(tmp_path):
    protocol = _need("protocol")
    res = _plan_dir(tmp_path, [_job(save=True)], files=("json",), statuses={0: "failed"})
    assert protocol.missing_result_files(res) == []
    assert protocol.missing_result_files(tmp_path / "empty") == ["plan.json"]


# --------------------------------------------------------------------------- hand-built result directories
def _tag(tool, dataset, profile, threads, repeat, control=None):
    return f"{tool}__{dataset}__{profile}__t{threads}__r{repeat}" + (f"__ctl-{control}" if control else "")


def _steps(**over):
    """Step seconds of a run: every core step at 1 s unless overridden."""
    return {s: over.get(s, 1.0) for s in CORE_STEPS}


def _write_json_run(raw: Path, tool, threads, steps, *, dataset="ds", repeat=0, profile="core", untracked=0.2,
                    startup=1.0, control=None, quiet=None, peak=2.0, baseline=0.5, n_cells=1000, kind=None,
                    parent=None, status="ok", with_endpoints=True):
    """One run JSON + timeline in the layout run_one writes; returns the plan entry of the run."""
    st = {s: {"seconds": v, "peak_rss_gb": peak, "parallelism": 1.0} for s, v in steps.items()}
    total = sum(steps.values()) + untracked
    rec = {"tool": tool, "dataset": dataset, "profile": profile, "threads": threads, "repeat": repeat, "status": status,
           "n_cells": n_cells, "total_seconds": total, "peak_rss_gb": peak, "baseline_rss_gb": baseline,
           "peak_rss_minus_baseline_gb": peak - baseline, "steps": st, "control": control,
           "env": {"logical_cores": 64, "physical_cores": 32}, "pinned": True}
    if with_endpoints:
        rec.update({"startup_seconds": startup, "import_seconds": startup * 0.9, "end_to_end_seconds": startup + total,
                    "untracked_seconds": untracked, "quiet": quiet or {"median_foreign_busy_cores": 0.1,
                                                                        "threshold": 1.0, "violated": False}})
    if kind:
        rec["dataset_meta"] = {"kind": kind, "parent": parent}
    tag = _tag(tool, dataset, profile, threads, repeat, control)
    (raw / f"{tag}.json").write_text(json.dumps(rec))
    (raw / f"{tag}.timeline.csv").write_text("t,step,rss_gb,proc_cores,util_0,util_1\n0.0,read,1.0,1.0,50,50\n1.0,read,1.0,1.0,50,50\n")
    return {"tool": tool, "dataset": dataset, "profile": profile, "threads": threads, "repeat": repeat, "save": False,
            "control": control}


def _write_results(res: Path, runs: list[dict], main_threads=(8, 64), attempts=()):
    """A results directory: raw/ files from ``runs`` (kwargs of _write_json_run, with tool, threads, steps), a
    complete plan.json and an attempts.jsonl."""
    raw = res / "raw"
    raw.mkdir(parents=True)
    jobs = [_write_json_run(raw, **r) for r in runs]
    (res / "plan.json").write_text(json.dumps({"args": {"main_threads": list(main_threads)}, "runs": jobs,
                                                "main_threads": list(main_threads), "ooc_threads": list(main_threads)}))
    if attempts:
        (res / "attempts.jsonl").write_text("\n".join(json.dumps(a) for a in attempts) + "\n")
    return res


# per-step seconds of the scenario (PBMC 68k-like, seconds): CREST normalises lazily, so the cost sits in hvg
CREST_STEPS = dict(read=1.0, qc_filter=0.5, normalize_log1p=LAZY_STEP_SECONDS, hvg=0.4, scale_pca=1.0, neighbors=1.0,
                   leiden=1.0, umap=2.0, de_ttest=1.0, de_wilcoxon=2.0)
SCANPY_STEPS = dict(read=2.0, qc_filter=1.0, normalize_log1p=0.3, hvg=0.1, scale_pca=3.0, neighbors=2.0, leiden=1.0,
                    umap=5.0, de_ttest=2.0, de_wilcoxon=34.7)
CREST_UNTRACKED, SCANPY_UNTRACKED = 0.2, 2.4           # untracked time differs by tool (gc, result extraction)
CREST_STARTUP, SCANPY_STARTUP = 0.6, 4.0               # start-up and imports differ by tool


def _scenario():
    """crest and scanpy at 8 and 64 threads, 2 repeats; scanpy's Wilcoxon step is 5x slower at 64 threads; a scanpy
    run that is not pinned (control); a scanpy-lean run; one real-subsample series and one synthetic series."""
    runs = []
    for rep in (0, 1):
        for p in (8, 64):
            runs.append(dict(tool="crest", threads=p, steps=CREST_STEPS, repeat=rep, untracked=CREST_UNTRACKED, startup=CREST_STARTUP))
            sc = dict(SCANPY_STEPS, de_wilcoxon=177.3 if p == 64 else 34.7, neighbors=2.0 if p == 8 else 2.1)
            runs.append(dict(tool="scanpy", threads=p, steps=sc, repeat=rep, untracked=SCANPY_UNTRACKED, startup=SCANPY_STARTUP,
                             peak=4.0, baseline=0.75))
    runs.append(dict(tool="scanpy", threads=8, steps=dict(SCANPY_STEPS, de_wilcoxon=400.0), untracked=SCANPY_UNTRACKED,
                     startup=SCANPY_STARTUP, control="unpinned"))                       # a control that must not count
    runs.append(dict(tool="scanpy-lean", threads=8, steps=SCANPY_STEPS, untracked=SCANPY_UNTRACKED, startup=SCANPY_STARTUP,
                     peak=3.0, baseline=0.75))
    for name, n, kind, parent in (("big_sub10k", 10_000, "real-subsample", "big"), ("big_sub20k", 20_000, "real-subsample", "big"),
                                  ("big_sub40k", 40_000, "real-subsample", "big"), ("synth_10k", 10_000, "synthetic", None),
                                  ("synth_20k", 20_000, "synthetic", None), ("synth_40k", 40_000, "synthetic", None)):
        f = n / 10_000
        for tool, base, un, su in (("crest", CREST_STEPS, CREST_UNTRACKED, CREST_STARTUP),
                                   ("scanpy", SCANPY_STEPS, SCANPY_UNTRACKED, SCANPY_STARTUP)):
            runs.append(dict(tool=tool, threads=8, dataset=name, n_cells=n, kind=kind, parent=parent, untracked=0.0, startup=0.0,
                             steps={k: v * f for k, v in base.items()}))     # no fixed costs: time is exactly linear in cells
    return runs


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    """The scenario written to disk, loaded, and summarised once with the command-line entry point."""
    import summarize
    res = tmp_path_factory.mktemp("scenario") / "res"
    _write_results(res, _scenario(), attempts=[
        {"tag": "crest__ds__core__t8__r0", "status": "ok", "quiet_violation": True, "median_foreign_busy_cores": 2.5,
         "started": "2026-01-01T00:00:00"},
        {"tag": "crest__ds__core__t8__r0", "status": "ok", "quiet_violation": False, "median_foreign_busy_cores": 0.1,
         "started": "2026-01-01T00:05:00"}])
    df = summarize.load_runs(res / "raw")
    old, error = sys.argv, None
    sys.argv = ["summarize.py", str(res)]
    try:
        summarize.main()
    except BaseException as e:  # noqa: BLE001  (the tests below assert on what was written; none means they fail)
        error = f"{type(e).__name__}: {e}"
    finally:
        sys.argv = old
    report = (res / "report.md").read_text() if (res / "report.md").exists() else ""
    return types.SimpleNamespace(res=res, df=df, report=report, tables=res / "tables", error=error)


# --------------------------------------------------------------------------- endpoints (end-to-end, sum of steps)
def _e2e(steps, untracked, startup):
    return startup + sum(steps.values()) + untracked


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"endpoint")
def test_headline_endpoint_is_the_end_to_end_time_and_the_step_sum_is_secondary(scenario, summarize):
    import inspect
    assert "endpoint" in inspect.signature(summarize.main_table).parameters, "main_table has no endpoint parameter"
    sets = {t: {8, 64} for t in ("crest", "scanpy", "crest-ooc")}
    sc8 = SCANPY_STEPS
    e = summarize.main_table(scenario.df, 64, sets, endpoint="end_to_end").filter(pl.col("dataset") == "ds").row(0, named=True)
    c = summarize.main_table(scenario.df, 64, sets, endpoint="core").filter(pl.col("dataset") == "ds").row(0, named=True)
    want_e2e = _e2e(sc8, SCANPY_UNTRACKED, SCANPY_STARTUP) / _e2e(CREST_STEPS, CREST_UNTRACKED, CREST_STARTUP)
    want_core = sum(sc8.values()) / sum(CREST_STEPS.values())
    assert e["speedup_t8"] == pytest.approx(want_e2e, rel=RTOL) and e["endpoint"] == "end_to_end"
    assert c["speedup_t8"] == pytest.approx(want_core, rel=RTOL) and c["endpoint"] == "core"
    assert want_e2e != pytest.approx(want_core, rel=1e-3)             # the two endpoints really differ here
    # the table carries both endpoints and the untracked time of each tool
    assert e["crest_core_seconds_median"] == pytest.approx(sum(CREST_STEPS.values()), rel=RTOL)
    assert e["scanpy_end_to_end_seconds_median"] is not None and e["scanpy_total_seconds_median"] is not None
    assert e["scanpy_untracked_seconds_median"] == pytest.approx(SCANPY_UNTRACKED, rel=RTOL)
    assert e["crest_untracked_seconds_median"] == pytest.approx(CREST_UNTRACKED, rel=RTOL)


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"end-to-end|endpoint")
def test_report_names_its_endpoint_and_lists_both_endpoints_per_tool(scenario):
    assert "end-to-end process wall time" in scenario.report, "the report does not name the primary endpoint"
    assert "Secondary endpoint" in scenario.report
    assert "## Core workflow" in scenario.report
    assert (scenario.tables / "endpoints.csv").exists(), f"no endpoints.csv ({scenario.error})"
    ep = pl.read_csv(scenario.tables / "endpoints.csv")
    for col in ("core_seconds", "total_seconds", "end_to_end_seconds", "untracked_seconds", "import_seconds"):
        assert col in ep.columns, f"endpoints.csv has no {col}"
    s = ep.filter((pl.col("dataset") == "ds") & (pl.col("tool") == "scanpy") & (pl.col("threads") == 8)).row(0, named=True)
    c = ep.filter((pl.col("dataset") == "ds") & (pl.col("tool") == "crest") & (pl.col("threads") == 8)).row(0, named=True)
    assert s["untracked_seconds"] == pytest.approx(SCANPY_UNTRACKED, rel=RTOL) and c["untracked_seconds"] == pytest.approx(CREST_UNTRACKED, rel=RTOL)
    assert s["end_to_end_seconds"] - s["total_seconds"] == pytest.approx(SCANPY_STARTUP, rel=RTOL)
    assert s["total_seconds"] - s["core_seconds"] == pytest.approx(SCANPY_UNTRACKED, rel=RTOL)


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_end_to_end_time_needs_runs_that_recorded_it(summarize, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for tool in ("crest", "scanpy"):
        _write_json_run(raw, tool, 8, _steps(), with_endpoints=False)
    df = summarize.load_runs(raw)
    _has(summarize, "ENDPOINT_COLUMNS")
    with pytest.raises(ValueError, match="end_to_end_seconds"):
        summarize.main_table(df, 8, endpoint="end_to_end")
    assert summarize.main_table(df, 8, endpoint="core").height == 1          # the step sum still works


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_end_to_end_seconds_is_start_up_plus_the_timed_block():
    protocol = _need("protocol")
    assert protocol.end_to_end_seconds(4.0, 17.0) == 21.0
    assert protocol.end_to_end_seconds(None, 17.0) is None and protocol.end_to_end_seconds(4.0, None) is None
    run_one = _need("run_one")
    got = _has(run_one, "import_tool")
    run_one.TOOL_IMPORTS = {"scanpy": ["json", "csv"], "crest": ["json", "no_such_module_for_the_test"]}
    out = got("scanpy", "core")
    assert set(out["per_module"]) == {"json", "csv"} and out["errors"] == {} and out["seconds"] >= 0.0
    out = got("crest", "core")                                    # a missing module is recorded, not fatal
    assert "no_such_module_for_the_test" in out["errors"] and "json" in out["per_module"]


# --------------------------------------------------------------------------- controls and quiet host in the report
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"control")
def test_controls_never_enter_the_headline_and_are_reported_separately(scenario, summarize):
    sets = {t: {8, 64} for t in ("crest", "scanpy", "crest-ooc")}
    row = summarize.main_table(scenario.df, 64, sets).filter(pl.col("dataset") == "ds").row(0, named=True)
    # the unpinned scanpy control has a 400 s Wilcoxon step: if it counted, scanpy's 8-thread time would not be 51.1 s
    assert row["scanpy_time_t8"] == pytest.approx(sum(SCANPY_STEPS.values()), rel=RTOL)
    assert len(summarize._core_samples(scenario.df)[("ds", "scanpy", 8)]["t"]) == 2, "a control run entered the statistics"
    assert (scenario.tables / "controls.csv").exists(), f"no controls.csv ({scenario.error})"
    ctl = pl.read_csv(scenario.tables / "controls.csv")
    r = ctl.filter((pl.col("tool") == "scanpy") & (pl.col("control") == "unpinned")).row(0, named=True)
    assert r["n_control"] == 1 and r["n_main"] == 2 and r["control_over_main"] > 1.0
    assert "## Controls" in scenario.report


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"quiet")
def test_report_lists_quiet_host_violations_and_their_outcome(scenario):
    assert "## Quiet host" in scenario.report, "no quiet-host section"
    assert (scenario.tables / "quiet_host.csv").exists(), f"no quiet_host.csv ({scenario.error})"
    q = pl.read_csv(scenario.tables / "quiet_host.csv")
    assert q.height == 1 and q["outcome"][0] == "re-run" and q["median_foreign_busy_cores"][0] == pytest.approx(2.5)


# --------------------------------------------------------------------------- fused stages (lazy steps)
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_stage_table_fuses_the_lazy_normalisation_into_one_stage(scenario, summarize):
    st = _has(summarize, "stage_table")(scenario.df)
    fused = st.filter((pl.col("dataset") == "ds") & (pl.col("step") == "norm+hvg"))
    assert sorted(fused["tool"].to_list()) == ["crest", "scanpy"]            # at the highest thread count of the runs
    got = {r["tool"]: r["median_s"] for r in fused.iter_rows(named=True)}
    assert got["crest"] == pytest.approx(CREST_STEPS["normalize_log1p"] + CREST_STEPS["hvg"], rel=RTOL)
    assert got["scanpy"] == pytest.approx(SCANPY_STEPS["normalize_log1p"] + SCANPY_STEPS["hvg"], rel=RTOL)
    assert "normalize_log1p" not in st["step"].to_list() and "hvg" not in st["step"].to_list()
    steps = summarize.step_table(scenario.df)                     # the raw steps stay available
    assert {"normalize_log1p", "hvg"} <= set(steps["step"].to_list())


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_no_stage_ratio_above_one_thousand_is_ever_formed(scenario, summarize):
    st = _has(summarize, "stage_table")(scenario.df)
    su = summarize.stage_speedup_table(st)
    ratios = [r for r in su["speedup"].to_list() if r is not None]
    assert ratios and max(ratios) < STEP_RATIO_CEILING
    row = su.filter((pl.col("dataset") == "ds") & pl.col("stage").str.contains("normalize")).row(0, named=True)
    assert row["speedup"] == pytest.approx((SCANPY_STEPS["normalize_log1p"] + SCANPY_STEPS["hvg"])
                                           / (CREST_STEPS["normalize_log1p"] + CREST_STEPS["hvg"]), rel=RTOL)
    # dividing the raw lazy step medians is what produces the 1000x ratios: the guard refuses it
    raw_ratio = SCANPY_STEPS["normalize_log1p"] / CREST_STEPS["normalize_log1p"]
    assert raw_ratio > STEP_RATIO_CEILING
    assert summarize.safe_ratio(SCANPY_STEPS["normalize_log1p"], CREST_STEPS["normalize_log1p"]) is None
    assert summarize.safe_ratio(1.0, 0.0) is None and summarize.safe_ratio(0.004, 0.003) is None
    assert summarize.safe_ratio(12.0, 4.0) == pytest.approx(3.0)
    assert summarize.safe_ratio(5000.0, 2.0) is None              # above the ceiling: not printed
    assert "normalize+log1p+HVG (fused)" in scenario.report


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"lazy step")
def test_not_faster_table_skips_the_raw_steps_whose_cost_is_deferred(summarize):
    steps = pl.DataFrame({"dataset": ["d"] * 6, "step": ["normalize_log1p", "normalize_log1p", "hvg", "hvg", "umap", "umap"],
                          "tool": ["crest", "scanpy"] * 3, "median_s": [0.4, 0.3, 0.5, 0.3, 1.0, 1.1]})
    got = summarize.not_faster_table(steps)["step"].to_list()
    assert got == ["umap"], f"lazy step compared as if it were one step: {got}"


# --------------------------------------------------------------------------- steps slower with more threads
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_thread_regression_table_flags_only_steps_that_pass_both_thresholds(scenario, summarize):
    reg = _has(summarize, "thread_regression_table")(scenario.df, {t: {8, 64} for t in ("crest", "scanpy")})
    flagged = reg.filter(pl.col("flagged"))
    assert flagged["tool"].to_list() == ["scanpy"] and flagged["step"].to_list() == ["de_wilcoxon"]
    r = flagged.row(0, named=True)
    assert (r["ref_threads"], r["threads"]) == (8, 64) and r["ratio"] == pytest.approx(177.3 / 34.7)
    assert r["extra_s"] == pytest.approx(177.3 - 34.7)
    nb = reg.filter((pl.col("tool") == "scanpy") & (pl.col("step") == "neighbors")).row(0, named=True)
    assert nb["ratio"] == pytest.approx(2.1 / 2.0) and not nb["flagged"]      # +5% and +0.1 s: neither threshold


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"slower with more threads")
def test_report_prints_the_steps_that_get_slower_with_more_threads(scenario):
    head = "Steps that get slower with more threads"
    assert head in scenario.report, "the report has no thread-regression section"
    section = scenario.report.split(head, 1)[1].split("\n## ", 1)[0]
    assert "de_wilcoxon" in section and "neighbors" not in section


# --------------------------------------------------------------------------- scaling series and datasets labels
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_scaling_series_are_real_subsamples_first_and_the_synthetic_set_is_a_labelled_supplement(scenario, summarize):
    series = _has(summarize, "scaling_series")(scenario.df)
    assert series == {"real subsamples of big": ["big_sub10k", "big_sub20k", "big_sub40k"],
                      "synthetic (supplement)": ["synth_10k", "synth_20k", "synth_40k"]}
    assert (scenario.tables / "scaling.csv").exists(), f"no scaling.csv ({scenario.error})"
    fits = pl.read_csv(scenario.tables / "scaling.csv")
    assert set(fits["series"].to_list()) == set(series)
    for s in series:
        assert sorted(fits.filter(pl.col("series") == s)["tool"].to_list()) == ["crest", "scanpy"]
    # the time of the scenario grows linearly with the cells: the fitted exponent is 1
    assert fits["alpha"].to_list() == pytest.approx([1.0] * fits.height, abs=0.02)
    assert "real subsamples" in scenario.report and "supplement" in scenario.report


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"series")
def test_a_real_series_is_not_mixed_with_the_synthetic_one_in_a_fit(scenario):
    cols = pl.read_csv(scenario.tables / "scaling.csv").columns if (scenario.tables / "scaling.csv").exists() else []
    assert "series" in cols, "scaling.csv has no series column"
    fits = pl.read_csv(scenario.tables / "scaling.csv")
    assert not any("big_sub" in d and "synth_" in d for d in fits["datasets"].to_list())


# --------------------------------------------------------------------------- memory profile
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_memory_table_compares_idiomatic_lean_and_crest_peak_memory(scenario, summarize):
    mem = _has(summarize, "memory_table")(scenario.df)
    got = {r["tool"]: r for r in mem.filter(pl.col("dataset") == "ds").iter_rows(named=True)}
    assert set(got) == {"scanpy", "scanpy-lean", "crest"}
    assert got["scanpy"]["peak_gb"] == pytest.approx(4.0) and got["scanpy-lean"]["peak_gb"] == pytest.approx(3.0)
    assert got["scanpy-lean"]["scanpy_over_this"] == pytest.approx(4.0 / 3.0)
    assert got["scanpy"]["peak_minus_baseline_gb"] == pytest.approx(3.25)


def test_lean_in_place_subsetting_gives_the_same_matrix_as_view_and_copy():
    """The memory-lean scanpy profile differs from the idiomatic one only in how the variable subset is taken."""
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(0)
    X = sp.random(40, 30, density=0.3, format="csr", random_state=1, dtype=np.float32)
    import pandas as pd
    a = ad.AnnData(X.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(30)]))
    a.var["highly_variable"] = rng.random(30) < 0.5
    a.raw = a
    ref = a[:, a.var.highly_variable].copy()
    lean = a.copy()
    lean._inplace_subset_var(lean.var["highly_variable"].to_numpy())
    assert (ref.X != lean.X).nnz == 0 and list(ref.var_names) == list(lean.var_names)
    assert lean.raw.shape == a.shape and (lean.raw.X != a.raw.X).nnz == 0       # the full-gene raw matrix is intact


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_the_lean_profile_is_a_scanpy_tool_with_the_scanpy_settings(run_all, monkeypatch):
    protocol = _need("protocol")
    assert protocol.base_tool("scanpy-lean") == "scanpy" and protocol.base_tool("crest-ooc") == "crest"
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    jobs = run_all.plan(_plan_args(tools=["scanpy", "scanpy-lean"], profile="full"))
    lean = [j for j in jobs if j["tool"] == "scanpy-lean"]
    assert {j["profile"] for j in lean} == {"core"} and not any(j["save"] for j in lean)   # core runs only, no outputs


# --------------------------------------------------------------------------- completeness gate of the summary
def _summarize_exit(summarize, res, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(res), *extra])
    try:
        summarize.main()
        return 0
    except SystemExit as e:
        return e.code


def _two_tool_results(tmp_path, missing_job=False):
    res = tmp_path / "gate"
    runs = [dict(tool="crest", threads=8, steps=CREST_STEPS), dict(tool="scanpy", threads=8, steps=SCANPY_STEPS)]
    _write_results(res, runs, main_threads=(8,))
    if missing_job:
        plan = json.loads((res / "plan.json").read_text())
        plan["runs"].append(_job(tool="crest", dataset="ds", threads=8, repeat=1))
        (res / "plan.json").write_text(json.dumps(plan))
    return res


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"missing required")
def test_the_summary_fails_when_a_required_result_file_is_missing(summarize, tmp_path, monkeypatch):
    res = _two_tool_results(tmp_path, missing_job=True)
    code = _summarize_exit(summarize, res, monkeypatch)
    assert code not in (0, None), "summarize accepted a results directory with missing required files"
    assert "crest__ds__core__t8__r1.json" in (res / "missing_files.txt").read_text()


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"INCOMPLETE")
def test_an_incomplete_draft_is_marked_and_a_complete_directory_is_not(summarize, tmp_path, monkeypatch):
    res = _two_tool_results(tmp_path, missing_job=True)
    assert _summarize_exit(summarize, res, monkeypatch, "--allow-incomplete") in (0, None)
    assert "INCOMPLETE" in (res / "report.md").read_text(), "a draft made from missing files is not marked INCOMPLETE"
    ok = _two_tool_results(tmp_path / "ok")
    assert _summarize_exit(summarize, ok, monkeypatch) in (0, None)
    assert "INCOMPLETE" not in (ok / "report.md").read_text() and not (ok / "missing_files.txt").exists()


# --------------------------------------------------------------------------- one real run (crest, tiny synthetic data)
def _tiny_dataset(path: Path, n_cells=450, n_genes=2300, seed=0):
    ds = _need("datasets")
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, n_genes)
    mult = np.ones((3, n_genes))
    for k in range(3):
        mult[k, k * 200:(k + 1) * 200] = 6.0
    labels = rng.integers(0, 3, n_cells)
    mu = base[None, :] * mult[labels] * rng.uniform(0.6, 1.4, (n_cells, 1))
    X = sp.csr_matrix(rng.poisson(mu).astype(np.int32))
    X.sort_indices()
    ds.write_10x_h5(path, X.indptr, X.indices, X.data, n_genes, [f"cell{i}-1" for i in range(n_cells)],
                    [f"ENSG{i:05d}" for i in range(n_genes)], [f"GENE{i}" for i in range(n_genes)])


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"startup_seconds")
def test_one_run_records_endpoints_load_pinning_and_memory_baseline(tmp_path):
    """run_one end to end with CREST on a tiny synthetic matrix: the new fields exist and are consistent."""
    pytest.importorskip("crest")
    pytest.importorskip("psutil")
    h5 = tmp_path / "tinyds.h5"
    _tiny_dataset(h5)
    out = tmp_path / "out"
    env = {**os.environ, "RAYON_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "CREST_BENCH_SPAWN_TIME": repr(time.time())}
    t_spawn = time.time()
    cmd = [sys.executable, str(REPO / "bench" / "paper" / "run_one.py"), "--tool", "crest", "--data", str(h5), "--out", str(out),
           "--profile", "core", "--threads", "2", "--repeat", "0", "--pin"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if "PermissionError" in proc.stderr:
        pytest.skip("psutil cannot list processes here")
    assert proc.returncode == 0, proc.stderr[-600:]
    rec = json.loads((out / "crest__tinyds__core__t2__r0.json").read_text())
    assert rec["status"] == "ok", rec.get("error")
    assert rec.get("startup_seconds", None) is not None, "startup_seconds not recorded"
    assert rec["startup_source"] == "spawn-time" and 0.0 < rec["startup_seconds"] < time.time() - t_spawn
    steps = sum(s["seconds"] for s in rec["steps"].values())
    assert rec["untracked_seconds"] == pytest.approx(rec["total_seconds"] - steps, abs=1e-6)
    assert rec["end_to_end_seconds"] == pytest.approx(rec["startup_seconds"] + rec["total_seconds"], abs=1e-6)
    assert 0.0 < rec["import_seconds"] <= rec["startup_seconds"] and "crest" in rec["import_breakdown"]
    assert rec["quiet"]["threshold"] == 1.0 and rec["quiet"]["n_samples"] > 0 and "violated" in rec["quiet"]
    assert rec["sample_interval_s"] == 0.05 and rec["rss_after_imports_gb"] > 0 and rec["baseline_rss_gb"] > 0
    assert rec["dataset_meta"]["kind"] == "real" and rec["variant"] == "default" and rec["control"] is None
    if hasattr(os, "sched_setaffinity"):
        assert rec["pin_cpus"] is not None and len(rec["pin_cpus"]) == 2
    assert (out / "crest__tinyds__core__t2__r0.timeline.csv").exists()


# --------------------------------------------------------------------------- datasets: independent conversion and verification
def _write_mtx_dir(d: Path, dense, genes, barcodes, gz=False):
    """A 10x MTX directory (genes x cells matrix, 1-based coordinate format): v2 layout (``genes.tsv``, two columns,
    plain text) or, with ``gz``, the v3 layout (``features.tsv.gz``, three columns, every file gzipped)."""
    import gzip
    features_name = "features.tsv" if gz else "genes.tsv"
    d.mkdir(parents=True, exist_ok=True)
    ng, nc = dense.shape
    coords = [(i + 1, j + 1, int(dense[i, j])) for j in range(nc) for i in range(ng) if dense[i, j]]
    mtx = f"%%MatrixMarket matrix coordinate integer general\n{ng} {nc} {len(coords)}\n" + "".join(f"{i} {j} {v}\n" for i, j, v in coords)
    texts = {"matrix.mtx": mtx, features_name: "".join(f"{a}\t{b}" + ("\tGene Expression" if gz else "") + "\n" for a, b in genes),
             "barcodes.tsv": "".join(f"{b}\n" for b in barcodes)}
    for name, text in texts.items():
        if gz:
            with gzip.open(d / f"{name}.gz", "wt") as f:
                f.write(text)
        else:
            (d / name).write_text(text)


GENES = [("ID1", "A"), ("ID2", "B"), ("ID3", "B"), ("ID4", "C"), ("ID5", "D"), ("ID6", "E")]   # a duplicated symbol, as in hg19
BARCODES = [f"AAAC{i}-1" for i in range(5)]


def _counts(seed=0):
    return np.random.default_rng(seed).integers(0, 6, (len(GENES), len(BARCODES))) * (np.random.default_rng(seed + 1).random((len(GENES), len(BARCODES))) < 0.6)


@pytest.fixture(scope="module")
def datasets():
    pytest.importorskip("h5py")
    return _need("datasets")


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
@pytest.mark.parametrize("gz", [False, True])
def test_independent_mtx_reader_returns_cells_by_genes_with_names_in_file_order(datasets, tmp_path, gz):
    dense = _counts()
    _write_mtx_dir(tmp_path / "m", dense, GENES, BARCODES, gz=gz)
    X, bc, ids, names = _has(datasets, "read_mtx_dir_independent")(tmp_path / "m")
    assert X.shape == (5, 6) and (X.toarray() == dense.T).all()
    assert bc == BARCODES and ids == [g[0] for g in GENES] and names == [g[1] for g in GENES]     # duplicates kept, in order


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"depends on crest")
def test_mtx_conversion_does_not_use_crests_reader(datasets, tmp_path, monkeypatch):
    dense = _counts()
    _write_mtx_dir(tmp_path / "m", dense, GENES, BARCODES)
    monkeypatch.setitem(sys.modules, "crest", None)                        # any import of crest now fails
    try:
        datasets._mtx_dir_to_h5(tmp_path / "m", tmp_path / "out.h5", "hg19")
    except ImportError as e:
        raise AssertionError(f"MTX conversion depends on crest's reader: {e}") from None
    X, bc, ids, names = datasets.read_h5_counts(tmp_path / "out.h5")
    assert (X.toarray() == dense.T).all() and bc == BARCODES and names == [g[1] for g in GENES]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_compare_counts_detects_a_changed_value_a_reordered_gene_and_a_swapped_cell(datasets):
    sp = pytest.importorskip("scipy.sparse")
    dense = _counts().T.astype(float)
    ref = (sp.csr_matrix(dense), BARCODES, [g[1] for g in GENES])
    cmp = _has(datasets, "compare_counts")
    assert cmp(ref, ref) == []
    changed = dense.copy()
    changed[2, 3] += 1 if changed[2, 3] else 0
    changed[np.nonzero(changed)[0][0], np.nonzero(changed)[1][0]] += 1
    assert any("values differ" in m for m in cmp((sp.csr_matrix(changed), BARCODES, ref[2]), ref))
    perm = [1, 0, 2, 3, 4, 5]                                                # two genes swapped, names kept in the old order
    assert any("column indices differ" in m or "values differ" in m for m in cmp((sp.csr_matrix(dense[:, perm]), BARCODES, ref[2]), ref))
    assert any("gene names differ" in m for m in cmp((ref[0], BARCODES, [g[1] for g in GENES][::-1]), ref))
    assert any("barcodes differ" in m for m in cmp((ref[0], BARCODES[::-1], ref[2]), ref))
    assert cmp((sp.csr_matrix(dense[:3]), BARCODES[:3], ref[2]), ref)[0].startswith("shape")


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_verification_passes_for_a_correct_conversion_and_fails_for_a_misordered_gene_axis(datasets, tmp_path):
    dense = _counts()
    _write_mtx_dir(tmp_path / "m", dense, GENES, BARCODES, gz=True)
    datasets._mtx_dir_to_h5(tmp_path / "m", tmp_path / "ok.h5", "hg19")
    checks = _has(datasets, "verify_mtx_dataset")(tmp_path / "m", tmp_path / "ok.h5")
    assert checks and all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]
    # a reader that mis-orders the genes writes the right counts under permuted names
    X, bc, ids, names = datasets.read_mtx_dir_independent(tmp_path / "m")
    datasets.write_10x_h5(tmp_path / "bad.h5", X.indptr, X.indices, X.data, X.shape[1], bc, ids[::-1], names[::-1], "hg19")
    bad = datasets.verify_mtx_dataset(tmp_path / "m", tmp_path / "bad.h5")
    assert not all(c["ok"] for c in bad)
    assert any("gene names differ" in c["detail"] for c in bad if not c["ok"])


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_reader_equality_checks_pass_on_a_converted_file(datasets, tmp_path):
    _write_mtx_dir(tmp_path / "m", _counts(), GENES, BARCODES)
    datasets._mtx_dir_to_h5(tmp_path / "m", tmp_path / "ok.h5", "hg19")
    checks = _has(datasets, "verify_reader_equality")(tmp_path / "ok.h5")
    assert checks and all(c["ok"] for c in checks), checks


# --------------------------------------------------------------------------- datasets: real-subsample series
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_subsample_rows_are_seeded_sorted_and_nested(datasets):
    rows = _has(datasets, "subset_rows")
    a, b = rows(1000, 100, 7), rows(1000, 400, 7)
    assert (np.diff(a) > 0).all() and len(a) == 100 and set(a) <= set(b)        # nested within one seed
    assert (rows(1000, 100, 7) == a).all() and not (rows(1000, 100, 8) == a).all()
    assert len(rows(50, 100, 0)) == 50                                         # never more than the parent


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_subsample_names_round_trip_and_name_a_real_dataset_kind(datasets):
    protocol = _need("protocol")
    assert protocol.subset_name("neurons_1m", 100_000) == "neurons_1m_sub100k"
    assert protocol.subset_name("p", 1_000_000) == "p_sub1m" and protocol.subset_name("p", 1234) == "p_sub1234"
    assert protocol.parse_subset_name("neurons_1m_sub100k") == ("neurons_1m", 100_000)
    assert protocol.parse_subset_name("p_sub1m") == ("p", 1_000_000) and protocol.parse_subset_name("pbmc3k") is None
    assert _has(datasets, "dataset_kind")("neurons_1m_sub100k") == "real-subsample"
    assert datasets.dataset_kind("synth_100k") == "synthetic" and datasets.dataset_kind("pbmc68k") == "real"
    assert datasets.dataset_kind("parse_pbmc_1m") == "real-subsample"


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_a_subsample_matches_its_parent_cell_for_cell_and_a_wrong_seed_does_not(datasets, tmp_path):
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(0.5, (80, 25)).astype(np.int32))
    X.sort_indices()
    parent = tmp_path / "parent.h5"
    datasets.write_10x_h5(parent, X.indptr, X.indices, X.data, 25, [f"c{i}" for i in range(80)],
                          [f"id{i}" for i in range(25)], [f"g{i}" for i in range(25)])
    out = tmp_path / "parent_sub20.h5"
    verify = _has(datasets, "verify_subset")
    datasets.subset_h5(parent, out, 20, seed=5)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["subset_of"] == "parent" and meta["subset_seed"] == 5 and meta["kind"] == "real-subsample"
    good = verify(parent, out, 20, 5)
    assert all(c["ok"] for c in good), good
    bad = verify(parent, out, 20, 6)
    assert not all(c["ok"] for c in bad)
    rows = datasets.subset_rows(80, 20, 5)
    Xs, bc, _, _ = datasets.read_h5_counts(out)
    assert (Xs.toarray() == X[rows].toarray()).all() and bc == [f"c{i}" for i in rows]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_manifest_never_fills_in_a_licence_and_labels_the_conversion(datasets, tmp_path):
    _write_mtx_dir(tmp_path / "src" / "pbmc3k" / "extracted" / "hg19", _counts(), GENES, BARCODES)
    datasets._mtx_dir_to_h5(tmp_path / "src" / "pbmc3k" / "extracted" / "hg19", tmp_path / "pbmc3k.h5", "hg19")
    m = _has(datasets, "build_manifest")(tmp_path, ["pbmc3k", "synth_25k", "parse_pbmc_1m"])
    assert all(r["licence"] is None for r in m.values())                      # stated by the provider, not by a script
    assert "scipy" in m["pbmc3k"]["conversion"] and len(m["pbmc3k"]["converted_sha256"]) == 64
    assert m["synth_25k"]["kind"] == "synthetic" and "supplement" in m["synth_25k"]["conversion"]
    assert m["parse_pbmc_1m"]["parent"] == "parse_pbmc" and m["parse_pbmc_1m"]["converted_sha256"] is None
    assert "pbmc3k.tar.gz" in m["pbmc3k"]["source_urls"]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"synthetic")
def test_the_dataset_listing_labels_synthetic_sets_as_a_supplement(datasets, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["datasets.py", "--list"])
    datasets.main()
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("synth_100k"))
    assert "supplement" in line.lower(), f"synthetic set not labelled as a supplement: {line}"
    assert "real" in next(ln for ln in out.splitlines() if ln.startswith("parse_pbmc_1m"))


# --------------------------------------------------------------------------- driver script and documents
@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"driver script")
def test_driver_script_has_the_pilot_controls_verification_and_environment_records():
    for needle in ("--thread-grid", "--pilot", "--controls", "--lean-scanpy", "--allow-incomplete", "--verify",
                   "dataset_manifest.json", "smt_siblings", "sessionInfo", "numba_threading_layer", "--mode pilot", "--tuning"):
        assert needle in SCRIPT, f"driver script has no {needle}"


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"not found|pilot")
def test_driver_script_dry_run_arguments_reach_the_planner():
    block = _shell_block(r"TOOL_ARGS=\(", r'\n\[\[ \$DRY -eq 1 \]\] && exit 0')
    assert "--mode pilot" in block and '--tuning "$OUT/tuning.json"' in block
    assert "--controls $CONTROLS" in block and "scanpy-lean" in block, "pilot, controls and the lean profile are wired in"


def test_driver_syntax_still_checks():
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "crest_paper_bench.sh")], capture_output=True).returncode == 0


def _doc(name):
    return (REPO / "docs" / name).read_text(encoding="utf-8")


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"bootstrap")
def test_docs_make_no_bootstrap_claim():
    hits = [f"{p.name}:{i}" for p in sorted((REPO / "docs").glob("*.md")) for i, ln in enumerate(p.read_text().splitlines(), 1)
            if "bootstrap" in ln.lower()]
    assert not hits, f"bootstrap interval mentioned (the harness computes none): {hits}"


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"pinning|endpoint")
def test_docs_describe_pinning_endpoints_and_tuning_as_the_harness_does():
    t = _doc("benchmarks.md")
    assert "every run" in t and "physical-core-first" in t, "pinning policy not described"
    assert "for the thread scan, the process is pinned" not in t
    assert "end-to-end process wall time" in t and "secondary endpoint" in t.lower(), "endpoints not described"
    assert "tuning pilot" in t.lower() and "default-settings control" in t
    assert "Quiet host" in t and "Required results" in t and "Memory." in t
    assert "PREREGISTRATION.md" in t


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"1\.42")
def test_docs_take_the_reference_versions_from_the_recorded_environment():
    t = _doc("benchmarks.md")
    bad = [ln for ln in t.splitlines() if re.search(r"DESeq2 1\.\d+", ln) and "unit-test fixtures" not in ln]
    assert not bad, f"a DESeq2 version is hard-coded as the benchmark's: {bad}"
    assert "R_versions" in SCRIPT and "Reference tools (R)" in SCRIPT


# --------------------------------------------------------------------------- pre-registration template
PREREG = REPO / "bench" / "paper" / "PREREGISTRATION.md"
PREREG_SECTIONS = ["Code and wheel", "Environments and lockfiles", "Datasets", "Endpoints", "Thread grid", "Tuning pilot",
                   "Repeats, interleaving, pinning", "Memory", "Statistics and headline rule", "Quiet host and re-run cap",
                   "Failures and required files", "Duration estimate", "Modules in the same run"]


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_preregistration_template_has_every_section_and_is_marked_draft():
    assert PREREG.exists(), "bench/paper/PREREGISTRATION.md is missing"
    t = PREREG.read_text(encoding="utf-8")
    heads = re.findall(r"^## \d+\. (.+)$", t, re.M)
    for s in PREREG_SECTIONS:
        assert any(h.startswith(s) for h in heads), f"section {s!r} is missing"
    assert "DRAFT" in t.splitlines()[0] and "FILL:" in t and "PENDING" in t
    assert "284" in t and "7.7 h" in t                                   # the published run the estimate starts from
    headline = t.split("Headline rule.", 1)[1].split("\n* ", 1)[0]
    assert "PENDING" in headline and "best-vs-best" in headline and "single-thread" in headline


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match=r"is missing")
def test_preregistration_variants_and_defaults_match_the_code():
    assert PREREG.exists(), "bench/paper/PREREGISTRATION.md is missing"
    t = PREREG.read_text(encoding="utf-8")
    protocol = _need("protocol")
    for tool in protocol.VARIANTS:
        for v in protocol.VARIANTS[tool]:
            assert f"`{v}`" in t, f"variant {v} of {tool} is not described"
    assert "1, 2, 4, 8, 16, 32, 64" in t and "5% faster" in t and "1 core or more" in t


def test_public_text_of_the_protocol_files_names_no_internal_item():
    files = [*(REPO / "bench" / "paper").glob("*.py"), *(REPO / "bench" / "paper").glob("*.md"), REPO / "docs" / "benchmarks.md",
             REPO / "scripts" / "crest_paper_bench.sh", REPO / "changes" / "bench-protocol-v2.md"]
    hits = [f"{p.name}:{i}: {ln.strip()[:80]}" for p in files if p.exists() for i, ln in enumerate(p.read_text().splitlines(), 1)
            if FORBIDDEN_PUBLIC.search(ln)]
    assert not hits, "\n".join(hits)
