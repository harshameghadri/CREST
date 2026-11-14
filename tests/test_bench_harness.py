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


def test_driver_default_main_threads_scan_below_the_maximum():
    block = _shell_block(r"clip\(\) \{", r"\n  esac\nfi\n")
    for ncpu, want in ((64, "8 16 32 64"), (16, "8 16"), (8, "8")):
        out = _bash(f'NCPU={ncpu}; TIER=standard; MAIN_THREADS=""; {block}\necho "$MAIN_THREADS"')
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == want, f"NCPU={ncpu}"


def test_run_plan_runs_every_main_thread_count_for_both_tools(run_all, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    a = Namespace(main_threads=[8, 16, 32, 64], ooc_threads=None, profile="core", datasets=["pbmc3k"],
                  tools=["crest", "scanpy"], scanpy_max_cells=0, no_pin=False, thread_dataset=None,
                  threads_scan=None, module_repeats=None, repeats=2, seed=0)
    jobs = run_all.plan(a)
    for tool in ("crest", "scanpy"):
        assert sorted({j["threads"] for j in jobs if j["tool"] == tool}) == [8, 16, 32, 64]
    assert all(j["pin"] is (j["threads"] < 64) for j in jobs)     # pinned unless all logical cores are used


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
    r = _bash(f'{func}\nrecord_r_versions "{env_dir}"', PATH=f"{empty}:/usr/bin:/bin")
    assert r.returncode == 0, r.stderr
    assert json.loads((env_dir / "R_versions.json").read_text())["DESeq2"] is None
    assert "not available" in (env_dir / "R_versions.txt").read_text()


def test_driver_script_passes_bash_syntax_check():
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "crest_paper_bench.sh")]).returncode == 0
