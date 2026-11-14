"""Wilcoxon rank sums with signed values, the lazy-normalisation tolerance, and the wording of
the reproducibility and parity statements.

* The rank-sum kernel ranks only the non-zero values and treats all implicit zeros as one tie block.
  The block sits between the negative and the positive values, so with signed input its rank is
  ``n_neg + (n_zero + 1) / 2`` and only the positive values are offset by the zero count.
* Lazy path: scanpy normalises the raw counts itself in float32; CREST's lazy ``normalize_total``
  differs from that by one ulp for many entries, so CREST has more exact ties and the Wilcoxon scores
  differ slightly (measured maximum 2.6e-4 on PBMC 3k). That path is pinned at ``LAZY_SCORE_ATOL``;
  with identical input values the tools agree to ``IDENTICAL_SCORE_ATOL``.
* The thread-count statement is limited to what is checked here: neighbour sets, and floating-point
  statistics that agree to ~1e-14 relative.

The PBMC 3k test runs when ``CREST_TEST_PBMC3K`` points at the directory holding ``matrix.mtx``,
``barcodes.tsv`` and ``genes.tsv``.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import crest
from crest import crest as _native

sp = pytest.importorskip("scipy.sparse")
stats = pytest.importorskip("scipy.stats")

baseline_fails = pytest.mark.baseline_fails
REPO = Path(__file__).resolve().parents[1]
PBMC3K = os.environ.get("CREST_TEST_PBMC3K")

# ---- named tolerances ---------------------------------------------------------------------------
RANK_ATOL = 1e-9             # rank sums are sums of half-integers, exact in float64
SIGNED_SCORE_ATOL = 1e-6     # signed input, identical values in both tools (the non-negative control gives 0.0)
IDENTICAL_SCORE_ATOL = 1e-6  # non-negative input, identical values in both tools (measured 1.2e-7 on PBMC 3k)
LAZY_SCORE_ATOL = 5e-4       # scanpy normalises the raw counts itself (measured 2.6e-4 on PBMC 3k)
LAZY_SCORE_RTOL = 1e-4
LAZY_P_ATOL = 1e-3
N_RANDOM_CASES = 400
THREAD_STATS_RTOL = 1e-12    # gene sums and PCA variance between thread counts (observed <= 4.2e-15)
THREAD_COUNTS = (1, 7)


# ======================================================================================
# signed values
# ======================================================================================
RANK_MSG = "rank sums differ from scipy.stats.rankdata"


def rank_sums(dense, grp, n_groups):
    """The native kernel on the non-zero entries of a dense (cells, genes) matrix."""
    r, g = np.nonzero(dense)
    sizes = np.bincount(grp, minlength=n_groups).astype(np.uint64)
    return _native.wilcoxon_rank_sums(g.astype(np.uint32), dense[r, g].astype(np.float32), grp[r].astype(np.uint32),
                                      dense.shape[1], sizes)


def reference_rank_sums(dense, grp, n_groups):
    ranks = np.apply_along_axis(stats.rankdata, 0, dense)
    rs = np.vstack([ranks[grp == k].sum(0) for k in range(n_groups)])
    ties = np.array([(np.unique(c, return_counts=True)[1].astype(float) ** 3 - np.unique(c, return_counts=True)[1]).sum()
                     for c in dense.T])
    return rs, ties


def check_against_scipy(dense, grp, n_groups, what):
    rs, ties = rank_sums(dense, grp, n_groups)
    ref_rs, ref_ties = reference_rank_sums(dense, grp, n_groups)
    if not (np.allclose(rs, ref_rs, atol=RANK_ATOL, rtol=0) and np.allclose(ties, ref_ties, atol=RANK_ATOL, rtol=0)):
        raise AssertionError(f"{what}: {RANK_MSG}: kernel {rs.ravel()[:6]} vs {ref_rs.ravel()[:6]}")


@baseline_fails(ref="v0.4.0", raises=AssertionError, match=RANK_MSG)
def test_six_cells_with_negative_values():
    dense = np.array([[-3.0], [0.0], [0.0], [2.0], [0.0], [-1.0]], dtype=np.float32)
    grp = np.array([0, 0, 0, 1, 1, 1])
    rs, ties = rank_sums(dense, grp, 2)
    assert rs.tolist() == [[9.0], [12.0]], f"{RANK_MSG}: {rs.tolist()}"
    assert ties.tolist() == [24.0]


@baseline_fails(ref="v0.4.0", raises=AssertionError, match=RANK_MSG)
def test_random_signed_cases_match_scipy_rankdata():
    rng = np.random.default_rng(11)
    saw_negative = saw_tie = 0
    for case in range(N_RANDOM_CASES):
        n, n_groups, n_genes = int(rng.integers(3, 60)), int(rng.integers(2, 5)), int(rng.integers(1, 4))
        mag = rng.integers(1, 5, (n, n_genes)).astype(np.float32)             # small integers: many ties
        sign = np.where(rng.random((n, n_genes)) < rng.random(), -1.0, 1.0)  # a random share of negatives
        dense = np.where(rng.random((n, n_genes)) < 0.5, 0.0, mag * sign).astype(np.float32)
        grp = rng.integers(0, n_groups, n)
        saw_negative += int((dense < 0).any())
        saw_tie += int(any(len(np.unique(c[c != 0])) < (c != 0).sum() for c in dense.T))
        check_against_scipy(dense, grp, n_groups, f"case {case}")
    assert saw_negative > N_RANDOM_CASES // 2 and saw_tie > N_RANDOM_CASES // 2


def test_non_negative_cases_match_scipy_rankdata():
    """The pin: non-negative input was always right and must stay so."""
    rng = np.random.default_rng(12)
    for case in range(N_RANDOM_CASES):
        n, n_groups = int(rng.integers(3, 60)), int(rng.integers(2, 5))
        dense = np.where(rng.random((n, 2)) < 0.6, 0.0, rng.integers(1, 5, (n, 2))).astype(np.float32)
        check_against_scipy(dense, rng.integers(0, n_groups, n), n_groups, f"case {case}")


def counts_frame(n_cells=600, n_genes=200, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, n_genes)
    lab = rng.choice(4, n_cells, p=[0.4, 0.3, 0.2, 0.1])
    prog = np.ones((4, n_genes))
    for c in range(4):
        prog[c, rng.choice(n_genes, 30, replace=False)] *= rng.uniform(2, 6, 30)
    mu = prog[lab] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * rng.lognormal(7.0, 0.3, n_cells)[:, None]
    return sp.csr_matrix(rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)), lab.astype(str)


def scanpy_scores(X, labels, method="wilcoxon"):
    sc = pytest.importorskip("scanpy")
    import anndata as ad
    import pandas as pd
    a = ad.AnnData(X.copy(), var=pd.DataFrame(index=[f"G{i}" for i in range(X.shape[1])]))
    a.obs["g"] = pd.Categorical(labels)
    a.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(a, "g", method=method, reference="rest", n_genes=a.n_vars, use_raw=False)
    return a.uns["rank_genes_groups"]


def crest_scores(X, labels, method="wilcoxon"):
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": [f"G{i}" for i in range(X.shape[1])]}))
    bf.obs = bf.obs.with_columns(pl.Series("g", list(labels)))
    return crest.tl.rank_genes_groups(bf, "g", method=method, n_genes=None)


def max_score_difference(res, rg):
    worst = 0.0
    for grp in rg["names"].dtype.names:
        sub = res.filter(pl.col("group") == grp)
        pos = {n: i for i, n in enumerate(rg["names"][grp])}
        ref = np.nan_to_num(np.asarray(rg["scores"][grp])[[pos[n] for n in sub["names"]]], nan=0.0)
        worst = max(worst, float(np.abs(sub["scores"].to_numpy() - ref).max()))
    return worst


def test_non_negative_control_matches_scanpy_with_identical_input():
    X, labels = counts_frame()
    X = X.copy()
    X.data = np.log1p(X.data)
    assert max_score_difference(crest_scores(X, labels), scanpy_scores(X, labels)) <= SIGNED_SCORE_ATOL


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="signed input")
def test_signed_input_matches_scanpy_with_identical_input():
    """Subtracting a constant from every stored value leaves zeros at zero and makes many values negative."""
    X, labels = counts_frame()
    X = X.copy()
    X.data = (np.log1p(X.data) - 1.2).astype(np.float32)
    assert (X.data < 0).mean() > 0.3
    worst = max_score_difference(crest_scores(X, labels), scanpy_scores(X, labels))
    assert worst <= SIGNED_SCORE_ATOL, f"signed input: max |score difference| to scanpy {worst:.3g}"


# ======================================================================================
# the lazy-normalisation path
# ======================================================================================
@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
def test_wilcoxon_lazy_path_and_identical_input_on_pbmc3k():
    sc = pytest.importorskip("scanpy")
    import anndata as ad
    import pandas as pd
    bf = crest.read_10x_mtx(PBMC3K)
    raw = bf.to_scipy().tocsr()
    bf = crest.pp.filter_cells(bf, min_genes=200)
    bf = crest.pp.filter_genes(bf, min_cells=3)
    keep = np.flatnonzero(np.asarray((raw > 0).sum(1)).ravel() >= 200)
    X = raw[keep]
    X = X[:, np.flatnonzero(np.asarray((X > 0).sum(0)).ravel() >= 3)].tocsr()
    assert X.shape == bf.shape
    names, seen = [], {}
    for n in bf.var_names:
        k = seen.get(n, 0)
        seen[n] = k + 1
        names.append(n if k == 0 else f"{n}-{k}")
    labels = np.random.default_rng(3).integers(0, 5, X.shape[0]).astype(str)
    cb = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    cb.obs = cb.obs.with_columns(pl.Series("g", list(labels)))
    crest.pp.normalize_total(cb, 1e4)
    crest.pp.log1p(cb)
    res = crest.tl.rank_genes_groups(cb, "g", method="wilcoxon", n_genes=None)

    def reference(a):
        a.obs["g"] = pd.Categorical(labels)
        sc.tl.rank_genes_groups(a, "g", method="wilcoxon", reference="rest", n_genes=a.n_vars, use_raw=False)
        return a.uns["rank_genes_groups"]

    ident = ad.AnnData(cb.to_scipy().tocsr(), var=pd.DataFrame(index=names))
    ident.uns["log1p"] = {"base": None}
    assert max_score_difference(res, reference(ident)) <= IDENTICAL_SCORE_ATOL
    lazy = ad.AnnData(X.copy(), var=pd.DataFrame(index=names))
    sc.pp.normalize_total(lazy, target_sum=1e4)
    sc.pp.log1p(lazy)
    rg = reference(lazy)
    for grp in rg["names"].dtype.names:
        sub = res.filter(pl.col("group") == grp)
        pos = {n: i for i, n in enumerate(rg["names"][grp])}
        idx = [pos[n] for n in sub["names"]]
        np.testing.assert_allclose(sub["scores"].to_numpy(), np.nan_to_num(np.asarray(rg["scores"][grp])[idx]),
                                   atol=LAZY_SCORE_ATOL, rtol=LAZY_SCORE_RTOL)
        np.testing.assert_allclose(sub["pvals"].to_numpy(), np.asarray(rg["pvals"][grp])[idx], atol=LAZY_P_ATOL, rtol=0)


@baseline_fails(ref="rc0", raises=AssertionError, match="parity statement")
def test_readme_and_benchmark_docs_state_the_measured_wilcoxon_agreement():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    bench = (REPO / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    for name, text in (("README.md", readme), ("docs/benchmarks.md", bench)):
        rows = [ln for ln in text.splitlines() if re.search(r"Wilcoxon", ln) and re.search(r"scores? within", ln)]
        assert rows, f"parity statement: no Wilcoxon agreement row in {name}"
        assert all("3e-4" in r for r in rows), f"parity statement in {name} does not state the lazy-path maximum 3e-4: {rows}"
        assert not any(re.search(r"scores within 1e-4 \(1e-7", r) for r in rows), f"parity statement in {name}: old 1e-4 claim"


# ======================================================================================
# thread counts
# ======================================================================================
THREAD_SCRIPT = textwrap.dedent('''
    import json, sys
    import numpy as np, polars as pl, scipy.sparse as sp
    import crest
    rng = np.random.default_rng(0)
    n, g = 3000, 500
    base = rng.gamma(0.4, 1.0, g)
    lab = rng.integers(0, 6, n)
    prog = np.ones((6, g))
    for c in range(6):
        prog[c, rng.choice(g, 40, replace=False)] *= rng.uniform(2, 6, 40)
    mu = prog[lab] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * rng.lognormal(7.0, 0.3, n)[:, None]
    X = sp.csr_matrix(rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32))
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": [f"G{i}" for i in range(g)]}))
    crest.pp.normalize_total(bf, 1e4); crest.pp.log1p(bf)
    st = crest.pp.gene_stats(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=200)
    crest.pp.scale(bf); crest.tl.pca(bf, n_comps=20); crest.pp.neighbors(bf, n_neighbors=15)
    crest.tl.leiden(bf)
    nb = bf.uns["neighbors"]
    out = dict(stats=st.tolist(), variance=np.asarray(bf.uns["pca"]["variance"]).tolist(), idx=np.asarray(nb["indices"]).tolist(),
               dist=np.asarray(nb["distances"]).tolist(), leiden=bf.obs["leiden"].to_list())
    json.dump(out, open(sys.argv[1], "w"))
''')


def _run_with_threads(tmp_path, threads):
    out = tmp_path / f"out_{threads}.json"
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads), "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "-c", THREAD_SCRIPT, str(out)], env=env, check=True, capture_output=True, timeout=600)
    return json.loads(out.read_text())


def test_thread_count_does_not_change_neighbours_labels_or_statistics(tmp_path):
    a, b = (_run_with_threads(tmp_path, t) for t in THREAD_COUNTS)
    assert a["idx"] == b["idx"], "kNN neighbour sets differ between thread counts"
    assert a["leiden"] == b["leiden"], "Leiden labels differ between thread counts"
    np.testing.assert_allclose(np.array(a["stats"]), np.array(b["stats"]), rtol=THREAD_STATS_RTOL, atol=0)
    np.testing.assert_allclose(a["variance"], b["variance"], rtol=THREAD_STATS_RTOL, atol=0)
    np.testing.assert_allclose(np.array(a["dist"]), np.array(b["dist"]), rtol=1e-5)


@baseline_fails(ref="rc0", raises=AssertionError, match="reproducibility statement")
def test_reproducibility_statement_is_limited_to_what_holds():
    text = (REPO / "docs" / "concepts.md").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", text)
    assert not re.search(r"identical for any number of threads", flat), "reproducibility statement: still claims identity for any thread count"
    sect = flat[flat.index("## Reproducibility"):]
    for needle in ("neighbour sets", "labels", "1e-14", "3e-8"):
        assert needle in sect, f"reproducibility statement: '{needle}' missing from the Reproducibility section"
