"""rank_genes_groups: "rest" is every other cell.

With ``reference="rest"`` scanpy compares a group with every other cell, including the cells of
groups that are not selected through ``groups`` and cells with a missing label. The tests compare
CREST with scanpy on clustered data (real Leiden labels on PBMC 3k, planted clusters on synthetic
counts: with random labels the adjusted p-values are 1 in both tools and the comparison is
vacuous) and with an independent scipy computation, and pin the argument validation.

Two input paths, one tolerance table:

* ``identical``: scanpy receives CREST's own normalised values (``AnnData(bf.to_scipy())``), so
  both tools see the same float32 numbers. This is the tight comparison.
* ``lazy``: scanpy normalises the raw counts itself. CREST's lazy ``normalize_total`` differs from
  scanpy's float32 arithmetic by one ulp for many entries, CREST then has more exact ties and the
  Wilcoxon statistics differ slightly (measured maximum score difference 2.7e-4 on PBMC 3k), so the
  Wilcoxon tolerances of this path are looser (atol 5e-4 + rtol 1e-4, p-values atol 1e-3). The
  t-test is unaffected. The size of the tie effect depends on the data (the planted synthetic
  counts below reach 5.6e-4 for one gene), so the Wilcoxon ``lazy`` comparison is made on PBMC 3k
  only, which is what the 5e-4 figure was measured on.

The PBMC 3k tests run when ``CREST_TEST_PBMC3K`` points at the directory holding ``matrix.mtx``,
``barcodes.tsv`` and ``genes.tsv``; the Leiden labels come from ``tests/data/pbmc3k_leiden.npz``
(scanpy + leidenalg), not from CREST's own pipeline.
"""

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import crest

sp = pytest.importorskip("scipy.sparse")
stats = pytest.importorskip("scipy.stats")
sc = pytest.importorskip("scanpy")
pd = pytest.importorskip("pandas")
ad = pytest.importorskip("anndata")

baseline_fails = pytest.mark.baseline_fails
DATA = Path(__file__).parent / "data"
PBMC3K = os.environ.get("CREST_TEST_PBMC3K")

# ---- one tolerance table (atol, rtol) -------------------------------------------------------
TOL = {
    # (method, path): scores, log fold changes, p-values and adjusted p-values
    ("t-test", "identical"): dict(score=(1e-4, 1e-4), lfc=(1e-4, 0.0), p=(1e-5, 0.0)),
    ("t-test", "lazy"): dict(score=(1e-4, 1e-4), lfc=(1e-4, 0.0), p=(1e-5, 0.0)),
    ("wilcoxon", "identical"): dict(score=(1e-5, 0.0), lfc=(1e-4, 0.0), p=(1e-5, 0.0)),
    ("wilcoxon", "lazy"): dict(score=(5e-4, 1e-4), lfc=(1e-4, 0.0), p=(1e-3, 0.0)),
}
SCIPY_ATOL = 1e-8          # CREST against an independent float64 scipy computation on the same values
N_NULL_LABELS = 300        # PBMC 3k; the synthetic data nulls 5 % of its cells
REST_DEFECT = pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match=r"(t-test|wilcoxon)/(identical|lazy)/")
# groups="all" without null labels was always right (every cell belongs to some group): a pin
GROUP_CASES = [
    pytest.param("all", id="all"),
    pytest.param(["0", "1"], id="subset", marks=REST_DEFECT),
    pytest.param(["2"], id="single", marks=REST_DEFECT),
    pytest.param("all+null", id="null-labels", marks=REST_DEFECT),
]
MIN_ADJUSTED_SIGNIFICANT = 50   # the comparison must see genes with adjusted p < 0.05


# ---- data ---------------------------------------------------------------------------------
def planted(n_cells=900, n_genes=300, n_clusters=5, seed=0):
    """Counts with cluster-specific programmes; the labels are the planted clusters."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, n_genes)
    labels = rng.choice(n_clusters, n_cells, p=np.array([0.35, 0.25, 0.2, 0.12, 0.08]))
    prog = np.ones((n_clusters, n_genes))
    for c in range(n_clusters):
        up = rng.choice(n_genes, 40, replace=False)
        prog[c, up] *= rng.uniform(2, 6, 40)
    lib = rng.lognormal(7.0, 0.3, n_cells)
    mu = prog[labels] * base[None, :]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    return sp.csr_matrix(counts), [f"G{i}" for i in range(n_genes)], labels.astype(str)


def with_nulls(labels, n, seed=1):
    out = np.array(labels, dtype=object)
    out[np.random.default_rng(seed).choice(len(out), n, replace=False)] = None
    return out


def crest_frame(X, names, labels):
    bf = crest.BioFrame.from_scipy(X, var=pl.DataFrame({"gene_name": names}))
    bf.obs = bf.obs.with_columns(pl.Series("g", list(labels), dtype=pl.Utf8))
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    return bf


def scanpy_input(bf, X, names, labels, path):
    if path == "identical":
        a = ad.AnnData(bf.to_scipy().tocsr(), var=pd.DataFrame(index=bf.var_names))
        a.uns["log1p"] = {"base": None}
    else:
        a = ad.AnnData(X.copy(), var=pd.DataFrame(index=bf.var_names))
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    a.obs["g"] = pd.Categorical(pd.Series(list(labels), dtype=object).where(pd.notna(pd.Series(list(labels), dtype=object)), np.nan))
    return a


def compare(res, a, method, groups, path):
    """CREST's table against scanpy's, group by group, within the table of tolerances."""
    tol = TOL[(method, path)]
    sc.tl.rank_genes_groups(a, "g", method=method, groups="all" if groups == "all" else groups,
                            reference="rest", n_genes=a.n_vars, use_raw=False)
    rg = a.uns["rank_genes_groups"]
    got_groups = list(dict.fromkeys(res["group"].to_list()))
    assert got_groups == list(rg["names"].dtype.names), (got_groups, rg["names"].dtype.names)
    n_sig = 0
    for grp in got_groups:
        sub = res.filter(pl.col("group") == grp)
        pos = {n: i for i, n in enumerate(rg["names"][grp])}
        idx = np.array([pos[n] for n in sub["names"]])
        ref = {k: np.asarray(rg[k][grp])[idx] for k in ("scores", "logfoldchanges", "pvals", "pvals_adj")}
        ref["scores"] = np.nan_to_num(ref["scores"], nan=0.0)
        ref["pvals"] = np.nan_to_num(ref["pvals"], nan=1.0)
        ref["pvals_adj"] = np.nan_to_num(ref["pvals_adj"], nan=1.0)
        for col, key, t in (("scores", "scores", tol["score"]), ("logfoldchanges", "logfoldchanges", tol["lfc"]),
                            ("pvals", "pvals", tol["p"]), ("pvals_adj", "pvals_adj", tol["p"])):
            np.testing.assert_allclose(sub[col].to_numpy(), ref[key], atol=t[0], rtol=t[1],
                                       err_msg=f"{method}/{path}/{groups}: group {grp}, {col}")
        n_sig += int((ref["pvals_adj"] < 0.05).sum())
    return n_sig


def select(groups_case):
    return ("all", True) if groups_case == "all+null" else (groups_case, False)


# ---- synthetic data with planted clusters (always runs) -----------------------------------------
@pytest.fixture(scope="module")
def synth():
    return planted()


@pytest.mark.parametrize("method, path", [("t-test", "identical"), ("t-test", "lazy"), ("wilcoxon", "identical")])
@pytest.mark.parametrize("case", GROUP_CASES)
def test_planted_clusters_match_scanpy(synth, method, path, case):
    X, names, labels = synth
    groups, nulls = select(case)
    lab = with_nulls(labels, int(0.05 * len(labels))) if nulls else labels
    bf = crest_frame(X, names, lab)
    a = scanpy_input(bf, X, names, lab, path)
    res = crest.tl.rank_genes_groups(bf, "g", method=method, groups=groups, n_genes=None)
    n_sig = compare(res, a, method, groups, path)
    assert n_sig >= MIN_ADJUSTED_SIGNIFICANT, f"only {n_sig} genes with adjusted p < 0.05: the comparison is vacuous"


# ---- independent scipy computation (no scanpy involved) ------------------------------------------
def scipy_reference(dense, labels, group, method):
    lab = np.array([("" if x is None else x) for x in labels], dtype=object)
    isg = lab == group
    n1, N = isg.sum(), len(lab)
    if method == "t-test":
        return stats.ttest_ind(dense[isg], dense[~isg], equal_var=False).statistic
    ranks = np.apply_along_axis(stats.rankdata, 0, dense)
    return (ranks[isg].sum(0) - n1 * (N + 1) / 2) / np.sqrt(n1 * (N - n1) * (N + 1) / 12)


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="scores differ from a scipy computation")
@pytest.mark.parametrize("method", ["t-test", "wilcoxon"])
@pytest.mark.parametrize("case", ["subset", "single", "null"])
def test_rest_is_every_other_cell_scipy(synth, method, case):
    X, names, labels = synth
    lab = with_nulls(labels, 60) if case == "null" else labels
    groups = {"subset": ["0", "1"], "single": ["2"], "null": "all"}[case]
    bf = crest_frame(X, names, lab)
    dense = bf.to_scipy().toarray().astype(np.float64)
    res = crest.tl.rank_genes_groups(bf, "g", method=method, groups=groups, n_genes=None)
    for grp in dict.fromkeys(res["group"].to_list()):
        sub = res.filter(pl.col("group") == grp)
        order = [names.index(n) for n in sub["names"]]
        ref = np.nan_to_num(scipy_reference(dense, lab, grp, method)[order])
        got = sub["scores"].to_numpy()
        if not np.allclose(got, ref, atol=SCIPY_ATOL, rtol=1e-6):
            raise AssertionError(f"{method}/{case}: group {grp}: scores differ from a scipy computation, "
                                 f"max |diff| {np.abs(got - ref).max():.3g}")


def test_all_groups_without_nulls_is_unchanged(synth):
    """The pin: groups='all' and no null labels never changed (every cell is in some group)."""
    X, names, labels = synth
    bf = crest_frame(X, names, labels)
    dense = bf.to_scipy().toarray().astype(np.float64)
    res = crest.tl.rank_genes_groups(bf, "g", method="t-test", n_genes=None)
    for grp in sorted(set(labels)):
        sub = res.filter(pl.col("group") == grp)
        ref = np.nan_to_num(scipy_reference(dense, labels, grp, "t-test")[[names.index(n) for n in sub["names"]]])
        np.testing.assert_allclose(sub["scores"].to_numpy(), ref, atol=SCIPY_ATOL, rtol=1e-6)


# ---- argument validation (unknown names, tiny groups, the group argument) -----------------------------------------
def _frame(labels, n_genes=30, seed=2):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(2.0, (len(labels), n_genes)).astype(np.float32))
    return crest_frame(X, [f"G{i}" for i in range(n_genes)], labels)


def raises_value_error(fn, match):
    try:
        out = fn()
    except ValueError as e:
        assert match in str(e), f"ValueError message {str(e)!r} does not contain {match!r}"
        return
    raise AssertionError(f"expected ValueError containing {match!r}, the call returned a table with {len(out)} rows")


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="expected ValueError")
def test_unknown_group_name_raises():
    bf = _frame(np.array(["a", "b", "c"] * 20))
    raises_value_error(lambda: crest.tl.rank_genes_groups(bf, "g", groups=["nonexistent"]), "nonexistent")
    raises_value_error(lambda: crest.tl.rank_genes_groups(bf, "g", groups=["a", "zz"]), "zz")
    raises_value_error(lambda: crest.tl.rank_genes_groups(bf, "g", groups="zz"), "zz")


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="expected ValueError")
def test_single_cell_group_raises():
    labels = np.array(["a"] * 20 + ["b"] * 20 + ["single"])
    bf = _frame(labels)
    raises_value_error(lambda: crest.tl.rank_genes_groups(bf, "g"), "single")
    raises_value_error(lambda: crest.tl.rank_genes_groups(bf, "g", method="wilcoxon", groups=["single", "a"]), "single")
    # the same group is fine as part of "rest" when it is not selected
    res = crest.tl.rank_genes_groups(bf, "g", groups=["a", "b"], n_genes=5)
    assert set(res["group"].to_list()) == {"a", "b"}


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="expected ValueError")
def test_group_holding_every_cell_raises():
    bf = _frame(np.array(["a"] * 30))
    raises_value_error(lambda: crest.tl.rank_genes_groups(bf, "g"), "every cell")


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="bare string")
def test_bare_string_is_one_group():
    labels = np.array((["1"] * 15 + ["0"] * 15 + ["10"] * 15))
    bf = _frame(labels)
    for fn in ("t-test", "wilcoxon"):
        res = crest.tl.rank_genes_groups(bf, "g", method=fn, groups="10", n_genes=5)
        assert set(res["group"].to_list()) == {"10"}, "a bare string was split into characters"
        assert set(crest.tl.rank_genes_groups(bf, "g", method=fn, groups=["10"], n_genes=5)["group"].to_list()) == {"10"}


@baseline_fails(ref="v0.4.0", raises=AssertionError, match="all-zero")
def test_single_group_does_not_return_zero_scores(synth):
    X, names, labels = synth
    bf = crest_frame(X, names, labels)
    res = crest.tl.rank_genes_groups(bf, "g", groups=["2"], n_genes=None)
    assert res["group"].unique().to_list() == ["2"]
    assert np.abs(res["scores"].to_numpy()).max() > 1.0, "all-zero scores for a single selected group"
    assert np.isfinite(res["logfoldchanges"].to_numpy()).all()


def test_group_order_and_result_are_stored(synth):
    X, names, labels = synth
    bf = crest_frame(X, names, labels)
    res = crest.tl.rank_genes_groups(bf, "g", groups=["3", "1"], n_genes=7)
    assert res["group"].unique(maintain_order=True).to_list() == ["1", "3"]
    assert bf.uns["rank_genes_groups"]["groups"] == ["1", "3"] and len(res) == 14
    with pytest.raises(ValueError, match="method"):
        crest.tl.rank_genes_groups(bf, "g", method="t-test_overestim_var")


UNSUPPORTED = ["reference", "corr_method", "pts", "t-test_overestim_var"]


@baseline_fails(ref="rc0", raises=AssertionError, match="not listed")
def test_unsupported_scanpy_arguments_are_documented():
    text = (Path(__file__).resolve().parents[1] / "docs" / "quickstart.md").read_text(encoding="utf-8")
    missing = [u for u in UNSUPPORTED if u not in text]
    assert not missing, f"not listed as unsupported in docs/quickstart.md: {missing}"
    doc = crest.tl.rank_genes_groups.__doc__
    assert all(u in doc for u in UNSUPPORTED[:3]), "the docstring does not list the unsupported arguments"


# ---- real Leiden labels on PBMC 3k ------------------------------------------------------------
@pytest.fixture(scope="module")
def pbmc():
    if not PBMC3K:
        pytest.skip("CREST_TEST_PBMC3K not set")
    fx = np.load(DATA / "pbmc3k_leiden.npz")
    bf = crest.read_10x_mtx(PBMC3K)
    raw = bf.to_scipy().tocsr()
    bf = crest.pp.filter_cells(bf, min_genes=200)
    bf = crest.pp.filter_genes(bf, min_cells=3)
    keep = fx["kept_cell_index"]
    assert bf.n_obs == len(keep) == len(fx["leiden"]), (bf.n_obs, len(keep))
    # the same cells and genes as scanpy's filters (scanpy's own pipeline made the labels)
    gene_keep = np.flatnonzero(np.asarray((raw[keep] > 0).sum(0)).ravel() >= 3)
    X = raw[keep][:, gene_keep].tocsr()
    labels = fx["leiden"].astype(str)
    assert len(set(labels)) == 9 and X.shape == bf.shape
    seen, names = {}, []
    for n in bf.var_names:  # six gene symbols occur twice: make them unique so that results can be matched by name
        k = seen.get(n, 0)
        seen[n] = k + 1
        names.append(n if k == 0 else f"{n}-{k}")
    return X, names, labels


@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
@pytest.mark.parametrize("path", ["identical", "lazy"])
@pytest.mark.parametrize("method", ["t-test", "wilcoxon"])
@pytest.mark.parametrize("case", GROUP_CASES)
def test_pbmc3k_leiden_labels_match_scanpy(pbmc, method, path, case):
    X, names, labels = pbmc
    groups, nulls = select(case)
    lab = with_nulls(labels, N_NULL_LABELS, seed=5) if nulls else labels
    bf = crest_frame(X, names, lab)
    a = scanpy_input(bf, X, names, lab, path)
    res = crest.tl.rank_genes_groups(bf, "g", method=method, groups=groups, n_genes=None)
    n_sig = compare(res, a, method, groups, path)
    assert n_sig >= MIN_ADJUSTED_SIGNIFICANT, f"only {n_sig} genes with adjusted p < 0.05: the comparison is vacuous"
