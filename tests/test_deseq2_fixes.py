"""DESeq2 port: refit dispersions, trend fit on high counts, substituted trend flag,
group-level error handling, level-order warning / contrast label, duplicate coefficient names.

Tests marked ``baseline_fails`` fail on the stated released build with the declared
exception and pass on this tree. Unmarked tests pin behaviour that is already correct.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import crest
from crest.deseq2 import DESeq2, model_matrix

sp = pytest.importorskip("scipy.sparse")

# --- tolerances (named constants) -------------------------------------------------
REFIT_DISP_LOW, REFIT_DISP_HIGH = 0.05, 0.5      # refit dispersion of the two outlier genes (MAP, not the 1e-7 floor)
TREND_A_RTOL = 1e-4                              # parametric trend coefficient a against R
TREND_B_RTOL = 1e-3                              # parametric trend coefficient b against R
DISPFIT_RTOL = 1e-4                              # fitted dispersion trend against R's
R_TREND_VERY_LARGE_COUNTS = (0.0450552718, 6704.217044)   # R 1.42, 800 genes, 4 vs 4, counts up to 1e8
GLM_SCALE_RTOL = 1e-6                            # trend coefficients invariant to rescaling the means (b scales with them)
DE_MEDIAN_LFC_MIN = 1.5                          # median log2FC of the 4x genes in the correct column (true value 2)


def _must_raise(exc, fn, match):
    """Call ``fn``; fail with an AssertionError if it does not raise ``exc`` matching ``match``."""
    try:
        fn()
    except exc as e:
        import re
        assert re.search(match, str(e)), f"raised {type(e).__name__}({e}) which does not match {match!r}"
        return e
    raise AssertionError(f"did not raise {exc.__name__}")


def _simulate(n_genes, n_per, seed, mean_log2=(5, 2), disp=lambda mu: 4 / mu + 0.1, sf_sd=0.25):
    rng = np.random.default_rng(seed)
    m = sum(n_per)
    cond = np.repeat([f"c{i}" for i in range(len(n_per))], n_per)
    mu0 = np.clip(2 ** rng.normal(mean_log2[0], mean_log2[1], n_genes), 0.05, 1e7)
    lfc = np.where(rng.random(n_genes) < 0.15, rng.normal(0, 1.0, n_genes), 0)
    sf = 2 ** rng.normal(0, sf_sd, m)
    sf /= np.exp(np.mean(np.log(sf)))
    mu = mu0[:, None] * sf[None, :] * (2 ** (lfc[:, None] * (cond == cond[-1])[None, :]))
    size = 1 / disp(mu0)[:, None]
    return rng.negative_binomial(size, size / (size + mu)).astype(np.int64), cond


def _fit(cts, cond, **kw):
    return DESeq2(cts.T.astype(float), pl.DataFrame({"condition": cond}), "~ condition", **kw)


# --------------------------------------------------------------------------- outlier-refit dispersion
_BASE = np.array([2, 1, 3, 2, 2, 1, 3, 2, 2, 3, 1, 2, 2, 2, 3, 1])


def _outlier_case(seed):
    cts, cond = _simulate(800, (8, 8), seed, disp=lambda mu: 0.04 + 0 * mu)
    rng = np.random.default_rng(seed * 7 + 1)
    for g in (0, 1):
        row = rng.permutation(_BASE).copy()
        row[rng.integers(0, 16)] = 60
        cts[g] = row
    return cts, cond


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="refit dispersion")
@pytest.mark.parametrize("seed", [41, 42, 43])
def test_refit_genes_get_map_dispersion(seed):
    cts, cond = _outlier_case(seed)
    f = _fit(cts, cond, quiet=True).fit
    assert sorted(np.flatnonzero(f["replace"]).tolist()) == [0, 1]
    d = f["dispersion"][[0, 1]]
    assert np.all((d > REFIT_DISP_LOW) & (d < REFIT_DISP_HIGH)), f"refit dispersion {d} outside the MAP range"


def test_all_genes_at_dispersion_floor_keeps_floor_shortcut():
    """First pass: when every gene-wise dispersion is at the floor, DESeq2 sets them to 10 x minDisp."""
    cts = np.tile(np.array([10, 12, 11, 10, 12, 11]), (50, 1)) * (1 + np.arange(50)[:, None] % 3)
    f = _fit(cts, np.repeat(["a", "b"], 3), quiet=True).fit
    assert np.allclose(f["dispersion"], 1e-7)
    assert f["fitType"] == "mean"


# --------------------------------------------------------------------------- trend fit on high counts
def _very_large_counts():
    cts, cond = _simulate(800, (4, 4), 36, mean_log2=(26, 2), disp=lambda mu: 0.05 + 0 * mu)
    assert cts.max() > 2 ** 24
    return cts, cond


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="fitType")
def test_parametric_trend_fits_on_very_large_counts():
    cts, cond = _very_large_counts()
    f = _fit(cts, cond, quiet=True).fit
    assert f["fitType"] == "parametric", f"fitType {f['fitType']!r}, messages {f['messages']}"
    assert f["fit_substituted"] is False
    a_r, b_r = R_TREND_VERY_LARGE_COUNTS
    a, b = f["trend"]
    assert abs(a / a_r - 1) < TREND_A_RTOL, (a, a_r)
    assert abs(b / b_r - 1) < TREND_B_RTOL, (b, b_r)
    fit_r = a_r + b_r / f["baseMean"]
    ok = f["baseMean"] > 0
    assert np.max(np.abs(f["dispFit"][ok] / fit_r[ok] - 1)) < DISPFIT_RTOL


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="fitType")
def test_parametric_trend_is_invariant_to_rescaling_counts():
    """Multiplying every count by 1e5 must not turn a fit that works into a failure."""
    cts, cond = _simulate(800, (4, 4), 5, mean_log2=(8, 2), disp=lambda mu: 0.05 + 4 / mu)
    small = _fit(cts, cond, quiet=True).fit
    big = _fit(cts * 100_000, cond, quiet=True).fit
    assert small["fitType"] == "parametric"
    assert big["fitType"] == "parametric", f"fitType {big['fitType']!r}"


# --------------------------------------------------------------------------- substituted trend
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="automatically substituted")
def test_substituted_trend_warns_and_flags():
    cts, cond = _simulate(120, (3, 3), 62, mean_log2=(2, 1.5))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        ds = _fit(cts, cond)
    msgs = [str(w.message) for w in wl if issubclass(w.category, UserWarning)]
    assert any("a local regression fit was automatically substituted" in m for m in msgs), f"no warning with R's note (automatically substituted): {msgs}"
    assert ds.fit["fitType"] == "local"
    assert ds.fit["fit_substituted"] is True


@pytest.mark.baseline_fails(ref="v0.4.0", raises=KeyError, match="fit_substituted")
def test_substituted_flag_survives_quiet_and_is_false_when_trend_fits():
    cts, cond = _simulate(120, (3, 3), 62, mean_log2=(2, 1.5))
    assert _fit(cts, cond, quiet=True).fit["fit_substituted"] is True
    cts, cond = _simulate(1000, (3, 3), 7)
    ok = _fit(cts, cond, quiet=True).fit
    assert ok["fitType"] == "parametric" and ok["fit_substituted"] is False
    asked = _fit(cts, cond, quiet=True, fit_type="mean").fit   # the user asked for the mean: not a substitution
    assert asked["fitType"] == "mean" and asked["fit_substituted"] is False


# --------------------------------------------------------------------------- level order and contrast label
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="no UserWarning")
def test_mixed_case_levels_warn_with_both_orders():
    obs = pl.DataFrame({"c": ["ctrl", "ctrl", "Stim", "Stim"]})
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        X, names, factors = model_matrix(obs, "~ c")
    msgs = [str(w.message) for w in wl if issubclass(w.category, UserWarning)]
    assert msgs, "no UserWarning"
    assert "['Stim', 'ctrl']" in msgs[0] and "['ctrl', 'Stim']" in msgs[0], msgs[0]
    # the default order itself is unchanged
    assert names == ["Intercept", "c_ctrl_vs_Stim"] and factors["c"] == ["Stim", "ctrl"]


def test_no_level_order_warning_when_order_is_unambiguous():
    obs = pl.DataFrame({"c": ["ctrl", "ctrl", "stim", "stim"], "b": ["x", "y", "x", "y"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model_matrix(obs, "~ b + c")
        mixed = pl.DataFrame({"c": ["ctrl", "ctrl", "Stim", "Stim"]})
        model_matrix(mixed, "~ c", ref_levels={"c": "ctrl"})                       # explicit reference


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AttributeError, match="get_categories")  # polars >= 2.0 removed the call
def test_enum_column_keeps_declared_order_without_warning():
    mixed = pl.DataFrame({"c": ["ctrl", "ctrl", "Stim", "Stim"]}).with_columns(pl.col("c").cast(pl.Enum(["ctrl", "Stim"])))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X, names, factors = model_matrix(mixed, "~ c")
    assert factors["c"] == ["ctrl", "Stim"] and names == ["Intercept", "c_Stim_vs_ctrl"]


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AttributeError, match="last_contrast")
def test_results_records_the_contrast_used():
    cts, cond = _simulate(60, (4, 4), 3, mean_log2=(6, 1))
    obs = pl.DataFrame({"condition": np.where(cond == "c0", "ctrl", "Stim")})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = DESeq2(cts.T.astype(float), obs, "~ condition", quiet=True)
    ds.results()
    assert ds.last_contrast == "condition_ctrl_vs_Stim"
    assert ds.last_results_info["contrast"] == "condition_ctrl_vs_Stim"
    ds.results(("condition", "Stim", "ctrl"))
    assert ds.last_contrast == "condition_Stim_vs_ctrl"
    ds.results("condition_ctrl_vs_Stim")
    assert ds.last_contrast == "condition_ctrl_vs_Stim"


# --------------------------------------------------------------------------- duplicate coefficient names
_A, _A_S = ["r"] * 4 + ["s_x"] * 4, ["r", "x"] * 4


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="did not raise")
def test_duplicate_coefficient_names_raise():
    obs = pl.DataFrame({"a": _A, "a_s": _A_S})
    _must_raise(ValueError, lambda: model_matrix(obs, "~ a + a_s"), "duplicate")
    cts = np.random.default_rng(3).poisson(50, (8, 20)).astype(float)
    _must_raise(ValueError, lambda: DESeq2(cts, obs, "~ a + a_s", quiet=True), "duplicate")
    _must_raise(ValueError, lambda: DESeq2(cts, design_matrix=np.column_stack([np.ones(8), np.arange(8.0), np.arange(8.0) ** 2]),
                                           coef_names=["Intercept", "x", "x"], quiet=True), "duplicate")


def test_unique_names_select_the_right_column():
    """With the level renamed so the names are distinct, the a_s contrast returns the a_s column."""
    rng = np.random.default_rng(3)
    mu = np.full((200, 8), 50.0)
    mu[:40, np.array(_A_S) == "x"] *= 4.0
    cts = rng.negative_binomial(20, 20 / (20 + mu)).astype(float)
    obs = pl.DataFrame({"a": ["r"] * 4 + ["s2"] * 4, "a_s": _A_S})
    ds = DESeq2(cts.T, obs, "~ a + a_s", quiet=True)
    names = ds.results_names()
    assert len(set(names)) == len(names)
    r = ds.results(("a_s", "x", "r"))
    assert np.median(r["log2FoldChange"][:40]) > DE_MEDIAN_LFC_MIN


# --------------------------------------------------------------------------- pseudobulk_de error handling
def _bf(extra_c_group=False, shift=0.0, only_c=False):
    rng = np.random.default_rng(1)
    donors, cells_per, G = 6, 40, 40
    n = donors * cells_per * 2
    sample = np.repeat(np.arange(donors * 2), cells_per)
    cond = np.where(sample % 2 == 0, "ctrl", "stim")
    ctype = rng.choice(["A", "B"], n)
    lam = rng.gamma(2.0, 1.0, G)
    X = rng.poisson(lam[None, :] * np.where(cond[:, None] == "stim", np.r_[np.full(5, 4.0), np.ones(G - 5)], 1.0)).astype(np.float64)
    obs = pl.DataFrame({"donor": (sample // 2).astype(str), "condition": cond, "ctype": ctype})
    if extra_c_group or only_c:   # cell types seen in one condition only: the condition effect is not estimable
        extra = rng.poisson(lam[None, :], (donors * 12, G)).astype(np.float64)
        eobs = pl.DataFrame({"donor": np.repeat(np.arange(donors), 12).astype(str), "condition": "ctrl", "ctype": "C"})
        if only_c:      # a control-only and a stim-only group; both conditions exist in the table
            X = np.vstack([extra, extra[:donors * 6]])
            obs = pl.concat([eobs, pl.DataFrame({"donor": np.repeat(np.arange(donors), 6).astype(str),
                                                 "condition": "stim", "ctype": "D"})])
        else:
            X, obs = np.vstack([X, extra]), pl.concat([obs, eobs])
    return crest.BioFrame.from_scipy(sp.csr_matrix((X + shift).astype(np.float32)), obs=obs)


_KW = dict(groupby="ctype", min_cells=1)
_CONTRAST = ("condition", "stim", "ctrl")


def test_pseudobulk_de_valid_design_runs():
    bf = _bf()
    res = crest.tl.pseudobulk_de(bf, ["donor", "condition"], "~ donor + condition", _CONTRAST, **_KW)
    assert res.height == 2 * 40
    assert set(res["group"].unique().to_list()) == {"A", "B"}
    assert bf.uns["pseudobulk_de"]["skipped"] == {}


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="did not raise")
def test_pseudobulk_de_raises_on_non_integer_counts():
    bf = _bf(shift=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _must_raise(ValueError, lambda: crest.tl.pseudobulk_de(bf, ["donor", "condition"], "~ donor + condition", _CONTRAST, **_KW),
                    "integer")
    assert "pseudobulk_de" not in bf.uns


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="did not raise")
def test_pseudobulk_de_raises_on_bad_formula_column():
    bf = _bf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _must_raise(KeyError, lambda: crest.tl.pseudobulk_de(bf, ["donor", "condition"], "~ donor + condx", _CONTRAST, **_KW),
                    "condx")
        _must_raise(ValueError, lambda: crest.tl.pseudobulk_de(bf, ["donor", "condition"], "~ donor * condition", _CONTRAST, **_KW),
                    "additive")


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="did not raise")
def test_pseudobulk_de_raises_on_bad_contrast():
    bf = _bf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _must_raise(ValueError, lambda: crest.tl.pseudobulk_de(bf, ["donor", "condition"], "~ donor + condition",
                                                               ("condition", "stim", "nope"), **_KW), "nope")
        _must_raise(KeyError, lambda: crest.tl.pseudobulk_de(bf, ["donor", "condition"], "~ donor + condition",
                                                             "condition_nope_vs_ctrl", **_KW), "nope")


def test_pseudobulk_de_skips_group_with_one_condition_and_reports_it():
    bf = _bf(extra_c_group=True)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        res = crest.tl.pseudobulk_de(bf, ["donor", "condition"], "~ donor + condition", _CONTRAST, **_KW)
    assert set(res["group"].unique().to_list()) == {"A", "B"}
    rec = bf.uns["pseudobulk_de"]
    assert list(rec["skipped"]) == ["C"] and isinstance(rec["skipped"]["C"], str)
    assert sum("skipped" in str(w.message) for w in wl) == 1   # one summary warning, not one per cause


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="did not raise")
def test_pseudobulk_de_raises_when_every_group_is_skipped():
    bf = _bf(only_c=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _must_raise(ValueError, lambda: crest.tl.pseudobulk_de(bf, ["donor", "condition"], "~ donor + condition", _CONTRAST, **_KW),
                    "no group")
