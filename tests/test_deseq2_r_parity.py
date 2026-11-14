"""DESeq2 against R: pinned reference output for the code paths the parity claims rest on.

Every fixture in ``tests/data`` is a count table (``deseq2_<name>_input.csv``) with the output of R's
``DESeq()`` + ``results()`` on it (``deseq2_<name>_R.csv``); the R and DESeq2 versions are recorded in
``deseq2_<name>_meta.json`` (key ``R``) and the scripts that make the files are ``make_pin_inputs.py``
and ``make_pin_fixtures.R`` (R 4.5.2, DESeq2 1.50.2, the reference version of this port).

Tests marked ``baseline_fails`` fail on the stated released build with the declared assertion and pass on
this tree. Unmarked tests pin behaviour that already matched R.

Tolerances: each ``PIN_RTOL`` entry is ten times the largest deviation measured on Linux x86-64 and on
macOS arm64 (rounded up to one digit, never below ``RTOL_FLOOR``). Deviations are |crest - R| / (ATOL +
|R|). The two groups of values that cannot be compared tightly are excluded by rule, not by tolerance:
gene-wise dispersions at the floor (the likelihood is flat there, see docs/deseq2.md, known difference 1),
and Cook's distances far above the cut-off (compared through the outlier flag only).
"""

import functools
import json
import pathlib
import warnings

import numpy as np
import polars as pl
import pytest

import crest

DATA = pathlib.Path(__file__).parent / "data"

# --- tolerances (named constants) -------------------------------------------------------------
ATOL = 1e-12                    # absolute part of the deviation measure
RTOL_FLOOR = 1e-12              # no pin is tighter than this (the reference CSVs carry 15 digits)
FLOOR_DISP = 1e-6               # gene-wise dispersions below this sit on the flat part of the likelihood
COOKS_COMPARE_FACTOR = 2.0      # maxCooks is compared where R's value is below this multiple of the cut-off
PRIOR_VAR_RTOL = 1e-9           # prior variance for <= 3 residual df against R's value (same grid point)
BAND_EPS = 1e-6                 # slack on the edges of the R seed band
REFIT_RTOL = 1e-6               # dispersion and p-value of genes refitted after outlier replacement
TREND_A_RTOL, TREND_B_RTOL = 1e-4, 1e-3   # parametric trend coefficients on counts up to 1e8
DISPFIT_RTOL = 1e-4             # dispersion trend on counts up to 1e8
MAXCOOKS_LRT_RTOL = 1e-9        # LRT Cook's distances against R 1.50.2
CUTOFF_RTOL = 1e-12             # Cook's cut-off qf(0.99, p, m - p) against R (the reference carries 15 digits)
TREND_FALLBACK_INDEPENDENT_RTOL = 1e-6   # quantities that do not depend on the dispersion trend, when R falls back to a local fit
CONTRAST_ZERO_MASK_RTOL = 1e-9  # rows whose c1 / c2 counts were all replaced, tuple and numeric contrast
PIN_RTOL = {}                   # filled below: per-fixture, per-column tolerances

PRIOR_DESIGNS = ["prior_2v2_sd10", "prior_2v2_sd15", "prior_2v3", "prior_3v2"]
COLUMNS = ["size_factors", "baseMean", "dispGeneEst", "dispFit", "dispMAP", "dispersion", "log2FoldChange", "lfcSE",
           "stat", "pvalue", "padj", "maxCooks"]
_FIT_COLUMNS = {"baseMean": "baseMean", "dispGeneEst": "dispGeneEst", "dispFit": "dispFit", "dispMAP": "dispMAP",
                "dispersion": "dispersion", "maxCooks": "maxCooks"}


def _must_raise(exc, fn, match):
    """Call ``fn``; fail with an AssertionError if it does not raise ``exc`` matching ``match``."""
    import re
    try:
        fn()
    except exc as e:
        assert re.search(match, str(e)), f"raised {type(e).__name__}({e}) which does not match {match!r}"
        return e
    raise AssertionError(f"did not raise {exc.__name__}")


@functools.lru_cache(maxsize=None)
def _case(name):
    """(DESeq2 object, results table, R table, meta) for one fixture."""
    inp = pl.read_csv(DATA / f"deseq2_{name}_input.csv")
    genes = [c for c in inp.columns if c.startswith("gene")]
    meta = json.loads((DATA / f"deseq2_{name}_meta.json").read_text())
    kw = {"test": "LRT", "reduced": meta["reduced"]} if meta.get("test") == "LRT" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = crest.tl.DESeq2(inp.select(genes).to_numpy().astype(float), inp.drop(genes), meta["design"],
                             ref_levels={k: v[0] for k, v in meta["levels"].items()}, var_names=genes, quiet=True,
                             sf_type=meta.get("sf_type", "ratio"), **kw)
    res = ds.results(tuple(meta["contrast"]) if meta["contrast"] else None)
    ref = pl.read_csv(DATA / f"deseq2_{name}_R.csv", null_values="NA")
    return ds, res, ref, meta


def _arr(x):
    return pl.Series(x).cast(pl.Float64).fill_null(np.nan).to_numpy() if not isinstance(x, np.ndarray) else x.astype(float)


def _dev(a, b):
    """(number of positions where exactly one side is NaN, max |a - b| / (ATOL + |b|) where both are finite)."""
    a, b = _arr(a), _arr(b)
    na, nb = np.isnan(a), np.isnan(b)
    ok = ~na & ~nb
    return int((na != nb).sum()), float(np.max(np.abs(a[ok] - b[ok]) / (ATOL + np.abs(b[ok])))) if ok.any() else 0.0


def measure(name):
    """Deviation of CREST from the stored R output, per column family; used to set ``PIN_RTOL``."""
    ds, res, ref, meta = _case(name)
    f, out = ds.fit, {}
    out["size_factors"] = _dev(ds.size_factors, np.array(meta["R"]["sf"]))
    for col in ("baseMean", "dispFit", "dispMAP", "dispersion"):
        out[col] = _dev(f[_FIT_COLUMNS[col]], ref[col])
    c_g, r_g = _arr(f["dispGeneEst"]), _arr(ref["dispGeneEst"])
    both = (c_g > FLOOR_DISP) & (r_g > FLOOR_DISP)
    out["dispGeneEst"] = (int(((c_g <= FLOOR_DISP) != (r_g <= FLOOR_DISP)).sum()), _dev(c_g[both], r_g[both])[1])
    for col in ("log2FoldChange", "lfcSE", "stat", "pvalue", "padj"):
        out[col] = _dev(res[col], ref[col])
    cut = meta["R"]["cooksCutoff"]
    c_m, r_m = _arr(f["maxCooks"]), _arr(ref["maxCooks"])
    near = np.isfinite(c_m) & np.isfinite(r_m) & (r_m < COOKS_COMPARE_FACTOR * cut)
    flag_mismatch = int(((c_m > cut) != (r_m > cut)).sum()) + int((np.isnan(c_m) != np.isnan(r_m)).sum())
    out["maxCooks"] = (flag_mismatch, _dev(c_m[near], r_m[near])[1])
    return out


def check_against_r(name):
    """Assert the whole fixture against R: fit type, NA patterns, every column within ``PIN_RTOL[name]``."""
    ds, res, ref, meta = _case(name)
    assert ds.fit["fitType"] == meta["R"]["fitType"], f"{name}: fitType {ds.fit['fitType']!r}, R {meta['R']['fitType']!r}"
    got = measure(name)
    for col in COLUMNS:
        n_na, dev = got[col]
        assert n_na == 0, f"{name}: {col}: {n_na} genes differ in NA / flag / floor status"
        assert dev <= PIN_RTOL[name][col], f"{name}: {col}: max deviation {dev:.3g} > {PIN_RTOL[name][col]:.3g}"
    sig_c = res["padj"].fill_null(1).to_numpy() < 0.1
    sig_r = ref["padj"].fill_null(1).to_numpy() < 0.1
    assert (sig_c == sig_r).all(), f"{name}: {(sig_c != sig_r).sum()} genes differ in padj < 0.1"


# ===================================================================================== prior variance
_BAND = json.loads((DATA / "deseq2_prior_var_band.json").read_text())


@pytest.mark.parametrize("design", [
    pytest.param("prior_2v2_sd10", marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="outside R's seed band")),
    "prior_2v2_sd15",
    "prior_2v3",
    pytest.param("prior_3v2", marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="outside R's seed band")),
])
def test_prior_variance_inside_r_seed_band(design):
    """R's value is one Monte-Carlo draw (set.seed(2)); over seeds 1-50 R itself spreads over this band."""
    pv = _case(design)[0].fit["dispPriorVar"]
    b = _BAND["designs"][design]
    assert b["band_min"] - BAND_EPS <= pv <= b["band_max"] + BAND_EPS, \
        f"prior variance {pv:.4f} outside R's seed band [{b['band_min']:.4f}, {b['band_max']:.4f}]"


@pytest.mark.parametrize("design", [pytest.param(d, marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="R's value"))
                                    for d in PRIOR_DESIGNS])
def test_prior_variance_equals_r_value(design):
    """The tabulated reference draws reproduce R's own value, not just its spread."""
    pv = _case(design)[0].fit["dispPriorVar"]
    r = _case(design)[3]["R"]["priorVar"]
    assert abs(pv / r - 1) <= PRIOR_VAR_RTOL, f"prior variance {pv:.6f} differs from R's value {r:.6f}"


def test_prior_variance_band_records_its_source():
    assert _BAND["deseq2_version"].startswith("1.50") and len(_BAND["seeds"]) >= 30
    for name, b in _BAND["designs"].items():
        assert b["residual_df"] in (2, 3) and len(b["values"]) == len(_BAND["seeds"])
        assert b["band_min"] <= b["r_default_seed2"] <= b["band_max"]


# ===================================================================================== whole-table pins
PIN_FIXTURES = [
    pytest.param("prior_2v2_sd10", marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="dispMAP")),
    pytest.param("prior_2v2_sd15", marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="dispMAP")),
    pytest.param("prior_2v3", marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="dispMAP")),
    pytest.param("prior_3v2", marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="dispMAP")),
    pytest.param("refit_outliers", marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="dispersion")),
    pytest.param("very_large_counts", marks=pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="fitType")),
    "filtering_cooks",
    "poscounts",
    "covariate",
    "lrt_cooks",
    "contrast_zero_mask",
]


@pytest.mark.parametrize("name", PIN_FIXTURES)
def test_matches_r(name):
    check_against_r(name)


# ===================================================================================== refit subset
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="refit dispersion")
def test_outlier_refit_genes_match_r():
    """Two genes replaced after outlier detection get a MAP dispersion, not the 1e-7 floor."""
    ds, res, ref, meta = _case("refit_outliers")
    assert sorted(np.flatnonzero(ds.fit["replace"]).tolist()) == [0, 1]
    for col, got in (("dispersion", ds.fit["dispersion"]), ("pvalue", res["pvalue"].to_numpy())):
        for g in (0, 1):
            r = ref[col][g]
            assert abs(got[g] / r - 1) <= REFIT_RTOL, f"refit dispersion / p-value: {col} of gene {g}: {got[g]:.6g} vs R {r:.6g}"


# ===================================================================================== high counts
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="fitType")
def test_parametric_trend_on_very_large_counts_matches_r():
    ds, res, ref, meta = _case("very_large_counts")
    f = ds.fit
    assert f["fitType"] == "parametric", f"fitType {f['fitType']!r}, messages {f['messages']}"
    a_r, b_r = meta["R"]["trend"]
    assert abs(f["trend"][0] / a_r - 1) <= TREND_A_RTOL and abs(f["trend"][1] / b_r - 1) <= TREND_B_RTOL, (f["trend"], a_r, b_r)
    ok = f["baseMean"] > 0
    assert np.max(np.abs(f["dispFit"][ok] / ref["dispFit"].to_numpy()[ok] - 1)) <= DISPFIT_RTOL


# ===================================================================================== filtering, Cook's
def test_filtering_fixture_exercises_independent_filtering_and_cooks_filtering():
    ds, res, ref, meta = _case("filtering_cooks")
    p_na, padj_na = ref["pvalue"].is_null().to_numpy(), ref["padj"].is_null().to_numpy()
    all_zero = (ref["baseMean"].to_numpy() == 0)
    filtered = padj_na & ~p_na
    cooks = p_na & ~all_zero
    assert filtered.sum() >= 100 and cooks.sum() >= 5, (int(filtered.sum()), int(cooks.sum()))
    # the three groups of missing values are the same genes in CREST
    assert (res["pvalue"].is_null().to_numpy() == p_na).all()
    assert (res["padj"].is_null().to_numpy() == padj_na).all()


# ===================================================================================== trend fallback
@pytest.mark.baseline_fails(ref="v0.4.0", raises=KeyError, match="fit_substituted")
def test_trend_fallback_is_a_flagged_known_difference():
    """R substitutes a local fit when the parametric trend fails, CREST the mean: flagged, not equal."""
    ds, res, ref, meta = _case("trend_fallback")
    assert meta["R"]["fitType"] == "local" and ds.fit["fitType"] == "mean" and ds.fit["fit_substituted"] is True
    f = ds.fit
    # what does not depend on the trend still agrees with R
    assert np.allclose(ds.size_factors, meta["R"]["sf"], rtol=1e-10, atol=0)
    assert np.allclose(f["baseMean"], ref["baseMean"].to_numpy(), rtol=1e-10, atol=0)
    n_na, dev = measure("trend_fallback")["dispGeneEst"]
    assert n_na == 0 and dev <= TREND_FALLBACK_INDEPENDENT_RTOL, f"gene-wise dispersion: {n_na} floor mismatches, deviation {dev:.3g}"
    # and what does is different, by a lot (the reason for the flag)
    assert not np.allclose(f["dispFit"], ref["dispFit"].to_numpy(), rtol=0.1)


# ===================================================================================== covariates
def test_numeric_covariate_coefficient_matches_r():
    """The coefficient of a numeric covariate (results by name) against R's results(dds, name = "age")."""
    ds, res, ref, meta = _case("covariate")
    ref_age = pl.read_csv(DATA / "deseq2_covariate_R_age.csv", null_values="NA")
    assert "age" in ds.results_names()
    got = ds.results("age")
    for col in ("log2FoldChange", "lfcSE", "stat", "pvalue", "padj"):
        n_na, dev = _dev(got[col], ref_age[col])
        assert n_na == 0 and dev <= PIN_RTOL["covariate"][col], f"age: {col}: {n_na} NA mismatches, deviation {dev:.3g}"


# ===================================================================================== LRT Cook's
def test_lrt_cooks_distances_follow_r_1_50():
    """LRT Cook's distances use the full-model means (DESeq2 >= 1.50); some genes sit within 5% of the cut-off."""
    ds, res, ref, meta = _case("lrt_cooks")
    assert meta["R"]["deseq2_version"].startswith("1.50")
    cut = meta["R"]["cooksCutoff"]
    r_m = ref["maxCooks"].to_numpy()
    near = np.flatnonzero(np.abs(r_m / cut - 1) <= 0.05)
    assert len(near) >= 4 and (r_m[near] > cut).any() and (r_m[near] < cut).any(), r_m[:4] / cut
    got = ds.fit["maxCooks"]
    assert np.allclose(got, r_m, rtol=MAXCOOKS_LRT_RTOL, atol=0, equal_nan=True)
    # the genes on either side of the cut-off keep or lose their p-value exactly as in R
    assert (res["pvalue"].is_null().to_numpy() == ref["pvalue"].is_null().to_numpy()).all()


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="Cook's cut-off")
@pytest.mark.parametrize("name", ["lrt_cooks", "filtering_cooks", "covariate", "refit_outliers", "prior_2v3"])
def test_cooks_cutoff_equals_r_quantile(name):
    ds, res, ref, meta = _case(name)
    got, want = ds.fit["cooksCutoff"], meta["R"]["cooksCutoff"]
    assert abs(got / want - 1) <= CUTOFF_RTOL, f"Cook's cut-off {got!r} vs R's qf(0.99, p, m - p) {want!r}"


# ===================================================================================== contrast zero mask
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="numeric contrast")
def test_numeric_and_tuple_contrast_agree_with_r_on_replaced_counts():
    """R's all-zero test of a contrast looks at the original counts, also after outlier replacement."""
    ds, res, ref, meta = _case("contrast_zero_mask")
    assert ds.fit["replace"][0]
    ref_num = pl.read_csv(DATA / "deseq2_contrast_zero_mask_R_numeric.csv", null_values="NA")
    num = ds.results(np.array(meta["numeric_contrast"], dtype=float))
    for col in ("log2FoldChange", "lfcSE", "stat", "pvalue", "padj"):
        for label, got, want in (("tuple contrast", res, ref), ("numeric contrast", num, ref_num)):
            n_na, dev = _dev(got[col], want[col])
            assert n_na == 0 and dev <= CONTRAST_ZERO_MASK_RTOL, f"{label}: {col}: {n_na} NA mismatches, deviation {dev:.3g}"


# ===================================================================================== reasons for NA
@pytest.mark.baseline_fails(ref="v0.4.0", raises=KeyError, match="na")
def test_missing_values_record_their_reason():
    """Genes without counts, with Cook's outliers and removed by independent filtering are told apart."""
    ds, res, ref, meta = _case("filtering_cooks")
    ds.results(tuple(meta["contrast"]))
    na = ds.last_results_info["na"]
    p_na, padj_na = ref["pvalue"].is_null().to_numpy(), ref["padj"].is_null().to_numpy()
    all_zero = ref["baseMean"].to_numpy() == 0
    assert (na["all_zero"] == all_zero).all()
    assert (na["cooks"] == (p_na & ~all_zero)).all()
    assert (na["filtered"] == (padj_na & ~p_na)).all()
    assert na["all_zero"].sum() > 0 and na["cooks"].sum() >= 5 and na["filtered"].sum() >= 100
    assert not (na["all_zero"] & na["cooks"]).any() and not (na["cooks"] & na["filtered"]).any()
