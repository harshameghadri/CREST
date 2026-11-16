"""DESeq2 edge cases against R: partial outlier replacement, non-converged gene-wise dispersions, lowess / loess with ties.

Fixtures (``tests/data``): ``deseq2_edge_*_input.csv`` and ``deseq2_lowess_ties_*`` are written by ``make_edge_inputs.py``;
the reference output next to them (``*_R.csv``) comes from ``make_edge_fixtures.R`` (R 4.5.2, DESeq2 1.50.2, recorded in
the ``R`` entry of every meta file). The tests pin behaviour that already matched R; they carry no ``baseline_fails``
marker unless a released build fails them.

Tolerances are named constants. ``EDGE_RTOL`` and the smoother tolerances are ten times the larger of the deviations
measured on macOS arm64 and Linux x86-64 (``deseq2_edge_measurements.json``; ``measure_all()`` regenerates it), rounded
up to one digit and never below ``RTOL_FLOOR``. Deviation = |crest - R| / (ATOL + |R|).

What is compared and what is not:

* gene-wise dispersions at the floor are compared by floor status only (the likelihood is flat there; docs/deseq2.md);
* ``dispGeneIter`` is pinned as a set: the genes that exhaust ``maxit`` and the genes that stop after one iteration away
  from the floor. Its value on floor genes differs by one to a few iterations and is not pinned;
* where R's parametric trend fails and R uses a local fit, so does CREST (``fitType`` "local", ``fit_substituted`` true);
  the columns that follow the trend are compared in ``test_deseq2_local_fit.py``;
* no gene fails the beta fit in R on these data (``betaConv`` is TRUE for every gene with counts); the pin is that CREST
  agrees and reports none either. A fixture with a non-converged beta fit could not be generated.
"""

import functools
import json
import pathlib
import warnings

import numpy as np
import polars as pl
import pytest

import crest
from crest import crest as _native
from crest._loess import loess_fit

DATA = pathlib.Path(__file__).parent / "data"

# --- tolerances (named constants) -------------------------------------------------------------
ATOL = 1e-12                 # absolute part of the deviation measure
RTOL_FLOOR = 1e-12           # no pin is tighter than this
FLOOR_DISP = 1e-6            # gene-wise dispersions below this sit on the flat part of the likelihood
NONCONV_DISP_ATOL = 6e-8     # gene-wise dispersion of genes that exhaust maxit: R's grid refit may land at 1.3e-8 where CREST has 1e-8
GENEWISE_RTOL = 4e-10        # gene-wise dispersion of genes that converge away from the floor
BASEMEAN_RTOL = 1e-12          # base means against R (sums of a few doubles)
LOWESS_RTOL = 3e-12          # lowess against R on random vectors with ties
LOESS_RTOL = 5e-12           # loess (degree 2) against R on random vectors with ties, where R's fit is free of warnings
MIN_REPLICATES = (7, 4, 9)   # boundary values of minReplicatesForReplace for groups of 8, 4 and 8 samples

COLUMNS = ["baseMean", "dispFit", "dispMAP", "dispersion", "maxCooks", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
FIT_COLUMNS = {"baseMean", "dispFit", "dispMAP", "dispersion", "maxCooks"}


def _arr(x):
    return pl.Series(x).cast(pl.Float64).fill_null(np.nan).to_numpy() if not isinstance(x, np.ndarray) else x.astype(float)


def _dev(a, b):
    """(positions where exactly one side is NaN, max |a - b| / (ATOL + |b|) where both are finite)."""
    a, b = _arr(a), _arr(b)
    na, nb = np.isnan(a), np.isnan(b)
    ok = ~na & ~nb
    return int((na != nb).sum()), float(np.max(np.abs(a[ok] - b[ok]) / (ATOL + np.abs(b[ok])))) if ok.any() else 0.0


def _flag(x, default):
    return pl.Series(x).cast(pl.Boolean).fill_null(default).to_numpy().astype(bool)


@functools.lru_cache(maxsize=None)
def _case(name, min_rep=7):
    """(DESeq2 object, R meta) for one fixture; the reference tables are read by ``_ref``."""
    inp = pl.read_csv(DATA / f"deseq2_{name}_input.csv")
    genes = [c for c in inp.columns if c.startswith("gene")]
    meta = json.loads((DATA / f"deseq2_{name}_meta.json").read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = crest.tl.DESeq2(inp.select(genes).to_numpy().astype(float), inp.drop(genes), meta["design"],
                             ref_levels={k: v[0] for k, v in meta["levels"].items()}, var_names=genes, quiet=True,
                             min_replicates_for_replace=min_rep)
    return ds, meta


def _ref(name, tag="", suffix=""):
    return pl.read_csv(DATA / f"deseq2_{name}{tag}_R{suffix}.csv", null_values="NA", infer_schema_length=None)


def _geneset(name, min_rep=7):
    ds, meta = _case(name, min_rep)
    tag = f"_minrep{min_rep}" if name == "edge_848" else ""
    return ds, meta, _ref(name, tag)


def measure_genewise(name, min_rep=7):
    """Gene-wise dispersion handling: set differences and deviations by group (floor / converged / maxit)."""
    ds, meta, ref = _geneset(name, min_rep)
    f = ds.fit
    nz = ~np.asarray(f["allZero"], bool)
    it_c, it_r = np.asarray(f["dispGeneIter"], float), _arr(ref["dispGeneIter"])
    ge_c, ge_r = np.asarray(f["dispGeneEst"], float), _arr(ref["dispGeneEst"])
    fl_c, fl_r = ge_c <= FLOOR_DISP, ge_r <= FLOOR_DISP
    out = {"floor_status_mismatch": int((nz & (fl_c != fl_r)).sum()),
           "maxit_set_mismatch": int((nz & ((it_c >= 100) != (it_r >= 100))).sum()),
           "iter1_off_floor_set_mismatch": int((nz & (((it_c == 1) & ~fl_c) != ((it_r == 1) & ~fl_r))).sum()),
           "n_maxit": int((nz & (it_r >= 100)).sum()), "n_iter1_off_floor": int((nz & (it_r == 1) & ~fl_r).sum()),
           "n_floor": int((nz & fl_r).sum()), "n_nonzero": int(nz.sum())}
    conv = nz & (it_r < 100) & ~fl_r & ~fl_c
    out["converged_genewise_rel"] = _dev(ge_c[conv], ge_r[conv])[1]
    mx = nz & (it_r >= 100)
    out["maxit_genewise_abs"] = float(np.max(np.abs(ge_c[mx] - ge_r[mx]))) if mx.any() else 0.0
    bc_r = _flag(ref["betaConv"], True)
    out["beta_nonconverged_R"] = int((nz & ~bc_r).sum())
    out["beta_nonconverged_CREST"] = int((nz & ~np.asarray(f["betaConv"], bool)).sum())
    out["beta_flag_mismatch"] = int((nz & (np.asarray(f["betaConv"], bool) != bc_r)).sum())
    return out


def measure_columns(name, min_rep=7, suffix="", contrast=None):
    """Deviation of every result column from R, in three groups: genes whose counts were replaced (in either tool), genes
    at the dispersion floor in R, and the rest. Returns {"replaced": {column: (NA mismatches, max deviation)}, "floor":
    {...}, "rest": {...}, "replace_flag_mismatch": n}."""
    ds, meta, ref = _geneset(name, min_rep)
    if suffix:
        ref = _ref(name, f"_minrep{min_rep}", suffix)
    res = ds.results(tuple(contrast or meta["contrast"]))
    f = ds.fit
    rep_c, rep_r = np.asarray(f["replace"], bool), _flag(ref["replace"], False)
    rep = rep_c | rep_r
    floor = ~rep & (_arr(ref["dispGeneEst"]) <= FLOOR_DISP)
    out = {"replace_flag_mismatch": int((rep_c != rep_r).sum()), "n_replaced": int(rep.sum()), "n_floor": int(floor.sum())}
    for label, m in (("replaced", rep), ("floor", floor), ("rest", ~rep & ~floor)):
        out[label] = {col: _dev((f[col] if col in FIT_COLUMNS else _arr(res[col]))[m], _arr(ref[col])[m]) for col in COLUMNS}
    return out


def measure_smoothers():
    """Largest deviation of lowess and of loess from R over the stored random cases."""
    cases = pl.read_csv(DATA / "deseq2_lowess_ties_cases.csv")
    inp = pl.read_csv(DATA / "deseq2_lowess_ties_input.csv")
    ref = pl.read_csv(DATA / "deseq2_lowess_ties_R.csv", null_values="NA")
    lo, lf, n_warn, n_raise = 0.0, 0.0, 0, 0
    for cs in cases.iter_rows(named=True):
        v = inp.filter(pl.col("case") == cs["case"])
        r = ref.filter(pl.col("case") == cs["case"])
        x, y = v["x"].to_numpy(), v["y"].to_numpy()
        lo = max(lo, _dev(_native.lowess(x, y, cs["f"], cs["iter"], cs["delta"]), r["lowess"])[1])
        if r["loess_warned"][0]:
            n_warn += 1
            continue
        lf = max(lf, _dev(loess_fit(x, y, span=cs["span"], degree=2), r["loess"])[1])
    return {"lowess": lo, "loess": lf, "loess_cases_with_R_warning": n_warn, "cases": cases.height}


def measure_all():
    out = {}
    for k in MIN_REPLICATES:
        out[f"edge_848_minrep{k}"] = {"columns": measure_columns("edge_848", k), "genewise": measure_genewise("edge_848", k)}
    out["edge_848_minrep7_c1"] = {"columns": measure_columns("edge_848", 7, "_c1", ["condition", "c1", "c0"])}
    for name in ("edge_sparse_param", "edge_sparse_extreme"):
        out[name] = {"columns": measure_columns(name), "genewise": measure_genewise(name)}
    out["smoothers"] = measure_smoothers()
    return out


# ===================================================================================== tolerances (measured)
# The largest entries of the "rest" groups (dispMAP, dispersion, lfcSE, stat, p-value of edge_848 and edge_sparse_param) come
# from one gene (edge_848, gene 80, gene-wise estimate within 2% of the trend) and four genes at the upper dispersion cap
# (edge_sparse_param), where the MAP objective is flat and the optimiser stops within ~1e-6 of R's value.
EDGE_RTOL = {   # 10 x the larger of the arm64 and x86-64 measurements, rounded up to one digit, never below RTOL_FLOOR
    "edge_848_minrep7": {
        "replaced": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 2e-12, "lfcSE": 1e-12, "stat": 2e-12, "pvalue": 1e-9, "padj": 1e-9},
        "floor": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 1e-12, "lfcSE": 1e-12, "stat": 1e-12, "pvalue": 5e-10, "padj": 5e-10},
        "rest": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 4e-6, "dispersion": 4e-6, "maxCooks": 3e-9, "log2FoldChange": 4e-9, "lfcSE": 2e-6, "stat": 2e-6, "pvalue": 5e-6, "padj": 1e-9},
    },
    "edge_848_minrep4": {
        "replaced": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 3e-12, "dispersion": 3e-12, "maxCooks": 1e-12, "log2FoldChange": 2e-12, "lfcSE": 1e-12, "stat": 2e-12, "pvalue": 1e-9, "padj": 1e-9},
        "floor": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 1e-12, "lfcSE": 1e-12, "stat": 1e-12, "pvalue": 5e-10, "padj": 5e-10},
        "rest": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 4e-6, "dispersion": 4e-6, "maxCooks": 1e-12, "log2FoldChange": 4e-9, "lfcSE": 2e-6, "stat": 2e-6, "pvalue": 5e-6, "padj": 1e-9},
    },
    "edge_848_minrep9": {
        "replaced": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 1e-12, "lfcSE": 1e-12, "stat": 1e-12, "pvalue": 1e-12, "padj": 1e-12},
        "floor": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 1e-12, "lfcSE": 1e-12, "stat": 1e-12, "pvalue": 5e-10, "padj": 8e-10},
        "rest": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 4e-6, "dispersion": 4e-6, "maxCooks": 3e-9, "log2FoldChange": 4e-9, "lfcSE": 2e-6, "stat": 2e-6, "pvalue": 5e-6, "padj": 1e-9},
    },
    "edge_848_minrep7_c1": {
        "replaced": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 2e-12, "lfcSE": 1e-12, "stat": 1e-12, "pvalue": 9e-10, "padj": 4e-10},
        "floor": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 1e-12, "lfcSE": 1e-12, "stat": 1e-12, "pvalue": 9e-10, "padj": 1e-12},
        "rest": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 4e-6, "dispersion": 4e-6, "maxCooks": 3e-9, "log2FoldChange": 2e-8, "lfcSE": 2e-6, "stat": 2e-6, "pvalue": 2e-6, "padj": 8e-10},
    },
    "edge_sparse_param": {
        "replaced": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 1e-12, "lfcSE": 1e-12, "stat": 1e-12, "pvalue": 1e-12, "padj": 1e-12},
        "floor": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 1e-12, "dispersion": 1e-12, "maxCooks": 1e-12, "log2FoldChange": 2e-12, "lfcSE": 1e-12, "stat": 2e-12, "pvalue": 1e-9, "padj": 1e-12},
        "rest": {"baseMean": 1e-12, "dispFit": 1e-12, "dispMAP": 8e-6, "dispersion": 8e-6, "maxCooks": 1e-12, "log2FoldChange": 3e-4, "lfcSE": 5e-6, "stat": 3e-4, "pvalue": 6e-4, "padj": 1e-12},
    },
}


def _tol(key, group, col):
    return max(EDGE_RTOL[key][group][col], RTOL_FLOOR)


def check_columns(key, name, min_rep=7, suffix="", contrast=None):
    got = measure_columns(name, min_rep, suffix, contrast)
    assert got["replace_flag_mismatch"] == 0, f"{key}: {got['replace_flag_mismatch']} genes differ in the replaced flag"
    for group in ("replaced", "floor", "rest"):
        for col in COLUMNS:
            n_na, dev = got[group][col]
            assert n_na == 0, f"{key}: {group}: {col}: {n_na} genes differ in NA status"
            assert dev <= _tol(key, group, col), f"{key}: {group}: {col}: max deviation {dev:.3g} > {_tol(key, group, col):.3g}"
    return got


def check_genewise(name, min_rep=7):
    got = measure_genewise(name, min_rep)
    for k in ("floor_status_mismatch", "maxit_set_mismatch", "iter1_off_floor_set_mismatch", "beta_flag_mismatch"):
        assert got[k] == 0, f"{name}: {k} = {got[k]}"
    assert got["beta_nonconverged_R"] == 0 and got["beta_nonconverged_CREST"] == 0, got
    assert got["converged_genewise_rel"] <= GENEWISE_RTOL, f"{name}: converged gene-wise dispersion deviation {got['converged_genewise_rel']:.3g}"
    assert got["maxit_genewise_abs"] <= NONCONV_DISP_ATOL, f"{name}: gene-wise dispersion of genes at maxit: {got['maxit_genewise_abs']:.3g}"
    return got


# ===================================================================================== partial outlier replacement
# groups of 8, 4 and 8 samples; Cook's p-values removed in R for genes whose outlier sits in a group that is not replaced
EXPECTED_COOKS_NA = {7: 3, 4: 0, 9: 9}     # genes with a removed p-value, from the stored R output
EXPECTED_REPLACED = {7: 9, 4: 9, 9: 0}     # genes with replaced counts, from the stored R output


@pytest.mark.parametrize("min_rep", MIN_REPLICATES)
def test_partial_replacement_matches_r(min_rep):
    """8-4-8 design: the groups of 8 are replaced at minReplicatesForReplace 7, all three at 4, none at 9."""
    ds, meta, ref = _geneset("edge_848", min_rep)
    assert meta["R"]["deseq2_version"].startswith("1.50")
    got = check_columns(f"edge_848_minrep{min_rep}", "edge_848", min_rep)
    f = ds.fit
    assert got["n_replaced"] == EXPECTED_REPLACED[min_rep] == int(_flag(ref["replace"], False).sum())
    res = ds.results(tuple(meta["contrast"]))
    p_na_r = ref["pvalue"].is_null().to_numpy() & (_arr(ref["baseMean"]) > 0)
    assert int(p_na_r.sum()) == EXPECTED_COOKS_NA[min_rep]
    assert (res["pvalue"].is_null().to_numpy() == ref["pvalue"].is_null().to_numpy()).all()
    assert (res["padj"].is_null().to_numpy() == ref["padj"].is_null().to_numpy()).all()
    assert np.array_equal(res["pvalue"].is_null().to_numpy() & (_arr(ref["baseMean"]) > 0), p_na_r)
    assert np.allclose(ds.size_factors, meta["R"]["by_min_replicates"][str(min_rep)]["sf"], rtol=1e-12, atol=0)


def test_partial_replacement_second_contrast_matches_r():
    ds, meta, ref = _geneset("edge_848", 7)
    check_columns("edge_848_minrep7_c1", "edge_848", 7, "_c1", ["condition", "c1", "c0"])
    ref_c1 = _ref("edge_848", "_minrep7", "_c1")
    res = ds.results(("condition", "c1", "c0"))
    assert (res["pvalue"].is_null().to_numpy() == ref_c1["pvalue"].is_null().to_numpy()).all()


# ===================================================================================== sparse design, non-converged fits
def test_sparse_design_nonconverged_dispersions_match_r():
    """6 samples, ``~ batch + condition``: 138 genes exhaust maxit in R, 2 stop after one iteration off the floor."""
    ds, meta, ref = _geneset("edge_sparse_param")
    assert meta["R"]["fitType"] == ds.fit["fitType"] == "parametric"
    got = check_genewise("edge_sparse_param")
    assert got["n_maxit"] >= 100 and got["n_iter1_off_floor"] >= 1 and got["n_floor"] >= 1000, got   # the fixture exercises the paths
    check_columns("edge_sparse_param", "edge_sparse_param")


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="fitType")
def test_sparse_design_without_trend_keeps_genewise_estimates():
    """R falls back to a local trend, and so does CREST; the gene-wise estimates and their flags agree."""
    ds, meta, ref = _geneset("edge_sparse_extreme")
    assert meta["R"]["fitType"] == "local" and ds.fit["fitType"] == "local", f"fitType {ds.fit['fitType']!r}"
    got = check_genewise("edge_sparse_extreme")
    assert got["n_maxit"] >= 100 and got["n_floor"] >= 500, got
    ok = _arr(ref["baseMean"]) > 0
    assert _dev(ds.fit["baseMean"][ok], ref["baseMean"].to_numpy()[ok])[1] <= BASEMEAN_RTOL


@pytest.mark.baseline_fails(ref="v0.4.0", raises=KeyError, match="fit_substituted")
def test_sparse_design_without_trend_is_flagged():
    ds, meta, ref = _geneset("edge_sparse_extreme")
    assert ds.fit["fit_substituted"] is True
    inp = pl.read_csv(DATA / "deseq2_edge_sparse_extreme_input.csv")
    genes = [c for c in inp.columns if c.startswith("gene")]
    with pytest.warns(UserWarning, match="local"):
        crest.tl.DESeq2(inp.select(genes).to_numpy().astype(float), inp.drop(genes), meta["design"],
                        ref_levels={k: v[0] for k, v in meta["levels"].items()})


# ===================================================================================== lowess / loess with ties
def _smoother_cases():
    cases = pl.read_csv(DATA / "deseq2_lowess_ties_cases.csv")
    inp = pl.read_csv(DATA / "deseq2_lowess_ties_input.csv")
    ref = pl.read_csv(DATA / "deseq2_lowess_ties_R.csv", null_values="NA")
    for cs in cases.iter_rows(named=True):
        yield cs, inp.filter(pl.col("case") == cs["case"]), ref.filter(pl.col("case") == cs["case"])


def test_lowess_with_tied_x_matches_r_on_random_vectors():
    n = 0
    for cs, v, r in _smoother_cases():
        assert cs["n_distinct"] < cs["n"]
        got = _native.lowess(v["x"].to_numpy(), v["y"].to_numpy(), cs["f"], cs["iter"], cs["delta"])
        n_na, dev = _dev(got, r["lowess"])
        assert n_na == 0 and dev <= LOWESS_RTOL, f"case {cs['case']} (n={cs['n']}, f={cs['f']:.3g}, iter={cs['iter']}, delta={cs['delta_mode']}): {dev:.3g}"
        n += 1
    assert n >= 50


def test_loess_with_tied_x_matches_r_on_random_vectors():
    """Where R's loess runs without a warning; the degenerate neighbourhoods are the next test."""
    n = 0
    for cs, v, r in _smoother_cases():
        if r["loess_warned"][0]:
            continue
        got = loess_fit(v["x"].to_numpy(), v["y"].to_numpy(), span=cs["span"], degree=2)
        n_na, dev = _dev(got, r["loess"])
        assert n_na == 0 and dev <= LOESS_RTOL, f"case {cs['case']} (n={cs['n']}, span={cs['span']}): {dev:.3g}"
        n += 1
    assert n >= 50


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="singular")
def test_loess_with_degenerate_ties_stops_where_r_warns():
    """R's loess warns and falls back to a pseudoinverse where fewer than three distinct x values carry weight; CREST
    follows skmisc.loess and stops with an error naming the singular neighbourhood (the released 0.4.0 returned values)."""
    n = 0
    for cs, v, r in _smoother_cases():
        if not r["loess_warned"][0]:
            continue
        err = None
        try:
            loess_fit(v["x"].to_numpy(), v["y"].to_numpy(), span=cs["span"], degree=2)
        except ValueError as e:
            err = e
        assert err is not None and "singular" in str(err), f"case {cs['case']}: no error naming a singular neighbourhood"
        n += 1
    assert n >= 1
