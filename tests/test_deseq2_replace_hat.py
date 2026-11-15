"""Outlier replacement on rows whose coefficient fit does not converge, against R DESeq2.

Fixture (``tests/data``): ``deseq2_replace_hat_input.csv`` and ``_meta.json`` are written by ``make_replace_inputs.py``; the
reference output (``deseq2_replace_hat_R.csv``, ``deseq2_replace_hat_noreplace_R.csv``) comes from
``make_replace_fixtures.R`` (R 4.5.2, DESeq2 1.50.2, recorded in the ``R`` entry of the meta file).

16 samples in two groups: ``a`` with 4 (below minReplicatesForReplace = 7: never replaced) and ``b`` with 12. Of the 750 genes,
450 have zero counts in all of ``a`` and heavy-tailed counts in ``b``; for 313 of them the iteratively reweighted least squares
fit diverges and both tools refit the row. R keeps the hat diagonals of the last iteratively reweighted iteration for such rows
(its refit replaces the coefficients and the fitted means only), and Cook's distance uses them. A build that takes the hat
diagonals from the refit instead gets Cook's distances that are too small by a few per cent, and with them a different
replacement decision where the distance sits next to the cut-off (gene629 here), and a different base mean, dispersion trend
and result for the genes that depend on it.

What is compared:

* the replaced flag and the base mean after replacement (all genes) and the dispersion (all genes, which follows the trend and
  with it every replacement decision),
* ``maxCooks`` of the first pass (``minReplicatesForReplace = Inf``: no counts are replaced): rows whose fit converges agree to
  round-off; rows that are refit agree to the extent that the refit lands on the same point of a flat likelihood
  (docs/deseq2.md, known difference 2), a deviation that is far smaller than the hat convention makes,
* the result columns of genes whose fit converged in CREST and whose counts were not replaced in either tool.

Tolerances are named constants: ten times the larger of the deviations measured on macOS arm64 and Linux x86-64
(``deseq2_replace_hat_measurements.json``; ``measure_all()`` regenerates it), rounded up to one digit and never below
``RTOL_FLOOR``. Deviation = |crest - R| / (ATOL + |R|).
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
NAME = "replace_hat"
BETA_MAXIT = 100             # betaIter at this value: the coefficient fit did not converge and the row was refit
ATOL = 1e-12
RTOL_FLOOR = 1e-12
FLIPPED_GENE = "gene629"     # its largest Cook's distance sits 0.2% above the cut-off in R and below it with the refit hat
RESULT_COLUMNS = ("log2FoldChange", "lfcSE", "stat", "pvalue", "dispMAP", "dispersion", "dispFit")   # no padj: Benjamini-Hochberg ranks over the refit rows


def _arr(x):
    return pl.Series(x).cast(pl.Float64).fill_null(np.nan).to_numpy() if not isinstance(x, np.ndarray) else x.astype(float)


def _dev(a, b):
    """(positions where exactly one side is NaN, max |a - b| / (ATOL + |b|) where both are finite)."""
    a, b = _arr(a), _arr(b)
    na, nb = np.isnan(a), np.isnan(b)
    ok = ~na & ~nb
    return int((na != nb).sum()), float(np.max(np.abs(a[ok] - b[ok]) / (ATOL + np.abs(b[ok])))) if ok.any() else 0.0


@functools.lru_cache(maxsize=None)
def _run(min_rep):
    inp = pl.read_csv(DATA / f"deseq2_{NAME}_input.csv")
    genes = [c for c in inp.columns if c.startswith("gene")]
    meta = json.loads((DATA / f"deseq2_{NAME}_meta.json").read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = crest.tl.DESeq2(inp.select(genes).to_numpy().astype(float), inp.drop(genes), meta["design"],
                             ref_levels={k: v[0] for k, v in meta["levels"].items()}, var_names=genes, quiet=True,
                             min_replicates_for_replace=min_rep)
    return ds, meta, genes


def _ref(tag):
    return pl.read_csv(DATA / f"deseq2_{NAME}{tag}_R.csv", null_values="NA", infer_schema_length=None)


def measure_all():
    """Deviations of CREST from R on the fixture; the numbers behind the tolerances below."""
    ds, meta, genes = _run(7)
    f, ref = ds.fit, _ref("")
    res = ds.results(tuple(meta["contrast"]))
    rep_c, rep_r = np.asarray(f["replace"], bool), pl.Series(ref["replace"]).cast(pl.Boolean).fill_null(False).to_numpy()
    nonconv = np.asarray(f["betaIter"]) >= BETA_MAXIT
    out = {"n_genes": len(genes), "n_replaced_R": int(rep_r.sum()), "n_replaced_CREST": int(rep_c.sum()),
           "replace_flag_mismatch": int((rep_c != rep_r).sum()), "n_nonconverged_irls": int(nonconv.sum()),
           "baseMean": _dev(f["baseMean"], ref["baseMean"]), "dispersion": _dev(f["dispersion"], ref["dispersion"])}
    # first pass: no replacement; Cook's distance of the observed counts
    f0, ref0 = _run(float("inf"))[0].fit, _ref("_noreplace")
    nc0 = np.asarray(f0["betaIter"]) >= BETA_MAXIT
    mc, rc = _arr(f0["maxCooks"]), _arr(ref0["maxCooks"])
    out["maxCooks_first_pass_converged"] = _dev(mc[~nc0], rc[~nc0])
    out["maxCooks_first_pass_refit"] = _dev(mc[nc0], rc[nc0])
    out["n_refit_first_pass"] = int(nc0.sum())
    # genes with a converged fit and without replacement in either tool
    rest = ~nonconv & ~rep_c & ~rep_r
    out["n_rest"] = int(rest.sum())
    out["rest"] = {col: _dev((_arr(res[col]) if col in res.columns else _arr(f[col]))[rest], _arr(ref[col])[rest]) for col in RESULT_COLUMNS}
    return out


# ===================================================================================== tolerances (measured)
BASEMEAN_RTOL = 1e-12        # base means after replacement (sums of a few doubles)
DISPERSION_RTOL = 2e-8       # final dispersions of all genes
COOKS_CONVERGED_RTOL = 1e-9  # first-pass maxCooks, rows whose coefficient fit converges
COOKS_REFIT_RTOL = 2e-2      # first-pass maxCooks, refit rows: the refit lands within ~0.1% of R's point; the stale-hat error is 4-7%
REST_RTOL = {"log2FoldChange": 5e-10, "lfcSE": 4e-9, "stat": 4e-9, "pvalue": 4e-9, "dispMAP": 1e-8, "dispersion": 1e-8, "dispFit": 8e-8}


def _tol(x):
    return max(x, RTOL_FLOOR)


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="replaced flag")
def test_replaced_genes_match_r():
    ds, meta, genes = _run(7)
    assert meta["R"]["deseq2_version"].startswith("1.50")
    f, ref = ds.fit, _ref("")
    rep_c, rep_r = np.asarray(f["replace"], bool), pl.Series(ref["replace"]).cast(pl.Boolean).fill_null(False).to_numpy()
    assert int(rep_r.sum()) == 337 and int(rep_c.sum()) == 337, f"replaced genes: CREST {rep_c.sum()}, R {rep_r.sum()}"
    assert (rep_c == rep_r).all(), f"{int((rep_c != rep_r).sum())} genes differ in the replaced flag: {[genes[i] for i in np.flatnonzero(rep_c != rep_r)]}"
    j = genes.index(FLIPPED_GENE)
    assert rep_r[j] and rep_c[j], f"{FLIPPED_GENE} is replaced in R and must be replaced here"
    assert int((np.asarray(f["betaIter"]) >= BETA_MAXIT).sum()) >= 300, "the fixture must exercise refit rows"


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="baseMean")
def test_base_means_after_replacement_match_r():
    ds, meta, genes = _run(7)
    ref = _ref("")
    n_na, dev = _dev(ds.fit["baseMean"], ref["baseMean"])
    assert n_na == 0 and dev <= _tol(BASEMEAN_RTOL), f"baseMean: {n_na} NA mismatches, max deviation {dev:.3g} > {_tol(BASEMEAN_RTOL):.3g}"
    j = genes.index(FLIPPED_GENE)
    assert abs(ds.fit["baseMean"][j] - ref["baseMean"][j]) <= _tol(BASEMEAN_RTOL) * ref["baseMean"][j], f"baseMean of {FLIPPED_GENE}"


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="dispersion")
def test_dispersions_follow_the_same_replacement_decisions():
    ds, meta, genes = _run(7)
    n_na, dev = _dev(ds.fit["dispersion"], _ref("")["dispersion"])
    assert n_na == 0 and dev <= _tol(DISPERSION_RTOL), f"dispersion: {n_na} NA mismatches, max deviation {dev:.3g} > {_tol(DISPERSION_RTOL):.3g}"


def test_first_pass_cooks_distance_of_converged_rows_matches_r():
    ds, meta, genes = _run(float("inf"))
    f, ref = ds.fit, _ref("_noreplace")
    nc = np.asarray(f["betaIter"]) >= BETA_MAXIT
    assert 300 <= int(nc.sum()) < len(genes)
    n_na, dev = _dev(_arr(f["maxCooks"])[~nc], _arr(ref["maxCooks"])[~nc])
    assert n_na == 0 and dev <= _tol(COOKS_CONVERGED_RTOL), f"maxCooks of converged rows: {n_na} NA mismatches, max deviation {dev:.3g}"


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="maxCooks of refit rows")
def test_first_pass_cooks_distance_of_refit_rows_uses_the_last_irls_hat():
    ds, meta, genes = _run(float("inf"))
    f, ref = ds.fit, _ref("_noreplace")
    nc = np.asarray(f["betaIter"]) >= BETA_MAXIT
    n_na, dev = _dev(_arr(f["maxCooks"])[nc], _arr(ref["maxCooks"])[nc])
    assert n_na == 0 and dev <= _tol(COOKS_REFIT_RTOL), f"maxCooks of refit rows: {n_na} NA mismatches, max deviation {dev:.3g} > {_tol(COOKS_REFIT_RTOL):.3g}"


def test_results_of_converged_unreplaced_genes_match_r():
    ds, meta, genes = _run(7)
    f, ref = ds.fit, _ref("")
    res = ds.results(tuple(meta["contrast"]))
    rep_c, rep_r = np.asarray(f["replace"], bool), pl.Series(ref["replace"]).cast(pl.Boolean).fill_null(False).to_numpy()
    rest = (np.asarray(f["betaIter"]) < BETA_MAXIT) & ~rep_c & ~rep_r
    assert int(rest.sum()) >= 250
    for col in RESULT_COLUMNS:
        got = _arr(res[col]) if col in res.columns else _arr(f[col])
        n_na, dev = _dev(got[rest], _arr(ref[col])[rest])
        assert n_na == 0 and dev <= _tol(REST_RTOL[col]), f"{col}: {n_na} NA mismatches, max deviation {dev:.3g} > {_tol(REST_RTOL[col]):.3g}"
