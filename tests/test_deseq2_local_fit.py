"""DESeq2's local dispersion trend (``fitType = "local"``, R's ``locfit``) against R.

Fixtures (``tests/data``): ``deseq2_local_fit_cases.csv`` / ``_at.csv`` are written by ``make_local_fit_inputs.py`` (synthetic
(mean, gene-wise dispersion) pairs built to hit the cases of locfit's evaluation structure), ``deseq2_local_fit_covid_cases.csv``
by ``bench/deseq2/covid_trend_inputs.py`` (R's base means and gene-wise dispersions for six strata of the Stephenson et al.
COVID-19 atlas where R's parametric trend fails), and ``deseq2_trend_mdc_inputs.csv`` likewise for the stratum where it does not.
The reference output (``deseq2_local_fit_R.csv``, ``deseq2_local_fit_R_status.json``, ``deseq2_trend_mdc_R.json``) comes from
``make_local_fit_fixtures.R`` and ``covid_trend_inputs.py`` (R 4.5.2, DESeq2 1.50.2, locfit 1.5-9.12, recorded in the files).

The comparison is of the trend function ``exp(predict(locfit(log(disp) ~ log(mean), weights = mean), log(at)))`` at the means
of each case and on a grid that reaches beyond the range of the means on both sides (R extrapolates with the quadratic of
locfit's parametric component), through ``crest.crest.local_dispersion_fit``, a test-only entry point of the native code.

Tolerances are named constants: ten times the larger of the deviations measured on Linux x86-64 and on macOS arm64
(``deseq2_local_fit_measurements.json``; ``measure_all()`` regenerates it), rounded up to one digit and never below 1e-12 for
quantities that carry 15 digits in the reference files. Deviation of the trend = |crest / R - 1|; of the table columns
|crest - R| / (1e-12 + |R|).
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

DATA = pathlib.Path(__file__).parent / "data"
MIN_DISP = 1e-8
LOCFIT_MIN_DISP_FACTOR = 10.0    # localDispersionFit leaves out gene-wise dispersions below 10 * minDisp

# --- tolerances (named constants) -------------------------------------------------------------
TREND_RTOL = 5e-13               # the trend function on the synthetic cases and the real strata (measured max 4.1e-14, both platforms)
PARAMETRIC_COEF_RTOL = 2e-12     # parametric coefficients on R's gene-wise values of the myeloid dendritic cell stratum (measured 2e-13)
PIN_RTOL_TREND_FALLBACK = {      # whole DESeq() fixture where R falls back to local (measured: see deseq2_local_fit_measurements.json)
    "size_factors": 1e-12, "baseMean": 1e-12, "dispGeneEst": 4e-12, "dispFit": 3e-12, "dispMAP": 2e-12,
    "dispersion": 2e-12, "log2FoldChange": 2e-12, "lfcSE": 1e-12, "stat": 2e-12, "pvalue": 1e-9,
    "padj": 5e-10, "maxCooks": 1e-12,
}
EXTREME_RTOL = {                 # edge_sparse_extreme, genes whose dispersion is below the cap: the trend and what follows
    "baseMean": 1e-12, "dispFit": 5e-10, "dispMAP": 4e-5, "dispersion": 4e-5, "maxCooks": 1e-12,
    "log2FoldChange": 2e-4, "lfcSE": 2e-5, "stat": 2e-4, "pvalue": 2e-4,
}
EXTREME_CAPPED_DISPFIT_RTOL = 5e-10   # genes at the dispersion cap: the trend agrees, the beta fit on the flat likelihood does not
EXTREME_SIG_COUNT_RTOL = 0.05    # significant genes at padj < alpha: CREST and R differ by at most this fraction of R's count (measured 3.4%)
EXTREME_SIG_SHARED_MIN = 0.9     # and share at least this fraction of R's genes (measured 95.5%)
MAX_DISP_REL = 1e-9              # a dispersion equal to the cap, 10 or the number of samples


def _arr(x):
    return pl.Series(x).cast(pl.Float64).fill_null(np.nan).to_numpy() if not isinstance(x, np.ndarray) else x.astype(float)


def _rel(a, b):
    a, b = _arr(a), _arr(b)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[ok] / b[ok] - 1))) if ok.any() else 0.0


def _read(name, **kw):
    return pl.read_csv(DATA / name, null_values="NA", infer_schema_length=None, **kw)


@functools.lru_cache(maxsize=None)
def _cases():
    """{case: (means, disps, at, R trend)} from the synthetic and the real-strata fixtures."""
    inputs = [_read("deseq2_local_fit_cases.csv")]
    if (DATA / "deseq2_local_fit_covid_cases.csv").exists():
        inputs.append(_read("deseq2_local_fit_covid_cases.csv"))
    cases = pl.concat(inputs)
    ref = _read("deseq2_local_fit_R.csv")
    out = {}
    for name in cases["case"].unique(maintain_order=True):
        c, r = cases.filter(pl.col("case") == name), ref.filter(pl.col("case") == name)
        out[name] = (c["mean"].to_numpy(), c["disp"].to_numpy(), r["at"].to_numpy(), r["trend"].to_numpy())
    return out


@functools.lru_cache(maxsize=None)
def _status():
    return json.loads((DATA / "deseq2_local_fit_R_status.json").read_text())


def _local(means, disps, at):
    keep = disps >= MIN_DISP * LOCFIT_MIN_DISP_FACTOR
    return _native.local_dispersion_fit(means[keep], disps[keep], at)


def measure_local_trend():
    """{case: (max relative deviation of the trend from R, number of points beyond the range of the means)}."""
    out = {}
    for name, (m, d, at, r) in _cases().items():
        if _status()[name] != "ok":
            continue
        got = _local(m, d, at)
        keep = d >= MIN_DISP * LOCFIT_MIN_DISP_FACTOR
        beyond = int(((at < m[keep].min()) | (at > m[keep].max())).sum())
        out[name] = (float(np.max(np.abs(got / r - 1))), beyond)
    return out


SYNTHETIC = [c for c in _read("deseq2_local_fit_cases.csv")["case"].unique(maintain_order=True)]


# ===================================================================================== the trend function
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AttributeError, match="local_dispersion_fit")
@pytest.mark.parametrize("case", [c for c in SYNTHETIC if _status()[c] == "ok"])
def test_local_trend_matches_r_on_synthetic_cases(case):
    m, d, at, r = _cases()[case]
    got = _local(m, d, at)
    dev = float(np.max(np.abs(got / r - 1)))
    assert dev <= TREND_RTOL, f"{case}: max relative deviation {dev:.3g} > {TREND_RTOL:.3g}"


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AttributeError, match="local_dispersion_fit")
def test_local_trend_extrapolates_like_r():
    """Beyond the range of the means R's curve is quadratic (locfit's parametric component), not linear: the grid of every
    synthetic case reaches 2.5 log units past the data on both sides and agrees."""
    n_beyond = 0
    for case in SYNTHETIC:
        if _status()[case] != "ok":
            continue
        m, d, at, r = _cases()[case]
        keep = d >= MIN_DISP * LOCFIT_MIN_DISP_FACTOR
        out = (at < m[keep].min()) | (at > m[keep].max())
        n_beyond += int(out.sum())
        got = _local(m, d, at)
        assert np.max(np.abs(got[out] / r[out] - 1)) <= TREND_RTOL, case
    assert n_beyond >= 500


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AttributeError, match="local_dispersion_fit")
def test_local_trend_stops_where_r_stops():
    """Two gene-wise dispersions: R's tree needs more vertices than it allocates ('out of vertex space'); so does this port."""
    errors = [c for c in SYNTHETIC if _status()[c] != "ok"]
    assert errors == ["few_genes_2"], errors
    m, d, at, r = _cases()["few_genes_2"]
    assert np.isnan(r).all() and "vertex space" in _status()["few_genes_2"]
    with pytest.raises(ValueError, match="vertex space|no points"):
        _local(m, d, at)


@pytest.mark.skipif(not (DATA / "deseq2_local_fit_covid_cases.csv").exists(), reason="real-strata fixture not generated")
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AttributeError, match="local_dispersion_fit")
def test_local_trend_matches_r_on_covid_strata_where_the_parametric_fit_fails():
    names = [c for c in _cases() if c not in SYNTHETIC]
    assert len(names) == 6
    for case in names:
        m, d, at, r = _cases()[case]
        assert _status()[case] == "ok" and len(m) >= 1000
        dev = float(np.max(np.abs(_local(m, d, at) / r - 1)))
        assert dev <= TREND_RTOL, f"{case}: max relative deviation {dev:.3g} > {TREND_RTOL:.3g}"


# ===================================================================================== the myeloid dendritic cell stratum
@pytest.mark.skipif(not (DATA / "deseq2_trend_mdc_inputs.csv").exists(), reason="stratum fixture not generated")
@pytest.mark.baseline_fails(ref="v0.4.0", raises=AttributeError, match="parametric_dispersion_fit")
def test_parametric_fit_follows_r_on_the_myeloid_dendritic_cell_stratum():
    """R's parametric trend succeeds on R's gene-wise dispersions of this stratum and fails (\"dispersion fit did not
    converge\") on CREST's, which differ for 19 genes whose likelihood is flat over four orders of magnitude of the
    dispersion. The native Gamma-GLM loop reproduces both outcomes: R's coefficients on the first set, the same failure on
    the second (CREST therefore falls back to the local trend there; R does not, because its gene-wise values differ)."""
    x = _read("deseq2_trend_mdc_inputs.csv")
    meta = json.loads((DATA / "deseq2_trend_mdc_R.json").read_text())
    assert meta["R_gene_wise"]["ok"] and not meta["CREST_gene_wise"]["ok"]
    assert "did not converge" in meta["CREST_gene_wise"]["error"]
    floor = 100 * MIN_DISP
    for key, col in (("R_gene_wise", "disp_R"), ("CREST_gene_wise", "disp_CREST")):
        use = x[col].is_not_null() & (x[col] > floor)
        m, d = x.filter(use)["mean"].to_numpy(), x.filter(use)[col].to_numpy()
        assert len(m) == meta[key]["n"]
        got = _native.parametric_dispersion_fit(m, d)
        if meta[key]["ok"]:
            assert got is not None
            for a, b in zip(got, meta[key]["coefficients"]):
                assert abs(a / b - 1) <= PARAMETRIC_COEF_RTOL, (key, got, meta[key]["coefficients"])
        else:
            assert got is None, f"{key}: R stops ({meta[key]['error']}), the native fit returned {got}"


# ===================================================================================== whole DESeq() fixtures
def _fixture(name):
    inp = _read(f"deseq2_{name}_input.csv")
    genes = [c for c in inp.columns if c.startswith("gene")]
    meta = json.loads((DATA / f"deseq2_{name}_meta.json").read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = crest.tl.DESeq2(inp.select(genes).to_numpy().astype(float), inp.drop(genes), meta["design"],
                             ref_levels={k: v[0] for k, v in meta["levels"].items()}, var_names=genes, quiet=True)
    res = ds.results(tuple(meta["contrast"]))
    return ds, res, _read(f"deseq2_{name}_R.csv"), meta


@functools.lru_cache(maxsize=None)
def _fixture_cached(name):
    return _fixture(name)


def _dev(a, b):
    a, b = _arr(a), _arr(b)
    na, nb = np.isnan(a), np.isnan(b)
    ok = ~na & ~nb
    return int((na != nb).sum()), float(np.max(np.abs(a[ok] - b[ok]) / (1e-12 + np.abs(b[ok])))) if ok.any() else 0.0


def measure_trend_fallback():
    ds, res, ref, meta = _fixture_cached("trend_fallback")
    f = ds.fit
    out = {"size_factors": _dev(ds.size_factors, np.array(meta["R"]["sf"]))}
    for col, key in (("baseMean", "baseMean"), ("dispGeneEst", "dispGeneEst"), ("dispFit", "dispFit"), ("dispMAP", "dispMAP"),
                     ("dispersion", "dispersion"), ("maxCooks", "maxCooks")):
        a, b = _arr(f[key]), _arr(ref[col])
        if col == "dispGeneEst":
            both = (a > 1e-6) & (b > 1e-6)
            out[col] = (int(((a <= 1e-6) != (b <= 1e-6)).sum()), _dev(a[both], b[both])[1])
        elif col == "maxCooks":
            near = np.isfinite(a) & np.isfinite(b) & (b < meta["R"]["cooksCutoff"])
            out[col] = (0, _dev(a[near], b[near])[1])
        else:
            out[col] = _dev(a, b)
    for col in ("log2FoldChange", "lfcSE", "stat", "pvalue", "padj"):
        out[col] = _dev(res[col], ref[col])
    return out


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="fitType")
def test_trend_fallback_to_local_matches_r():
    """Where R's parametric trend fails and DESeq2 substitutes the local regression, so does this port, and the whole table
    (dispersions, fold changes, p-values, adjusted p-values) agrees with R."""
    ds, res, ref, meta = _fixture_cached("trend_fallback")
    assert meta["R"]["fitType"] == "local" and ds.fit["fitType"] == "local", f"fitType {ds.fit['fitType']!r}, R {meta['R']['fitType']!r}"
    assert ds.fit["fit_substituted"] is True
    got = measure_trend_fallback()
    for col, (n_na, dev) in got.items():
        assert n_na == 0, f"{col}: {n_na} genes differ in NA / floor status"
        assert dev <= PIN_RTOL_TREND_FALLBACK[col], f"{col}: max deviation {dev:.3g} > {PIN_RTOL_TREND_FALLBACK[col]:.3g}"
    sig = lambda p: set(np.flatnonzero(np.nan_to_num(_arr(p), nan=1.0) < 0.1))
    assert sig(res["padj"]) == sig(ref["padj"])


def measure_extreme():
    """edge_sparse_extreme (six samples, counts over six orders of magnitude; R's parametric fit fails): deviations of the
    columns that follow the trend, over the rows below and at the dispersion cap, and the significant sets."""
    ds, res, ref, meta = _extreme()
    f = ds.fit
    disp_r = _arr(ref["dispersion"])
    cap = disp_r >= 10 * (1 - MAX_DISP_REL)
    live = _arr(ref["baseMean"]) > 0
    out = {"n_capped": int((cap & live).sum()), "n_live": int(live.sum())}
    cols = {"baseMean": f["baseMean"], "dispFit": f["dispFit"], "dispMAP": f["dispMAP"], "dispersion": f["dispersion"],
            "maxCooks": f["maxCooks"], "log2FoldChange": _arr(res["log2FoldChange"]), "lfcSE": _arr(res["lfcSE"]),
            "stat": _arr(res["stat"]), "pvalue": _arr(res["pvalue"])}
    out["below_cap"] = {k: _dev(np.asarray(v)[~cap & live], _arr(ref[k])[~cap & live]) for k, v in cols.items()}
    out["at_cap"] = {k: _dev(np.asarray(v)[cap & live], _arr(ref[k])[cap & live]) for k, v in cols.items()}
    pc, pr = np.nan_to_num(_arr(res["padj"]), nan=1.0), np.nan_to_num(_arr(ref["padj"]), nan=1.0)
    for a in (0.1, 0.05):
        sc, sr = pc < a, pr < a
        out[f"sig_{a}"] = {"CREST": int(sc.sum()), "R": int(sr.sum()), "shared": int((sc & sr).sum())}
    return out


@functools.lru_cache(maxsize=None)
def _extreme():
    return _fixture("edge_sparse_extreme")


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="fitType")
def test_sparse_extreme_fixture_follows_r_through_the_local_trend():
    """Six samples, counts over six orders of magnitude, R's parametric trend fails. With the local trend the dispersion trend
    agrees with R's to 1e-9 on every gene, and every column agrees on the genes whose dispersion is below the cap. On the
    392 genes whose dispersion sits at the cap (10) the likelihood in the coefficients is flat, the optimisers of R (L-BFGS-B)
    and CREST (Newton) stop at different places, fold changes differ by up to several log2 units with equal standard errors
    (docs/deseq2.md, known difference 2); the adjusted p-values are then compared by the number of significant genes only."""
    ds, res, ref, meta = _extreme()
    assert meta["R"]["fitType"] == "local" and ds.fit["fitType"] == "local", f"fitType {ds.fit['fitType']!r}"
    assert ds.fit["fit_substituted"] is True
    got = measure_extreme()
    assert got["n_capped"] >= 300
    for k, tol in EXTREME_RTOL.items():
        n_na, dev = got["below_cap"][k]
        assert n_na == 0 and dev <= tol, f"below the cap: {k}: {n_na} NA mismatches, deviation {dev:.3g} > {tol:.3g}"
    n_na, dev = got["at_cap"]["dispFit"]
    assert n_na == 0 and dev <= EXTREME_CAPPED_DISPFIT_RTOL, got["at_cap"]["dispFit"]
    for a in ("0.1", "0.05"):
        s = got[f"sig_{float(a)}"]
        assert abs(s["CREST"] - s["R"]) <= EXTREME_SIG_COUNT_RTOL * s["R"] and s["shared"] >= EXTREME_SIG_SHARED_MIN * s["R"], s


# ===================================================================================== the public interface
def _small(seed=3, genes=200, m=6):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(3, 1.5, genes))
    size = 1 / (0.1 + 1.5 / mu)
    cts = rng.negative_binomial(size[:, None], size[:, None] / (size[:, None] + mu[:, None]), (genes, m)).astype(float)
    return cts, pl.DataFrame({"c": ["a"] * (m // 2) + ["b"] * (m // 2)})


@pytest.mark.baseline_fails(ref="v0.4.0", raises=ValueError, match="fit_type")
def test_local_fit_type_can_be_requested():
    cts, obs = _small()
    ds = crest.tl.DESeq2(cts.T, obs, "~ c", fit_type="local", quiet=True)
    assert ds.fit["fitType"] == "local" and ds.fit["fit_substituted"] is False and len(ds.fit["trend"]) == 0
    assert np.all(np.isfinite(ds.fit["dispFit"][ds.fit["baseMean"] > 0]))
    par = crest.tl.DESeq2(cts.T, obs, "~ c", fit_type="parametric", quiet=True)
    assert par.fit["fitType"] == "parametric"
    # both describe the same simulated curve (the local fit is weighted by the means and runs below the parametric one here)
    ok = (par.fit["baseMean"] > 5) & (par.fit["baseMean"] < 5000)
    ratio = ds.fit["dispFit"][ok] / par.fit["dispFit"][ok]
    assert 0.5 < np.median(ratio) < 2.0


def test_unknown_fit_type_is_refused():
    cts, obs = _small(genes=20)
    with pytest.raises(ValueError, match="parametric.*local.*mean"):
        crest.tl.DESeq2(cts.T, obs, "~ c", fit_type="glmGamPoi", quiet=True)


@pytest.mark.baseline_fails(ref="v0.4.0", raises=AssertionError, match="local regression fit was automatically substituted")
def test_failed_parametric_trend_is_replaced_by_the_local_fit_with_rs_note():
    """The trend-fallback fixture (120 genes, 3 v 3) is one where R's parametric fit fails; the note is R's own."""
    inp = _read("deseq2_trend_fallback_input.csv")
    genes = [c for c in inp.columns if c.startswith("gene")]
    meta = json.loads((DATA / "deseq2_trend_fallback_meta.json").read_text())
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        ds = crest.tl.DESeq2(inp.select(genes).to_numpy().astype(float), inp.drop(genes), meta["design"],
                             ref_levels={k: v[0] for k, v in meta["levels"].items()})
    msgs = [str(w.message) for w in wl if issubclass(w.category, UserWarning)]
    assert any("local regression fit was automatically substituted" in m for m in msgs), f"no warning with R's note (local regression fit was automatically substituted): {msgs}"
    assert ds.fit["fitType"] == "local" and ds.fit["fit_substituted"] is True
    r_note = meta["R"]["messages"] if isinstance(meta["R"]["messages"], str) else meta["R"]["messages"][0]
    assert ds.fit["messages"][0] == r_note


@pytest.mark.baseline_fails(ref="v0.4.0", raises=ValueError, match="fit_type")
def test_local_fit_that_stops_in_r_uses_the_mean_and_says_so():
    """Two genes with gene-wise dispersions above the floor: R's locfit stops ('out of vertex space'); CREST continues with the
    mean trend, flags the substitution and says the results can differ from R (R itself would stop)."""
    rng = np.random.default_rng(11)
    cts = np.array([[5, 40, 3, 60, 7, 20], [200, 10, 90, 400, 30, 150]], dtype=float)
    obs = pl.DataFrame({"c": ["a", "a", "a", "b", "b", "b"]})
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        ds = crest.tl.DESeq2(cts.T, obs, "~ c", fit_type="local")
    assert ds.fit["fitType"] == "mean" and ds.fit["fit_substituted"] is True
    assert any("differ from R" in str(w.message) for w in wl), [str(w.message) for w in wl]


def measure_mdc():
    x = _read("deseq2_trend_mdc_inputs.csv")
    meta = json.loads((DATA / "deseq2_trend_mdc_R.json").read_text())
    use = x["disp_R"].is_not_null() & (x["disp_R"] > 100 * MIN_DISP)
    got = _native.parametric_dispersion_fit(x.filter(use)["mean"].to_numpy(), x.filter(use)["disp_R"].to_numpy())
    return float(max(abs(a / b - 1) for a, b in zip(got, meta["R_gene_wise"]["coefficients"])))


def measure_all():
    out = {"trend": {k: {"max_rel_dev": v[0], "points_beyond_range": v[1]} for k, v in measure_local_trend().items()},
           "trend_fallback": {k: {"na_mismatch": v[0], "max_dev": v[1]} for k, v in measure_trend_fallback().items()},
           "sparse_extreme": measure_extreme()}
    out["sparse_extreme"]["below_cap"] = {k: {"na_mismatch": v[0], "max_dev": v[1]} for k, v in out["sparse_extreme"]["below_cap"].items()}
    out["sparse_extreme"]["at_cap"] = {k: {"na_mismatch": v[0], "max_dev": v[1]} for k, v in out["sparse_extreme"]["at_cap"].items()}
    out["mdc_parametric_coefficients_max_rel_dev"] = measure_mdc()
    out["reference"] = _status()["versions"]
    return out
