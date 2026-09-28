"""Pseudobulk differential expression with DESeq2, implemented in Rust.

``DESeq2`` reproduces R's ``DESeq()`` + ``results()`` (DESeq2 1.42): median-of-
ratios size factors, Cox-Reid gene-wise dispersions, parametric trend, MAP
shrinkage, NB-GLM Wald tests, Cook's distance filtering / outlier replacement and
independent filtering. The per-gene work runs in parallel in native code.

    pb = crest.tl.pseudobulk(bf, sample_key="donor", groupby="cell_type")
    res = crest.tl.pseudobulk_de(bf, sample_key="donor", design="~ condition",
                                 contrast=("condition", "stim", "ctrl"), groupby="cell_type")

Differences from R, all small and documented in ``docs/deseq2.md``: when the
parametric trend fails DESeq2 substitutes a local regression, here the mean is
used (reported in ``messages``); rows IRLS cannot fit are refined by Newton's
method rather than L-BFGS-B; for designs with <= 3 residual degrees of freedom
the prior variance uses exact integration instead of R's 10⁴-draw simulation.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from . import crest as _native
from .core import BioFrame

__all__ = ["Pseudobulk", "pseudobulk", "DESeq2", "deseq2", "pseudobulk_de"]


# --------------------------------------------------------------------------- pseudobulk
@dataclass
class Pseudobulk:
    """Summed raw counts: ``counts`` is (samples × genes), ``obs`` one row per sample."""

    counts: np.ndarray
    obs: pl.DataFrame
    var_names: List[str]

    def __repr__(self) -> str:
        return f"Pseudobulk({self.counts.shape[0]} samples × {self.counts.shape[1]} genes)"

    def subset(self, mask: np.ndarray) -> "Pseudobulk":
        mask = np.asarray(mask, bool)
        return Pseudobulk(self.counts[mask], self.obs.filter(pl.Series(mask)), self.var_names)

    def to_anndata(self):
        import anndata as ad
        import pandas as pd

        return ad.AnnData(X=self.counts, obs=self.obs.to_pandas(), var=pd.DataFrame(index=self.var_names))


def pseudobulk(bf: BioFrame, sample_key: Union[str, Sequence[str]], groupby: Optional[str] = None,
               min_cells: int = 10) -> Pseudobulk:
    """Sum raw counts per sample (and per ``groupby`` group, e.g. cell type).

    ``sample_key`` is the obs column identifying a biological sample, or several
    columns that together do (e.g. ``["donor", "condition"]`` when each donor was
    measured in both conditions).

    Uses the raw counts of the kept cells/genes regardless of any recorded
    normalisation; streamed chunk by chunk, so it works out-of-core. Pseudobulk
    samples with fewer than ``min_cells`` cells are dropped. ``obs`` keeps every
    cell-level column that is constant within each pseudobulk sample (donor
    covariates such as condition or batch), plus ``n_cells``.
    """
    keys = ([sample_key] if isinstance(sample_key, str) else list(sample_key)) + ([groupby] if groupby else [])
    obs = bf.obs.with_row_index("__row")
    valid = obs.filter(pl.all_horizontal([pl.col(k).is_not_null() for k in keys]))
    groups = valid.group_by(keys, maintain_order=True).agg(pl.col("__row"), pl.len().alias("n_cells"))
    groups = groups.filter(pl.col("n_cells") >= min_cells).sort(keys)
    if groups.height == 0:
        raise ValueError(f"no ({', '.join(keys)}) combination has >= {min_cells} cells")
    code = np.full(bf.n_obs, np.iinfo(np.uint32).max, np.uint32)
    for i, rows in enumerate(groups["__row"].to_list()):
        code[np.asarray(rows, np.int64)] = i
    sums = np.zeros((groups.height, bf.n_vars), np.float64)
    for ctx in bf.iter_ctx(transform=False):
        _native.group_gene_totals(*ctx, code, sums)

    # sample-level covariates: columns constant within every pseudobulk sample
    other = [c for c in bf.obs.columns if c not in keys]
    const = []
    if other:
        nu = valid.group_by(keys).agg([pl.col(c).n_unique().alias(c) for c in other])
        const = [c for c in other if nu[c].max() == 1]
    meta = valid.group_by(keys, maintain_order=True).agg([pl.col(c).first() for c in const])
    pobs = groups.select(keys + ["n_cells"]).join(meta, on=keys, how="left", maintain_order="left")
    name = pobs.select(pl.concat_str([pl.col(k).cast(pl.Utf8) for k in keys], separator="|")).to_series()
    pobs = pobs.with_columns(name.alias("pseudobulk"))
    return Pseudobulk(sums, pobs, list(bf.var_names))


# --------------------------------------------------------------------------- design
_TERM = re.compile(r"^[A-Za-z_.][A-Za-z0-9_.]*$")


def _levels(s: pl.Series, ref: Optional[str]) -> List[str]:
    if s.dtype == pl.Enum:
        levels = [str(x) for x in s.cat.get_categories().to_list() if x in set(s.cast(pl.Utf8).to_list())]
    else:
        levels = sorted({str(x) for x in s.cast(pl.Utf8).to_list()})
    if ref is not None:
        if ref not in levels:
            raise ValueError(f"reference level {ref!r} not found among {levels}")
        levels = [ref] + [x for x in levels if x != ref]
    return levels


def model_matrix(obs: pl.DataFrame, design: str, ref_levels: Optional[Dict[str, str]] = None):
    """Treatment-coded model matrix for an additive formula ``"~ a + b"``.

    Categorical (string / categorical / boolean) columns get one indicator per
    non-reference level, named like DESeq2's ``resultsNames``
    (``condition_treated_vs_control``); numeric columns enter as is. Levels are
    sorted, the first is the reference unless ``ref_levels`` says otherwise.
    Returns (X, coef_names, factors) with ``factors[name] = levels``.
    """
    ref_levels = ref_levels or {}
    rhs = design.strip()
    if not rhs.startswith("~"):
        raise ValueError('design must be a formula like "~ condition" or "~ batch + condition"')
    terms = [t.strip() for t in rhs[1:].split("+") if t.strip()]
    if any(not _TERM.match(t) for t in terms if t != "1"):
        raise ValueError("only additive designs of column names are supported (no ':', '*', '-', '0'); "
                         "pass a numeric model matrix to DESeq2(design_matrix=...) for anything else")
    terms = [t for t in terms if t != "1"]
    n = obs.height
    cols, names, factors = [np.ones(n)], ["Intercept"], {}
    for t in terms:
        if t not in obs.columns:
            raise KeyError(f"design variable {t!r} is not a column of obs")
        s = obs[t]
        if s.null_count():
            raise ValueError(f"design variable {t!r} has missing values")
        if s.dtype.is_numeric() and s.dtype != pl.Boolean:
            cols.append(s.cast(pl.Float64).to_numpy())
            names.append(t)
            continue
        lv = _levels(s, ref_levels.get(t))
        factors[t] = lv
        v = np.array(s.cast(pl.Utf8).to_list(), dtype=object)
        for level in lv[1:]:
            cols.append((v == level).astype(np.float64))
            names.append(f"{t}_{level}_vs_{lv[0]}")
    X = np.ascontiguousarray(np.column_stack(cols))
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("the model matrix is not full rank: some variables are linear combinations of others "
                         "or some levels have no samples")
    return X, names, factors


# --------------------------------------------------------------------------- results helpers
def _bh(p: np.ndarray) -> np.ndarray:
    """p.adjust(method='BH') with NaNs left as NaN (n = number of non-NaN p-values)."""
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = len(q)
    if n:
        o = np.argsort(q)[::-1]
        adj = np.minimum.accumulate(q[o] * n / np.arange(n, 0, -1))
        r = np.empty(n)
        r[o] = np.minimum(adj, 1.0)
        out[ok] = r
    return out


def _independent_filtering(pvalue: np.ndarray, filt: np.ndarray, alpha: float):
    """DESeq2 ``pvalueAdjustment``: pick the baseMean quantile that maximises rejections."""
    lower = float(np.mean(filt == 0))
    upper = 0.95 if lower < 0.95 else 1.0
    theta = np.linspace(lower, upper, 50)
    cutoffs = np.quantile(filt, theta)  # R quantile type 7
    padj_all = np.full((len(filt), len(theta)), np.nan)
    for i, c in enumerate(cutoffs):
        use = filt >= c
        if use.any():
            padj_all[use, i] = _bh(pvalue[use])
    with np.errstate(invalid="ignore"):
        num_rej = np.nansum(padj_all < alpha, axis=0).astype(np.float64)
    fit = _native.lowess(np.ascontiguousarray(theta), np.ascontiguousarray(num_rej), 1.0 / 5.0, 3, None)
    if num_rej.max() <= 10:
        j = 0
    else:
        resid = np.array([0.0]) if np.all(num_rej == 0) else num_rej[num_rej > 0] - fit[num_rej > 0]
        max_fit = fit.max()
        thresh = max_fit - np.sqrt(np.mean(resid ** 2))
        for cand in (num_rej > thresh, num_rej > 0.9 * max_fit, num_rej > 0.8 * max_fit):
            if cand.any():
                j = int(np.argmax(cand))
                break
        else:
            j = 0
    info = {"filterThreshold": float(cutoffs[j]), "filterTheta": float(theta[j]),
            "filterNumRej": pl.DataFrame({"theta": theta, "numRej": num_rej})}
    return padj_all[:, j], info


# --------------------------------------------------------------------------- main class
class DESeq2:
    """Fit DESeq2 to a (samples × genes) integer count matrix.

    Parameters
    ----------
    counts : (samples, genes) array or :class:`Pseudobulk`
    obs : sample metadata (one row per sample); taken from the Pseudobulk if omitted
    design : additive formula (``"~ batch + condition"``)
    design_matrix, coef_names : a numeric model matrix instead of a formula
    ref_levels : reference level per factor (default: first in sorted order)
    sf_type : ``"ratio"`` (DESeq2 default) or ``"poscounts"`` (for data where every gene has a zero)
    fit_type : ``"parametric"`` (default) or ``"mean"``
    min_replicates_for_replace : replace Cook's outliers in cells with at least this many samples (7)
    test : ``"Wald"`` (default) or ``"LRT"``; the LRT compares ``design`` with ``reduced``
    reduced : reduced formula for the LRT (e.g. ``"~ batch"``, or ``"~ 1"``), or a numeric
        model matrix when ``design_matrix`` is used
    """

    def __init__(self, counts, obs: Optional[pl.DataFrame] = None, design: str = "~ condition", *,
                 design_matrix: Optional[np.ndarray] = None, coef_names: Optional[Sequence[str]] = None,
                 ref_levels: Optional[Dict[str, str]] = None, var_names: Optional[Sequence[str]] = None,
                 size_factors: Optional[np.ndarray] = None, sf_type: str = "ratio", fit_type: str = "parametric",
                 min_replicates_for_replace: float = 7, min_mu: float = 0.5, quiet: bool = False,
                 test: str = "Wald", reduced: Union[None, str, np.ndarray] = None):
        if isinstance(counts, Pseudobulk):
            obs = counts.obs if obs is None else obs
            var_names = counts.var_names if var_names is None else var_names
            counts = counts.counts
        counts = np.asarray(counts, dtype=np.float64)
        if counts.ndim != 2:
            raise ValueError("counts must be a 2-D (samples × genes) array")
        m, g = counts.shape
        if design_matrix is not None:
            X = np.ascontiguousarray(design_matrix, dtype=np.float64)
            self.coef_names = list(coef_names) if coef_names is not None else [f"x{i}" for i in range(X.shape[1])]
            self.factors: Dict[str, List[str]] = {}
            self.design = None
        else:
            if obs is None:
                raise ValueError("obs (sample metadata) is required with a design formula")
            X, self.coef_names, self.factors = model_matrix(obs, design, ref_levels)
            self.design = design
        if X.shape[0] != m:
            raise ValueError(f"design has {X.shape[0]} rows but counts has {m} samples")
        self.obs = obs
        self.var_names = list(var_names) if var_names is not None else [str(i) for i in range(g)]
        self.X = X
        self.counts = counts  # samples × genes, original
        sf = None if size_factors is None else np.ascontiguousarray(size_factors, dtype=np.float64)
        if test not in ("Wald", "LRT"):
            raise ValueError('test must be "Wald" or "LRT"')
        self.test = test
        Xr = None
        if test == "LRT":
            if reduced is None:
                raise ValueError('test="LRT" needs a reduced design, e.g. reduced="~ 1"')
            if isinstance(reduced, str):
                if obs is None:
                    raise ValueError("a reduced formula needs obs")
                Xr, self.reduced_names, _ = model_matrix(obs, reduced, ref_levels)
            else:
                Xr = np.ascontiguousarray(reduced, dtype=np.float64)
                self.reduced_names = [f"x{i}" for i in range(Xr.shape[1])]
            if Xr.shape[1] >= X.shape[1]:
                raise ValueError("the reduced design must have fewer coefficients than the full design")
            self.reduced = reduced
        self.fit = _native.deseq2_fit(np.ascontiguousarray(counts.T), X, sf, sf_type, fit_type,
                                      float(min_replicates_for_replace), float(min_mu), Xr)
        if not quiet:
            for msg in self.fit["messages"]:
                warnings.warn(msg, stacklevel=2)

    # ---------------------------------------------------------------- accessors
    @property
    def size_factors(self) -> np.ndarray:
        return self.fit["size_factors"]

    def results_names(self) -> List[str]:
        return list(self.coef_names)

    def dispersions(self) -> pl.DataFrame:
        f = self.fit
        return pl.DataFrame({"gene": self.var_names, "baseMean": f["baseMean"], "dispGeneEst": f["dispGeneEst"],
                             "dispFit": f["dispFit"], "dispMAP": f["dispMAP"], "dispersion": f["dispersion"],
                             "dispOutlier": f["dispOutlier"]})

    # ---------------------------------------------------------------- contrasts
    def _contrast(self, contrast):
        """(lfc, se, stat, pvalue, all-zero mask) on the log2 scale."""
        f, names = self.fit, self.coef_names
        beta, cov = f["beta"], f["beta_cov"]
        counts = f.get("replaceCounts", self.counts.T)  # genes × samples used for the fit

        def coef(i, sign=1.0):
            se = np.sqrt(np.maximum(cov[:, i, i], 0))
            return sign * beta[:, i], se

        zero = np.zeros(beta.shape[0], bool)
        if contrast is None:
            contrast = names[-1]
        if isinstance(contrast, str):
            if contrast not in names:
                raise KeyError(f"{contrast!r} not in results_names(): {names}")
            lfc, se = coef(names.index(contrast))
        elif (isinstance(contrast, (tuple, list)) and len(contrast) == 3 and all(isinstance(c, str) for c in contrast)):
            fac, num, den = contrast
            if fac not in self.factors:
                raise KeyError(f"{fac!r} is not a factor in the design")
            lv = self.factors[fac]
            if num not in lv or den not in lv:
                raise ValueError(f"{num!r} and {den!r} must be levels of {fac}: {lv}")
            ref = lv[0]
            col = lambda level: names.index(f"{fac}_{level}_vs_{ref}")  # noqa: E731
            if den == ref:
                lfc, se = coef(col(num))
            elif num == ref:
                lfc, se = coef(col(den), -1.0)
            else:
                c = np.zeros(len(names))
                c[col(num)], c[col(den)] = 1.0, -1.0
                lfc = beta @ c
                se = np.sqrt(np.maximum(np.einsum("i,gij,j->g", c, cov, c), 0))
            v = np.array(self.obs[fac].cast(pl.Utf8).to_list(), dtype=object)
            sel = np.isin(v, [num, den])
            zero = np.all(self.counts.T[:, sel] == 0, axis=1)
        else:
            c = np.asarray(contrast, dtype=np.float64)
            if c.shape != (len(names),):
                raise ValueError(f"numeric contrast must have length {len(names)}")
            lfc = beta @ c
            se = np.sqrt(np.maximum(np.einsum("i,gij,j->g", c, cov, c), 0))
            if not (np.all(c >= 0) or np.all(c <= 0)):
                which = (self.X @ (c != 0).astype(float)) != 0
                zero = counts[:, which].sum(axis=1) == 0
        zero &= ~f["allZero"]
        if self.test == "LRT":  # LFC from the contrast, statistic and p-value from the LRT
            stat, p = f["LRTStatistic"].copy(), f["LRTPvalue"].copy()
            lfc = lfc.copy()
            lfc[zero] = 0.0
            return lfc, se, stat, p
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = lfc / se
        p = _native.normal_pvalues(np.ascontiguousarray(stat))
        lfc, stat, p = lfc.copy(), stat.copy(), p.copy()
        lfc[zero], stat[zero], p[zero] = 0.0, 0.0, 1.0
        return lfc, se, stat, p

    def results(self, contrast=None, *, alpha: float = 0.1, cooks_cutoff: Union[None, bool, float] = None,
                independent_filtering: bool = True) -> pl.DataFrame:
        """DESeq2 ``results()``.

        ``contrast``: a coefficient name from :meth:`results_names`, a
        ``(factor, numerator, denominator)`` tuple, or a numeric vector over the
        coefficients. Default: the last coefficient. Returns gene, baseMean,
        log2FoldChange, lfcSE, stat, pvalue, padj (null where DESeq2 gives NA).
        With ``test="LRT"``, stat/pvalue are the likelihood-ratio test of the full
        against the reduced design (whatever the contrast); the contrast only picks
        the reported fold change, as in DESeq2.
        """
        f = self.fit
        m, p = self.X.shape
        lfc, se, stat, pvalue = self._contrast(contrast)
        pvalue = pvalue.astype(np.float64).copy()
        info: dict = {}
        # Cook's distance filtering
        if cooks_cutoff is not False and m > p:
            cutoff = float(cooks_cutoff) if isinstance(cooks_cutoff, (int, float)) and not isinstance(cooks_cutoff, bool) \
                else f["cooksCutoff"]
            with np.errstate(invalid="ignore"):
                out = f["maxCooks"] > cutoff
            if out.any() and len(self.factors) == 1 and self.X.shape[1] == 2 and self.design is not None \
                    and len(next(iter(self.factors.values()))) == 2 and _single_var(self.design):
                cooks, cts = f["cooks"], self.counts.T
                for i in np.flatnonzero(out):
                    row = cooks[i]
                    jmax = int(np.nanargmax(row))
                    if np.sum(cts[i] > cts[i, jmax]) >= 3:
                        out[i] = False
            pvalue[out] = np.nan
            info["cooksCutoff"] = cutoff
        # rows whose counts became all zero after outlier replacement
        if f["replace"].any():
            now_zero = f["replace"] & (f["baseMean"] == 0)
            lfc, se, stat = lfc.copy(), se.copy(), stat.copy()
            lfc[now_zero], se[now_zero] = 0.0, 0.0
            if self.test == "Wald":
                stat[now_zero], pvalue[now_zero] = 0.0, 1.0
        base_mean = f["baseMean"]
        if independent_filtering:
            padj, filt_info = _independent_filtering(pvalue, base_mean, alpha)
            info.update(filt_info)
        else:
            padj = _bh(pvalue)
        self.last_results_info = info
        nan_to_null = lambda a: pl.Series(a, dtype=pl.Float64).fill_nan(None)  # noqa: E731
        return pl.DataFrame({
            "gene": self.var_names, "baseMean": base_mean,
            "log2FoldChange": nan_to_null(lfc), "lfcSE": nan_to_null(se), "stat": nan_to_null(stat),
            "pvalue": nan_to_null(pvalue), "padj": nan_to_null(padj),
        })


def _single_var(design: str) -> bool:
    return len([t for t in design.strip()[1:].split("+") if t.strip() and t.strip() != "1"]) == 1


# --------------------------------------------------------------------------- convenience
def deseq2(counts, obs: Optional[pl.DataFrame] = None, design: str = "~ condition", contrast=None, *,
           alpha: float = 0.1, **kwargs) -> pl.DataFrame:
    """One call: fit :class:`DESeq2` and return ``results(contrast)``."""
    return DESeq2(counts, obs, design, **kwargs).results(contrast, alpha=alpha)


def pseudobulk_de(bf: BioFrame, sample_key: Union[str, Sequence[str]], design: str = "~ condition", contrast=None, *,
                  groupby: Optional[str] = None, min_cells: int = 10, alpha: float = 0.1,
                  key_added: str = "pseudobulk_de", **kwargs) -> pl.DataFrame:
    """Pseudobulk DESeq2 per ``groupby`` group (e.g. per cell type), in one pass over the data.

    Counts are summed per (sample, group); each group is then tested with
    ``design``/``contrast``. Groups whose design is not estimable (e.g. a
    condition missing, or no replicates) are skipped with a warning. Returns a
    long DataFrame with a ``group`` column; also stored in ``bf.uns[key_added]``.
    """
    pb = pseudobulk(bf, sample_key, groupby, min_cells=min_cells)
    frames, skipped = [], {}
    groups = [None] if groupby is None else pb.obs[groupby].unique(maintain_order=True).to_list()
    for gname in groups:
        sub = pb if gname is None else pb.subset((pb.obs[groupby] == gname).to_numpy())
        try:
            res = DESeq2(sub, design=design, **kwargs).results(contrast, alpha=alpha)
        except (ValueError, KeyError) as e:
            skipped[str(gname)] = str(e)
            continue
        if gname is not None:
            res = res.with_columns(pl.lit(str(gname)).alias("group")).select(["group"] + res.columns)
        frames.append(res)
    for gname, why in skipped.items():
        warnings.warn(f"pseudobulk_de: skipped group {gname!r}: {why}", stacklevel=2)
    out = pl.concat(frames) if frames else pl.DataFrame()
    bf.uns[key_added] = {"params": {"sample_key": sample_key, "design": design, "contrast": contrast,
                                    "groupby": groupby, "min_cells": min_cells}, "skipped": skipped, "table": out}
    return out
