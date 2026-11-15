"""Pseudobulk differential expression with DESeq2, implemented in Rust.

``DESeq2`` reproduces R's ``DESeq()`` + ``results()`` (reference: R 4.5.2, DESeq2 1.50.2): median-of-
ratios size factors, Cox-Reid gene-wise dispersions, parametric trend, MAP
shrinkage, NB-GLM Wald tests, Cook's distance filtering / outlier replacement and
independent filtering. The per-gene work runs in parallel in native code.

    pb = crest.tl.pseudobulk(bf, sample_key="donor", groupby="cell_type")
    res = crest.tl.pseudobulk_de(bf, sample_key="donor", design="~ condition",
                                 contrast=("condition", "stim", "ctrl"), groupby="cell_type")

Differences from R are documented in ``docs/deseq2.md``: when the parametric
trend fails DESeq2 substitutes a local regression (``locfit``), and so does
this port (a port of ``locfit`` 1.5-9.12's evaluation, reported in ``messages`` and as
``fit["fit_substituted"]``); rows IRLS cannot fit are refined by Newton's method
rather than L-BFGS-B. Under ``test="LRT"`` the Cook's distances use the full-model
means, as DESeq2 >= 1.50 (1.42 differs). For designs with <= 3 residual degrees of freedom the
prior variance uses R's own simulated reference distribution (tabulated from ``set.seed(2)``),
so it equals R's value, which is one draw of a Monte-Carlo estimate (``docs/deseq2.md``).
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

__all__ = ["Pseudobulk", "pseudobulk", "DESeq2", "deseq2", "pseudobulk_de", "NotEstimableError", "LevelOrderWarning"]


class NotEstimableError(ValueError):
    """The data of a group (or of the whole table) cannot support the requested model:
    a factor with a single level, a rank-deficient design, no replicates, or no usable counts."""


class DesignNotFullRank(NotEstimableError):
    """The model matrix is rank deficient."""


class LevelOrderWarning(UserWarning):
    """Sorted factor levels differ between code-point order (used here) and case-insensitive order (R, en_US)."""


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
        """Keep the samples where ``mask`` is True."""
        mask = np.asarray(mask, bool)
        return Pseudobulk(self.counts[mask], self.obs.filter(pl.Series(mask)), self.var_names)

    def to_anndata(self):
        """AnnData with samples as rows (for pydeseq2 or scanpy)."""
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


def _levels(s: pl.Series, ref: Optional[str], name: Optional[str] = None) -> List[str]:
    if s.dtype == pl.Enum:
        levels = [str(x) for x in s.dtype.categories.to_list() if x in set(s.cast(pl.Utf8).to_list())]
    else:
        levels = sorted({str(x) for x in s.cast(pl.Utf8).to_list()})
        folded = sorted(levels, key=lambda v: (v.casefold(), v))
        if ref is None and folded != levels:
            warnings.warn(
                f"levels of {name!r} are sorted by code point: {levels} (reference {levels[0]!r}); R sorts factor "
                f"levels case-insensitively in an en_US locale: {folded} (reference {folded[0]!r}). The reference "
                f"level, the coefficient names and the sign of the default contrast follow the first order. Pass "
                f"ref_levels={{{name!r}: ...}} or an ordered pl.Enum column to choose the order explicitly.",
                LevelOrderWarning, stacklevel=4)
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
    sorted by code point (upper case before lower case) and the first is the
    reference unless ``ref_levels`` says otherwise; an ordered ``pl.Enum`` column
    keeps its declared order. R in an en_US locale sorts case-insensitively, so
    when the two orders differ and no reference is given a
    :class:`LevelOrderWarning` names both. Coefficient names must be unique, else
    ``ValueError`` (e.g. factor ``a`` with level ``s_x`` and factor ``a_s`` with
    level ``x``). Returns (X, coef_names, factors) with ``factors[name] = levels``.
    """
    X, names, factors = _build_design(obs, design, ref_levels)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise DesignNotFullRank("the model matrix is not full rank: some variables are linear combinations of others "
                                "or some levels have no samples")
    return X, names, factors


def _check_unique_names(names: Sequence[str]) -> None:
    seen, dup = set(), []
    for n in names:
        if n in seen and n not in dup:
            dup.append(n)
        seen.add(n)
    if dup:
        raise ValueError(f"duplicate coefficient names {dup}: two design terms produce the same name, so a contrast "
                         "could not tell them apart; rename the factor levels or the columns")


def _build_design(obs: pl.DataFrame, design: str, ref_levels: Optional[Dict[str, str]] = None):
    """``model_matrix`` without the rank check (formula, column and name validation only)."""
    if not isinstance(obs, pl.DataFrame):
        raise TypeError(f"obs must be a polars DataFrame, not {type(obs).__name__} (convert a pandas table with pl.from_pandas)")
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
        lv = _levels(s, ref_levels.get(t), t)
        factors[t] = lv
        v = np.array(s.cast(pl.Utf8).to_list(), dtype=object)
        for level in lv[1:]:
            cols.append((v == level).astype(np.float64))
            names.append(f"{t}_{level}_vs_{lv[0]}")
    _check_unique_names(names)
    return np.ascontiguousarray(np.column_stack(cols)), names, factors


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
    counts : numpy.ndarray or Pseudobulk
        Integer counts, one row per sample and one column per gene, or a
        :class:`Pseudobulk` (its ``obs`` and ``var_names`` are then used unless
        given explicitly).
    obs : polars.DataFrame, optional
        Sample metadata, one row per sample. Taken from the :class:`Pseudobulk`
        if omitted. Required with a ``design`` formula.
    design : str
        Additive formula, e.g. ``"~ batch + condition"``.
    design_matrix : numpy.ndarray, optional
        A numeric (samples × coefficients) model matrix to use instead of a
        formula (for interactions or other designs the formula parser does not
        handle).
    coef_names : list of str, optional
        Names of the columns of ``design_matrix`` (default ``x0, x1, ...``).
    ref_levels : dict of str to str, optional
        Reference level per factor. Default: the first level in code-point sorted
        order (upper case before lower case; R's en_US locale sorts case-insensitively,
        which a :class:`LevelOrderWarning` points out when the two differ). An ordered
        ``pl.Enum`` column keeps its declared order, as an R factor does.
    var_names : list of str, optional
        Gene names, used in the result tables. Default: the Pseudobulk's
        ``var_names``, or ``"0", "1", ...``.
    size_factors : numpy.ndarray, optional
        Per-sample size factors to use instead of estimating them (as assigning
        ``sizeFactors(dds)`` in R).
    sf_type : str
        ``"ratio"`` (median of ratios, the DESeq2 default) or ``"poscounts"``
        (for data where every gene has a zero). R's ``"iterate"`` is not available.
    fit_type : str
        Dispersion trend: ``"parametric"`` (default), ``"local"`` (DESeq2's ``locfit`` regression of
        the log gene-wise dispersions on the log means, weighted by the means) or ``"mean"``.
        When the parametric fit does not converge R substitutes the local regression, and so does
        this port: ``fit["fitType"]`` is ``"local"``, ``fit["fit_substituted"]`` is ``True`` and a
        ``UserWarning`` repeats R's note (``quiet=True`` suppresses the warning, not the flag).
        Where R's local regression itself stops with an error, R stops; here the mean trend is used,
        ``fit["fit_substituted"]`` is ``True`` and the warning says the results can differ from R.
    min_replicates_for_replace : float
        Replace Cook's outliers in sample groups with at least this many
        samples (default 7, as DESeq2's ``minReplicatesForReplace``). At least 3, as in R;
        ``float("inf")`` switches replacement off.
    min_mu : float
        Lower bound on the fitted means in the GLM fit (default 0.5, as
        DESeq2's ``minmu``).
    quiet : bool
        If ``False`` (default), fit messages (e.g. a failed parametric trend
        fit) are emitted as warnings. They are always kept in
        ``fit["messages"]``.
    test : str
        ``"Wald"`` (default) or ``"LRT"``. The LRT compares ``design`` with
        ``reduced``.
    reduced : str or numpy.ndarray, optional
        Reduced design for the LRT: a formula (e.g. ``"~ batch"`` or ``"~ 1"``),
        or a numeric model matrix when ``design_matrix`` is used.

    Attributes
    ----------
    fit : dict
        Everything the Rust fit returns: per-gene arrays (``baseMean``,
        ``dispGeneEst``, ``dispFit``, ``dispMAP``, ``dispersion``, ``beta``,
        ``beta_cov``, ``cooks``, and with ``test="LRT"`` ``LRTStatistic`` and
        ``LRTPvalue``, ...), ``size_factors``, ``dispPriorVar`` and
        ``messages`` (a list of notes such as a fallback to the mean
        dispersion trend), ``fitType`` (the trend used: ``"parametric"``, ``"local"`` or ``"mean"``) and
        ``fit_substituted`` (``True`` when the trend requested was replaced: parametric by local, as R
        does, or local by the mean, where R stops with an error).
    last_results_info : dict
        Set by :meth:`results`: the contrast, the Cook's cut-off, the independent-filtering threshold
        and ``"na"``, the boolean masks that say why a p-value or adjusted p-value is missing.
    last_contrast : str
        Label of the contrast ``results()`` last computed (a coefficient name,
        ``factor_num_vs_den`` for a tuple, or ``numeric: [...]``); ``None`` before.
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
        if not min_replicates_for_replace >= 3:
            raise ValueError("min_replicates_for_replace must be at least 3: a count outlier cannot be identified with fewer "
                             "replicates (DESeq2 stops in the same way); pass float('inf') to switch replacement off")
        m, g = counts.shape
        if design_matrix is not None:
            X = np.ascontiguousarray(design_matrix, dtype=np.float64)
            self.coef_names = list(coef_names) if coef_names is not None else [f"x{i}" for i in range(X.shape[1])]
            _check_unique_names(self.coef_names)
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
        self.last_contrast: Optional[str] = None
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
        """Per-sample size factors (median of ratios, or those passed in)."""
        return self.fit["size_factors"]

    def results_names(self) -> List[str]:
        """Coefficient names, as DESeq2's ``resultsNames()`` (e.g. ``condition_stim_vs_ctrl``)."""
        return list(self.coef_names)

    def dispersions(self) -> pl.DataFrame:
        """Per-gene dispersion estimates: gene-wise, trend, MAP, final, and the outlier flag."""
        f = self.fit
        return pl.DataFrame({"gene": self.var_names, "baseMean": f["baseMean"], "dispGeneEst": f["dispGeneEst"],
                             "dispFit": f["dispFit"], "dispMAP": f["dispMAP"], "dispersion": f["dispersion"],
                             "dispOutlier": f["dispOutlier"]})

    # ---------------------------------------------------------------- contrasts
    def _contrast(self, contrast):
        """(lfc, se, stat, pvalue, all-zero mask) on the log2 scale."""
        f, names = self.fit, self.coef_names
        beta, cov = f["beta"], f["beta_cov"]
        counts = self.counts.T  # genes × samples, original: R's contrast zero test ignores replaced counts

        def coef(i, sign=1.0):
            se = np.sqrt(np.maximum(cov[:, i, i], 0))
            return sign * beta[:, i], se

        zero = np.zeros(beta.shape[0], bool)
        if contrast is None:
            contrast = names[-1]
        if isinstance(contrast, str):
            label = contrast
            if contrast not in names:
                raise KeyError(f"{contrast!r} not in results_names(): {names}")
            lfc, se = coef(names.index(contrast))
        elif (isinstance(contrast, (tuple, list)) and len(contrast) == 3 and all(isinstance(c, str) for c in contrast)):
            fac, num, den = contrast
            label = f"{fac}_{num}_vs_{den}"
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
            label = "numeric: " + np.array2string(c, separator=", ")
            if c.shape != (len(names),):
                raise ValueError(f"numeric contrast must have length {len(names)}")
            lfc = beta @ c
            se = np.sqrt(np.maximum(np.einsum("i,gij,j->g", c, cov, c), 0))
            if not (np.all(c >= 0) or np.all(c <= 0)):
                which = (self.X @ (c != 0).astype(float)) != 0
                zero = counts[:, which].sum(axis=1) == 0
        zero &= ~f["allZero"]
        self.last_contrast = label
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
        coefficients. Default: the last coefficient (check ``last_contrast`` for
        the one used; its sign depends on the reference level, see ``ref_levels``). Returns gene, baseMean,
        log2FoldChange, lfcSE, stat, pvalue, padj (null where DESeq2 gives NA). The reason for a
        missing value is kept in ``last_results_info["na"]``, a dict of boolean arrays over genes:
        ``"all_zero"`` (no counts; p-value and adjusted p-value are NA), ``"cooks"`` (p-value removed
        because of a Cook's distance above the cut-off; adjusted p-value is NA too) and ``"filtered"``
        (p-value present, adjusted p-value NA because independent filtering removed the gene).
        With ``test="LRT"``, stat/pvalue are the likelihood-ratio test of the full
        against the reduced design (whatever the contrast); the contrast only picks
        the reported fold change, as in DESeq2.
        """
        f = self.fit
        m, p = self.X.shape
        lfc, se, stat, pvalue = self._contrast(contrast)
        pvalue = pvalue.astype(np.float64).copy()
        info: dict = {"contrast": self.last_contrast}
        cooks_na = np.zeros(len(pvalue), dtype=bool)
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
            cooks_na = out.copy()
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
        # why a p-value / adjusted p-value is missing (they are null in the table in every case)
        info["na"] = {"all_zero": np.asarray(f["allZero"], dtype=bool).copy(), "cooks": cooks_na,
                      "filtered": np.isnan(padj) & ~np.isnan(pvalue)}
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


def _group_not_estimable(sub: Pseudobulk, design: str, contrast, kwargs: dict) -> Optional[str]:
    """Reason why ``sub`` cannot support the model, or None. Pre-checks only (no native call)."""
    m = sub.counts.shape[0]
    if not np.any(sub.counts):
        return "all counts are zero"
    if "design_matrix" not in kwargs:
        X, names, factors = _build_design(sub.obs, design, kwargs.get("ref_levels"))
        if np.linalg.matrix_rank(X) < X.shape[1]:
            return "the design is not full rank in this group (a factor has a single level, or variables are collinear)"
        if m <= X.shape[1]:
            return f"{m} samples for {X.shape[1]} coefficients: no replicates to estimate the dispersion"
        if isinstance(contrast, str):
            if contrast not in names:
                return f"coefficient {contrast!r} is not in this group's design (a level is absent)"
        elif isinstance(contrast, (tuple, list)) and len(contrast) == 3 and all(isinstance(c, str) for c in contrast):
            fac, num, den = contrast
            missing = [x for x in (num, den) if x not in factors.get(fac, [])]
            if missing:
                return f"level(s) {missing} of {fac!r} are absent from this group"
        elif contrast is not None and np.asarray(contrast).shape != (len(names),):
            return "the numeric contrast does not match the design columns of this group"
    if kwargs.get("size_factors") is None:
        if kwargs.get("sf_type", "ratio") == "ratio":
            if not np.any(np.all(sub.counts > 0, axis=0)):
                return "every gene has a zero count, so ratio size factors are undefined (try sf_type=\"poscounts\")"
        elif np.any(sub.counts.sum(axis=1) <= 0):
            return "a sample has no counts"
    return None


def pseudobulk_de(bf: BioFrame, sample_key: Union[str, Sequence[str]], design: str = "~ condition", contrast=None, *,
                  groupby: Optional[str] = None, min_cells: int = 10, alpha: float = 0.1,
                  key_added: str = "pseudobulk_de", **kwargs) -> pl.DataFrame:
    """Pseudobulk DESeq2 per ``groupby`` group (e.g. per cell type), in one pass over the data.

    Counts are summed per (sample, group); each group is then tested with
    ``design``/``contrast``. A group whose data cannot support the model (a factor
    with a single level, a rank-deficient design, no replicates, no usable counts, a
    contrast level that is absent from the group) is skipped; all skipped groups are
    named in one warning and recorded, with the reason, in
    ``bf.uns[key_added]["skipped"]``. Mistakes in the call are not skipped but raised:
    non-integer or negative counts (e.g. normalised data), a malformed formula, a
    design variable that is not a column, a contrast naming an unknown level or
    coefficient, invalid options. If no group can be tested a
    :class:`NotEstimableError` (a ``ValueError``) is raised.

    Returns a long DataFrame with a ``group`` column; also stored in
    ``bf.uns[key_added]["table"]`` next to ``params``, ``skipped`` and ``contrast``
    (the contrast label each tested group used, see ``DESeq2.last_contrast``).
    """
    pb = pseudobulk(bf, sample_key, groupby, min_cells=min_cells)
    contrast_arg = contrast
    # errors in the call itself: raised before any group is touched
    if np.any(pb.counts < 0) or np.any(pb.counts != np.floor(pb.counts)):
        raise ValueError("counts must be non-negative integers: pseudobulk_de sums the raw counts of the cells, "
                         "but this matrix has negative or fractional entries (normalised or imputed data?)")
    if "design_matrix" not in kwargs:
        _, names, factors = _build_design(pb.obs, design, kwargs.get("ref_levels"))   # formula / column / name errors
        if isinstance(contrast, str):
            if contrast not in names:
                raise KeyError(f"{contrast!r} not in results_names(): {names}")
        elif isinstance(contrast, (tuple, list)) and len(contrast) == 3 and all(isinstance(c, str) for c in contrast):
            fac, num, den = contrast
            if fac not in factors:
                raise KeyError(f"{fac!r} is not a factor in the design")
            if num not in factors[fac] or den not in factors[fac]:
                raise ValueError(f"{num!r} and {den!r} must be levels of {fac}: {factors[fac]}")
        elif contrast is not None and np.asarray(contrast).shape != (len(names),):
            raise ValueError(f"numeric contrast must have length {len(names)}")
        if contrast is None:
            contrast = names[-1]   # the same coefficient in every group, not whichever happens to be last there
    frames, skipped, labels = [], {}, {}
    groups = [None] if groupby is None else pb.obs[groupby].unique(maintain_order=True).to_list()
    for gname in groups:
        sub = pb if gname is None else pb.subset((pb.obs[groupby] == gname).to_numpy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LevelOrderWarning)   # already reported once for the whole table
            why = _group_not_estimable(sub, design, contrast, kwargs)
            if why is not None:
                skipped[str(gname)] = why
                continue
            try:
                ds = DESeq2(sub, design=design, **kwargs)
                res = ds.results(contrast, alpha=alpha)
            except NotEstimableError as e:
                skipped[str(gname)] = str(e)
                continue
            except (ValueError, KeyError) as e:
                if hasattr(e, "add_note"):
                    e.add_note(f"while testing group {gname!r}")
                raise
        labels[str(gname)] = ds.last_contrast
        if gname is not None:
            res = res.with_columns(pl.lit(str(gname)).alias("group")).select(["group"] + res.columns)
        frames.append(res)
    if skipped:
        listing = "; ".join(f"{g!r}: {why}" for g, why in skipped.items())
        if not frames:
            raise NotEstimableError(f"pseudobulk_de: no group could be tested (skipped {len(skipped)}): {listing}")
        warnings.warn(f"pseudobulk_de: skipped {len(skipped)} group(s): {listing}", stacklevel=2)
    out = pl.concat(frames)
    bf.uns[key_added] = {"params": {"sample_key": sample_key, "design": design, "contrast": contrast_arg,
                                    "groupby": groupby, "min_cells": min_cells},
                         "skipped": skipped, "contrast": labels, "table": out}
    return out
