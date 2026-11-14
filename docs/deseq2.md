# Pseudobulk DESeq2 in CREST

`crest.tl.DESeq2` is a Rust port of Bioconductor **DESeq2 1.42** (`DESeq()` +
`results()`, with the Wald test or the likelihood-ratio test), written against the R and
C++ sources, not re-derived from the paper. Use it for pseudobulk DE on single-cell data
(and for bulk RNA-seq).

```python
import crest

# one call: sum counts per donor × condition within each cell type, test each type
res = crest.tl.pseudobulk_de(bf, sample_key=["donor", "condition"], design="~ donor + condition",
                             contrast=("condition", "stim", "ctrl"), groupby="cell_type")

# or step by step
pb = crest.tl.pseudobulk(bf, ["donor", "condition"], groupby="cell_type", min_cells=10)
sub = pb.subset((pb.obs["cell_type"] == "B cells").to_numpy())
ds = crest.tl.DESeq2(sub, design="~ donor + condition")
ds.results_names()                                   # ['Intercept', 'donor_…', 'condition_stim_vs_ctrl']
table = ds.results(("condition", "stim", "ctrl"))    # gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
ds.dispersions()                                     # dispGeneEst, dispFit, dispMAP, dispersion, dispOutlier
```

`pseudobulk` streams the raw counts chunk by chunk (in-memory or out-of-core
Parquet). It ignores any recorded `normalize_total`/`log1p`, so it always sums
raw integer counts.

`pseudobulk_de` tests every group in turn. A group that cannot support the model
(a factor with a single level, a rank-deficient design, no replicates, no usable
counts, a contrast level that is absent from it) is skipped; the skipped groups and the
reason for each are named in one warning and stored in
`bf.uns["pseudobulk_de"]["skipped"]`, next to the contrast label each tested group used
(`["contrast"]`). A mistake in the call is raised instead of skipped: negative or
fractional counts, a malformed formula, a design variable that is not a column of `obs`,
a contrast with an unknown level or coefficient. If no group can be tested,
`crest.deseq2.NotEstimableError` (a `ValueError`) is raised.

Factor levels are sorted by code point (upper case before lower case) and the first is
the reference, so `["ctrl", "Stim"]` gives the reference `Stim` and the default contrast
`condition_ctrl_vs_Stim`. R in an en_US locale sorts case-insensitively and would pick
`ctrl`; when the two orders differ and no `ref_levels` is given, a
`crest.deseq2.LevelOrderWarning` names both. `DESeq2.last_contrast` records the contrast
that `results()` computed. Two terms that produce the same coefficient name raise a
`ValueError`, since a contrast could not tell them apart.

## What is implemented (and matches R)

| DESeq2 step | CREST |
|---|---|
| `estimateSizeFactors` (`ratio`, `poscounts`) | same |
| `estimateDispersionsGeneEst` (rough/moments init, `linearMu`, `fitDisp` line search on the Cox-Reid profile likelihood, grid refit, `noIncrease` rule) | same |
| `estimateDispersionsFit` `parametric` (Gamma-identity GLM, residual trimming, 10 outer iterations) and `mean` | same |
| `estimateDispersionsPriorVar` (`mad²` − `trigamma((m−p)/2)`, floor 0.25) | same |
| `estimateDispersionsMAP` (+ `dispOutlier`) | same |
| `nbinomWaldTest` (IRLS with ridge 1e-6, `betaConv`, Wald p-values) | same (refit of non-converged rows: see below) |
| `nbinomLRT` (`DESeq2(test="LRT", reduced="~ ...")`: deviance of the full vs the reduced design, χ² p-values) | same (LRT statistics within 1e-6 of R) |
| Cook's distances, `maxCooks`, `refitWithoutOutliers` (≥ 7 replicates) | same |
| `results()`: coefficient / `c(factor, a, b)` / numeric contrasts, `contrastAllZero`, Cook's cutoff incl. the 2-group heuristic, independent filtering (lowess port of R's `clowess`), BH | same |

Not implemented: `lfcShrink` (apeglm/ashr/normal), `betaPrior=TRUE`,
`fitType="local"`/`glmGamPoi`, observation weights, `lfcThreshold`,
interaction terms in formulas (pass `design_matrix=` instead).

## Validation against R

`bench/deseq2/compare_r.py` (simulated data, 5000 genes) and
`bench/deseq2/kang_pseudobulk.py` (Kang et al. 2018, GSE96583: 24,679 PBMCs,
8 donors × ctrl/stim, 8 cell types, design `~ ind + stim`) feed identical
inputs to R and CREST. Results are in `bench/deseq2/results/`.

* Simulated designs (2×3, batch + condition, 3 levels with a non-reference
  contrast, 2×8 and 2×12 with outlier replacement): size factors, dispersions,
  log2 fold changes, standard errors and p-values agree to ~1e-9 relative. The
  significant gene sets are identical.
* Kang 2018: identical significant sets (padj < 0.1) in 6 of 8 cell types. In
  FCGR3A+ monocytes 1468/1469 match, in CD4 T cells 1350/1365. Every difference
  traces to a few genes at the dispersion floor (*Known differences* 1).
  For comparison, pydeseq2's number of calls differs from R's by 3–25% per cell
  type (e.g. 1630 vs 1365 in CD4 T cells).
* `tests/test_crest.py` checks against stored R 1.42 output (`tests/data/`), so CI
  guards R-parity without needing R.

## Speed

On the Kang 2018 cell types (≈ 12–16k expressed genes, 10–16 samples, 9
coefficients), on the 4-core benchmark machine:

R ran single-threaded (`DESeq()` without BiocParallel) and pydeseq2 on all cores
(`n_cpus=None`), as in `bench/deseq2/kang_pseudobulk.py`:

| | per cell type | all 8 cell types |
|---|---|---|
| R DESeq2 1.42 | 12–19 s | 131 s |
| pydeseq2 0.5.4 | 20–28 s | 202 s |
| **CREST** | **0.4–1.2 s** | **7.2 s** (plus 0.4 s to pseudobulk 24k cells) |

## Known differences

1. **Dispersion floor.** For genes whose likelihood is flat near
   `minDisp = 1e-8` (practically Poisson), R evaluates
   `lgamma(y + 1e8) − lgamma(1e8)` directly, which carries ~1e-7 absolute error.
   That is larger than the likelihood differences it compares, so R's line
   search ends in numerical noise. CREST evaluates the same quantities
   without cancellation (Stirling-remainder and `log1p` forms), so a few such
   genes (4 of 15,720 in Kang CD4 T cells) can land elsewhere on the flat
   region. When one of them crosses the `1e-6` threshold for inclusion in the
   trend fit, the trend shifts by ~0.1% and a few borderline calls flip.
2. **Rows IRLS does not converge on.** R refits them with L-BFGS-B (`optim`),
   CREST with Newton's method on the same concave penalised likelihood and the
   same bounds. The optimum is the same, but R's optimizer tolerance leaves
   differences of ~1e-4 in log2 fold change for these rows.
3. **Prior variance with ≤ 3 residual degrees of freedom** (e.g. 2 vs 2). R
   estimates it by simulating 10⁴ draws with `set.seed(2)`, then smooths with
   `loess`. CREST integrates the same reference distribution exactly, which is
   deterministic but differs from R's Monte-Carlo value by a few percent. With
   3+ residual df the closed form is used and the results agree exactly.
4. **Parametric trend failure.** R substitutes a local regression (`locfit`).
   CREST uses the mean dispersion, which is not equivalent: on small or low-count
   inputs the dispersion trend, the prior variance and the significant genes can
   differ markedly from R's. When it happens `DESeq2.fit["fit_substituted"]` is `True`,
   `fit["fitType"]` is `"mean"`, `fit["messages"]` says so, and a `UserWarning` states
   that the results will differ from R (`quiet=True` silences the warning, not the
   flag). Check the flag before comparing with R. A parametric fit that fails for
   numerical reasons only (for example at counts in the hundreds of millions) is no
   longer a cause: the normal equations of the trend fit are equilibrated before the
   pivot test.
