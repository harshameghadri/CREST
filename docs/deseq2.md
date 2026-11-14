# Pseudobulk DESeq2 in CREST

`crest.tl.DESeq2` is a Rust port of Bioconductor **DESeq2** (`DESeq()` +
`results()`, with the Wald test or the likelihood-ratio test), written against the R and
C++ sources, not re-derived from the paper. The reference version is **R 4.5.2 with
DESeq2 1.50.2**: new R fixtures are generated with it and record `packageVersion` in their
`.json` file. Fixtures made earlier with DESeq2 1.42 are kept; see *Reference version* below. Use it for pseudobulk DE on single-cell data
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
| Cook's distances, `maxCooks`, `refitWithoutOutliers` (≥ 7 replicates) | same; under the LRT the distances use the full-model means, as DESeq2 ≥ 1.50 does |
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
* `tests/test_crest.py` and `tests/test_deseq2_r_parity.py` check against stored R output
  (`tests/data/`), so CI guards R-parity without needing R. The R-parity file pins eleven
  count tables made with R 1.50.2 (scripts `make_pin_inputs.py` and `make_pin_fixtures.R`
  next to them): heterogeneous dispersions with 2 and 3 residual degrees of freedom, outlier
  replacement and refit, counts up to 10⁸, independent filtering together with Cook's
  p-value filtering, `poscounts` size factors, a numeric covariate with all-zero contrast
  rows, an LRT with Cook's distances at the cut-off, and a contrast over replaced counts.
  Each tolerance is ten times the larger of the deviations measured on Linux x86-64 and macOS
  arm64; what is excluded from a comparison (gene-wise dispersions on the flat part of the
  likelihood, Cook's distances above the cut-off) is stated in the test module.

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
   differences of about 1e-4 in log2 fold change for most of these rows, and larger ones for
   sparse genes (up to about 1 log2 unit in the Kang comparison).
3. **Prior variance with ≤ 3 residual degrees of freedom** (e.g. 2 vs 2). R estimates it by
   simulating 10⁴ draws after `set.seed(2)` for each of 200 candidate variances, comparing
   the histogram of the draws with the observed residuals (KL divergence) and smoothing
   with `loess(span = 0.2)`. The draws do not depend on the data, so CREST uses the same
   draws, tabulated once from R (`src/deseq/data/`), with the same pseudo-count, the same
   histogram bins and an interpolating `loess` as R's; on the test designs it returns R's
   value (to 1e-9). R's value is itself one draw of a Monte-Carlo estimate: repeating R's
   computation over seeds 1 to 50 on the 1,000-gene test tables gives
   2v2 (sd 1.0): 0.25 to 0.36 (mean 0.29); 2v2 (sd 1.5): 1.01 to 1.21 (mean 1.08); 2v3: 0.61 to 0.66 (mean 0.63); 3v2: 0.73 to 0.74 (mean 0.73).
   With 4 or more residual degrees of freedom the closed form is used and the results agree.
   Before this was done CREST integrated the reference distribution exactly, which moved
   the estimate outside R's seed range for two of the four designs (0.41 against 0.27 at
   2 vs 2, 0.67 against 0.73 at 3 vs 2).
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
5. **Cook's cut-off and missing values.** `qf(0.99, p, m - p)` is evaluated to double precision,
   as R does. A p-value or adjusted p-value that is missing is `null` in the table; the reason
   (no counts, Cook's outlier, removed by independent filtering) is in
   `DESeq2.last_results_info["na"]`, a dict of boolean masks over genes.
6. **Checks R makes and variants it has.** `min_replicates_for_replace < 3` raises, as R's
   `replaceOutliers` stops (`float("inf")` switches replacement off). R's `sfType = "iterate"`
   is not available (`"ratio"` and `"poscounts"` are). When every gene-wise dispersion is at the
   floor R's `estimateDispersionsFit` stops; CREST continues with the mean trend and sets the
   dispersions to 10 times the floor (the first pass only; genes refitted after outlier
   replacement always get their MAP dispersion). `obs` must be a polars `DataFrame`
   (`pl.from_pandas` converts a pandas table). The all-zero test of numeric and tuple contrasts
   uses the original counts, as R's `contrastAllZeroNumeric` / `contrastAllZeroCharacter` do,
   also for genes whose outliers were replaced.

## Reference version

New fixtures and the numbers in this page refer to R 4.5.2 / DESeq2 1.50.2. The older
fixtures (`deseq2_{two_vs_two,three_level_batch,outlier_replacement}_*`, the LRT outputs from
`make_lrt_fixtures.R`) were made with DESeq2 1.42 and stay in the tests. Known differences
between the two R versions, so that they are not mistaken for CREST differences:

* Under `test="LRT"`, DESeq2 1.42.1 does not store the full model's fitted means before it
  computes Cook's distances, 1.50.2 does; CREST follows 1.50.2. On a 3-level design with
  4 + 4 + 4 samples the earlier comparison found `maxCooks` differing by up to 0.21 in
  absolute terms between the two R versions and by about 3e-14 between CREST and 1.50.2
  (carried over from that comparison; 1.42 is not installed where the fixtures are made).
  The 1.50.2 fixture `lrt_cooks` has four genes within 5 % of the cut-off, on either side,
  and CREST agrees with R there to 1e-9.
* The two 1.42 LRT fixtures have no group with three replicates, so R reports `maxCooks` as
  `NA` for them in both versions. Re-running `make_lrt_fixtures.R` with 1.50.2 reproduces
  the stored statistics, p-values and adjusted p-values to 2.2e-11 relative, log2 fold
  changes to 4.3e-13 and dispersions to 9.7e-14.
