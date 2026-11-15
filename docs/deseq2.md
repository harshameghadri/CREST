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
| `estimateDispersionsFit` `parametric` (Gamma-identity GLM, residual trimming, 10 outer iterations), `local` (`locfit`, see below) and `mean`, with R's fallback from `parametric` to `local` | same (the local trend to 5e-13 relative on the test cases) |
| `estimateDispersionsPriorVar` (`mad²` − `trigamma((m−p)/2)`, floor 0.25) | same |
| `estimateDispersionsMAP` (+ `dispOutlier`) | same |
| `nbinomWaldTest` (IRLS with ridge 1e-6, `betaConv`, Wald p-values) | same (refit of non-converged rows: see below) |
| `nbinomLRT` (`DESeq2(test="LRT", reduced="~ ...")`: deviance of the full vs the reduced design, χ² p-values) | same (LRT statistics within 1e-6 of R) |
| Cook's distances, `maxCooks`, `refitWithoutOutliers` (≥ 7 replicates) | same; under the LRT the distances use the full-model means, as DESeq2 ≥ 1.50 does |
| `results()`: coefficient / `c(factor, a, b)` / numeric contrasts, `contrastAllZero`, Cook's cutoff incl. the 2-group heuristic, independent filtering (lowess port of R's `clowess`), BH | same |

### The local dispersion trend

`fit_type="local"` is DESeq2's `localDispersionFit`: `locfit(log(disp) ~ log(mean), weights = mean)` on the genes whose
gene-wise dispersion is at least `10 * minDisp`, evaluated as `exp(predict(fit, log(mean)))`. It is a port of the
evaluation of locfit 1.5-9.12 for this one call (`src/deseq/locfit.rs`), not an approximation of it: the defaults
(`alpha = 0.7`, degree 2, tricube kernel, family `qgaussian`) with locfit's adaptive tree on the bounding box of the data
(`ev = rbox()`: a cell is split while its width exceeds 0.8 of the smaller bandwidth at its ends, at most 19 vertices), a
local quadratic at each vertex solved as locfit solves it (one Newton step through an eigen-decomposition of the scaled
normal equations, sums in the order of the C code), cubic Hermite interpolation between vertices, and locfit's parametric
component, a global weighted quadratic that is subtracted at the vertices and added back at every evaluation (it makes the
curve beyond the range of the means quadratic, as R's is). Where R's `locfit` stops with an error (for example two
gene-wise dispersions: "newsplit: out of vertex space") R's `DESeq()` stops; CREST then continues with the mean trend, sets
`fit["fit_substituted"]` and warns that the results can differ from R.

When the parametric fit does not converge (or one of its coefficients is not positive) R prints a note and substitutes the
local trend; CREST does the same, with R's note as the message: `fit["fitType"]` is `"local"`, `fit["fit_substituted"]` is
`True` and `fit["trend"]` is empty (there are no coefficients). The mean trend is used where R uses it: when every gene-wise
dispersion is within two orders of magnitude of `minDisp`, and where the local fit fails as above. Agreement with R's local
trend (R 4.5.2, DESeq2 1.50.2, locfit 1.5-9.12): over 22 synthetic cases built to hit the cases of the evaluation structure
(ties in the means, 3 to 30 genes, extreme weights, a trend that needs many vertices and pseudo-vertices, a gap in the means,
outliers, gene-wise values at the floor) and six COVID-19 atlas strata where R's parametric fit fails, the largest relative
difference of the trend is 4.1e-14 on Linux x86-64 and on macOS arm64, at the means and at points up to 2.5 log units beyond
their range (`tests/test_deseq2_local_fit.py`, `tests/data/deseq2_local_fit_measurements.json`). On two whole `DESeq()` fixtures where R falls back the agreement is as follows. In the first (120 genes, 3 v 3) every column
agrees to 1e-9 or better, adjusted p-values to 5e-10, and the significant sets are identical. In the second (`edge_sparse_extreme`:
6 samples, 1,500 genes, counts over six orders of magnitude) the trend agrees to 5e-10 on every gene, and every column agrees
within 2e-4 on the 1,095 genes whose dispersion is below the cap (10). The other 392 genes sit at the cap, where the likelihood in
the coefficients is flat: R's optimiser (L-BFGS-B) and CREST's (Newton) stop at different places, fold changes differ by up to 3.7
log2 units with equal standard errors (*Known differences* 2), and the changed p-values move the `baseMean` threshold of
independent filtering, so adjusted p-values are missing for 1,021 genes in CREST and 820 in R. The significant sets then differ:
CREST 92, R 89, 85 shared at padj < 0.1, and 77, 78 and 77 at padj < 0.05.

Not implemented: `lfcShrink` (apeglm/ashr/normal), `betaPrior=TRUE`, `glmGamPoi`, observation weights, `lfcThreshold`,
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
* Kang 2018 (R 4.5.2 / DESeq2 1.50.2, BiocParallel 1.44.0; CREST built from release/0.5 at
  commit d6a1f16, Linux x86-64; `bench/deseq2/kang_concordance.py`): identical significant
  sets in 6 of 8 cell types at padj < 0.1 and at padj < 0.05. In CD4 T cells CREST calls
  1356 genes at padj < 0.1 and R 1365, with 1350 shared (15 R-only, 6 CREST-only); at
  padj < 0.05 the numbers are 1107 and 1108 with 1104 shared. In FCGR3A+ monocytes CREST
  calls 1468 and R 1469 at padj < 0.1 (1468 shared; the one gene only R calls has padj
  0.1004 in CREST and 0.0999 in R) and both call 1213 at padj < 0.05. The 21 + 7
  discordant CD4 T genes are all removed by independent filtering choosing a different
  `baseMean` threshold in the two tools, which follows from a slightly different
  dispersion trend (*Known differences* 1). The script re-measures these figures, and
  also lists every discordant gene with both adjusted p-values, both dispersions and its
  cause. For comparison, pydeseq2's number of calls differs from R's by 3–25% per cell
  type (e.g. 1630 vs 1365 in CD4 T cells).
* COVID-19 atlas (Stephenson et al. 2021; donor-level pseudobulk, design `~ Status`, Covid vs Healthy, 109 donors, 30 cell
  types; R 4.5.2 / DESeq2 1.50.2 / locfit 1.5-9.12; CREST at commit a38d065, Linux x86-64;
  `bench/deseq2/covid_concordance.py`, tables in `bench/deseq2/results/covid_concordance_*.csv`): R's parametric trend
  fails in 11 of the 30 strata and DESeq2 substitutes its local trend. CREST substitutes it in the same 11 and in one more
  (myeloid dendritic cell, *Known differences* 4). In the 11 strata CREST and R call the same genes: 9,043 at padj < 0.1 and
  6,494 at padj < 0.05 (before the local trend was ported CREST used the mean and called 6% to 22% fewer genes than R in each
  of the 12 strata where it did so, 9,723 against 10,901 in total at padj < 0.1). Over all 30 strata the significant sets are
  identical in 28 at padj < 0.1 and in 29 at padj < 0.05. The exceptions at 0.1: "dendritic cell, human" (CREST 1,502, R 1,499,
  1,499 shared; the three CREST-only genes sit at the threshold) and "myeloid dendritic cell" (CREST 2,110, R 1,858, 1,856
  shared at 0.1; 1,584, 1,436 and 1,432 at 0.05; independent filtering picks the `baseMean` threshold 0.344 in CREST and 0.438
  in R because the two trends differ). R's `localDispersionFit` given CREST's gene-wise dispersions reproduces CREST's fitted dispersions to 5.8e-15 at the genes
that enter the trend fit (gene-wise dispersion above 100 * `minDisp`: 6,403 to 9,457 of the 24,245 genes per stratum) in eight of
these strata (`bench/deseq2/covid_local_trend_swap.py`), so the trend function is not what differs. The fitted dispersions of
the two tools do differ in some strata. Over the 11 strata where R uses its local trend, the largest relative difference of the
fitted dispersion per stratum is at most 2e-9 in three, 3e-5 or less in three more, and between 0.011 and 0.22 in five. In
"IgA plasma cell" (`bench/deseq2/covid_trend_gap.py`) the difference has median 1.8e-3, 99th percentile 2.0e-2 and maximum 0.22
over 16,639 genes. It is above 0.1 at two genes only, ENSG00000211946 (0.22) and ENSG00000211942 (0.16), whose base means differ
between the tools (60.0 against 42.6 and 64.8 against 49.9) and the second of which has its counts replaced in R and not in
CREST; why those base means differ was not examined. The broad part of the difference comes from the gene-wise inputs: 15
first-pass gene-wise estimates differ between the tools, R's trend moves by a median 1.8e-3 and at most 2.2e-2 when CREST's
values replace R's, and with R's values put back at those 15 genes alone it is reproduced to 3.5e-8.
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
* `tests/test_deseq2_edge_pins.py` adds four more R-pinned cases (scripts
  `make_edge_inputs.py` and `make_edge_fixtures.R`): three groups of 8, 4 and 8 samples with
  outliers inside and outside the groups that are replaced (at `minReplicatesForReplace` 7, 4
  and 9), two sparse 6-sample designs in which hundreds of gene-wise dispersions stop at
  `maxit` (one where R falls back to a local trend, which CREST now does too), and random vectors with tied x for
  `lowess` and `loess`. No gene fails the beta fit in R in these data, so a non-converged
  beta fit is not pinned.

## Speed

Timings against R DESeq2 and pydeseq2 on the Kang 2018 cell types are thread-matched:
CREST, R (`DESeq()` serial or with BiocParallel workers) and pydeseq2 (`n_cpus`) are timed
at the same thread counts, and each tool is also compared at its own best thread count
(`bench/deseq2/kang_pseudobulk.py`). No speed-up is quoted here until that comparison has been
re-run; the results will be published in {doc}`benchmarks`.

## Known differences

1. **Dispersion floor.** For genes whose likelihood is flat near
   `minDisp = 1e-8` (practically Poisson), R evaluates
   `lgamma(y + 1e8) − lgamma(1e8)` directly, which carries ~1e-7 absolute error.
   That is larger than the likelihood differences it compares, so R's line
   search ends in numerical noise. CREST evaluates the same quantities
   without cancellation (Stirling-remainder and `log1p` forms), so a few such
   genes can land elsewhere on the flat region. `dispGeneIter` is not comparable
   there: in Kang CD4 T cells it differs on 11,567 of 15,720 genes, and all of
   them sit at the floor in both tools. A gene-wise estimate above `1e-6` enters
   the trend fit, so when a gene crosses that threshold in one tool only, the
   trend moves (coefficients by 1e-4 to 1e-3, fitted dispersions by ~0.1%, prior
   variance by ~1%), the number of rejections at each filter quantile changes,
   independent filtering picks another `baseMean` threshold (CD4 T cells: 1.81 in R,
   1.56 in CREST) and borderline calls flip. In CD4 T cells two genes that exhaust
   `maxit` are at 1e-8 in R and at 2.6e-5 and 7e-6 in CREST (adjusted
   log-likelihood differing by 6e-5 and 4e-6). In FCGR3A+ monocytes one gene is not at the
   floor: R stops at 2e-6 after 23 iterations, where the adjusted log-likelihood is 1.1
   below its value at CREST's 0.25. Replacing, in R, the gene-wise estimates of the four
   genes per cell type that differ by CREST's values reproduces CREST's trend to 1e-13
   and removes every discordant call at padj < 0.05 and < 0.1 in both cell types
   (`bench/deseq2/kang_trend_swap.py`).
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
4. **Parametric trend failure.** R substitutes a local regression (`locfit`) and so does CREST
   (*The local dispersion trend* above); `DESeq2.fit["fit_substituted"]` is `True`,
   `fit["fitType"]` is `"local"`, and `fit["messages"]` and a `UserWarning` carry R's note
   (`quiet=True` silences the warning, not the flag). Whether the parametric fit fails depends on
   the gene-wise dispersions, so a gene that differs between the tools in the way *Known differences* 1
   describes can make the fit fail in one tool and not in the other: in the COVID-19 atlas stratum
   "myeloid dendritic cell" (56 donors) two genes with a flat likelihood sit at 1e-8 in R and at 3.7e-5 and
   4.7e-5 in CREST; with CREST's values R's own `parametricDispersionFit` stops with "dispersion fit did not
   converge" (the fit keeps changing its coefficients in the fourth digit), with R's values it converges, and CREST's
   Gamma-GLM loop returns R's coefficients to 2e-13 on R's values and fails on CREST's as R does
   (`tests/test_deseq2_local_fit.py`). CREST then uses the local trend where R uses its parametric one. Where the
   mean was substituted before (versions up to 0.4.0), results differed markedly from R's; the local trend removes
   that difference. A parametric fit that fails for numerical reasons only (for example at counts in the hundreds
   of millions) is no longer a cause: the normal equations of the trend fit are equilibrated before the pivot test.
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

New fixtures and the numbers in this page refer to R 4.5.2 / DESeq2 1.50.2 (the Kang
comparison above was run with BiocParallel 1.44.0). The older
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
