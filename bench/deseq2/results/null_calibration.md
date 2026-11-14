# Null calibration of pseudobulk DESeq2 on Kang et al. 2018

Condition (stim / ctrl) permuted within donor at the cell level, 30 permutations (seed 1); counts summed per donor x condition within each of 8 cell types (annotated singlets, samples with >= 10 cells, donors seen in both conditions); design `~ ind + stim`, stim vs ctrl; all genes with a non-zero count. Software: crest 0.4.0, R DESeq2 1.50.2, pydeseq2 0.5.4. Adjusted p-values come from each tool's own `results()` / `summary()` with alpha = 0.1 for independent filtering. Nothing is differentially expressed under this null, so every call is a false positive. Evidence from one run (one seed); wall times are not a result of this study.

## All cell types pooled

| method | fits | genes tested | FPR padj < 0.05 | FPR padj < 0.1 | fits with >= 1 call (0.05) | fits with >= 1 call (0.1) | raw p < 0.05 | raw p < 0.05, baseMean >= 10 | raw p < 0.05, baseMean >= 100 |
|---|---|---|---|---|---|---|---|---|---|
| CREST | 240 | 3,140,018 | 2.23e-06 (7) | 3.18e-06 (10) | 4 / 240 | 6 / 240 | 0.0037 | 0.0110 (456,371 genes) | 0.0209 (65,801 genes) |
| R DESeq2 | 240 | 3,140,018 | 2.23e-06 (7) | 3.18e-06 (10) | 4 / 240 | 6 / 240 | 0.0038 | 0.0115 (456,371 genes) | 0.0214 (65,801 genes) |
| pydeseq2 | 240 | 3,140,018 | 2.87e-06 (9) | 3.82e-06 (12) | 6 / 240 | 8 / 240 | 0.0065 | 0.0165 (456,371 genes) | 0.0219 (65,801 genes) |

FPR = genes with padj below the threshold / genes with a non-missing adjusted p-value, summed over fits (calls in brackets). Under a calibrated test the raw-p columns are 0.05 and BH keeps the share of fits with any call at or below the threshold. Counts of 0-3 give a discrete Wald statistic that rarely exceeds 1.96, so the raw-p rate over all genes is expected to sit below 0.05; the baseMean strata show the rate for genes with enough counts.

## By cell type (padj < 0.1: calls summed over permutations / fits with a call; raw p < 0.05)

| cell type | fits | CREST | R DESeq2 | pydeseq2 |
|---|---|---|---|---|
| B cells | 30 | 5 / 3; 0.005 | 5 / 3; 0.005 | 5 / 3; 0.008 |
| CD14+ Monocytes | 30 | 3 / 1; 0.004 | 3 / 1; 0.004 | 5 / 3; 0.007 |
| CD4 T cells | 30 | 1 / 1; 0.005 | 1 / 1; 0.005 | 1 / 1; 0.009 |
| CD8 T cells | 30 | 0 / 0; 0.003 | 0 / 0; 0.004 | 0 / 0; 0.006 |
| Dendritic cells | 30 | 0 / 0; 0.002 | 0 / 0; 0.002 | 0 / 0; 0.004 |
| FCGR3A+ Monocytes | 30 | 1 / 1; 0.004 | 1 / 1; 0.004 | 1 / 1; 0.007 |
| Megakaryocytes | 30 | 0 / 0; 0.001 | 0 / 0; 0.001 | 0 / 0; 0.002 |
| NK cells | 30 | 0 / 0; 0.004 | 0 / 0; 0.005 | 0 / 0; 0.008 |

## Positive control: the observed labels (padj < 0.1)

| cell type | CREST | R DESeq2 | pydeseq2 |
|---|---|---|---|
| B cells | 790 | 790 | 927 |
| CD14+ Monocytes | 3660 | 3660 | 3753 |
| CD4 T cells | 1350 | 1349 | 1630 |
| CD8 T cells | 320 | 320 | 388 |
| Dendritic cells | 488 | 488 | 531 |
| FCGR3A+ Monocytes | 1480 | 1468 | 1568 |
| Megakaryocytes | 51 | 51 | 50 |
| NK cells | 505 | 505 | 635 |

## Calls shared between methods under the null (padj < 0.1)

| pair | calls in first | calls in second | in both |
|---|---|---|---|
| CREST / R DESeq2 | 10 | 10 | 10 |
| CREST / pydeseq2 | 10 | 12 | 9 |
| R DESeq2 / pydeseq2 | 10 | 12 | 9 |

## Fits and fit types

Fits per method: 248. Failed fits: 0.

Dispersion trend used by CREST (fits): mean: 14, parametric: 234.

Dispersion trend used by R DESeq2 (fits): local: 14, parametric: 234. R substitutes a local regression when the parametric fit fails; CREST substitutes the mean (`fit_substituted`).

### Fits where R fell back to a local trend

| method | fits | calls padj < 0.1 | raw p < 0.05 |
|---|---|---|---|
| CREST | 14 | 0 | 0.0031 |
| R DESeq2 | 14 | 0 | 0.0055 |
| pydeseq2 | 14 | 0 | 0.0066 |

## Run

CREST built from the tree at commit 69d36d7 (Linux x86-64), R 4.5.2, pydeseq2 0.5.4, on a 64-thread
AMD Threadripper 3975WX; 30 permutations (seed 1), 8 cell types, `--jobs 12`. The script, the
permutation seeds and the per-fit counts (`null_calibration.csv`) reproduce the table; the 8 observed-label
fits are included in the fit counts but not in the null tables.

## Positive control: where CREST and R differ

Observed labels: CREST and R call the same genes in 6 of 8 cell types. FCGR3A+ monocytes: CREST 1480, R 1468
(22 genes differ: 17 are called by CREST only, with adjusted p-values within 0.01 of R's on both sides of 0.1;
5 are called by R only, with adjusted p-values of 0.03 to 0.08 where CREST's is missing, which was not examined
further); CD4 T cells: 1350 and 1349 (1 gene). Re-fitting these two tables: the parametric trend coefficients
differ from R's by 1.9e-4 (a) and 5.1e-4 (b) for FCGR3A+ monocytes (CREST 0.029874, 1.74372; R 0.029868, 1.74462),
the prior variance by 0.9 % (0.7461 against 0.7530) and the fitted trend by up to 5.1e-4. Running R's trend
algorithm (reimplemented, with R's `glm` stopping rule) on R's gene-wise dispersions returns R's coefficients;
on CREST's gene-wise dispersions it returns CREST's. The shift in the trend therefore comes from four genes whose
gene-wise dispersion differs between the two (for example gene index 5069 of the table: 0.248 in CREST, 2.2e-6 in
R; two genes at the floor 1e-8 in CREST and 1.0e-7 and 3.4e-8 in R), not from the trend solver. This is the case
described in `docs/deseq2.md`, known difference 1. The same counts give the same results with the released 0.4.0
build, so the difference is not new in this tree. The figures quoted for these two cell types in `docs/deseq2.md`
(1468 / 1469 and 1350 / 1365) were obtained from a different preparation of the data and are not reproduced
by this run.
