- Changed: the README and the DESeq2 page now name the reference version (R 4.5.2 / DESeq2 1.50.2), give the Kang 2018
  agreement at padj < 0.05 and < 0.1 with its cause, and no longer mention bootstrap confidence intervals for the benchmark.
- Added: R-pinned tests for outlier replacement in groups of unequal size, sparse designs with non-converged gene-wise
  dispersions, and `lowess` / `loess` on vectors with tied x.
- Added: `bench/deseq2/kang_concordance.py`, `kang_trend_swap.py`, `covid_concordance.py` and `thread_consistency.py`,
  which compare CREST with R DESeq2 per cell type, attribute discordant genes to a cause, run donor-level pseudobulk of the
  COVID-19 atlas, and check that results do not depend on the thread count.
