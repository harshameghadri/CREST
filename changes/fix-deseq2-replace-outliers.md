- Fixed: `crest.tl.DESeq2` now takes the hat diagonals behind Cook's distance from the last iteration of the iterative fit for rows
  whose coefficient fit does not converge and is refit, as R does; outlier replacement at such rows (for example a gene with zero
  counts in one group and heavy-tailed counts in the other) now replaces the same counts as R, where it could skip a replacement
  at the cut-off and report a different base mean, dispersion and result.
- Added: R-pinned tests of outlier replacement and Cook's distance on refit rows; the COVID-19 concordance table now lists the genes with
  replaced counts in R and in CREST for every stratum.
