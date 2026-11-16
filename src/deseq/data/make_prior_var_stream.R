# Tabulate the random draws behind R's prior variance for <= 3 residual degrees of freedom.
# estimateDispersionsPriorVar() calls set.seed(2) and then, for each of 200 grid values x of the prior
# variance, draws 10^4 chisq and 10^4 normal variates and histograms log(chisq) + N(0, x) - log(df) on the
# breaks -10, -9.5, ..., 10. The draws do not depend on the data, so the histogram counts for df = 1, 2, 3 are
# fixed numbers; this script writes them (200 rows of 40 counts per df) to prior_var_stream_counts.txt.
# Run:  Rscript src/deseq/data/make_prior_var_stream.R
suppressMessages(library(DESeq2))
out <- c(sprintf("# counts of the 10^4 simulated residuals per bin (-10, -9.5, ..., 10), grid x = seq(0, 8, length = 200), set.seed(2); DESeq2 %s, %s",
                 as.character(packageVersion("DESeq2")), R.version.string))
brks <- -20:20/2
grid <- seq(from = 0, to = 8, length = 200)
for (df in 1:3) {
  set.seed(2)
  out <- c(out, sprintf("df %d", df))
  for (x in grid) {
    r <- log(rchisq(10000, df = df)) + rnorm(10000, 0, sqrt(x)) - log(df)
    r <- r[r > min(brks) & r < max(brks)]
    h <- hist(r, breaks = brks, plot = FALSE)
    out <- c(out, paste(h$counts, collapse = " "))
  }
}
writeLines(out, Sys.getenv("OUT_FILE", "src/deseq/data/prior_var_stream_counts.txt"))
