# Reference output for the outlier-replacement fixture replace_hat in this directory (R DESeq2 DESeq() + results()).
# Run from the repository root:  Rscript tests/data/make_replace_fixtures.R
# Reads deseq2_replace_hat_input.csv and deseq2_replace_hat_meta.json (written by make_replace_inputs.py) and writes
#   deseq2_replace_hat_R.csv          DESeq() with the default minReplicatesForReplace = 7 (group "b" is replaced, group "a" is not),
#   deseq2_replace_hat_noreplace_R.csv  DESeq(minReplicatesForReplace = Inf): no replacement, so maxCooks is the largest Cook's
#                                     distance of the observed counts (columns baseMean, maxCooks, betaConv, allZero only),
# and adds the R-side quantities and package versions to the meta json.
suppressMessages({library(DESeq2); library(jsonlite)})
d <- Sys.getenv("CREST_FIXTURE_DIR", "tests/data")
name <- "replace_hat"
inp <- read.csv(file.path(d, paste0("deseq2_", name, "_input.csv")), check.names = FALSE)
meta <- fromJSON(file.path(d, paste0("deseq2_", name, "_meta.json")), simplifyVector = TRUE)
genes <- grep("^gene", colnames(inp), value = TRUE)
cts <- t(as.matrix(inp[, genes])); colnames(cts) <- paste0("s", seq_len(ncol(cts)))
cd <- inp[, setdiff(colnames(inp), genes), drop = FALSE]
for (v in names(meta$levels)) cd[[v]] <- factor(cd[[v]], levels = meta$levels[[v]])
rownames(cd) <- colnames(cts)

write_res <- function(res, dds, file) {
  mc <- mcols(dds); col <- function(x) if (x %in% names(mc)) mc[[x]] else rep(NA, nrow(dds))
  out <- data.frame(baseMean = res$baseMean, log2FoldChange = res$log2FoldChange, lfcSE = res$lfcSE,
                    stat = res$stat, pvalue = res$pvalue, padj = res$padj,
                    dispGeneEst = col("dispGeneEst"), dispGeneIter = col("dispGeneIter"), dispFit = col("dispFit"),
                    dispMAP = col("dispMAP"), dispersion = col("dispersion"), dispIter = col("dispIter"),
                    dispOutlier = col("dispOutlier"), betaConv = col("betaConv"), maxCooks = col("maxCooks"),
                    replace = col("replace"), allZero = col("allZero"))
  write.csv(out, file, row.names = FALSE, na = "NA")
}
run <- function(min_rep, file, keep = NULL) {
  dds <- DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))
  dds <- DESeq(dds, minReplicatesForReplace = min_rep, quiet = TRUE)
  write_res(results(dds, contrast = meta$contrast), dds, file.path(d, file))
  if (!is.null(keep)) { o <- read.csv(file.path(d, file)); write.csv(o[, keep], file.path(d, file), row.names = FALSE, na = "NA") }
  dfun <- dispersionFunction(dds)
  list(fitType = attr(dfun, "fitType"), sf = unname(sizeFactors(dds)),
       n_beta_not_converged = sum(!mcols(dds)$betaConv, na.rm = TRUE))
}
m <- ncol(cts); p <- 2
r7 <- run(7, paste0("deseq2_", name, "_R.csv"))
rinf <- run(Inf, paste0("deseq2_", name, "_noreplace_R.csv"), keep = c("baseMean", "maxCooks", "betaConv", "allZero"))
meta$R <- list(deseq2_version = as.character(packageVersion("DESeq2")), r_version = R.version.string,
               generated_by = "tests/data/make_replace_fixtures.R", fitType = r7$fitType, sf = r7$sf,
               cooksCutoff = qf(0.99, p, m - p), n_beta_not_converged = r7$n_beta_not_converged,
               n_beta_not_converged_noreplace = rinf$n_beta_not_converged)
writeLines(toJSON(meta, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null"), file.path(d, paste0("deseq2_", name, "_meta.json")))
cat("fitType", r7$fitType, "betaConv FALSE:", r7$n_beta_not_converged, "\n")
