# Regenerate the LRT reference outputs: Rscript tests/data/make_lrt_fixtures.R  (DESeq2 1.42)
suppressMessages({library(DESeq2); library(jsonlite)})
d <- "tests/data"
run <- function(name, reduced) {
  inp <- read.csv(file.path(d, paste0("deseq2_", name, "_input.csv")), check.names = FALSE)
  meta <- fromJSON(file.path(d, paste0("deseq2_", name, "_meta.json")))
  genes <- grep("^gene", colnames(inp), value = TRUE)
  cts <- t(as.matrix(inp[, genes])); colnames(cts) <- paste0("s", seq_len(ncol(cts)))
  cd <- inp[, setdiff(colnames(inp), genes), drop = FALSE]
  for (v in names(meta$levels)) cd[[v]] <- factor(cd[[v]], levels = meta$levels[[v]])
  rownames(cd) <- colnames(cts)
  dds <- DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))
  dds <- DESeq(dds, test = "LRT", reduced = as.formula(reduced), quiet = TRUE)
  res <- if (is.null(meta$contrast)) results(dds) else results(dds, contrast = meta$contrast)
  out <- data.frame(baseMean = res$baseMean, log2FoldChange = res$log2FoldChange, lfcSE = res$lfcSE,
                    stat = res$stat, pvalue = res$pvalue, padj = res$padj, dispersion = mcols(dds)$dispersion)
  write.csv(out, file.path(d, paste0("deseq2_", name, "_LRT_R.csv")), row.names = FALSE, na = "NA")
  writeLines(reduced, file.path(d, paste0("deseq2_", name, "_LRT_reduced.txt")))
}
run("three_level_batch", "~ batch")
run("outlier_replacement", "~ 1")
