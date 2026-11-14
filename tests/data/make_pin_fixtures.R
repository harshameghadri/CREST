# Reference output for the DESeq2 fixtures in this directory (R DESeq2 DESeq() + results()).
# Run from the repository root:  Rscript tests/data/make_pin_fixtures.R [name ...]
# Reads deseq2_<name>_input.csv and deseq2_<name>_meta.json (written by make_pin_inputs.py), writes
# deseq2_<name>_R.csv (plus deseq2_<name>_R_<coef>.csv for each entry of meta$extra_names and deseq2_<name>_R_numeric.csv
# for meta$numeric_contrast) and adds the
# R-side quantities and the package versions to the meta json.
suppressMessages({library(DESeq2); library(jsonlite)})
d <- Sys.getenv("CREST_FIXTURE_DIR", "tests/data")
all_names <- c("prior_2v2_sd10", "prior_2v2_sd15", "prior_2v3", "prior_3v2", "refit_outliers", "very_large_counts",
               "filtering_cooks", "trend_fallback", "poscounts", "covariate", "lrt_cooks", "contrast_zero_mask")
args <- commandArgs(trailingOnly = TRUE)
names_to_run <- if (length(args)) args else all_names

write_res <- function(res, dds, file) {
  out <- data.frame(baseMean = res$baseMean, log2FoldChange = res$log2FoldChange, lfcSE = res$lfcSE,
                    stat = res$stat, pvalue = res$pvalue, padj = res$padj,
                    dispGeneEst = mcols(dds)$dispGeneEst, dispFit = mcols(dds)$dispFit,
                    dispMAP = mcols(dds)$dispMAP, dispersion = mcols(dds)$dispersion,
                    maxCooks = mcols(dds)$maxCooks)
  write.csv(out, file, row.names = FALSE, na = "NA")
}

for (name in names_to_run) {
  inp <- read.csv(file.path(d, paste0("deseq2_", name, "_input.csv")), check.names = FALSE)
  mf <- file.path(d, paste0("deseq2_", name, "_meta.json"))
  meta <- fromJSON(mf, simplifyVector = TRUE)
  genes <- grep("^gene", colnames(inp), value = TRUE)
  cts <- t(as.matrix(inp[, genes])); colnames(cts) <- paste0("s", seq_len(ncol(cts)))
  cd <- inp[, setdiff(colnames(inp), genes), drop = FALSE]
  for (v in names(meta$levels)) cd[[v]] <- factor(cd[[v]], levels = meta$levels[[v]])
  rownames(cd) <- colnames(cts)
  dds <- DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))
  sf_type <- if (is.null(meta$sf_type)) "ratio" else meta$sf_type
  fitdisp <- NULL
  msgs <- character()
  withCallingHandlers({
    if (!is.null(meta$test) && meta$test == "LRT") {
      dds <- DESeq(dds, test = "LRT", reduced = as.formula(meta$reduced), sfType = sf_type, quiet = TRUE)
    } else {
      dds <- DESeq(dds, sfType = sf_type, quiet = TRUE)
    }
  }, message = function(m) { msgs <<- c(msgs, trimws(conditionMessage(m))); invokeRestart("muffleMessage") },
     warning = function(w) { msgs <<- c(msgs, paste("warning:", trimws(conditionMessage(w)))); invokeRestart("muffleWarning") })
  res <- if (is.null(meta$contrast)) results(dds) else results(dds, contrast = meta$contrast)
  write_res(res, dds, file.path(d, paste0("deseq2_", name, "_R.csv")))
  for (cn in meta$extra_names) {
    write_res(results(dds, name = cn), dds, file.path(d, paste0("deseq2_", name, "_R_", cn, ".csv")))
  }
  if (!is.null(meta$numeric_contrast)) {
    write_res(results(dds, contrast = as.numeric(meta$numeric_contrast)), dds,
              file.path(d, paste0("deseq2_", name, "_R_numeric.csv")))
  }
  dfun <- dispersionFunction(dds)
  m <- ncol(dds); p <- ncol(model.matrix(design(dds), colData(dds)))
  meta$R <- list(
    deseq2_version = as.character(packageVersion("DESeq2")),
    r_version = R.version.string,
    generated_by = "tests/data/make_pin_fixtures.R",
    fitType = attr(dfun, "fitType"),
    trend = if (identical(attr(dfun, "fitType"), "parametric")) unname(attr(dfun, "coefficients")) else NULL,
    priorVar = attr(dfun, "dispPriorVar"),
    sf = unname(sizeFactors(dds)),
    cooksCutoff = qf(0.99, p, m - p),
    messages = msgs)
  writeLines(toJSON(meta, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null"), mf)
  cat(name, attr(dfun, "fitType"), "priorVar", attr(dfun, "dispPriorVar"), "\n")
}
