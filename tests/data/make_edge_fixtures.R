# Reference output for the edge-case fixtures in this directory (R DESeq2 DESeq() + results(); R stats lowess / loess).
# Run from the repository root:  Rscript tests/data/make_edge_fixtures.R
# Reads the files written by make_edge_inputs.py and writes
#   deseq2_edge_848_minrep<N>_R.csv (N in meta$min_replicates_for_replace) and deseq2_edge_848_minrep<N>_R_c1.csv,
#   deseq2_edge_sparse_param_R.csv, deseq2_edge_sparse_extreme_R.csv,
#   deseq2_lowess_ties_R.csv (lowess and loess fitted values per case),
# (loess_warned = 1 where R's loess warned, for example about a pseudoinverse on a degenerate neighbourhood)
# and adds the R-side quantities and package versions to each meta json.
suppressMessages({library(DESeq2); library(jsonlite)})
d <- Sys.getenv("CREST_FIXTURE_DIR", "tests/data")

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

fit_one <- function(name, min_rep = NULL, tag = "") {
  inp <- read.csv(file.path(d, paste0("deseq2_", name, "_input.csv")), check.names = FALSE)
  meta <- fromJSON(file.path(d, paste0("deseq2_", name, "_meta.json")), simplifyVector = TRUE)
  genes <- grep("^gene", colnames(inp), value = TRUE)
  cts <- t(as.matrix(inp[, genes])); colnames(cts) <- paste0("s", seq_len(ncol(cts)))
  cd <- inp[, setdiff(colnames(inp), genes), drop = FALSE]
  for (v in names(meta$levels)) cd[[v]] <- factor(cd[[v]], levels = meta$levels[[v]])
  rownames(cd) <- colnames(cts)
  dds <- DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))
  msgs <- character()
  withCallingHandlers({
    dds <- if (is.null(min_rep)) DESeq(dds, quiet = TRUE) else DESeq(dds, minReplicatesForReplace = min_rep, quiet = TRUE)
  }, message = function(m) { msgs <<- c(msgs, trimws(conditionMessage(m))); invokeRestart("muffleMessage") },
     warning = function(w) { msgs <<- c(msgs, paste("warning:", trimws(conditionMessage(w)))); invokeRestart("muffleWarning") })
  write_res(results(dds, contrast = meta$contrast), dds, file.path(d, paste0("deseq2_", name, tag, "_R.csv")))
  if (!is.null(meta$extra_contrasts))
    write_res(results(dds, contrast = meta$extra_contrasts[1, ]), dds, file.path(d, paste0("deseq2_", name, tag, "_R_c1.csv")))
  dfun <- dispersionFunction(dds)
  m <- ncol(dds); p <- ncol(model.matrix(design(dds), colData(dds)))
  list(deseq2_version = as.character(packageVersion("DESeq2")), r_version = R.version.string,
       generated_by = "tests/data/make_edge_fixtures.R", fitType = attr(dfun, "fitType"),
       trend = if (identical(attr(dfun, "fitType"), "parametric")) unname(attr(dfun, "coefficients")) else NULL,
       priorVar = attr(dfun, "dispPriorVar"), sf = unname(sizeFactors(dds)), cooksCutoff = qf(0.99, p, m - p),
       min_replicates_for_replace = if (is.null(min_rep)) 7 else min_rep, messages = msgs)
}

save_meta <- function(name, R) {
  mf <- file.path(d, paste0("deseq2_", name, "_meta.json"))
  meta <- fromJSON(mf, simplifyVector = TRUE)
  meta$R <- R
  writeLines(toJSON(meta, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null"), mf)
}

# --- three groups of 8, 4, 8: partial outlier replacement
meta848 <- fromJSON(file.path(d, "deseq2_edge_848_meta.json"), simplifyVector = TRUE)
by_rep <- list()
for (k in meta848$min_replicates_for_replace) {
  by_rep[[as.character(k)]] <- fit_one("edge_848", k, paste0("_minrep", k))
  cat("edge_848 minrep", k, by_rep[[as.character(k)]]$fitType, "\n")
}
save_meta("edge_848", c(by_rep[["7"]], list(by_min_replicates = lapply(by_rep, function(x) x[c("sf", "cooksCutoff", "messages")]))))

# --- sparse designs
for (nm in c("edge_sparse_param", "edge_sparse_extreme")) {
  R <- fit_one(nm); save_meta(nm, R); cat(nm, R$fitType, "priorVar", R$priorVar, "\n")
}

# --- lowess and loess on random vectors with tied x
cases <- read.csv(file.path(d, "deseq2_lowess_ties_cases.csv"))
inp <- read.csv(file.path(d, "deseq2_lowess_ties_input.csv"))
out <- list()
for (k in cases$case) {
  cs <- cases[cases$case == k, ]; v <- inp[inp$case == k, ]
  lo <- lowess(v$x, v$y, f = cs$f, iter = cs$iter, delta = cs$delta)$y
  warned <- FALSE
  lf <- withCallingHandlers(tryCatch(unname(fitted(loess(y ~ x, data = v, span = cs$span, degree = 2))), error = function(e) rep(NA_real_, nrow(v))),
                            warning = function(w) { warned <<- TRUE; invokeRestart("muffleWarning") })
  out[[length(out) + 1]] <- data.frame(case = k, i = v$i, lowess = lo, loess = lf, loess_warned = as.integer(warned))
}
write.csv(do.call(rbind, out), file.path(d, "deseq2_lowess_ties_R.csv"), row.names = FALSE, na = "NA")
writeLines(toJSON(list(r_version = R.version.string, generated_by = "tests/data/make_edge_fixtures.R",
                       note = "lowess(x, y, f, iter, delta)$y and fitted(loess(y ~ x, span, degree = 2))"), auto_unbox = TRUE, pretty = TRUE),
           file.path(d, "deseq2_lowess_ties_meta.json"))
