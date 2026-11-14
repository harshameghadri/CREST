# Spread of R's prior variance (estimateDispersionsPriorVar, <= 3 residual df) over random seeds.
# Run from the repository root:  Rscript tests/data/make_prior_var_band.R
# R fixes set.seed(2) inside the function, so its value is one draw of a Monte-Carlo estimate; this script
# repeats the identical computation (body copied from DESeq2, seed made a parameter) for seeds 1..N on the
# dispersion residuals of each design and records the observed range as the acceptance band, together with
# the value of the unmodified DESeq2 function (seed 2).
suppressMessages({library(DESeq2); library(jsonlite)})
d <- Sys.getenv("CREST_FIXTURE_DIR", "tests/data")
designs <- c("prior_2v2_sd10", "prior_2v2_sd15", "prior_2v3", "prior_3v2")
n_seeds <- 50

prior_var_seed <- function(object, seed, minDisp = 1e-08) {
  objectNZ <- object[!mcols(object)$allZero, , drop = FALSE]
  aboveMinDisp <- mcols(objectNZ)$dispGeneEst >= minDisp * 100
  modelMatrix <- getModelMatrix(object)
  dispResiduals <- log(mcols(objectNZ)$dispGeneEst) - log(mcols(objectNZ)$dispFit)
  m <- nrow(modelMatrix); p <- ncol(modelMatrix)
  set.seed(seed)
  obsDist <- dispResiduals[aboveMinDisp]
  brks <- -20:20/2
  obsDist <- obsDist[obsDist > min(brks) & obsDist < max(brks)]
  obsVarGrid <- seq(from = 0, to = 8, length = 200)
  obsDistHist <- hist(obsDist, breaks = brks, plot = FALSE)
  klDivs <- sapply(obsVarGrid, function(x) {
    randDist <- log(rchisq(10000, df = (m - p))) + rnorm(10000, 0, sqrt(x)) - log(m - p)
    randDist <- randDist[randDist > min(brks) & randDist < max(brks)]
    randDistHist <- hist(randDist, breaks = brks, plot = FALSE)
    z <- c(obsDistHist$density, randDistHist$density)
    small <- min(z[z > 0])
    sum(obsDistHist$density * (log(obsDistHist$density + small) - log(randDistHist$density + small)))
  })
  lofit <- loess(klDivs ~ obsVarGrid, span = 0.2)
  obsVarFineGrid <- seq(from = 0, to = 8, length = 1000)
  lofitFitted <- predict(lofit, obsVarFineGrid)
  pmax(obsVarFineGrid[which.min(lofitFitted)], 0.25)
}

out <- list(deseq2_version = as.character(packageVersion("DESeq2")), r_version = R.version.string,
            generated_by = "tests/data/make_prior_var_band.R", seeds = 1:n_seeds, designs = list())
for (name in designs) {
  inp <- read.csv(file.path(d, paste0("deseq2_", name, "_input.csv")), check.names = FALSE)
  meta <- fromJSON(file.path(d, paste0("deseq2_", name, "_meta.json")), simplifyVector = TRUE)
  genes <- grep("^gene", colnames(inp), value = TRUE)
  cts <- t(as.matrix(inp[, genes])); colnames(cts) <- paste0("s", seq_len(ncol(cts)))
  cd <- inp[, setdiff(colnames(inp), genes), drop = FALSE]
  for (v in names(meta$levels)) cd[[v]] <- factor(cd[[v]], levels = meta$levels[[v]])
  rownames(cd) <- colnames(cts)
  dds <- DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))
  dds <- estimateSizeFactors(dds)
  dds <- estimateDispersionsGeneEst(dds, quiet = TRUE)
  dds <- estimateDispersionsFit(dds, quiet = TRUE)
  unmodified <- DESeq2:::estimateDispersionsPriorVar(dds)
  vals <- vapply(1:n_seeds, function(s) prior_var_seed(dds, s), numeric(1))
  stopifnot(isTRUE(all.equal(vals[2], unmodified, tolerance = 0)))
  out$designs[[name]] <- list(fitType = attr(dispersionFunction(dds), "fitType"), residual_df = nrow(colData(dds)) - ncol(getModelMatrix(dds)),
                              r_default_seed2 = unmodified, values = vals, band_min = min(vals), band_max = max(vals),
                              mean = mean(vals), sd = sd(vals))
  cat(name, "seed2", unmodified, "band", min(vals), max(vals), "\n")
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(d, "deseq2_prior_var_band.json"))
