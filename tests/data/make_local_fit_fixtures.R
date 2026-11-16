# Reference output for the local dispersion-trend fixtures in this directory (R DESeq2 localDispersionFit / locfit).
# Run from the repository root:  Rscript tests/data/make_local_fit_fixtures.R
# Reads deseq2_local_fit_cases.csv and deseq2_local_fit_at.csv (written by make_local_fit_inputs.py), and
# deseq2_local_fit_covid_cases.csv (real strata, written by bench/deseq2/covid_trend_inputs.py), and writes
#   deseq2_local_fit_R.csv  (case, at, trend; also the cases of deseq2_local_fit_covid_cases.csv when present): exp(predict(locfit(log(disp) ~ log(mean), weights = mean), log(at))) as
#                           DESeq2 computes it; NA where R's fit stops with an error, and
#   deseq2_local_fit_R_status.json: per case "ok" or R's error message, and the package versions.
suppressMessages({library(DESeq2); library(jsonlite)})
d <- Sys.getenv("CREST_FIXTURE_DIR", "tests/data")
cases <- read.csv(file.path(d, "deseq2_local_fit_cases.csv"))
at <- read.csv(file.path(d, "deseq2_local_fit_at.csv"))
# real strata: (case, mean, disp) from bench/deseq2/covid_trend_inputs.py; compared at the stratum's own means and a grid
cov_file <- file.path(d, "deseq2_local_fit_covid_cases.csv")
if (file.exists(cov_file)) {
  cv <- read.csv(cov_file, check.names = FALSE)
  cases <- rbind(cases, cv)
  at <- rbind(at, do.call(rbind, lapply(unique(cv$case), function(nm) {
    x <- cv[cv$case == nm & cv$disp >= 1e-7, ]
    lm <- log(x$mean)
    data.frame(case = nm, at = sort(unique(c(x$mean[seq(1, nrow(x), length.out = min(nrow(x), 400))], exp(seq(min(lm) - 2.5, max(lm) + 2.5, length.out = 80))))))
  })))
}
minDisp <- 1e-8
local_fit <- getFromNamespace("localDispersionFit", "DESeq2")
out <- list(); status <- list()
for (nm in unique(cases$case)) {
  cs <- cases[cases$case == nm, ]
  pts <- at[at$case == nm, "at"]
  res <- tryCatch({
    fn <- local_fit(cs$mean, cs$disp, minDisp)
    list(ok = TRUE, trend = fn(pts))
  }, error = function(e) list(ok = FALSE, msg = conditionMessage(e)))
  status[[nm]] <- if (res$ok) "ok" else res$msg
  out[[nm]] <- data.frame(case = nm, at = pts, trend = if (res$ok) res$trend else NA_real_)
}
o <- do.call(rbind, out)
write.csv(o, file.path(d, "deseq2_local_fit_R.csv"), row.names = FALSE, na = "NA")
status$versions <- list(R = R.version.string, DESeq2 = as.character(packageVersion("DESeq2")),
                        locfit = as.character(packageVersion("locfit")))
writeLines(toJSON(status, auto_unbox = TRUE, pretty = TRUE), file.path(d, "deseq2_local_fit_R_status.json"))
