"""Does R's local trend, fed CREST's gene-wise dispersions, reproduce CREST's trend on the whole stratum?

    python bench/deseq2/covid_local_trend_swap.py --h5 <covid_stephenson.h5> --obs <covid_stephenson.obs.parquet> --out <dir>

``covid_concordance.py`` reports the largest relative difference of the fitted dispersion trend per stratum; where R's and
CREST's gene-wise dispersions differ for a few genes (``docs/deseq2.md``, known difference 1) the trend differs, most where the
fit is weighted by large means. This script separates that cause from the trend function itself. For each stratum where R
substitutes its local trend it takes CREST's first-pass base means and gene-wise dispersions (outlier replacement switched off:
the trend is fitted before any replacement), gives them to R's ``localDispersionFit`` and compares R's trend at every gene with
the ``dispFit`` CREST returns; it also takes R's own first-pass values and counts the genes where the two tools' gene-wise
estimates differ by more than 1e-6 (relative). Writes ``covid_local_trend_swap.csv``.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import covid_concordance as cc  # noqa: E402
import covid_trend_inputs as ti  # noqa: E402

R_SWAP = r"""
suppressMessages(library(DESeq2))
d <- commandArgs(TRUE)[1]
x <- read.csv(file.path(d, "crest_inputs.csv"))
fn <- getFromNamespace("localDispersionFit", "DESeq2")(x$mean, x$disp, 1e-8)
write.csv(data.frame(trend = fn(x$mean)), file.path(d, "r_trend.csv"), row.names = FALSE)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--strata", nargs="*", default=None, help="default: the strata of covid_trend_inputs.LOCAL_STRATA plus IgG plasma cell, immature B cell")
    a = ap.parse_args()
    import crest
    genes, tabs = ti.stratum_tables(a.h5, a.obs)
    names = a.strata or ti.LOCAL_STRATA + ["IgG plasma cell", "immature B cell"]
    rows = []
    for name in names:
        counts, obs = tabs[name]
        ds = crest.tl.DESeq2(counts, obs.select("Status"), "~ Status", ref_levels={"Status": "Healthy"}, var_names=list(genes), quiet=True,
                             min_replicates_for_replace=float("inf"))
        f = ds.fit
        bm, ge, fit = np.asarray(f["baseMean"], float), np.asarray(f["dispGeneEst"], float), np.asarray(f["dispFit"], float)
        use = ~np.asarray(f["allZero"], bool) & np.isfinite(ge) & (ge > 100 * ti.MIN_DISP)
        live = ~np.asarray(f["allZero"], bool)
        d = Path(tempfile.mkdtemp())
        pl.DataFrame({"mean": bm[use], "disp": ge[use]}).write_csv(d / "crest_inputs.csv")
        (d / "run.R").write_text(R_SWAP)
        p = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr[-2000:])
        r_trend = pl.read_csv(d / "r_trend.csv")["trend"].to_numpy()
        # R's trend is evaluated at the genes used for the fit; CREST's dispFit for the same genes
        rel = np.abs(fit[use] / r_trend - 1)
        r, _ = ti.r_first_pass(genes, counts, obs)
        rg = r["dispGeneEst"].to_numpy()
        both = live & np.isfinite(rg) & np.isfinite(ge)
        diff = both & (np.abs(ge / rg - 1) > 1e-6)
        rows.append({"stratum": name, "genes_in_fit": int(use.sum()), "R_trend_on_CREST_values_max_rel_diff_to_CREST_dispFit": float(rel.max()),
                     "genes_with_different_gene_wise_estimate": int(diff.sum()), "of_which_either_at_floor": int((diff & ((ge <= 1.0001e-8) | (rg <= 1.0001e-8))).sum()),
                     "largest_baseMean_among_them": float(np.max(bm[diff])) if diff.any() else 0.0})
        print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(Path(a.out) / "covid_local_trend_swap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
