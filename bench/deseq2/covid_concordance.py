"""Concordance of CREST with R DESeq2 on donor-level pseudobulk of the Stephenson et al. 2021 COVID-19 atlas.

    python bench/deseq2/covid_concordance.py --h5 <covid_stephenson.h5> --obs <covid_stephenson.obs.parquet> --out <dir> [--workers 6]

Design ``~ Status``, contrast Covid vs Healthy. Status ``LPS`` (healthy volunteers stimulated with LPS) and ``Non_covid`` are
left out. Each donor contributes ONE sample per cell type: its earliest collection day (``Collection_Day``, D0 < D7 < ... < D28;
the sample of that day; a donor with two libraries on that day contributes the 'Initial' one, then the smaller
sample_id), so that donors are independent. A cell type is analysed if at least ``MIN_DONORS`` donors per Status
group have at least ``MIN_CELLS`` cells of it. Counts are ``raw.X`` of the source file (integer-valued, checked by the data
set's manifest). The per-stratum comparison with R is the one of ``kang_concordance.py`` (same code); strata are fitted in
parallel processes, each with one R thread and ``--rayon`` CREST threads.
"""
import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kang_concordance as kc  # noqa: E402

MIN_CELLS = 10
MIN_DONORS = 3
KEEP = ("Covid", "Healthy")
DAY_ORDER = {"D0": 0, "D7": 7, "D9": 9, "D12": 12, "D13": 13, "D28": 28}
BLOCK_CELLS = 40000


def pseudobulk(h5_path, obs, group_codes, n_groups):
    """Dense (n_groups x genes) sums of raw counts per group code (-1 = cell not used), read block-wise from a 10x-style h5."""
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        n_genes, n_cells = (int(x) for x in m["shape"][:])
        assert n_cells == len(group_codes), (n_cells, len(group_codes))
        assert [b.decode() if isinstance(b, bytes) else b for b in m["barcodes"][:5]] == obs["barcode"].head(5).to_list()
        genes = [g.decode() if isinstance(g, bytes) else g for g in m["features"]["id"][:]]
        indptr = m["indptr"][:]
        out = np.zeros((n_groups, n_genes))
        for lo in range(0, n_cells, BLOCK_CELLS):
            hi = min(lo + BLOCK_CELLS, n_cells)
            codes = group_codes[lo:hi]
            if not (codes >= 0).any():
                continue
            a, b = indptr[lo], indptr[hi]
            blk = sp.csc_matrix((m["data"][a:b], m["indices"][a:b], indptr[lo:hi + 1] - a), shape=(n_genes, hi - lo))
            use = np.flatnonzero(codes >= 0)
            ind = sp.csr_matrix((np.ones(len(use)), (codes[use], use)), shape=(n_groups, hi - lo))
            out += np.asarray((ind @ blk.T.tocsr()).todense())
    return genes, out


def build_strata(h5_path, obs_path):
    obs = pl.read_parquet(obs_path).with_row_index("row")
    obs = obs.filter(pl.col("Status").is_in(list(KEEP)))
    obs = obs.with_columns(pl.col("Collection_Day").replace_strict(DAY_ORDER, default=None).alias("day_num"))
    assert obs["day_num"].null_count() == 0, "unknown Collection_Day"
    # the earliest collection day of each donor (one sample per donor)
    first = obs.group_by("donor_id").agg(pl.col("day_num").min().alias("first_day"))
    obs = obs.join(first, on="donor_id").filter(pl.col("day_num") == pl.col("first_day"))
    # two donors have two libraries on their earliest day: take the "Initial" one, then the smallest sample_id
    pick = (obs.group_by("donor_id", "sample_id", "Resample").agg(pl.len().alias("cells"))
            .with_columns((pl.col("Resample") != "Initial").alias("later"))
            .sort("donor_id", "later", "sample_id").group_by("donor_id", maintain_order=True).first())
    n_ties = int((obs.group_by("donor_id").agg(pl.col("sample_id").n_unique().alias("n"))["n"] > 1).sum())
    obs = obs.join(pick.select("donor_id", "sample_id"), on=["donor_id", "sample_id"])
    assert obs.group_by("donor_id").agg(pl.col("sample_id").n_unique().alias("n"))["n"].max() == 1
    sizes = obs.group_by("donor_id", "Status", "cell_type").agg(pl.len().alias("cells")).filter(pl.col("cells") >= MIN_CELLS)
    ok_ct = (sizes.group_by("cell_type", "Status").agg(pl.len().alias("donors")).filter(pl.col("donors") >= MIN_DONORS)
             .group_by("cell_type").agg(pl.len().alias("g")).filter(pl.col("g") == len(KEEP))["cell_type"].to_list())
    sizes = sizes.filter(pl.col("cell_type").is_in(ok_ct)).with_row_index("gid")
    n_all = pl.read_parquet(obs_path, columns=["barcode"]).height
    codes = np.full(n_all, -1, dtype=np.int64)
    key = obs.join(sizes.select("donor_id", "cell_type", "gid"), on=["donor_id", "cell_type"], how="inner")
    codes[key["row"].to_numpy()] = key["gid"].to_numpy()
    genes, mat = pseudobulk(h5_path, pl.read_parquet(obs_path, columns=["barcode"]), codes, sizes.height)
    strata = []
    for ct in sorted(ok_ct):
        s = sizes.filter(pl.col("cell_type") == ct).sort("donor_id")
        strata.append((ct, mat[s["gid"].to_numpy()], s.select(pl.col("donor_id"), pl.col("Status"), pl.col("cells"))))
    return genes, strata, {"cells_used": int((codes >= 0).sum()), "donors": obs["donor_id"].n_unique(), "donors_with_two_libraries_on_earliest_day": n_ties,
                           "donors_by_status": {r["Status"]: r["n"] for r in obs.group_by("Status").agg(pl.col("donor_id").n_unique().alias("n")).to_dicts()},
                           "strata": len(strata)}


def _one(args):
    name, counts, obs, genes = args
    t0 = time.time()
    row, disc, fit, dispo = kc.concordance_stratum(name, counts, obs.select("Status"), genes, "~ Status", ("Status", "Covid", "Healthy"),
                                                   {"Status": "Healthy"}, {"Status": ["Healthy", "Covid"]})
    row["donors_Covid"] = int((obs["Status"] == "Covid").sum())
    row["donors_Healthy"] = int((obs["Status"] == "Healthy").sum())
    row["seconds"] = round(time.time() - t0, 1)
    return row, disc, fit, dispo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", default="bench/deseq2/results")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rayon", type=int, default=2, help="CREST threads per worker")
    a = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.rayon)
    t0 = time.time()
    genes, strata, info = build_strata(a.h5, a.obs)
    print(f"{info}  (pseudobulk {time.time() - t0:.0f} s)", flush=True)
    jobs = [(n, c, o, genes) for n, c, o in strata]
    rows, disc, fits, dispo = [], [], [], []
    # 'spawn': forking a process that already runs polars / rayon threads can deadlock
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for row, d, fit, dout in ex.map(_one, jobs):
            rows.append(row); disc += d; fits.append(fit); dispo += dout
            print({k: row[k] for k in ("stratum", "donors_Covid", "donors_Healthy", "sig_CREST_0.1", "sig_R_0.1", "shared_0.1")}, fit["R_fitType"], fit["CREST_fitType"], flush=True)
    import crest
    kc.write_report(Path(a.out), "covid_concordance", rows, disc, fits, dispo,
                    {"dataset": "Stephenson 2021 COVID-19 PBMC atlas (CELLxGENE), raw.X", "design": "~ Status", "contrast": "Covid vs Healthy",
                     "one sample per donor": "earliest Collection_Day", "excluded Status": ["LPS", "Non_covid"], "min_cells": MIN_CELLS,
                     "min_donors_per_group": MIN_DONORS, "crest_version": crest.__version__, **info})
    return 0


if __name__ == "__main__":
    sys.exit(main())
