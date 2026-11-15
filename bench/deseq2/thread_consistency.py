"""Consistency of R DESeq2 (BiocParallel workers) and CREST (rayon threads) across thread counts on Kang 2018.

    python bench/deseq2/thread_consistency.py --data <kang dir> --out <dir> --threads 1 8 64

For every cell type and every k: R through kang_pseudobulk.run_r(workers=k) and CREST in a fresh process with
RAYON_NUM_THREADS=k. Everything is compared with k = 1 of the same tool: p-values, adjusted p-values and log2 fold changes
(max abs and max relative difference, NA masks), and the number of genes whose call at padj < 0.1 differs. Crash and
consistency check only; nothing here is a timing result.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("KANG_BENCH_DIR", str(HERE)))
import crest  # noqa: E402


def worker(npz, out):
    z = np.load(npz, allow_pickle=False)
    obs = pl.DataFrame({"ind": z["ind"].astype(str), "stim": z["stim"].astype(str)})
    ds = crest.tl.DESeq2(z["counts"], obs, "~ ind + stim", ref_levels={"stim": "ctrl"}, quiet=True)
    res = ds.results(("stim", "stim", "ctrl"))
    f = lambda c: res[c].fill_null(np.nan).to_numpy()  # noqa: E731
    np.savez(out, padj=f("padj"), pvalue=f("pvalue"), lfc=f("log2FoldChange"), dispersion=np.asarray(ds.fit["dispersion"], float))


def diff(a, b):
    na, nb = np.isnan(a), np.isnan(b)
    ok = ~na & ~nb
    ab = np.abs(a[ok] - b[ok])
    return {"na_mismatch": int((na != nb).sum()), "max_abs": float(ab.max()) if ok.any() else 0.0,
            "max_rel": float((ab / (1e-300 + np.abs(b[ok]))).max()) if ok.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--threads", nargs="+", type=int, default=[1, 8, 64])
    ap.add_argument("--worker", nargs=2)
    a = ap.parse_args()
    if a.worker:
        return worker(*a.worker)
    import kang_concordance as kc
    import kang_pseudobulk as kp
    genes, strata, n_cells = kc.kang_strata(Path(a.data))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for ct, counts, obs in strata:
        ob = obs.with_columns(pl.Series("pseudobulk", [f"s{i}" for i in range(obs.height)]))
        ref = {}
        for k in a.threads:
            rec = {"stratum": ct, "threads": k}
            # R
            t0 = time.time()
            try:
                r, _diag = kp.run_r(counts, ob, genes, workers=k)
                R = {c: r[c].fill_null(np.nan).to_numpy().astype(float) for c in ("padj", "pvalue", "log2FoldChange")}
                R["lfc"] = R.pop("log2FoldChange")
                rec["R_status"] = "ok"
            except Exception as e:  # noqa: BLE001
                R = None; rec["R_status"] = f"FAILED: {str(e)[-300:]}"; failures.append((ct, "R", k))
            rec["R_wall_s"] = round(time.time() - t0, 1)
            # CREST
            d = Path(tempfile.mkdtemp())
            np.savez(d / "in.npz", counts=counts, ind=obs["ind"].to_numpy().astype(str), stim=obs["stim"].to_numpy().astype(str))
            env = {**os.environ, **{v: str(k) for v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
            p = subprocess.run([sys.executable, __file__, "--worker", str(d / "in.npz"), str(d / "out.npz")], env=env, capture_output=True, text=True)
            if p.returncode:
                C = None; rec["CREST_status"] = f"FAILED: {p.stderr[-300:]}"; failures.append((ct, "CREST", k))
            else:
                C = dict(np.load(d / "out.npz")); rec["CREST_status"] = "ok"
            for tool, X in (("R", R), ("CREST", C)):
                if X is None:
                    continue
                if (tool, "base") not in ref:
                    ref[(tool, "base")] = X
                    ref[(tool, "k")] = k
                b = ref[(tool, "base")]
                for col in ("padj", "pvalue", "lfc"):
                    d_ = diff(X[col], b[col])
                    rec.update({f"{tool}_{col}_{kk}": vv for kk, vv in d_.items()})
                rec[f"{tool}_sig_0.1"] = int((np.nan_to_num(X["padj"], nan=1.0) < 0.1).sum())
                rec[f"{tool}_sig_call_diff_vs_k{ref[(tool, 'k')]}"] = int(((np.nan_to_num(X["padj"], nan=1.0) < 0.1) != (np.nan_to_num(b["padj"], nan=1.0) < 0.1)).sum())
            if R is not None and C is not None:
                rec["shared_sig_0.1"] = int(((np.nan_to_num(R["padj"], nan=1.0) < 0.1) & (np.nan_to_num(C["padj"], nan=1.0) < 0.1)).sum())
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(out / "thread_consistency.csv")
    summary = {"threads": a.threads, "failures": failures, "cells": n_cells}
    for tool in ("R", "CREST"):
        for col in ("padj", "pvalue", "lfc"):
            vals = [r.get(f"{tool}_{col}_max_abs") for r in rows if r.get(f"{tool}_{col}_max_abs") is not None]
            rels = [r.get(f"{tool}_{col}_max_rel") for r in rows if r.get(f"{tool}_{col}_max_rel") is not None]
            summary[f"{tool}_{col}_max_abs_over_all"] = max(vals) if vals else None
            summary[f"{tool}_{col}_max_rel_over_all"] = max(rels) if rels else None
        summary[f"{tool}_na_mismatch_total"] = int(sum(r.get(f"{tool}_padj_na_mismatch", 0) + r.get(f"{tool}_pvalue_na_mismatch", 0) for r in rows))
    (out / "thread_consistency.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
