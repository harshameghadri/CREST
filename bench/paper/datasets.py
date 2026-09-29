"""Benchmark datasets: download (resumable, size-checked, SHA-256 recorded) and
conversion to one common format, a Cell Ranger v3 ``.h5`` (read natively by
both scanpy and CREST) plus an ``<name>.obs.parquet`` sidecar with per-cell
metadata (barcode-aligned) where the source has any.

    python bench/paper/datasets.py --data-dir bench_data --datasets pbmc3k kang
    python bench/paper/datasets.py --list
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

TENX = "https://cf.10xgenomics.com/samples/cell-exp"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
CXG = "https://datasets.cellxgene.cziscience.com"

# name -> source files, expected download sizes (bytes, from the servers' Content-Length
# on 2026-09-28), description, citation, tier and the obs columns useful downstream.
REGISTRY: dict[str, dict] = {
    "pbmc3k": {
        "files": {"pbmc3k.tar.gz": (f"{TENX}/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz", 7621991)},
        "cells": 2700, "tier": "quick", "chemistry": "10x 3' v1",
        "description": "10x PBMC 3k (scanpy tutorial dataset)",
        "citation": "10x Genomics (2016), pbmc3k",
    },
    "pbmc10k": {
        "files": {"pbmc10k.h5": (f"{TENX}/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5", 37491093)},
        "cells": 11769, "tier": "quick", "chemistry": "10x 3' v3",
        "description": "10x PBMC 10k, v3 chemistry",
        "citation": "10x Genomics (2018), pbmc_10k_v3",
    },
    "kang": {
        "files": {
            "GSM2560248_2.1.mtx.gz": (f"{GEO}/samples/GSM2560nnn/GSM2560248/suppl/GSM2560248_2.1.mtx.gz", 28545918),
            "GSM2560248_barcodes.tsv.gz": (f"{GEO}/samples/GSM2560nnn/GSM2560248/suppl/GSM2560248_barcodes.tsv.gz", None),
            "GSM2560249_2.2.mtx.gz": (f"{GEO}/samples/GSM2560nnn/GSM2560249/suppl/GSM2560249_2.2.mtx.gz", None),
            "GSM2560249_barcodes.tsv.gz": (f"{GEO}/samples/GSM2560nnn/GSM2560249/suppl/GSM2560249_barcodes.tsv.gz", None),
            "GSE96583_batch2.genes.tsv.gz": (f"{GEO}/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.genes.tsv.gz", 277054),
            "GSE96583_batch2.total.tsne.df.tsv.gz": (f"{GEO}/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.total.tsne.df.tsv.gz", None),
        },
        "cells": 29065, "tier": "quick", "chemistry": "10x 3' v2",
        "description": "Kang 2018 IFN-beta PBMCs, 8 donors x ctrl/stim, demuxlet doublet calls",
        "citation": "Kang et al. (2018) Nat Biotechnol 36:89, GEO GSE96583",
        "batch": "stim", "sample": ["ind", "stim"], "condition": "stim", "contrast": ["stim", "stim", "ctrl"],
        "celltype": "cell", "doublet": "multiplets",
    },
    "pbmc68k": {
        "files": {"pbmc68k.tar.gz": (f"{TENX}/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz", 124442812)},
        "cells": 68579, "tier": "standard", "chemistry": "10x 3' v1",
        "description": "10x fresh 68k PBMCs, donor A (Zheng et al. 2017)",
        "citation": "Zheng et al. (2017) Nat Commun 8:14049",
    },
    "covid_stephenson": {
        "files": {"covid_stephenson.h5ad": (f"{CXG}/fe2e847c-1602-4f1b-86a4-112e4dc7a8e3.h5ad", 7058204031)},
        "cells": 647366, "tier": "full", "chemistry": "10x 3'",
        "description": "Stephenson 2021 COVID-19 PBMC atlas, 120 donors, 3 sites (CELLxGENE)",
        "citation": "Stephenson et al. (2021) Nat Med 27:904",
        "batch": "donor_id", "sample": ["donor_id"], "condition": "disease", "contrast": ["disease", "COVID-19", "normal"],
        "celltype": "cell_type",
    },
    "neurons_1m": {
        "files": {"neurons_1m.h5": (f"{TENX}/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5", 4216018749)},
        "cells": 1306127, "tier": "full", "chemistry": "10x 3' v2",
        "description": "10x 1.3M mouse brain cells (E18)",
        "citation": "10x Genomics (2017), 1M_neurons",
    },
}
# Controlled scaling series: synthetic cells interpolating real PBMC 68k neighbours.
SYNTHETIC = {f"synth_{n // 1000}k": n for n in (100_000, 200_000, 500_000, 1_000_000)}
SYNTH_TIER = {"synth_100k": "standard", "synth_200k": "standard", "synth_500k": "full", "synth_1000k": "full"}


def tier_datasets(tier: str) -> list[str]:
    order = {"quick": 0, "standard": 1, "full": 2}[tier]
    real = [k for k, v in REGISTRY.items() if order >= {"quick": 0, "standard": 1, "full": 2}[v["tier"]]]
    syn = [k for k, t in SYNTH_TIER.items() if order >= {"quick": 0, "standard": 1, "full": 2}[t]]
    return real + syn


# --------------------------------------------------------------------------- download
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int | None, retries: int = 5) -> None:
    """Resumable download (HTTP Range), retried with exponential backoff, size-checked."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print(f"  [ok]       {dest.name}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "crest-bench/1.0",
                                                       **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                total = int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
                print(f"  [download] {dest.name} ({total / 1e6:.1f} MB)" + (f", resuming at {have / 1e6:.1f} MB" if mode == "ab" else ""))
                with open(part, mode) as f:
                    t0, done = time.time(), (have if mode == "ab" else 0)
                    while chunk := r.read(1 << 22):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - t0 > 10:
                            print(f"             {done / 1e6:9.1f} / {total / 1e6:.1f} MB", flush=True)
                            t0 = time.time()
            if size is not None and part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size} != expected {size}")
            part.rename(dest)
            return
        except Exception as e:  # noqa: BLE001
            wait = 2 ** (attempt + 1)
            print(f"  [retry]    {dest.name}: {e}; again in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"could not download {url}")


def record_checksums(data_dir: Path, names: list[Path]) -> None:
    """SHA-256 of every source file, recorded on first sight and verified afterwards."""
    f = data_dir / "checksums.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    for p in names:
        digest = _sha256(p)
        if p.name in known and known[p.name] != digest:
            raise RuntimeError(f"checksum mismatch for {p.name}: {digest} != recorded {known[p.name]}")
        known[p.name] = digest
    f.write_text(json.dumps(known, indent=1, sort_keys=True))


# --------------------------------------------------------------------------- conversion
def _enc(xs) -> np.ndarray:
    return np.array([str(x).encode() for x in xs])


def write_10x_h5(out: Path, indptr, indices, data, n_genes: int, barcodes, gene_ids, gene_names, genome="NA") -> None:
    """Cell Ranger v3 layout: CSC genes x cells == CSR cells x genes."""
    tmp = out.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=_enc(barcodes))
        m.create_dataset("data", data=np.asarray(data).astype(np.int32), compression="gzip", compression_opts=1)
        m.create_dataset("indices", data=np.asarray(indices).astype(np.int32), compression="gzip", compression_opts=1)
        m.create_dataset("indptr", data=np.asarray(indptr).astype(np.int64))
        m.create_dataset("shape", data=np.array([n_genes, len(indptr) - 1], np.int32))
        ft = m.create_group("features")
        ft.create_dataset("id", data=_enc(gene_ids))
        ft.create_dataset("name", data=_enc(gene_names))
        ft.create_dataset("feature_type", data=_enc(["Gene Expression"] * n_genes))
        ft.create_dataset("genome", data=_enc([genome] * n_genes))
    tmp.rename(out)


def _mtx_dir_to_h5(mtx_dir: Path, out: Path, genome: str) -> None:
    import crest
    bf = crest.read_10x_mtx(mtx_dir)
    X = bf.to_scipy(transform=False).tocsr()
    X.sort_indices()
    write_10x_h5(out, X.indptr, X.indices, X.data, X.shape[1], bf.obs["barcode"].to_list(),
                 bf.var["gene_ids"].to_list(), bf.var["gene_name"].to_list(), genome)


def _build_kang(d: Path, out: Path) -> None:
    import polars as pl
    import scipy.io
    import scipy.sparse as sp
    mats, bcs = [], []
    for mtx, bc in [("GSM2560248_2.1.mtx.gz", "GSM2560248_barcodes.tsv.gz"),
                    ("GSM2560249_2.2.mtx.gz", "GSM2560249_barcodes.tsv.gz")]:
        mats.append(scipy.io.mmread(gzip.open(d / mtx)).T.tocsr())
        bcs += gzip.open(d / bc).read().decode().split()
    X = sp.vstack(mats).tocsr()
    X.sort_indices()
    ann = pl.read_csv(d / "GSE96583_batch2.total.tsne.df.tsv.gz", separator="\t", has_header=False, skip_rows=1,
                      new_columns=["barcode", "tsne1", "tsne2", "ind", "stim", "cluster", "cell", "multiplets"])
    stem = [b.rsplit("-", 1)[0] for b in ann["barcode"].to_list()]
    if ann.height != X.shape[0] or stem != [b.rsplit("-", 1)[0] for b in bcs]:
        raise RuntimeError("Kang annotation rows do not match the matrix barcodes")
    genes = pl.read_csv(d / "GSE96583_batch2.genes.tsv.gz", separator="\t", has_header=False,
                        new_columns=["gene_ids", "gene_name"])
    # unique barcodes: the two lanes reuse some, the annotation's own names are unique
    barcodes = ann["barcode"].to_list()
    write_10x_h5(out, X.indptr, X.indices, X.data, X.shape[1], barcodes, genes["gene_ids"].to_list(),
                 genes["gene_name"].to_list(), "hg19")
    ann.select("barcode", pl.col("ind").cast(pl.Utf8), "stim", "cell", "multiplets").write_parquet(
        out.with_suffix(".obs.parquet"))


def _build_h5ad(src: Path, out: Path) -> None:
    """CELLxGENE h5ad -> 10x h5, streaming the raw counts (raw/X if present, else X)."""
    import polars as pl
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from crest.io import _h5ad_frame
    with h5py.File(src, "r") as f:
        X = f["raw/X"] if "raw" in f and "X" in f["raw"] else f["X"]
        var = _h5ad_frame(f["raw/var"] if "raw" in f and "var" in f["raw"] else f["var"], "gene_ids")
        obs = _h5ad_frame(f["obs"], "barcode")
        enc = X.attrs.get("encoding-type", "csr_matrix")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        if not enc.startswith("csr"):
            raise RuntimeError(f"expected CSR counts in {src.name}, found {enc}")
        n_genes = int(X.attrs["shape"][1])
        indptr = X["indptr"][:].astype(np.int64)
        tmp = out.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as g:
            m = g.create_group("matrix")
            nnz = int(indptr[-1])
            dd = m.create_dataset("data", (nnz,), np.int32, compression="gzip", compression_opts=1, chunks=(1 << 20,))
            di = m.create_dataset("indices", (nnz,), np.int32, compression="gzip", compression_opts=1, chunks=(1 << 20,))
            step = 1 << 26
            for lo in range(0, nnz, step):
                hi = min(lo + step, nnz)
                vals = X["data"][lo:hi]
                if not np.all(np.mod(vals, 1) == 0):
                    raise RuntimeError(f"{src.name}: non-integer values; no raw counts found")
                dd[lo:hi] = vals.astype(np.int32)
                di[lo:hi] = X["indices"][lo:hi].astype(np.int32)
            m.create_dataset("indptr", data=indptr)
            m.create_dataset("shape", data=np.array([n_genes, len(indptr) - 1], np.int32))
            m.create_dataset("barcodes", data=_enc(obs["barcode"].to_list()))
            names = var["feature_name"].to_list() if "feature_name" in var.columns else var["gene_ids"].to_list()
            ft = m.create_group("features")
            ft.create_dataset("id", data=_enc(var["gene_ids"].to_list()))
            ft.create_dataset("name", data=_enc(names))
            ft.create_dataset("feature_type", data=_enc(["Gene Expression"] * n_genes))
            ft.create_dataset("genome", data=_enc(["GRCh38"] * n_genes))
        tmp.rename(out)
    keep = [c for c in ["barcode", "donor_id", "disease", "cell_type", "sex", "assay", "tissue",
                        "development_stage", "sample_id", "Site", "Status"] if c in obs.columns]
    obs.select([pl.col(c).cast(pl.Utf8) for c in keep]).write_parquet(out.with_suffix(".obs.parquet"))


def build(name: str, data_dir: Path) -> Path:
    """Download (if needed) and convert; returns the path of ``<name>.h5``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{name}.h5"
    if name in SYNTHETIC:
        if not out.exists():
            src = build("pbmc68k", data_dir)
            from make_dataset import synthesize
            print(f"  [build]    {out.name} from {src.name}")
            synthesize(src, SYNTHETIC[name], out, seed=0)
        return out
    spec = REGISTRY[name]
    src_dir = data_dir / "src" / name
    src_dir.mkdir(parents=True, exist_ok=True)
    for fname, (url, size) in spec["files"].items():
        download(url, src_dir / fname, size)
    record_checksums(data_dir, [src_dir / f for f in spec["files"]])
    if out.exists():
        return out
    print(f"  [convert]  {name}")
    if name in ("pbmc3k", "pbmc68k"):
        tgz = src_dir / f"{name}.tar.gz"
        ex = src_dir / "extracted"
        if not ex.exists():
            with tarfile.open(tgz) as t:
                t.extractall(ex)
        mtx = next(ex.rglob("matrix.mtx*")).parent
        _mtx_dir_to_h5(mtx, out, "hg19")
    elif name in ("pbmc10k", "neurons_1m"):
        shutil.copyfile(src_dir / f"{name}.h5", out)
    elif name == "kang":
        _build_kang(src_dir, out)
    elif name == "covid_stephenson":
        _build_h5ad(src_dir / "covid_stephenson.h5ad", out)
    else:
        raise KeyError(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="bench_data")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--tier", choices=["quick", "standard", "full"])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in REGISTRY.items():
            print(f"{k:18s} {v['cells']:>9,d} cells  tier={v['tier']:8s} {v['description']}")
        for k, n in SYNTHETIC.items():
            print(f"{k:18s} {n:>9,d} cells  tier={SYNTH_TIER[k]:8s} synthetic, interpolated from pbmc68k")
        return
    names = a.datasets or tier_datasets(a.tier or "standard")
    for n in names:
        print(f"== {n}")
        print(f"   -> {build(n, Path(a.data_dir))}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
