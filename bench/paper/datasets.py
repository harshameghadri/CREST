"""Benchmark datasets: download (resumable, size-checked, SHA-256 recorded) and
conversion to one common format, a Cell Ranger v3 ``.h5`` (read natively by
both scanpy and CREST) plus an ``<name>.obs.parquet`` sidecar with per-cell
metadata (barcode-aligned) where the source has any.

    python bench/paper/datasets.py --data-dir bench_data --datasets pbmc3k kang
    python bench/paper/datasets.py --list
    python bench/paper/datasets.py --data-dir bench_data --datasets pbmc3k pbmc68k --verify
    python bench/paper/datasets.py --data-dir bench_data --subsample neurons_1m --sizes 25000 100000 500000 --seed 0
    python bench/paper/datasets.py --data-dir bench_data --datasets pbmc3k --manifest

Independent conversion. The 10x MTX datasets are converted with ``scipy.io.mmread`` and plain text
parsing, not with CREST's reader, so a reader defect cannot be baked into the file both tools read.
``--verify`` checks every converted file count for count (equal CSR arrays, gene and barcode names):
against the source files read by scipy, against ``scanpy.read_10x_mtx`` / ``scanpy.read_10x_h5``, and
CREST's own readers against the independent read. It exits non-zero on any mismatch.

Real subsamples. ``<parent>_sub<N>`` (N cells, e.g. ``neurons_1m_sub100k``) is a seeded random subset of a
real dataset; the subsets of one parent and seed are nested (they are prefixes of one random
permutation), so the scaling series differs only in the number of cells. The synthetic ``synth_*`` sets
(built from 68k real cells with CREST's own PCA and kNN) are a labelled supplement, never the basis
of a scaling statement.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol  # noqa: E402

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
    "parse_pbmc": {
        # Parse Biosciences "10 million human PBMCs" (12 donors x 90 cytokines + PBS), as
        # processed for CellFlow: all 9.7M cells, raw counts of 2,000 selected genes in X.
        # Figshare may refuse scripted downloads
        # (bot check); then download it in a browser and put it in the data directory under
        # any of `local_names`, it is picked up from there.
        "files": {"parse_pbmc.h5ad": ("https://figshare.com/ndownloader/files/53372768", 13101592422)},
        "local_names": ["Parse_1M_adata_for_cellflow_datasets_with_embeddings.h5ad",
                        "adata_for_cellflow_datasets_with_embeddings.h5ad"],
        "md5": "d89559875a6c75ab37e0adf84c6e177d",
        "cells": 9_697_974, "tier": "full", "chemistry": "Parse Evercode WT v3 (split-pool)",
        "description": "Parse 10M PBMC cytokine atlas, 9.7M cells x 2,000 genes (CellFlow; figshare 28589774)",
        "citation": "Parse Biosciences (2024), 10 million human PBMCs; figshare 28589774",
        # batch / sample / condition / cell type columns are detected at conversion
        # (written to parse_pbmc.meta.json), since the file's obs schema is not fixed here
    },
    "neurons_1m": {
        "files": {"neurons_1m.h5": (f"{TENX}/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5", 4216018749)},
        "cells": 1306127, "tier": "full", "chemistry": "10x 3' v2",
        "description": "10x 1.3M mouse brain cells (E18)",
        "citation": "10x Genomics (2017), 1M_neurons",
    },
}
# Random cell subsets of a real dataset: name -> (parent, cells). parse_pbmc_1m is the size
# at which scanpy still fits in 128 GB (its dense scaled matrix at 9.7M cells would be 78 GB).
SUBSETS = {"parse_pbmc_1m": ("parse_pbmc", 1_000_000)}
SUBSET_TIER = {"parse_pbmc_1m": "full"}
# Real-subsample scaling series (the primary scaling evidence): seeded random subsets of a large real
# dataset, named ``<parent>_sub<N>`` and nested (see ``subset_rows``). Sizes are a draft pending the
# owner's choice of datasets; ``--subsample PARENT --sizes ...`` builds any other series.
SUBSAMPLE_SERIES = {"parse_pbmc": (250_000, 500_000),            # with parse_pbmc_1m: 250k / 500k / 1M
                    "neurons_1m": (25_000, 50_000, 100_000, 200_000, 500_000)}
SUBSAMPLE_SEED = 0
for _parent, _sizes in SUBSAMPLE_SERIES.items():
    for _n in _sizes:
        SUBSETS[protocol.subset_name(_parent, _n)] = (_parent, _n)
        SUBSET_TIER[protocol.subset_name(_parent, _n)] = "full"
# Supplement: synthetic cells interpolating real PBMC 68k neighbours with CREST's own normalisation,
# PCA and kNN. A labelled supplement only: its structure is defined by CREST, so no scaling statement
# rests on it (one generator, one source, only the cell count varies; the standard tier has 4 points).
SYNTHETIC = {f"synth_{n // 1000}k": n for n in (25_000, 50_000, 100_000, 200_000, 500_000, 1_000_000)}
SYNTH_TIER = {"synth_25k": "standard", "synth_50k": "standard", "synth_100k": "standard", "synth_200k": "standard",
              "synth_500k": "full", "synth_1000k": "full"}
# A tiny series interpolated from PBMC 3k, for smoke-testing the harness end to end on a laptop
# (--tier smoke); never part of the paper tiers.
SYNTHETIC.update({"synth_smoke_2k": 2_000, "synth_smoke_4k": 4_000, "synth_smoke_8k": 8_000})
SYNTH_TIER.update({k: "smoke" for k in ("synth_smoke_2k", "synth_smoke_4k", "synth_smoke_8k")})
SYNTH_SOURCE = {k: "pbmc3k" for k in ("synth_smoke_2k", "synth_smoke_4k", "synth_smoke_8k")}  # default pbmc68k
TIER_ORDER = {"quick": 0, "standard": 1, "full": 2}


def dataset_kind(name: str) -> str:
    """``real``, ``real-subsample`` or ``synthetic`` (the supplement)."""
    if name in SYNTHETIC:
        return "synthetic"
    return "real-subsample" if subset_spec(name) else "real"


def subset_spec(name: str):
    """``(parent, n_cells)`` of a real-subsample dataset name, else None."""
    return SUBSETS.get(name) or protocol.parse_subset_name(name)


def tier_datasets(tier: str) -> list[str]:
    if tier == "smoke":
        return ["pbmc3k"] + [k for k, t in SYNTH_TIER.items() if t == "smoke"]
    order = TIER_ORDER[tier]
    real = [k for k, v in REGISTRY.items() if order >= TIER_ORDER[v["tier"]]]
    syn = [k for k, t in SYNTH_TIER.items() if t in TIER_ORDER and order >= TIER_ORDER[t]]
    sub = [k for k, t in SUBSET_TIER.items() if order >= TIER_ORDER[t]]
    return real + sub + syn


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


def _text_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def read_mtx_dir_independent(mtx_dir: Path):
    """A 10x MTX directory (``matrix.mtx``, ``genes.tsv`` or ``features.tsv``, ``barcodes.tsv``; each
    optionally gzipped) read with ``scipy.io.mmread`` and plain text parsing, without CREST or scanpy.
    Returns ``(X, barcodes, gene_ids, gene_names)``: X is CSR, cells x genes, with sorted indices."""
    import scipy.io
    import scipy.sparse as sp
    mtx_dir = Path(mtx_dir)

    def find(*stems):
        for stem in stems:
            for suffix in ("", ".gz"):
                f = mtx_dir / f"{stem}{suffix}"
                if f.exists():
                    return f
        raise FileNotFoundError(f"none of {stems} (.gz) in {mtx_dir}")

    mtx = find("matrix.mtx")
    opener = gzip.open if mtx.suffix == ".gz" else open
    with opener(mtx, "rb") as f:
        M = scipy.io.mmread(f)                      # genes x cells in the 10x convention
    X = sp.csr_matrix(M.T)
    X.sum_duplicates()
    X.sort_indices()
    genes = [ln.split("\t") for ln in _text_lines(find("features.tsv", "genes.tsv"))]
    ids = [g[0] for g in genes]
    names = [g[1] if len(g) > 1 else g[0] for g in genes]
    barcodes = [ln.split("\t")[0] for ln in _text_lines(find("barcodes.tsv"))]
    if X.shape != (len(barcodes), len(ids)):
        raise RuntimeError(f"{mtx_dir}: matrix {X.shape} does not match {len(barcodes)} barcodes x {len(ids)} genes")
    return X, barcodes, ids, names


def _mtx_dir_to_h5(mtx_dir: Path, out: Path, genome: str) -> None:
    """MTX directory -> Cell Ranger v3 h5 through the independent reader (not CREST's)."""
    X, barcodes, ids, names = read_mtx_dir_independent(mtx_dir)
    if X.data.size and (not np.all(np.mod(X.data, 1) == 0) or X.data.min() < 0):
        raise RuntimeError(f"{mtx_dir}: the matrix does not hold non-negative integer counts")
    write_10x_h5(out, X.indptr, X.indices, X.data, X.shape[1], barcodes, ids, names, genome)


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
            ch = (max(1, min(1 << 20, nnz)),)
            dd = m.create_dataset("data", (nnz,), np.int32, compression="gzip", compression_opts=1, chunks=ch)
            di = m.create_dataset("indices", (nnz,), np.int32, compression="gzip", compression_opts=1, chunks=ch)
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


# --------------------------------------------------------------------------- generic h5ad
def _h5_str(a) -> list:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def _obs_column(node, max_levels: int):
    """A string / categorical obs column (either anndata encoding) as a NumPy object array,
    or None if it is numeric or has more than ``max_levels`` distinct values."""
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        cats = np.array(_h5_str(node["categories"][:]) + [None], dtype=object)
        if len(cats) - 1 > max_levels:
            return None
        codes = node["codes"][:].astype(np.int64)
        codes[codes < 0] = len(cats) - 1
        return cats[codes]
    if isinstance(node, h5py.Dataset) and node.dtype.kind in ("S", "O", "U") and node.ndim == 1:
        v = np.array(_h5_str(node[:]), dtype=object)
        return v if len(np.unique(v)) <= max_levels else None
    return None


def read_h5ad_obs(group, max_levels: int = 20_000) -> dict:
    """obs index + every string/categorical column with at most ``max_levels`` levels
    (NumPy object arrays; vectorised, so 10M cells take seconds)."""
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    cols = {"barcode": np.array(_h5_str(group[key][:]), dtype=object)}
    order = group.attrs.get("column-order", list(group.keys()))
    for c in [x.decode() if isinstance(x, bytes) else str(x) for x in order]:
        if c == key or c not in group:
            continue
        v = _obs_column(group[c], max_levels)
        if v is not None:
            cols[c] = v
    return cols


def _level_counts(v) -> dict:
    u, n = np.unique(np.asarray(v, dtype=str), return_counts=True)
    return dict(zip(u.tolist(), n.tolist()))


def _counts_node(f):
    """(path, group) of the raw-count matrix: layers/counts, raw/X, then X; the first CSR
    matrix whose first million values are non-negative integers."""
    for path in ("layers/counts", "layers/raw_counts", "raw/X", "X"):
        if path not in f or not isinstance(f[path], h5py.Group):
            continue
        g = f[path]
        enc = g.attrs.get("encoding-type", "csr_matrix")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        nnz = g["data"].shape[0]
        k = min(1 << 20, nnz)
        sample = np.concatenate([g["data"][o:o + k] for o in sorted({0, max(0, nnz // 2 - k // 2), max(0, nnz - k)})])
        if len(sample) and np.all(sample >= 0) and np.all(np.mod(sample, 1) == 0):
            if not enc.startswith("csr"):
                raise RuntimeError(f"raw counts at {path} are stored as {enc}; CSR (cells x genes) is required")
            return path, g
        print(f"  [skip]     {path}: not raw counts (non-integer or negative values)")
    raise RuntimeError("no raw-count matrix found (looked at layers/counts, layers/raw_counts, raw/X, X)")


def inspect_h5ad(path: Path) -> None:
    """Print the layout of an .h5ad: matrices, obs columns and their levels."""
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset) and not name.startswith(("obs/", "var/")):
                print(f"  {name:45s} {str(obj.shape):>18s} {obj.dtype}")
            elif isinstance(obj, h5py.Group) and "encoding-type" in obj.attrs and not name.startswith(("obs/", "var/")):
                shape = obj.attrs.get("shape")
                print(f"  {name + '/':45s} {obj.attrs['encoding-type']} shape={None if shape is None else tuple(int(x) for x in shape)}")
        f.visititems(show)
        try:
            print(f"counts: {_counts_node(f)[0]}")
        except RuntimeError as e:
            print(f"counts: {e}")
        obs = read_h5ad_obs(f["obs"])
        print(f"obs: {len(obs['barcode'])} cells")
        for c, v in obs.items():
            if c == "barcode":
                continue
            levels = _level_counts(v)
            top = sorted(levels.items(), key=lambda kv: -kv[1])[:6]
            print(f"  {c:30s} {len(levels):6d} levels  e.g. " + ", ".join(f"{k} ({n})" for k, n in top))


def _pick(cols, names):
    low = {c.lower(): c for c in cols}
    return next((low[n.lower()] for n in names if n.lower() in low), None)


TREATED_PREFERENCE = ["IFN-beta", "IFN-gamma", "TNF-alpha", "IL-6", "stim", "stimulated", "treated"]


def _resolve_meta(obs: dict, n_genes: int) -> dict:
    """Batch / sample / condition / cell-type columns, a treated-vs-control contrast, and the
    QC threshold. A matrix restricted to few genes gets ``min_genes`` scaled to the same
    fraction of genes as the usual 200 of ~20,000 (both tools use the same value)."""
    import re
    cols = [c for c in obs if c != "barcode"]
    donor = _pick(cols, ["donor", "donor_id", "Donor", "individual", "patient"])
    cond = _pick(cols, ["cytokine", "cytokines", "treatment", "condition", "stim", "perturbation"])
    ctype = _pick(cols, ["cell_type", "celltype", "cell_type_annotation", "cell_type_label", "annotation"])
    meta = {"detected_columns": {"donor": donor, "condition": cond, "celltype": ctype}}
    if n_genes < 10_000:
        meta["min_genes"] = max(10, int(round(200 * n_genes / 20_000)))
    if donor:
        meta["batch"] = donor
    if ctype:
        meta["celltype"] = ctype
    if donor and cond:
        counts = _level_counts(obs[cond])
        ctrl = next((k for k in counts if k and re.fullmatch(r"(?i)pbs|control|ctrl|untreated|none|unstim(ulated)?", k)), None)
        if ctrl:
            others = [k for k in counts if k not in (ctrl, "None")]
            treated = next((k for k in TREATED_PREFERENCE if k in others), None) or max(others, key=counts.get)
            meta.update({"sample": [donor, cond], "condition": cond, "contrast": [cond, treated, ctrl],
                         "pb_levels_only": True})
    return meta


def _build_generic_h5ad(src: Path, out: Path) -> None:
    """Any .h5ad with CSR raw counts -> 10x h5 (streamed) + obs sidecar + meta sidecar."""
    import polars as pl
    with h5py.File(src, "r") as f:
        path, X = _counts_node(f)
        print(f"  [counts]   {path}")
        var_group = f["raw/var"] if path == "raw/X" and "raw/var" in f else f["var"]
        vkey = var_group.attrs.get("_index", "_index")
        vkey = vkey.decode() if isinstance(vkey, bytes) else vkey
        names = _h5_str(var_group[vkey][:])
        gid = next((c for c in ("gene_ids", "gene_id", "ensembl_id", "gene_ids-0") if c in var_group), None)
        idv = _obs_column(var_group[gid], 10 ** 9) if gid else None
        ids = list(idv) if idv is not None else names
        obs = read_h5ad_obs(f["obs"])
        shape = X.attrs.get("shape")
        n_cells, n_genes = int(shape[0]), int(shape[1])
        if len(names) != n_genes or len(obs["barcode"]) != n_cells:
            raise RuntimeError(f"shape {n_cells} x {n_genes} does not match obs/var ({len(obs['barcode'])}/{len(names)})")
        indptr = X["indptr"][:].astype(np.int64)
        tmp = out.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as g:
            m = g.create_group("matrix")
            nnz = int(indptr[-1])
            ch = (max(1, min(1 << 20, nnz)),)
            dd = m.create_dataset("data", (nnz,), np.int32, compression="gzip", compression_opts=1, chunks=ch)
            di = m.create_dataset("indices", (nnz,), np.int32, compression="gzip", compression_opts=1, chunks=ch)
            step = 1 << 26
            for lo in range(0, nnz, step):
                hi = min(lo + step, nnz)
                vals = X["data"][lo:hi]
                if not np.all(np.mod(vals, 1) == 0):
                    raise RuntimeError(f"{src.name}: non-integer values in {path}")
                dd[lo:hi] = vals.astype(np.int32)
                di[lo:hi] = X["indices"][lo:hi].astype(np.int32)
                print(f"             {hi / nnz:6.1%} of {nnz:,} non-zeros", flush=True)
            m.create_dataset("indptr", data=indptr)
            m.create_dataset("shape", data=np.array([n_genes, n_cells], np.int32))
            m.create_dataset("barcodes", data=np.asarray(obs["barcode"], dtype=str).astype(np.bytes_))
            ft = m.create_group("features")
            ft.create_dataset("id", data=_enc(ids))
            ft.create_dataset("name", data=_enc(names))
            ft.create_dataset("feature_type", data=_enc(["Gene Expression"] * n_genes))
            ft.create_dataset("genome", data=_enc(["GRCh38"] * n_genes))
        tmp.rename(out)
    meta = _resolve_meta(obs, n_genes)
    keep = ["barcode"] + [c for c in dict.fromkeys([meta.get("batch"), meta.get("condition"), meta.get("celltype")]) if c]
    pl.DataFrame({c: pl.Series(c, np.asarray(obs[c], dtype=str)) for c in keep}).write_parquet(out.with_suffix(".obs.parquet"))
    meta.update({"source": src.name, "counts_path": path, "cells": n_cells, "genes": n_genes})
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  [meta]     {json.dumps({k: meta.get(k) for k in ('batch', 'sample', 'condition', 'contrast', 'celltype', 'min_genes')})}")


def subset_rows(n_total: int, n: int, seed: int = 0) -> np.ndarray:
    """Sorted row indices of the seeded random ``n``-cell subset of ``n_total`` cells. The subsets of one
    seed are nested: they are the first ``n`` entries of one random permutation."""
    return np.sort(np.random.default_rng(seed).permutation(n_total)[: min(n, n_total)])


def subset_h5(src: Path, out: Path, n: int, seed: int = 0) -> None:
    """Random ``n``-cell subset of a 10x h5 (+ its obs/meta sidecars), streamed in row windows."""
    import polars as pl
    with h5py.File(src, "r") as f:
        m = f["matrix"]
        indptr = m["indptr"][:].astype(np.int64)
        N = len(indptr) - 1
        rows = subset_rows(N, n, seed)
        lens = indptr[rows + 1] - indptr[rows]
        new_ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        tmp = out.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as g:
            o = g.create_group("matrix")
            total = int(new_ptr[-1])
            ch = (max(1, min(1 << 20, total)),)
            dd = o.create_dataset("data", (total,), np.int32, compression="gzip", compression_opts=1, chunks=ch)
            di = o.create_dataset("indices", (total,), np.int32, compression="gzip", compression_opts=1, chunks=ch)
            step, r0, pos = 1 << 26, 0, 0
            while r0 < N:  # windows of whole rows, ~64M non-zeros each
                r1 = int(np.searchsorted(indptr, indptr[r0] + step, side="right")) - 1
                r1 = min(max(r1, r0 + 1), N)
                lo, hi = int(indptr[r0]), int(indptr[r1])
                sel = rows[(rows >= r0) & (rows < r1)]
                if len(sel):
                    data, idx = m["data"][lo:hi], m["indices"][lo:hi]
                    st, ln = indptr[sel] - lo, indptr[sel + 1] - indptr[sel]
                    take = np.repeat(st - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(int(ln.sum()))
                    dd[pos:pos + len(take)] = data[take]
                    di[pos:pos + len(take)] = idx[take]
                    pos += len(take)
                print(f"             {r1 / N:6.1%} of {N:,} cells", flush=True)
                r0 = r1
            o.create_dataset("indptr", data=new_ptr)
            o.create_dataset("shape", data=np.array([int(m["shape"][0]), len(rows)], np.int32))
            o.create_dataset("barcodes", data=m["barcodes"][:][rows])
            m.copy("features", o)
        tmp.rename(out)
    side = src.with_suffix(".obs.parquet")
    if side.exists():
        pl.read_parquet(side)[rows.tolist()].write_parquet(out.with_suffix(".obs.parquet"))
    meta = json.loads(src.with_suffix(".meta.json").read_text()) if src.with_suffix(".meta.json").exists() else {}
    meta.update({"cells": int(len(rows)), "subset_of": src.stem, "subset_seed": seed, "kind": "real-subsample"})
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))


def _find_local(spec: dict, data_dir: Path, dest: Path) -> None:
    """Link a manually downloaded copy (any of spec['local_names']) into the source folder."""
    if dest.exists():
        return
    for n in spec.get("local_names", []):
        for d in (data_dir, dest.parent, Path.cwd()):
            p = d / n
            if p.exists():
                size = spec["files"][dest.name][1]
                if size and p.stat().st_size != size:
                    print(f"  [note]     {p} is {p.stat().st_size:,} bytes, figshare's copy {size:,}; using it anyway")
                dest.symlink_to(p.resolve())
                print(f"  [local]    {dest.name} -> {p}")
                return


def build(name: str, data_dir: Path) -> Path:
    """Download (if needed) and convert; returns the path of ``<name>.h5``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{name}.h5"
    if subset_spec(name):
        if not out.exists():
            parent, n = subset_spec(name)
            src = build(parent, data_dir)
            print(f"  [subset]   {out.name}: {n:,} random cells of {src.name} (seed {SUBSAMPLE_SEED})")
            subset_h5(src, out, n, seed=SUBSAMPLE_SEED)
        return out
    if name in SYNTHETIC:
        if not out.exists():
            src = build(SYNTH_SOURCE.get(name, "pbmc68k"), data_dir)
            from make_dataset import synthesize
            print(f"  [build]    {out.name} from {src.name}")
            synthesize(src, SYNTHETIC[name], out, seed=0)
            out.with_suffix(".meta.json").write_text(json.dumps(
                {"kind": "synthetic", "synthetic_from": src.stem, "seed": 0, "cells": SYNTHETIC[name],
                 "note": "supplement: built with CREST's own normalisation, PCA and kNN"}, indent=1))
        return out
    spec = REGISTRY[name]
    src_dir = data_dir / "src" / name
    src_dir.mkdir(parents=True, exist_ok=True)
    for fname, (url, size) in spec["files"].items():
        _find_local(spec, data_dir, src_dir / fname)
        if (src_dir / fname).is_symlink():
            continue
        try:
            download(url, src_dir / fname, size)
        except RuntimeError:
            if spec.get("local_names"):
                raise RuntimeError(f"could not download {name}. Download {url} in a browser and put it in "
                                   f"{data_dir} as {spec['local_names'][0]}") from None
            raise
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
    elif name == "parse_pbmc":
        _build_generic_h5ad(src_dir / "parse_pbmc.h5ad", out)
    else:
        raise KeyError(name)
    return out


# --------------------------------------------------------------------------- verification
def read_h5_counts(path: Path):
    """A Cell Ranger v3 h5 read with h5py only (no scanpy, no CREST): ``(X, barcodes, gene_ids, gene_names)``,
    X CSR cells x genes with sorted indices."""
    import scipy.sparse as sp
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        n_genes, n_cells = (int(x) for x in m["shape"][:])
        X = sp.csr_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=(n_cells, n_genes))
        X.sort_indices()
        barcodes = _h5_str(m["barcodes"][:])
        ids, names = _h5_str(m["features"]["id"][:]), _h5_str(m["features"]["name"][:])
    return X, barcodes, ids, names


def compare_counts(a, b, names=("a", "b")) -> list[str]:
    """Count-for-count comparison of two ``(X, barcodes, gene_ids_or_names)`` triples (``None`` skips a
    name list). Equal means: same shape, equal CSR arrays (indptr, indices, values), equal gene names in
    the same order, equal barcodes in the same order. Returns the mismatches (empty list = equal)."""
    import scipy.sparse as sp
    out = []
    (Xa, oa, va), (Xb, ob, vb) = a, b
    if Xa.shape != Xb.shape:
        return [f"shape {names[0]} {Xa.shape} != {names[1]} {Xb.shape}"]
    Xa, Xb = sp.csr_matrix(Xa, copy=True), sp.csr_matrix(Xb, copy=True)
    for X in (Xa, Xb):
        X.sum_duplicates()
        X.sort_indices()
    if not np.array_equal(Xa.indptr, Xb.indptr):
        bad = int(np.argmax(Xa.indptr != Xb.indptr))
        out.append(f"row pointers differ from cell {bad} on (per-cell non-zero counts)")
    elif not np.array_equal(Xa.indices, Xb.indices):
        out.append(f"column indices differ at non-zero {int(np.argmax(Xa.indices != Xb.indices))} (genes mis-ordered?)")
    elif not np.array_equal(Xa.data, Xb.data):
        out.append(f"values differ at non-zero {int(np.argmax(Xa.data != Xb.data))}")
    for what, x, y in (("barcodes", oa, ob), ("gene names", va, vb)):
        if x is None or y is None:
            continue
        x, y = list(map(str, x)), list(map(str, y))
        if x != y:
            k = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            out.append(f"{what} differ at position {k}: {x[k] if k < len(x) else None!r} vs {y[k] if k < len(y) else None!r}")
    return out


def _check(name: str, mism: list[str]) -> dict:
    return {"name": name, "ok": not mism, "detail": "; ".join(mism) if mism else "equal"}


def _skipped(name: str, why: str) -> dict:
    return {"name": name, "ok": True, "skipped": True, "detail": why}


def _scanpy_csr(adata):
    X = adata.X.tocsr().astype(np.float64)
    X.sort_indices()
    return X, list(adata.obs_names), list(adata.var_names)


def verify_reader_equality(h5: Path) -> list[dict]:
    """The converted h5 read by scanpy and by CREST against h5py's raw read, count for count."""
    X0, bc0, ids0, names0 = read_h5_counts(h5)
    ref = (X0.astype(np.float64), bc0, names0)
    checks = []
    try:
        import scanpy as sc
        a = sc.read_10x_h5(h5)
        checks.append(_check("scanpy.read_10x_h5 == h5py read", compare_counts(_scanpy_csr(a), ref, ("scanpy", "h5py"))))
    except ImportError:
        checks.append(_skipped("scanpy.read_10x_h5 == h5py read", "scanpy is not installed"))
    try:
        import crest
        b = crest.read_10x_h5(h5)
        Xc = b.to_scipy(transform=False).tocsr().astype(np.float64)
        checks.append(_check("crest.read_10x_h5 == h5py read",
                             compare_counts((Xc, b.obs["barcode"].to_list(), b.var["gene_name"].to_list()), ref, ("crest", "h5py"))))
    except ImportError:
        checks.append(_skipped("crest.read_10x_h5 == h5py read", "crest is not installed"))
    return checks


def verify_mtx_dataset(mtx_dir: Path, h5: Path) -> list[dict]:
    """Source MTX files vs the converted h5: the independent read (scipy), scanpy's MTX reader and CREST's
    MTX reader, all against the h5 as h5py reads it."""
    X0, bc0, ids0, names0 = read_h5_counts(h5)
    conv = (X0.astype(np.float64), bc0, names0)
    Xi, bci, idsi, namesi = read_mtx_dir_independent(mtx_dir)
    checks = [_check("independent MTX read (scipy) == converted h5", compare_counts((Xi.astype(np.float64), bci, namesi), conv, ("scipy mtx", "h5")))]
    try:
        import scanpy as sc
        a = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=False)
        checks.append(_check("scanpy.read_10x_mtx == converted h5", compare_counts(_scanpy_csr(a), conv, ("scanpy mtx", "h5"))))
    except ImportError:
        checks.append(_skipped("scanpy.read_10x_mtx == converted h5", "scanpy is not installed"))
    try:
        import crest
        b = crest.read_10x_mtx(mtx_dir)
        Xc = b.to_scipy(transform=False).tocsr().astype(np.float64)
        checks.append(_check("crest.read_10x_mtx == independent MTX read",
                             compare_counts((Xc, b.obs["barcode"].to_list(), b.var["gene_name"].to_list()),
                                            (Xi.astype(np.float64), bci, namesi), ("crest mtx", "scipy mtx"))))
    except ImportError:
        checks.append(_skipped("crest.read_10x_mtx == independent MTX read", "crest is not installed"))
    return checks


def _h5ad_counts_source(f):
    """(indptr, indices ds, data ds, var index, obs index) of the raw-count matrix of an open .h5ad."""
    path, g = _counts_node(f)
    var = f["raw/var"] if path == "raw/X" and "raw/var" in f else f["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    okey = f["obs"].attrs.get("_index", "_index")
    okey = okey.decode() if isinstance(okey, bytes) else okey
    return g["indptr"][:].astype(np.int64), g["indices"], g["data"], _h5_str(var[key][:]), _h5_str(f["obs"][okey][:])


def _stream_equal(src_idx, src_dat, dst_idx, dst_dat, window: int = 1 << 26) -> str | None:
    """Compare two equally long non-zero arrays in windows; the first mismatch as text, else None."""
    n = src_idx.shape[0]
    if n != dst_idx.shape[0]:
        return f"{n} vs {dst_idx.shape[0]} non-zeros"
    for lo in range(0, n, window):
        hi = min(lo + window, n)
        if not np.array_equal(src_idx[lo:hi].astype(np.int64), dst_idx[lo:hi].astype(np.int64)):
            return f"column indices differ in non-zeros {lo}-{hi}"
        if not np.array_equal(src_dat[lo:hi].astype(np.int64), dst_dat[lo:hi].astype(np.int64)):
            return f"values differ in non-zeros {lo}-{hi}"
    return None


def verify_h5ad_dataset(h5ad: Path, h5: Path) -> list[dict]:
    """An .h5ad source (CELLxGENE, Parse) vs the converted h5: row pointers, column indices and values of
    every non-zero (streamed in windows), barcodes and gene identifiers."""
    with h5py.File(h5ad, "r") as f, h5py.File(h5, "r") as g:
        indptr, idx, dat, var, obs = _h5ad_counts_source(f)
        m = g["matrix"]
        mism = []
        if not np.array_equal(indptr, m["indptr"][:].astype(np.int64)):
            mism.append("row pointers differ")
        else:
            bad = _stream_equal(idx, dat, m["indices"], m["data"])
            if bad:
                mism.append(bad)
        if obs != _h5_str(m["barcodes"][:]):
            mism.append("barcodes differ from the source obs index")
        ids, names = _h5_str(m["features"]["id"][:]), _h5_str(m["features"]["name"][:])
        if var != ids and var != names:
            mism.append("gene identifiers differ from the source var index")
    return [_check("source h5ad counts == converted h5 (all non-zeros)", mism)]


def verify_subset(parent_h5: Path, h5: Path, n: int, seed: int, n_check: int = 2000) -> list[dict]:
    """A real subsample vs its parent: the rows are those ``subset_rows`` picks, and ``n_check`` random
    cells (plus the first and last) match the parent cell for cell (count for count, genes in order)."""
    with h5py.File(parent_h5, "r") as pf, h5py.File(h5, "r") as sf:
        pm, sm = pf["matrix"], sf["matrix"]
        pptr, sptr = pm["indptr"][:].astype(np.int64), sm["indptr"][:].astype(np.int64)
        rows = subset_rows(len(pptr) - 1, n, seed)
        mism = []
        if len(sptr) - 1 != len(rows):
            return [_check("subsample == parent rows", [f"{len(sptr) - 1} cells, expected {len(rows)}"])]
        pick = np.unique(np.r_[0, len(rows) - 1, np.random.default_rng(1).choice(len(rows), min(n_check, len(rows)), replace=False)])
        pbc, sbc = pm["barcodes"][:], sm["barcodes"][:]
        for i in pick:
            r = rows[i]
            a, b = slice(pptr[r], pptr[r + 1]), slice(sptr[i], sptr[i + 1])
            if (not np.array_equal(pm["indices"][a], sm["indices"][b]) or not np.array_equal(pm["data"][a], sm["data"][b])
                    or pbc[r] != sbc[i]):
                mism.append(f"cell {i} (parent row {r}) differs")
                break
        if _h5_str(pm["features"]["name"][:]) != _h5_str(sm["features"]["name"][:]):
            mism.append("gene names differ from the parent")
    return [_check(f"subsample == parent rows ({len(pick)} cells compared, seed {seed})", mism)]


def verify_dataset(name: str, data_dir: Path) -> dict:
    """Verify one built dataset; ``ok`` is False when any check fails. Synthetic sets have no source to
    verify against (they are built from another file): only the readers are compared."""
    data_dir = Path(data_dir)
    h5 = data_dir / f"{name}.h5"
    res = {"dataset": name, "kind": dataset_kind(name), "h5": h5.name, "checks": []}
    if not h5.exists():
        res["checks"].append(_check("converted file exists", [f"{h5} is missing"]))
    else:
        src_dir = data_dir / "src" / name
        spec = subset_spec(name)
        try:
            if name in ("pbmc3k", "pbmc68k"):
                mtx = next((src_dir / "extracted").rglob("matrix.mtx*")).parent
                res["checks"] += verify_mtx_dataset(mtx, h5)
            elif name in ("pbmc10k", "neurons_1m"):
                same = _sha256(src_dir / f"{name}.h5") == _sha256(h5)
                res["checks"].append(_check("converted file == downloaded file (sha256; no conversion)", [] if same else ["differ"]))
            elif name in ("covid_stephenson", "parse_pbmc"):
                res["checks"] += verify_h5ad_dataset(src_dir / f"{name}.h5ad", h5)
            elif spec:
                side = h5.with_suffix(".meta.json")
                seed = json.loads(side.read_text()).get("subset_seed", SUBSAMPLE_SEED) if side.exists() else SUBSAMPLE_SEED
                res["checks"] += verify_subset(data_dir / f"{spec[0]}.h5", h5, spec[1], seed)
            elif name == "kang":
                res["checks"].append(_skipped("source vs converted", "built from GEO files with scipy; reader checks below"))
            res["checks"] += verify_reader_equality(h5)
        except Exception as e:  # noqa: BLE001
            res["checks"].append(_check("verification ran", [f"{type(e).__name__}: {e}"]))
    res["ok"] = all(c["ok"] for c in res["checks"])
    return res


# --------------------------------------------------------------------------- provenance manifest
def build_manifest(data_dir: Path, names: list[str]) -> dict:
    """Provenance of every dataset: source URLs, the SHA-256 of the source files and of the converted file,
    sizes, the conversion path, the verification result and the licence. The licence is NOT filled in
    here: it is stated by the data provider, so it stays ``null`` until it has been checked at the source."""
    data_dir = Path(data_dir)
    known = json.loads((data_dir / "checksums.json").read_text()) if (data_dir / "checksums.json").exists() else {}
    vf = data_dir / "verification.json"
    ver = {v["dataset"]: v for v in json.loads(vf.read_text())} if vf.exists() else {}
    out = {}
    for n in names:
        spec, h5 = REGISTRY.get(n), data_dir / f"{n}.h5"
        sub = subset_spec(n)
        meta_f = h5.with_suffix(".meta.json")
        rec = {"kind": dataset_kind(n), "converted_file": h5.name, "converted_sha256": _sha256(h5) if h5.exists() else None,
               "converted_bytes": h5.stat().st_size if h5.exists() else None, "licence": None,
               "licence_note": "not recorded: confirm at the data provider before the run",
               "verification": ver.get(n, {}).get("ok"),
               "conversion": ("scipy.io.mmread (independent of CREST)" if n in ("pbmc3k", "pbmc68k") else
                              "seeded random subset of the parent" if sub else
                              "CREST synthesize (supplement)" if n in SYNTHETIC else
                              "h5ad -> h5 streaming copy" if n in ("covid_stephenson", "parse_pbmc") else
                              "none (downloaded h5) or scipy (GEO files)")}
        if spec:
            rec.update(source_urls={f: u for f, (u, _) in spec["files"].items()}, citation=spec.get("citation"),
                       source_sha256={f: known.get(f) for f in spec["files"]})
            rec["source_retrieved_utc"] = {f: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((data_dir / "src" / n / f).stat().st_mtime))
                                           for f in spec["files"] if (data_dir / "src" / n / f).exists()}
        if sub:
            rec.update(parent=sub[0], cells_requested=sub[1],
                       seed=json.loads(meta_f.read_text()).get("subset_seed", SUBSAMPLE_SEED) if meta_f.exists() else SUBSAMPLE_SEED)
        if meta_f.exists():
            rec["meta"] = json.loads(meta_f.read_text())
        out[n] = rec
    return out


def main():
    global SUBSAMPLE_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="bench_data")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--tier", choices=["smoke", "quick", "standard", "full"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", metavar="H5AD", help="print the layout of an .h5ad and exit")
    ap.add_argument("--verify", action="store_true",
                    help="check the converted files count for count against independent readers; non-zero exit on a mismatch")
    ap.add_argument("--manifest", action="store_true", help="write dataset_manifest.json (sources, sha256, licence field)")
    ap.add_argument("--subsample", metavar="PARENT", help="build a real-subsample series of PARENT (with --sizes)")
    ap.add_argument("--sizes", nargs="*", type=int, default=[], help="cell counts of the --subsample series")
    ap.add_argument("--seed", type=int, default=SUBSAMPLE_SEED, help="seed of the --subsample series (default 0)")
    a = ap.parse_args()
    if a.inspect:
        inspect_h5ad(Path(a.inspect))
        return
    if a.subsample:
        SUBSAMPLE_SEED = a.seed
        for n in a.sizes:
            SUBSETS[protocol.subset_name(a.subsample, n)] = (a.subsample, n)
        a.datasets = (a.datasets or []) + [protocol.subset_name(a.subsample, n) for n in a.sizes]
    if a.list:
        for k, v in REGISTRY.items():
            print(f"{k:18s} {v['cells']:>9,d} cells  tier={v['tier']:8s} {v['description']}")
        for k, (parent, n) in SUBSETS.items():
            print(f"{k:18s} {n:>9,d} cells  tier={SUBSET_TIER[k]:8s} seeded random subset of {parent} (real)")
        for k, n in SYNTHETIC.items():
            print(f"{k:18s} {n:>9,d} cells  tier={SYNTH_TIER[k]:8s} synthetic SUPPLEMENT, interpolated from "
                  f"{SYNTH_SOURCE.get(k, 'pbmc68k')} with CREST's own PCA / kNN")
        return
    names = a.datasets or tier_datasets(a.tier or "standard")
    data_dir = Path(a.data_dir)
    if not (a.verify or a.manifest) or a.subsample or a.tier:
        for n in names:
            print(f"== {n}")
            print(f"   -> {build(n, data_dir)}")
    failed = False
    if a.verify:
        results = []
        for n in names:
            r = verify_dataset(n, data_dir)
            results.append(r)
            print(f"== verify {n}: {'ok' if r['ok'] else 'MISMATCH'}")
            for c in r["checks"]:
                print(f"   [{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
        (data_dir / "verification.json").write_text(json.dumps(results, indent=1))
        failed = not all(r["ok"] for r in results)
    if a.manifest:
        (data_dir / "dataset_manifest.json").write_text(json.dumps(build_manifest(data_dir, names), indent=1))
        print(f"   -> {data_dir / 'dataset_manifest.json'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
