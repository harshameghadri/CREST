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
    "parse_pbmc": {
        # Parse Biosciences "10 million human PBMCs" (12 donors x 90 cytokines + PBS), the
        # processed subset distributed with CellFlow. Figshare may refuse scripted downloads
        # (bot check); then download it in a browser and put it in the data directory under
        # any of `local_names`, it is picked up from there.
        "files": {"parse_pbmc.h5ad": ("https://figshare.com/ndownloader/files/53372768", 13101592422)},
        "local_names": ["Parse_1M_adata_for_cellflow_datasets_with_embeddings.h5ad",
                        "adata_for_cellflow_datasets_with_embeddings.h5ad"],
        "md5": "d89559875a6c75ab37e0adf84c6e177d",
        "cells": 1_000_000, "tier": "full", "chemistry": "Parse Evercode WT v3 (split-pool)",
        "description": "Parse 10M PBMC cytokine atlas, ~1M-cell CellFlow subset (figshare 28589774)",
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


# --------------------------------------------------------------------------- generic h5ad
def _h5_str(a) -> list:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def _obs_column(node) -> list | None:
    """A string / categorical obs column in either anndata encoding, or None."""
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        cats = _h5_str(node["categories"][:])
        codes = node["codes"][:]
        return [cats[c] if c >= 0 else None for c in codes]
    if isinstance(node, h5py.Dataset) and node.dtype.kind in ("S", "O", "U") and node.ndim == 1:
        return _h5_str(node[:])
    return None


def read_h5ad_obs(group, max_levels: int = 20_000) -> dict:
    """obs index + every string/categorical column with at most ``max_levels`` levels."""
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    cols = {"barcode": _h5_str(group[key][:])}
    order = group.attrs.get("column-order", list(group.keys()))
    for c in [x.decode() if isinstance(x, bytes) else str(x) for x in order]:
        if c == key or c not in group:
            continue
        v = _obs_column(group[c])
        if v is not None and len(set(v)) <= max_levels:
            cols[c] = v
    return cols


def _counts_node(f):
    """(path, group) of the raw-count matrix: layers/counts, raw/X, then X; the first CSR
    matrix whose first million values are non-negative integers."""
    for path in ("layers/counts", "layers/raw_counts", "raw/X", "X"):
        if path not in f or not isinstance(f[path], h5py.Group):
            continue
        g = f[path]
        enc = g.attrs.get("encoding-type", "csr_matrix")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        sample = g["data"][: 1 << 20]
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
                print(f"  {name + '/':45s} {obj.attrs['encoding-type']} shape={None if shape is None else tuple(shape)}")
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
            levels = {}
            for x in v:
                levels[x] = levels.get(x, 0) + 1
            top = sorted(levels.items(), key=lambda kv: -kv[1])[:6]
            print(f"  {c:30s} {len(levels):6d} levels  e.g. " + ", ".join(f"{k} ({n})" for k, n in top))


def _pick(cols, names):
    low = {c.lower(): c for c in cols}
    return next((low[n.lower()] for n in names if n.lower() in low), None)


def _resolve_meta(obs: dict) -> dict:
    """Batch / sample / condition / cell-type columns and a treated-vs-control contrast."""
    import re
    cols = [c for c in obs if c != "barcode"]
    donor = _pick(cols, ["donor", "donor_id", "Donor", "individual", "patient"])
    cond = _pick(cols, ["cytokine", "cytokines", "treatment", "condition", "stim", "perturbation"])
    ctype = _pick(cols, ["cell_type", "celltype", "cell_type_annotation", "cell_type_label", "annotation"])
    meta = {"detected_columns": {"donor": donor, "condition": cond, "celltype": ctype}}
    if donor:
        meta["batch"] = donor
    if ctype:
        meta["celltype"] = ctype
    if donor and cond:
        counts = {}
        for x in obs[cond]:
            counts[x] = counts.get(x, 0) + 1
        ctrl = next((k for k in counts if k and re.fullmatch(r"(?i)pbs|control|ctrl|untreated|none|unstim(ulated)?", k)), None)
        if ctrl:
            treated = max((k for k in counts if k not in (ctrl, None)), key=counts.get)
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
        ids = (_obs_column(var_group[gid]) if gid else None) or names
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
            dd = m.create_dataset("data", (nnz,), np.int32, compression="gzip", compression_opts=1, chunks=(1 << 20,))
            di = m.create_dataset("indices", (nnz,), np.int32, compression="gzip", compression_opts=1, chunks=(1 << 20,))
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
            m.create_dataset("barcodes", data=_enc(obs["barcode"]))
            ft = m.create_group("features")
            ft.create_dataset("id", data=_enc(ids))
            ft.create_dataset("name", data=_enc(names))
            ft.create_dataset("feature_type", data=_enc(["Gene Expression"] * n_genes))
            ft.create_dataset("genome", data=_enc(["GRCh38"] * n_genes))
        tmp.rename(out)
    meta = _resolve_meta(obs)
    keep = ["barcode"] + [c for c in dict.fromkeys([meta.get("batch"), meta.get("condition"), meta.get("celltype")]) if c]
    pl.DataFrame({c: obs[c] for c in keep}).write_parquet(out.with_suffix(".obs.parquet"))
    meta.update({"source": src.name, "counts_path": path, "cells": n_cells, "genes": n_genes})
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  [meta]     {json.dumps({k: meta.get(k) for k in ('batch', 'sample', 'condition', 'contrast', 'celltype')})}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="bench_data")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--tier", choices=["quick", "standard", "full"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", metavar="H5AD", help="print the layout of an .h5ad and exit")
    a = ap.parse_args()
    if a.inspect:
        inspect_h5ad(Path(a.inspect))
        return
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
