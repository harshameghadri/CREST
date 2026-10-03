"""Readers and writers. Only ``h5py`` (for .h5/.h5ad) and ``polars`` are needed."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl

from .core import BioFrame, read_parquet

__all__ = ["read_10x_h5", "read_h5ad", "read_10x_mtx", "read_parquet", "convert_h5_to_parquet_stream"]

_READ_BLOCK = 1 << 24


def _decode(a) -> list:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a[:]]


def _read_array(ds, dtype) -> np.ndarray:
    """Read an HDF5 dataset into a new array of ``dtype`` block by block (no full-size temporary)."""
    out = np.empty(ds.shape[0], dtype=dtype)
    for lo in range(0, ds.shape[0], _READ_BLOCK):
        hi = min(lo + _READ_BLOCK, ds.shape[0])
        out[lo:hi] = ds[lo:hi]
    return out


def _open_10x(f):
    """Return (group, barcodes, var DataFrame) for Cell Ranger v3+ or v2 layouts."""
    if "matrix" in f:
        m = f["matrix"]
        feats = m["features"]
        var = pl.DataFrame({
            "gene_ids": _decode(feats["id"]),
            "gene_name": _decode(feats["name"]),
            "feature_types": _decode(feats["feature_type"]) if "feature_type" in feats else ["Gene Expression"] * feats["id"].shape[0],
        })
        return m, _decode(m["barcodes"]), var
    genome = next(iter(f.keys()))  # v2: one group per genome
    m = f[genome]
    var = pl.DataFrame({
        "gene_ids": _decode(m["genes"]),
        "gene_name": _decode(m["gene_names"]),
        "feature_types": ["Gene Expression"] * m["genes"].shape[0],
    })
    return m, _decode(m["barcodes"]), var


def read_10x_h5(path: Union[str, Path], gex_only: bool = True, backed: Optional[Union[str, Path]] = None,
                nnz_per_part: int = 1 << 23) -> BioFrame:
    """Read a Cell Ranger ``filtered_feature_bc_matrix.h5``.

    With ``backed=<dir>`` the matrix is streamed to a Parquet dataset in that
    directory (one cell range at a time) and returned as an out-of-core
    BioFrame; otherwise it is loaded as compact in-memory CSR.
    """
    import h5py

    with h5py.File(path, "r") as f:
        m, barcodes, var = _open_10x(f)
        n_genes = var.height
        obs = pl.DataFrame({"barcode": barcodes})
        if backed is None:
            indptr = m["indptr"][:].astype(np.int64)
            bf = BioFrame.from_csr(indptr, _read_array(m["indices"], np.uint32), _read_array(m["data"], np.float32),
                                   n_genes, obs, var)
        else:
            out = Path(backed)
            out.mkdir(parents=True, exist_ok=True)
            for old in out.glob("part-*.parquet"):
                old.unlink()
            indptr = m["indptr"][:].astype(np.int64)
            n = len(indptr) - 1
            lo, part = 0, 0
            while lo < n:
                # whole cells, ~nnz_per_part non-zeros per part (bounded memory for any cell depth)
                hi = int(np.searchsorted(indptr, indptr[lo] + nnz_per_part, side="right")) - 1
                hi = min(max(hi, lo + 1), n)
                a, b = int(indptr[lo]), int(indptr[hi])
                cells = np.repeat(np.arange(lo, hi, dtype=np.uint32), np.diff(indptr[lo:hi + 1]))
                pl.DataFrame({
                    "cell_id": cells,
                    "gene_id": m["indices"][a:b].astype(np.uint32),
                    "count": m["data"][a:b].astype(np.float32),
                }).write_parquet(out / f"part-{part:05d}.parquet", compression="zstd")
                lo, part = hi, part + 1
            obs.with_columns(pl.Series("cell_id", np.arange(obs.height, dtype=np.uint32))).write_parquet(out / "obs.parquet")
            var.with_columns(pl.Series("gene_id", np.arange(n_genes, dtype=np.uint32))).write_parquet(out / "var.parquet")
            bf = read_parquet(out)
    if gex_only and "feature_types" in bf.var.columns:
        bf = bf.filter_genes(pl.col("feature_types") == "Gene Expression")
    return bf


def _h5ad_series(group, key):
    """One obs/var column from the 'group of columns' h5ad layout. Plain numeric/string
    datasets are read directly; a categorical column (anndata's usual encoding for pandas
    ``category`` dtype: a sub-group holding ``categories`` + integer ``codes``, -1 =
    missing) is expanded to strings, matching what ``BioFrame.from_anndata`` already does
    for a categorical column arriving via pandas. Returns ``None`` for an encoding this
    doesn't recognise (e.g. nullable-integer/-boolean) rather than guessing at it."""
    import h5py

    node = group[key]
    if isinstance(node, h5py.Group):
        if "categories" in node and "codes" in node:
            cats = _decode(node["categories"])
            codes = node["codes"][:]
            return [cats[c] if c >= 0 else "nan" for c in codes]
        return None
    arr = node[:]
    return _decode(arr) if arr.dtype.kind in "SUO" else arr


def _h5ad_frame(node, index_name: str) -> pl.DataFrame:
    """obs/var columns from either h5ad layout: a group of columns (anndata >= 0.7,
    including the categorical encoding) or a legacy structured dataset."""
    import h5py

    if isinstance(node, h5py.Dataset):  # legacy: structured array
        arr = node[:]
        cols = {index_name: _decode(arr["index"])}
        for name in arr.dtype.names or ():
            if name == "index":
                continue
            col = arr[name]
            cols[name] = _decode(col) if col.dtype.kind in "SUO" else col
        return pl.DataFrame(cols)
    key = node.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    cols = {index_name: _decode(node[key])}
    order = node.attrs.get("column-order", list(node.keys()))
    for raw in order:
        name = raw.decode() if isinstance(raw, bytes) else raw
        if name == key:
            continue
        value = _h5ad_series(node, name)
        if value is not None:
            cols[name] = value
    return pl.DataFrame(cols)


def read_h5ad(path: Union[str, Path]) -> BioFrame:
    """Read ``X`` (CSR, or dense) plus every obs/var column from an .h5ad file, without
    needing ``anndata`` installed. Categorical columns are expanded to strings (matching
    ``BioFrame.from_anndata``). ``obsm``/``varm``/``uns``/other layers are not read; for
    those, load with `anndata.read_h5ad` and use :meth:`BioFrame.from_anndata` instead.
    """
    import h5py

    with h5py.File(path, "r") as f:
        X = f["X"]
        obs = _h5ad_frame(f["obs"], "barcode")
        var = _h5ad_frame(f["var"], "gene_name")
        if isinstance(X, h5py.Group):
            enc = X.attrs.get("encoding-type", X.attrs.get("h5sparse_format", "csr_matrix"))
            enc = enc.decode() if isinstance(enc, bytes) else enc
            shape = tuple(int(v) for v in X.attrs.get("shape", X.attrs.get("h5sparse_shape")))
            indptr = X["indptr"][:].astype(np.int64)
            indices = _read_array(X["indices"], np.uint32)
            data = _read_array(X["data"], np.float32)
            if enc.startswith("csr"):
                return BioFrame.from_csr(indptr, indices, data, shape[1], obs, var)
            import scipy.sparse as sp  # CSC: transpose once
            return BioFrame.from_scipy(sp.csc_matrix((data, indices, indptr), shape=shape), obs, var)
        dense = X[:]
        rows, cols = np.nonzero(dense)
        df = pl.DataFrame({"cell_id": rows.astype(np.uint32), "gene_id": cols.astype(np.uint32),
                           "count": dense[rows, cols].astype(np.float32)})
        return BioFrame.from_triplets(df, dense.shape[0], dense.shape[1], obs, var)


def read_10x_mtx(path: Union[str, Path], gex_only: bool = True) -> BioFrame:
    """Read a Cell Ranger ``matrix.mtx(.gz)`` directory (with barcodes/features files)."""
    path = Path(path)

    def find(*names):
        for n in names:
            for suf in ("", ".gz"):
                p = path / (n + suf)
                if p.exists():
                    return p
        raise FileNotFoundError(f"none of {names} in {path}")

    mtx = find("matrix.mtx")
    opener = gzip.open if mtx.suffix == ".gz" else open
    with opener(mtx, "rt") as fh:
        line = fh.readline()
        n_header = 1
        while line.startswith("%"):
            line = fh.readline()
            n_header += 1
        n_genes, n_cells, _ = (int(x) for x in line.split())
    df = pl.read_csv(mtx, separator=" ", has_header=False, skip_rows=n_header,
                     new_columns=["gene_id", "cell_id", "count"],
                     schema_overrides={"gene_id": pl.UInt32, "cell_id": pl.UInt32, "count": pl.Float32})
    df = df.with_columns(pl.col("gene_id") - 1, pl.col("cell_id") - 1)
    bc = pl.read_csv(find("barcodes.tsv"), separator="\t", has_header=False)
    feats = pl.read_csv(find("features.tsv", "genes.tsv"), separator="\t", has_header=False)
    var = pl.DataFrame({
        "gene_ids": feats[:, 0].cast(pl.Utf8),
        "gene_name": feats[:, 1].cast(pl.Utf8) if feats.width > 1 else feats[:, 0].cast(pl.Utf8),
        "feature_types": feats[:, 2].cast(pl.Utf8) if feats.width > 2 else pl.Series(["Gene Expression"] * feats.height),
    })
    obs = pl.DataFrame({"barcode": bc[:, 0].cast(pl.Utf8)})
    bf = BioFrame.from_triplets(df, n_cells, n_genes, obs, var)
    if gex_only:
        bf = bf.filter_genes(pl.col("feature_types") == "Gene Expression")
    return bf


def convert_h5_to_parquet_stream(file_path: Union[str, Path], output_path: Union[str, Path]) -> BioFrame:
    """Stream a 10x HDF5 into a Parquet dataset directory (kept for backward compatibility)."""
    return read_10x_h5(file_path, gex_only=False, backed=output_path)
