"""Readers and writers. Only ``h5py`` (for .h5/.h5ad) and ``polars`` are needed."""

from __future__ import annotations

import gzip
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl

from .core import BioFrame, _publish_parquet_parts, _refuse_live_directory, _warn_if_not_counts, read_parquet

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


def _open_10x(f, genome: Optional[str] = None):
    """Return (group, barcodes, var DataFrame) for Cell Ranger v3+ or v2 layouts.

    v3+ files hold one matrix; their ``features/genome`` column (if any) becomes the ``genome``
    ``var`` column. v2 files hold one group per genome: ``genome`` selects it, and a file with several
    genomes and no ``genome`` raises ``ValueError`` (nothing is read silently).
    """
    import h5py

    if "matrix" in f:
        m = f["matrix"]
        feats = m["features"]
        cols = {
            "gene_ids": _decode(feats["id"]),
            "gene_name": _decode(feats["name"]),
            "feature_types": _decode(feats["feature_type"]) if "feature_type" in feats else ["Gene Expression"] * feats["id"].shape[0],
        }
        if "genome" in feats:
            cols["genome"] = _decode(feats["genome"])
        return m, _decode(m["barcodes"]), pl.DataFrame(cols)
    genomes = [k for k in f.keys() if isinstance(f[k], h5py.Group) and "barcodes" in f[k]]  # v2: one group per genome
    if genome is None:
        if len(genomes) > 1:
            raise ValueError(f"the file holds {len(genomes)} genomes {genomes}; pass genome=<one of them> to choose")
        if not genomes:
            raise ValueError("no Cell Ranger matrix found: expected a 'matrix' group (v3+) or one group per genome (v2)")
        genome = genomes[0]
    elif genome not in genomes:
        raise ValueError(f"genome {genome!r} is not in the file; available: {genomes}")
    m = f[genome]
    var = pl.DataFrame({
        "gene_ids": _decode(m["genes"]),
        "gene_name": _decode(m["gene_names"]),
        "feature_types": ["Gene Expression"] * m["genes"].shape[0],
    })
    return m, _decode(m["barcodes"]), var


def read_10x_h5(path: Union[str, Path], gex_only: bool = True, backed: Optional[Union[str, Path]] = None,
                nnz_per_part: int = 1 << 23, genome: Optional[str] = None) -> BioFrame:
    """Read a Cell Ranger ``filtered_feature_bc_matrix.h5``.

    ``genome`` selects one genome. A legacy (v2) file stores one group per genome (for example
    ``hg19`` and ``mm10``): with several genomes and ``genome=None`` a ``ValueError`` lists them,
    and an unknown name raises ``ValueError`` too, as in ``sc.read_10x_h5``. A v3+ file has a single
    matrix whose features may carry a ``genome`` column (kept in ``var["genome"]``); there,
    ``genome=`` keeps only the genes of that genome. The warning below applies to every reader.

    With ``backed=<dir>`` the matrix is streamed to a Parquet dataset in that
    directory (one cell range at a time) and returned as an out-of-core
    BioFrame; otherwise it is loaded as compact in-memory CSR. Existing ``part-*.parquet``
    files in ``backed`` are replaced, but only once the new dataset is complete; a
    ``ValueError`` is raised (and nothing is touched) if an open BioFrame still reads from
    that directory.

    The values are expected to be raw integer UMI counts. A ``UserWarning`` is raised when the
    stored values are negative or (in the first part of the file) not integers, for example for a
    log-normalised matrix; counts are stored as ``float32``, so integers above 2**24 are not exact.
    """
    import h5py

    if backed is not None:
        _refuse_live_directory(backed, "read_10x_h5")

    with h5py.File(path, "r") as f:
        m, barcodes, var = _open_10x(f, genome)
        if genome is not None and "genome" in var.columns and genome not in set(var["genome"].to_list()):
            raise ValueError(f"genome {genome!r} is not in the file; available: {sorted(set(var['genome'].to_list()))}")
        if genome is not None and "genome" not in var.columns and "matrix" in f:
            raise ValueError("genome= was given but this v3 file has no 'genome' feature column")
        n_genes = var.height
        obs = pl.DataFrame({"barcode": barcodes})
        if backed is None:
            indptr = m["indptr"][:].astype(np.int64)
            bf = BioFrame.from_csr(indptr, _read_array(m["indices"], np.uint32), _read_array(m["data"], np.float32),
                                   n_genes, obs, var)
        else:
            out = Path(backed)
            out.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".write-", dir=out))  # same file system as the target
            try:
                indptr = m["indptr"][:].astype(np.int64)
                n = len(indptr) - 1
                lo, part = 0, 0
                seen = {"checked": 0, "n_negative": 0, "n_non_integer": 0, "min": 0.0, "example": None}
                while lo < n:
                    # whole cells, ~nnz_per_part non-zeros per part (bounded memory for any cell depth)
                    hi = int(np.searchsorted(indptr, indptr[lo] + nnz_per_part, side="right")) - 1
                    hi = min(max(hi, lo + 1), n)
                    a, b = int(indptr[lo]), int(indptr[hi])
                    cells = np.repeat(np.arange(lo, hi, dtype=np.uint32), np.diff(indptr[lo:hi + 1]))
                    genes = m["indices"][a:b].astype(np.uint32)
                    if len(genes) and int(genes.max()) >= n_genes:  # checked once, at conversion
                        raise ValueError(f"gene index {int(genes.max())} out of range for n_genes={n_genes}")
                    vals = m["data"][a:b].astype(np.float32)
                    if part == 0:  # non-integers: the first part; negatives: every part
                        frac = (vals != np.floor(vals)) & np.isfinite(vals)
                        seen["checked"], seen["n_non_integer"] = len(vals), int(frac.sum())
                        seen["example"] = float(vals[frac][0]) if frac.any() else None
                    if len(vals) and float(vals.min()) < 0:
                        seen["n_negative"] += int((vals < 0).sum())
                        seen["min"] = min(seen["min"], float(vals.min()))
                    pl.DataFrame({
                        "cell_id": cells,
                        "gene_id": genes,
                        "count": vals,
                    }).write_parquet(staging / f"part-{part:05d}.parquet", compression="zstd")
                    lo, part = hi, part + 1
                obs.with_columns(pl.Series("cell_id", np.arange(obs.height, dtype=np.uint32))).write_parquet(staging / "obs.parquet")
                var.with_columns(pl.Series("gene_id", np.arange(n_genes, dtype=np.uint32))).write_parquet(staging / "var.parquet")
                _publish_parquet_parts(staging, out)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            bf = read_parquet(out)
            _warn_if_not_counts(None, "read_10x_h5", stacklevel=3, report=seen)
    if genome is not None and "genome" in bf.var.columns:  # v3+: one matrix, several genomes
        bf = bf.filter_genes(pl.col("genome") == genome)
    if gex_only and "feature_types" in bf.var.columns:
        bf = bf.filter_genes(pl.col("feature_types") == "Gene Expression")
    return bf


def _nullable(values: np.ndarray, mask: np.ndarray, strings: bool = False) -> pl.Series:
    """A Polars Series from anndata's nullable encodings (``values`` + boolean ``mask``, True = missing)."""
    base = pl.Series(_decode(values) if strings else values)
    return pl.select(pl.when(pl.lit(pl.Series(mask.astype(bool)))).then(None).otherwise(pl.lit(base))).to_series()


def _h5ad_series(group, key):
    """One obs/var column from the 'group of columns' h5ad layout. Plain numeric/string
    datasets are read directly; a categorical column (anndata's usual encoding for pandas
    ``category`` dtype: a sub-group holding ``categories`` + integer ``codes``, -1 =
    missing) is expanded to strings, matching what ``BioFrame.from_anndata`` already does
    for a categorical column arriving via pandas; anndata's nullable integer / boolean /
    string columns (a group of ``values`` + ``mask``) become Polars columns with nulls.
    Returns ``None`` for an encoding this doesn't recognise; the caller warns about it."""
    import h5py

    node = group[key]
    if isinstance(node, h5py.Group):
        if "categories" in node and "codes" in node:
            cats = _decode(node["categories"])
            codes = node["codes"][:]
            return [cats[c] if c >= 0 else "nan" for c in codes]
        if "values" in node and "mask" in node:
            enc = node.attrs.get("encoding-type", "")
            enc = enc.decode() if isinstance(enc, bytes) else enc
            vals = node["values"][:]
            if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array", ""):
                return _nullable(vals, node["mask"][:], strings=vals.dtype.kind in "SUO")
        return None
    arr = node[:]
    return _decode(arr) if arr.dtype.kind in "SUO" else arr


def _h5ad_frame(node, index_name: str, what: str = "obs") -> pl.DataFrame:
    """obs/var columns from either h5ad layout: a group of columns (anndata >= 0.7,
    including the categorical and nullable encodings) or a legacy structured dataset. Columns in an
    encoding that is not recognised are left out with a ``UserWarning`` naming them."""
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
    skipped = []
    for raw in order:
        name = raw.decode() if isinstance(raw, bytes) else raw
        if name == key:
            continue
        value = _h5ad_series(node, name)
        if value is None:
            skipped.append(name)
        else:
            cols[name] = value
    if skipped:
        warnings.warn(f"read_h5ad: {what} columns {skipped} use an encoding that is not supported and were not read; "
                      "load the file with anndata.read_h5ad and BioFrame.from_anndata to keep them",
                      UserWarning, stacklevel=4)
    return pl.DataFrame(cols)


def read_h5ad(path: Union[str, Path]) -> BioFrame:
    """Read ``X`` (CSR, or dense) plus every obs/var column from an .h5ad file, without
    needing ``anndata`` installed. Categorical columns are expanded to strings (matching
    ``BioFrame.from_anndata``); nullable integer, boolean and string columns keep their missing
    values as nulls. A column in an encoding that is not supported is skipped with a ``UserWarning``
    naming it. ``X`` is expected to hold raw counts: a ``UserWarning`` is raised when it holds negative
    or non-integer values (for example log-normalised data), which CREST would normalise again;
    counts are stored as ``float32`` (integers above 2**24 are not exact). A column named ``cell_id`` / ``gene_id`` is kept as
    ``cell_id_orig`` / ``gene_id_orig``; the frame's own ids are assigned fresh.
    ``obsm``/``varm``/``uns``/other layers are not read; for
    those, load with `anndata.read_h5ad` and use :meth:`BioFrame.from_anndata` instead.
    """
    import h5py

    with h5py.File(path, "r") as f:
        X = f["X"]
        obs = _h5ad_frame(f["obs"], "barcode", "obs")
        var = _h5ad_frame(f["var"], "gene_name", "var")
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
    """Stream a 10x HDF5 file into a Parquet dataset directory and open it out-of-core.

    Kept for backward compatibility: it is ``read_10x_h5(file_path,
    gex_only=False, backed=output_path)``, so all feature types are kept.

    Parameters
    ----------
    file_path : str or pathlib.Path
        A Cell Ranger ``.h5`` matrix.
    output_path : str or pathlib.Path
        Directory for the Parquet parts (created if needed; existing
        ``part-*.parquet`` files in it are replaced once the new dataset is complete,
        and a ``ValueError`` is raised if an open BioFrame still reads from it).

    Returns
    -------
    crest.BioFrame
        An out-of-core BioFrame backed by the new Parquet dataset.
    """
    return read_10x_h5(file_path, gex_only=False, backed=output_path)
