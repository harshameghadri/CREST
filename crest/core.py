"""BioFrame: a single-cell dataset processed in cell-sorted chunks.

Expression data (``X``) lives in one of three stores:

* :class:`CSRStore` - compact in-memory CSR (``indptr``, ``indices``, ``data``);
* :class:`FrameStore` - a Polars DataFrame of ``(cell_id, gene_id, count)`` triplets;
* :class:`ParquetStore` - a directory of Parquet parts, streamed from disk (out-of-core).

Raw counts are never modified. Cell/gene filters and the ``normalize_total`` /
``log1p`` transforms are recorded and applied on the fly to each chunk by native
kernels, so no normalised, scaled or densified copy of the matrix is ever held.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import polars as pl

from . import crest as _native  # compiled extension

Chunk = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (row, var index, value)
# Raw chunk: (genes, values, cells or None, indptr or None, first_cell)
RawChunk = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]

DEFAULT_CHUNK_NNZ = 1 << 24  # ~16.8M non-zeros (~200 MB of working buffers)


# --------------------------------------------------------------------------- stores
class CSRStore:
    """Raw counts in memory as CSR (cells × genes).

    Row ``i`` holds entries ``indptr[i]:indptr[i+1]`` of ``indices`` (gene ids, ``uint32``)
    and ``data`` (counts, ``float32``); ``indptr`` is ``int64``. About 8 bytes per non-zero.
    Chunks are views of these arrays (no copy). Made by :func:`crest.read_10x_h5`,
    :func:`crest.read_h5ad` and :meth:`BioFrame.from_csr` / ``from_scipy`` / ``from_anndata``.
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n_genes: int):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.uint32)
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n_cells = len(self.indptr) - 1
        self.n_genes = int(n_genes)

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    def chunks(self, chunk_nnz: int = DEFAULT_CHUNK_NNZ) -> Iterator[RawChunk]:
        n = self.n_cells
        start = 0
        while start < n:
            target = self.indptr[start] + chunk_nnz
            end = int(np.searchsorted(self.indptr, target, side="right")) - 1
            end = min(max(end, start + 1), n)
            lo, hi = int(self.indptr[start]), int(self.indptr[end])
            yield self.indices[lo:hi], self.data[lo:hi], None, self.indptr[start:end + 1], start
            start = end

    def nbytes(self) -> int:
        return self.indptr.nbytes + self.indices.nbytes + self.data.nbytes


class FrameStore:
    """Raw counts in memory as a Polars DataFrame of ``(cell_id, gene_id, count)`` triplets.

    Columns are ``uint32``, ``uint32``, ``float32`` (about 12 bytes per non-zero), sorted by
    ``cell_id`` (sorted on construction if needed). Made by :func:`crest.read_10x_mtx`,
    :func:`crest.read_h5ad` for a dense ``X``, and :meth:`BioFrame.from_triplets`.
    """

    def __init__(self, df: pl.DataFrame, n_cells: int, n_genes: int,
                 cell_col: str = "cell_id", gene_col: str = "gene_id", value_col: str = "count"):
        df = df.select(
            pl.col(cell_col).cast(pl.UInt32).alias("cell_id"),
            pl.col(gene_col).cast(pl.UInt32).alias("gene_id"),
            pl.col(value_col).cast(pl.Float32).alias("count"),
        )
        cells = df["cell_id"].to_numpy()
        if len(cells) > 1 and np.any(cells[1:] < cells[:-1]):
            df = df.sort("cell_id", maintain_order=True)
        self.df = df.rechunk()
        self.n_cells, self.n_genes = int(n_cells), int(n_genes)
        self._cells = self.df["cell_id"].to_numpy()
        self._genes = self.df["gene_id"].to_numpy()
        self._vals = self.df["count"].to_numpy()

    @property
    def nnz(self) -> int:
        return self.df.height

    def chunks(self, chunk_nnz: int = DEFAULT_CHUNK_NNZ) -> Iterator[RawChunk]:
        n = len(self._cells)
        lo = 0
        while lo < n:
            hi = min(lo + chunk_nnz, n)
            if hi < n:  # extend to the end of the current cell
                hi = int(np.searchsorted(self._cells, self._cells[hi - 1], side="right"))
            yield self._genes[lo:hi], self._vals[lo:hi], self._cells[lo:hi], None, 0
            lo = hi


class ParquetStore:
    """Raw counts on disk (out-of-core): a directory of ``part-*.parquet`` files.

    Each file holds ``(cell_id, gene_id, count)`` triplets for a range of whole cells (about
    8.4 M non-zeros per file by default). Analysis steps read one file at a time, so the count
    matrix never has to fit in memory; ``BioFrame.chunk_nnz`` is ignored (one chunk = one
    file). Made by ``crest.read_10x_h5(path, backed=dir)`` or :meth:`BioFrame.write_parquet`,
    reopened with :func:`crest.read_parquet`.
    """

    def __init__(self, path: Union[str, Path], n_cells: int, n_genes: int):
        self.path = Path(path)
        self.parts = sorted(self.path.glob("part-*.parquet"))
        if not self.parts:
            raise FileNotFoundError(f"no part-*.parquet files in {self.path}")
        self.n_cells, self.n_genes = int(n_cells), int(n_genes)

    @property
    def nnz(self) -> int:
        return int(sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in self.parts))

    def chunks(self, chunk_nnz: int = DEFAULT_CHUNK_NNZ) -> Iterator[RawChunk]:
        for p in self.parts:
            df = pl.read_parquet(p, columns=["cell_id", "gene_id", "count"]).rechunk()
            cols = [df[c].cast(t).to_numpy() for c, t in (("gene_id", pl.UInt32), ("count", pl.Float32), ("cell_id", pl.UInt32))]
            del df
            yield cols[0], cols[1], cols[2], None, 0


Store = Union[CSRStore, FrameStore, ParquetStore]


# --------------------------------------------------------------------------- BioFrame
@dataclass
class BioFrame:
    """A single-cell dataset: raw counts plus annotations (CREST's counterpart of AnnData).

    The raw counts in ``store`` are never modified. Filtering creates a new BioFrame with
    smaller ``obs``/``var`` tables over the same store; ``normalize_total``/``log1p`` are
    recorded in ``ops`` and applied on the fly by every step that reads the counts. There
    is no ``X``: use :meth:`to_scipy` or :meth:`to_anndata` to materialise the matrix.

    Parameters
    ----------
    store : CSRStore, FrameStore or ParquetStore
        The raw counts (in memory, or on disk for :class:`ParquetStore`).
    obs : polars.DataFrame
        One row per kept cell. ``cell_id`` is the cell's row in ``store``.
    var : polars.DataFrame
        One row per kept gene. ``gene_id`` is the gene's column in ``store``.
    obsm : dict
        Per-cell arrays (``X_pca``, ``X_umap``, ``X_pca_harmony``), rows aligned with ``obs``.
    varm : dict
        Per-gene arrays (``PCs``), rows aligned with ``var``.
    uns : dict
        Everything else: parameters, the neighbour graph, result tables.
    ops : list of tuple
        Recorded transforms in order, e.g. ``[("normalize_total", 10000.0), ("log1p",)]``.
    chunk_nnz : int
        Target non-zeros per chunk for the in-memory stores (default 2**24).
    """

    store: Store
    obs: pl.DataFrame
    var: pl.DataFrame
    obsm: dict = field(default_factory=dict)
    varm: dict = field(default_factory=dict)
    uns: dict = field(default_factory=dict)
    ops: list = field(default_factory=list)  # [("normalize_total", target), ("log1p",)]
    chunk_nnz: int = DEFAULT_CHUNK_NNZ

    # ---- construction
    @classmethod
    def from_triplets(cls, df: pl.DataFrame, n_cells: Optional[int] = None, n_genes: Optional[int] = None,
                      obs: Optional[pl.DataFrame] = None, var: Optional[pl.DataFrame] = None,
                      cell_col: str = "cell_id", gene_col: str = "gene_id", value_col: str = "count") -> "BioFrame":
        """Wrap a Polars DataFrame of (cell, gene, count) triplets with integer ids."""
        n_cells = n_cells if n_cells is not None else (obs.height if obs is not None else int(df[cell_col].max()) + 1)
        n_genes = n_genes if n_genes is not None else (var.height if var is not None else int(df[gene_col].max()) + 1)
        store = FrameStore(df, n_cells, n_genes, cell_col, gene_col, value_col)
        return cls._with_store(store, obs, var)

    @classmethod
    def from_csr(cls, indptr, indices, data, n_genes: int, obs=None, var=None) -> "BioFrame":
        """From CSR arrays of raw counts (cells × genes); ``obs``/``var`` are optional Polars frames."""
        return cls._with_store(CSRStore(indptr, indices, data, n_genes), obs, var)

    @classmethod
    def from_scipy(cls, X, obs=None, var=None) -> "BioFrame":
        """From a scipy.sparse matrix (cells × genes)."""
        X = X.tocsr()
        X.sort_indices()
        return cls.from_csr(X.indptr, X.indices, X.data, X.shape[1], obs, var)

    @classmethod
    def _with_store(cls, store: Store, obs, var) -> "BioFrame":
        if obs is None:
            obs = pl.DataFrame({"cell_id": np.arange(store.n_cells, dtype=np.uint32)})
        elif "cell_id" not in obs.columns:
            obs = obs.with_columns(pl.Series("cell_id", np.arange(obs.height, dtype=np.uint32)))
        if var is None:
            var = pl.DataFrame({"gene_id": np.arange(store.n_genes, dtype=np.uint32)})
        elif "gene_id" not in var.columns:
            var = var.with_columns(pl.Series("gene_id", np.arange(var.height, dtype=np.uint32)))
        if obs.height != store.n_cells or var.height != store.n_genes:
            raise ValueError(f"obs/var sizes {obs.height}/{var.height} do not match store {store.n_cells}/{store.n_genes}")
        return cls(store=store, obs=obs, var=var)

    # ---- shape / names
    @property
    def n_obs(self) -> int:
        return self.obs.height

    @property
    def n_vars(self) -> int:
        return self.var.height

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_obs, self.n_vars

    @property
    def var_names(self) -> list:
        for c in ("gene_name", "gene_symbols", "gene_ids"):
            if c in self.var.columns:
                return self.var[c].cast(pl.Utf8).to_list()
        return self.var["gene_id"].cast(pl.Utf8).to_list()

    def __repr__(self) -> str:
        kind = type(self.store).__name__
        ops = ", ".join(o[0] for o in self.ops) or "raw counts"
        return (f"BioFrame {self.n_obs} cells × {self.n_vars} genes ({kind}; {ops})\n"
                f"    obs: {self.obs.columns}\n    var: {self.var.columns}\n"
                f"    obsm: {list(self.obsm)}  uns: {list(self.uns)}")

    # ---- maps from store ids to current rows
    def _cell_map(self) -> np.ndarray:
        m = np.full(self.store.n_cells, -1, dtype=np.int64)
        m[self.obs["cell_id"].to_numpy().astype(np.int64)] = np.arange(self.n_obs, dtype=np.int64)
        return m

    def _gene_map(self) -> np.ndarray:
        m = np.full(self.store.n_genes, -1, dtype=np.int32)
        m[self.var["gene_id"].to_numpy().astype(np.int64)] = np.arange(self.n_vars, dtype=np.int32)
        return m

    def _transform(self) -> Tuple[float, bool]:
        target, log = 0.0, False
        for op in self.ops:
            if op[0] == "normalize_total":
                if log:
                    raise ValueError("normalize_total after log1p is not supported")
                target = float(op[1])
            elif op[0] == "log1p":
                log = True
        return target, log

    def iter_ctx(self, transform: bool = True) -> Iterator[tuple]:
        """Yield argument tuples for the fused native kernels:
        ``(genes, values, cell_map, gene_map, target_sum, log1p, cells, indptr, first_cell)``.

        Kernels apply the cell/gene filters and (if ``transform``) the recorded
        normalize_total/log1p on the fly; raw data is never copied.
        """
        cmap, gmap = self._cell_map(), self._gene_map()
        target, log = self._transform() if transform else (0.0, False)
        for genes, values, cells, indptr, first in self.store.chunks(self.chunk_nnz):
            yield (np.ascontiguousarray(genes, dtype=np.uint32), np.ascontiguousarray(values, dtype=np.float32),
                   cmap, gmap, target, log,
                   None if cells is None else np.ascontiguousarray(cells, dtype=np.uint32),
                   None if indptr is None else np.ascontiguousarray(indptr, dtype=np.int64), int(first))

    def iter_chunks(self, transform: bool = True) -> Iterator[Chunk]:
        """Yield materialised (row, var index, value) chunks of kept cells/genes."""
        for ctx in self.iter_ctx(transform):
            yield _native.materialize(*ctx)

    # ---- subsetting
    def _subset(self, obs_mask=None, var_mask=None) -> "BioFrame":
        obs = self.obs if obs_mask is None else self.obs.filter(pl.Series(obs_mask))
        var = self.var if var_mask is None else self.var.filter(pl.Series(var_mask))
        obsm = {k: v[obs_mask] for k, v in self.obsm.items()} if obs_mask is not None else dict(self.obsm)
        varm = {k: v[var_mask] for k, v in self.varm.items()} if var_mask is not None else dict(self.varm)
        uns = dict(self.uns)
        if obs_mask is not None:
            uns.pop("neighbors", None)
        return BioFrame(self.store, obs, var, obsm, varm, uns, list(self.ops), self.chunk_nnz)

    def filter_cells(self, condition: Union[pl.Expr, np.ndarray]) -> "BioFrame":
        """Keep cells where ``condition`` (Polars expression on obs, or boolean mask) holds."""
        mask = condition if isinstance(condition, np.ndarray) else self.obs.select(condition).to_series().to_numpy()
        return self._subset(obs_mask=np.asarray(mask, dtype=bool))

    def filter_genes(self, condition: Union[pl.Expr, np.ndarray]) -> "BioFrame":
        """Keep genes where ``condition`` (Polars expression on var, or boolean mask) holds."""
        mask = condition if isinstance(condition, np.ndarray) else self.var.select(condition).to_series().to_numpy()
        return self._subset(var_mask=np.asarray(mask, dtype=bool))

    def copy(self) -> "BioFrame":
        """New BioFrame with copied metadata; the raw-count store is shared (it is never modified)."""
        return BioFrame(self.store, self.obs.clone(), self.var.clone(), dict(self.obsm), dict(self.varm),
                        dict(self.uns), list(self.ops), self.chunk_nnz)

    # ---- export
    def to_scipy(self, transform: bool = True):
        """Materialise the current (filtered, transformed) matrix as scipy CSR."""
        import scipy.sparse as sp
        rows, cols, vals = [], [], []
        for r, g, v in self.iter_chunks(transform):
            rows.append(r); cols.append(g); vals.append(v)
        r = np.concatenate(rows) if rows else np.zeros(0, np.uint32)
        return sp.csr_matrix((np.concatenate(vals) if vals else np.zeros(0, np.float32),
                              (r, np.concatenate(cols) if cols else np.zeros(0, np.uint32))),
                             shape=self.shape)

    def to_anndata(self, transform: bool = False):
        """Convert to AnnData (requires ``anndata``). ``X`` holds raw counts unless ``transform``."""
        import anndata as ad
        obs, var = _to_pandas(self.obs), _to_pandas(self.var)
        obs.index = obs.get("barcode", obs["cell_id"]).astype(str).values
        var.index = np.array(self.var_names, dtype=str)
        a = ad.AnnData(X=self.to_scipy(transform), obs=obs, var=var)
        for k, v in self.obsm.items():
            a.obsm[k] = v
        for k, v in self.varm.items():
            a.varm[k] = v
        for k, v in self.uns.items():
            if k != "neighbors":
                a.uns[k] = v
        if "neighbors" in self.uns:
            from .pp import connectivities_matrix
            a.obsp["connectivities"] = connectivities_matrix(self)
        return a

    @classmethod
    def from_anndata(cls, adata) -> "BioFrame":
        """From AnnData with a sparse (or dense) X of counts."""
        import scipy.sparse as sp
        X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        obs = _from_pandas(adata.obs, "barcode")
        var = _from_pandas(adata.var, "gene_name")
        bf = cls.from_scipy(X, obs, var)
        for k in adata.obsm.keys():
            bf.obsm[k] = np.asarray(adata.obsm[k])
        return bf

    def write_parquet(self, path: Union[str, Path], nnz_per_part: int = 1 << 23) -> None:
        """Write raw counts (kept cells/genes, current ids) as a Parquet dataset
        directory of whole-cell parts of ~``nnz_per_part`` non-zeros each."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for old in path.glob("part-*.parquet"):
            old.unlink()
        buf, n_buf, part = [], 0, 0

        def flush():
            nonlocal buf, n_buf, part
            if buf:
                pl.concat(buf).write_parquet(path / f"part-{part:05d}.parquet", compression="zstd")
                part += 1
            buf, n_buf = [], 0

        for r, g, v in self.iter_chunks(transform=False):
            lo = 0
            while lo < len(r):
                hi = min(lo + nnz_per_part - n_buf, len(r))
                if hi < len(r):  # end on a cell boundary
                    hi = int(np.searchsorted(r, r[hi - 1], side="right"))
                buf.append(pl.DataFrame({"cell_id": r[lo:hi], "gene_id": g[lo:hi], "count": v[lo:hi]}))
                n_buf += hi - lo
                lo = hi
                if n_buf >= nnz_per_part:
                    flush()
        flush()
        self.obs.with_columns(pl.Series("cell_id", np.arange(self.n_obs, dtype=np.uint32))).write_parquet(path / "obs.parquet")
        self.var.with_columns(pl.Series("gene_id", np.arange(self.n_vars, dtype=np.uint32))).write_parquet(path / "var.parquet")


def _to_pandas(df: pl.DataFrame):
    """polars -> pandas without pyarrow (column by column through numpy)."""
    import pandas as pd
    return pd.DataFrame({c: df[c].to_numpy() for c in df.columns})


def _from_pandas(df, index_name: str) -> pl.DataFrame:
    """pandas -> polars without pyarrow; the index becomes column ``index_name``."""
    cols = {index_name: np.asarray(df.index.astype(str))}
    for c in df.columns:
        v = df[c]
        if str(v.dtype) in ("category", "object", "string"):
            cols[str(c)] = np.asarray(v.astype(str))
        else:
            cols[str(c)] = v.to_numpy()
    return pl.DataFrame(cols)


def read_parquet(path: Union[str, Path], chunk_nnz: int = DEFAULT_CHUNK_NNZ) -> BioFrame:
    """Open a Parquet dataset written by :meth:`BioFrame.write_parquet` (out-of-core)."""
    path = Path(path)
    obs = pl.read_parquet(path / "obs.parquet")
    var = pl.read_parquet(path / "var.parquet")
    bf = BioFrame._with_store(ParquetStore(path, obs.height, var.height), obs, var)
    bf.chunk_nnz = chunk_nnz
    return bf
