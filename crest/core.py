"""BioFrame: a single-cell dataset processed in cell-sorted chunks.

Expression data (``X``) lives in one of three stores:

* :class:`CSRStore` - compact in-memory CSR (``indptr``, ``indices``, ``data``);
* :class:`FrameStore` - a Polars DataFrame of ``(cell_id, gene_id, count)`` triplets;
* :class:`ParquetStore` - a directory of Parquet parts, streamed from disk (out-of-core).

Raw counts are never modified. Cell/gene filters and the ``normalize_total`` /
``log1p`` transforms are recorded and applied on the fly to each chunk by native
kernels, so no normalised, scaled or densified copy of the matrix is ever held.
As in scanpy, ``normalize_total`` divides each cell by its total over the genes kept
when it is called: it computes those per-cell totals in one streamed pass and stores
them (``BioFrame.cell_totals``), so later gene or cell filters do not renormalise.
"""

from __future__ import annotations

import copy as _copy
import hashlib
import os
import shutil
import tempfile
import warnings
import weakref
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
        # validated once here, so the per-chunk kernels need not scan ids (they
        # would otherwise silently drop out-of-range entries)
        # the same rules as scipy's check_format (storage may be over-allocated)
        if self.n_cells < 0 or self.indptr[0] != 0 or self.indptr[-1] > len(self.indices) \
                or len(self.data) != len(self.indices) or np.any(np.diff(self.indptr) < 0):
            raise ValueError("CSR indptr must start at 0, be non-decreasing and not exceed len(indices) == len(data)")
        if len(self.indices) and int(self.indices[:self.indptr[-1]].max(initial=0)) >= self.n_genes:
            raise ValueError(f"gene index {int(self.indices[:self.indptr[-1]].max())} out of range "
                             f"for n_genes={self.n_genes}")

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
        if len(self._cells):  # sorted by cell, so the last id is the largest
            if int(self._cells[-1]) >= self.n_cells:
                raise ValueError(f"cell index {int(self._cells[-1])} out of range for n_cells={self.n_cells}")
            if int(self._genes.max()) >= self.n_genes:
                raise ValueError(f"gene index {int(self._genes.max())} out of range for n_genes={self.n_genes}")

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


# ParquetStore objects that are still referenced, so that no writer replaces the part files a
# live frame is reading (see :func:`_refuse_live_directory`).
_LIVE_PARQUET_STORES: "weakref.WeakSet" = weakref.WeakSet()


def _refuse_live_directory(path: Union[str, Path], what: str) -> None:
    """Raise ``ValueError`` if ``path`` is the directory that any open :class:`ParquetStore`
    in this process reads from. Replacing its part files would delete or renumber the data
    those frames still point at. Nothing is touched before this check."""
    target = Path(path).resolve()
    for store in list(_LIVE_PARQUET_STORES):
        if store.resolved_path == target:
            raise ValueError(
                f"{what}: {target} is the directory an open BioFrame reads its counts from, and "
                "replacing its part files would destroy that data. Write to a different directory "
                "(or release every frame backed by this one first, e.g. `del bf`); nothing was changed.")


def _publish_parquet_parts(staging: Path, target: Path) -> None:
    """Move a finished dataset (part files, obs.parquet, var.parquet) from ``staging`` into
    ``target``. The previous parts are removed only now, after the new ones are complete."""
    for old in target.glob("part-*.parquet"):
        old.unlink()
    # parts first, obs/var last: the dataset only looks complete once all three kinds are in place
    names = sorted(f.name for f in staging.glob("part-*.parquet")) + ["obs.parquet", "var.parquet"]
    for name in names:
        if (staging / name).exists():
            os.replace(staging / name, target / name)


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
        self.resolved_path = self.path.resolve()
        _LIVE_PARQUET_STORES.add(self)

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


# --------------------------------------------------------------------------- counts check
COUNTS_CHECK_ENTRIES = 1 << 22  # stored values inspected for non-integers (the first ~4 M)


def counts_report(store: "Store", max_entries: int = COUNTS_CHECK_ENTRIES, full_scan: bool = True) -> dict:
    """Cheap look at the raw values of ``store``: how many are negative and how many are not integers.

    Negative values are counted over the whole store when it is in memory (``full_scan``);
    non-integers are looked for in the first ``max_entries`` stored values only (for a
    :class:`ParquetStore`, in the first part file). Returns ``{"checked", "n_negative",
    "n_non_integer", "min", "example"}``.
    """
    if isinstance(store, CSRStore):
        vals = store.data[:store.indptr[-1]]
    elif isinstance(store, FrameStore):
        vals = store._vals
    else:
        first = next(iter(store.chunks()), None)
        vals = np.zeros(0, np.float32) if first is None else first[1]
    head = vals[:max_entries]
    with np.errstate(invalid="ignore"):
        frac = head != np.floor(head)
        frac &= np.isfinite(head)
        n_non_int = int(np.count_nonzero(frac))
        scan = vals if (full_scan and not isinstance(store, ParquetStore)) else head
        vmin = float(scan.min()) if len(scan) else 0.0
        n_neg = int(np.count_nonzero(scan < 0)) if vmin < 0 else 0
    example = float(head[frac][0]) if n_non_int else None
    return {"checked": int(len(head)), "n_negative": n_neg, "n_non_integer": n_non_int, "min": vmin, "example": example}


def _warn_if_not_counts(store: "Store", where: str, stacklevel: int = 3, report: Optional[dict] = None) -> None:
    """``UserWarning`` when the raw values look like something other than non-negative integer counts."""
    r = counts_report(store) if report is None else report
    msgs = []
    if r["n_non_integer"]:
        msgs.append(f"{r['n_non_integer']} of the first {r['checked']} stored values are not integers "
                    f"(e.g. {r['example']:.6g}); is this normalised or log-transformed data?")
    if r["n_negative"]:
        msgs.append(f"{r['n_negative']} stored values are negative (minimum {r['min']:.6g})")
    if msgs:
        warnings.warn(f"{where}: the matrix does not look like raw counts: " + "; ".join(msgs) +
                      ". CREST validates and expects raw UMI counts (normalize_total, highly variable genes "
                      "and DESeq2 assume them), so results on other data are not comparable to scanpy's.",
                      UserWarning, stacklevel=stacklevel)


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
        One row per kept cell. ``cell_id`` is the cell's row in ``store``; the constructors
        assign it, and an input column of that name is kept as ``cell_id_orig``.
    var : polars.DataFrame
        One row per kept gene. ``gene_id`` is the gene's column in ``store``; the constructors
        assign it, and an input column of that name is kept as ``gene_id_orig``.
    obsm : dict
        Per-cell arrays (``X_pca``, ``X_umap``, ``X_pca_harmony``), rows aligned with ``obs``.
    varm : dict
        Per-gene arrays (``PCs``), rows aligned with ``var``.
    uns : dict
        Everything else: parameters, the neighbour graph, result tables.
    ops : list of tuple
        Recorded transforms in order, e.g.
        ``[("normalize_total", 10000.0, "<totals fingerprint>"), ("log1p",)]``. The third
        element of ``normalize_total`` identifies the stored :attr:`cell_totals`, so caches
        keyed on ``ops`` notice when the totals change.
    chunk_nnz : int
        Target non-zeros per chunk for the in-memory stores (default 2**24).
    cell_totals : numpy.ndarray or None
        Per-cell totals that ``normalize_total`` divides by, indexed by store cell id
        (``float64``, length ``store.n_cells``; 8 bytes per cell). ``normalize_total``
        computes them in one streamed pass over the raw counts, as each cell's total over
        the genes kept at that moment (what scanpy divides by), and every later cell or
        gene filter carries them unchanged, so dropping genes afterwards does not
        renormalise. Calling ``normalize_total`` again replaces them. ``None`` (no
        ``normalize_total`` yet, or ``ops`` set by hand) makes the kernels fall back to
        each cell's total over the currently kept genes.
    """

    store: Store
    obs: pl.DataFrame
    var: pl.DataFrame
    obsm: dict = field(default_factory=dict)
    varm: dict = field(default_factory=dict)
    uns: dict = field(default_factory=dict)
    ops: list = field(default_factory=list)  # [("normalize_total", target), ("log1p",)]
    chunk_nnz: int = DEFAULT_CHUNK_NNZ
    cell_totals: Optional[np.ndarray] = None  # normalize_total divisors by store cell id; see above

    # ---- construction
    @classmethod
    def from_triplets(cls, df: pl.DataFrame, n_cells: Optional[int] = None, n_genes: Optional[int] = None,
                      obs: Optional[pl.DataFrame] = None, var: Optional[pl.DataFrame] = None,
                      cell_col: str = "cell_id", gene_col: str = "gene_id", value_col: str = "count") -> "BioFrame":
        """Wrap a Polars DataFrame of (cell, gene, count) triplets with integer ids.
        Raises ``ValueError`` if a cell or gene id is ``>= n_cells`` / ``n_genes``."""
        n_cells = n_cells if n_cells is not None else (obs.height if obs is not None else int(df[cell_col].max()) + 1)
        n_genes = n_genes if n_genes is not None else (var.height if var is not None else int(df[gene_col].max()) + 1)
        store = FrameStore(df, n_cells, n_genes, cell_col, gene_col, value_col)
        return cls._with_store(store, obs, var)

    @classmethod
    def from_csr(cls, indptr, indices, data, n_genes: int, obs=None, var=None) -> "BioFrame":
        """From CSR arrays of raw counts (cells × genes); ``obs``/``var`` are optional Polars frames.
        Raises ``ValueError`` for a malformed ``indptr`` or a gene index ``>= n_genes``."""
        return cls._with_store(CSRStore(indptr, indices, data, n_genes), obs, var)

    @classmethod
    def from_scipy(cls, X, obs=None, var=None) -> "BioFrame":
        """From a scipy.sparse matrix (cells × genes)."""
        X = X.tocsr()
        X.sort_indices()
        return cls.from_csr(X.indptr, X.indices, X.data, X.shape[1], obs, var)

    @classmethod
    def _with_store(cls, store: Store, obs, var, keep_ids: bool = False) -> "BioFrame":
        """Pair ``store`` with ``obs``/``var``. ``obs["cell_id"]`` and ``var["gene_id"]`` are the
        store's own row/column ids and are always assigned here as ``0..n-1``; a column of
        that name in the input is user data and is kept as ``cell_id_orig`` / ``gene_id_orig``.
        ``keep_ids`` (used by :func:`read_parquet`, whose ids were written by
        :meth:`write_parquet`) leaves an existing id column alone."""
        obs = _with_store_ids(obs, "cell_id", store.n_cells, keep_ids)
        var = _with_store_ids(var, "gene_id", store.n_genes, keep_ids)
        if obs.height != store.n_cells or var.height != store.n_genes:
            raise ValueError(f"obs/var sizes {obs.height}/{var.height} do not match store {store.n_cells}/{store.n_genes}")
        if not keep_ids and not isinstance(store, ParquetStore):  # parquet readers check while streaming
            _warn_if_not_counts(store, "BioFrame", stacklevel=4)
        return cls(store=store, obs=obs, var=var)

    # ---- shape / names
    @property
    def n_obs(self) -> int:
        """Number of cells kept (rows of ``obs``), as ``AnnData.n_obs``."""
        return self.obs.height

    @property
    def n_vars(self) -> int:
        """Number of genes kept (rows of ``var``), as ``AnnData.n_vars``."""
        return self.var.height

    @property
    def shape(self) -> Tuple[int, int]:
        """``(n_obs, n_vars)``."""
        return self.n_obs, self.n_vars

    @property
    def var_names(self) -> list:
        """Gene names of the kept genes, as a list of str.

        Taken from the first ``var`` column that exists of ``gene_name``,
        ``gene_symbols``, ``gene_ids`` and ``gene_id_orig`` (a ``gene_id`` column that came
        with the input); otherwise the store's integer ``gene_id`` as strings. The names
        need not be unique (see :attr:`gene_ids`).
        """
        for c in ("gene_name", "gene_symbols", "gene_ids", "gene_id_orig"):
            if c in self.var.columns:
                return self.var[c].cast(pl.Utf8).to_list()
        return self.var["gene_id"].cast(pl.Utf8).to_list()

    @property
    def gene_ids(self) -> Optional[list]:
        """Stable gene identifiers (e.g. Ensembl ids) of the kept genes, or ``None``.

        Taken from the ``gene_ids`` ``var`` column (Cell Ranger's ``id``), else from
        ``gene_id_orig``. Gene symbols are not unique in Cell Ranger feature tables; steps that
        match genes between objects (:func:`crest.tl.ingest`, :func:`crest.tl.score_genes`) use these
        ids when both sides have them.
        """
        for c in ("gene_ids", "gene_id_orig"):
            if c in self.var.columns:
                return self.var[c].cast(pl.Utf8).to_list()
        return None

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
        ``(genes, values, cell_map, gene_map, target_sum, log1p, cell_totals, cells, indptr, first_cell)``.

        Kernels apply the cell/gene filters and (if ``transform``) the recorded
        normalize_total/log1p on the fly; raw data is never copied. ``cell_totals``
        is :attr:`cell_totals` when the data are normalised (``None`` otherwise). A
        legacy two-element ``("normalize_total", target)`` op without stored totals
        falls back to each cell's total over the kept genes; a fingerprinted op whose
        totals are missing raises ``ValueError``.
        """
        cmap, gmap = self._cell_map(), self._gene_map()
        target, log = self._transform() if transform else (0.0, False)
        totals = None
        if target > 0:
            if self.cell_totals is not None:
                totals = np.ascontiguousarray(self.cell_totals, dtype=np.float64)
            elif any(op[0] == "normalize_total" and len(op) > 2 for op in self.ops):
                raise ValueError("bf.ops records normalize_total with stored per-cell totals, but "
                                 "bf.cell_totals is None; call crest.pp.normalize_total again")
        for genes, values, cells, indptr, first in self.store.chunks(self.chunk_nnz):
            yield (np.ascontiguousarray(genes, dtype=np.uint32), np.ascontiguousarray(values, dtype=np.float32),
                   cmap, gmap, target, log, totals,
                   None if cells is None else np.ascontiguousarray(cells, dtype=np.uint32),
                   None if indptr is None else np.ascontiguousarray(indptr, dtype=np.int64), int(first))

    def iter_chunks(self, transform: bool = True) -> Iterator[Chunk]:
        """Yield materialised (row, var index, value) chunks of kept cells/genes."""
        for ctx in self.iter_ctx(transform):
            yield _native.materialize(*ctx)

    # ---- normalisation totals
    def _kept_gene_totals(self) -> np.ndarray:
        """Raw per-cell totals over the currently kept genes, by store cell id (NaN for
        cells not kept). One streamed pass; used by ``normalize_total``."""
        ct = np.zeros(self.n_obs, np.float64)
        cn, cf = np.zeros(self.n_obs, np.uint32), np.zeros(self.n_obs, np.float64)
        gn, gt = np.zeros(self.n_vars, np.uint32), np.zeros(self.n_vars, np.float64)
        flag = np.zeros(self.n_vars, dtype=bool)
        for ctx in self.iter_ctx(transform=False):
            _native.qc(*ctx, flag, ct, cn, cf, gn, gt)
        totals = np.full(self.store.n_cells, np.nan, np.float64)
        totals[self.obs["cell_id"].to_numpy().astype(np.int64)] = ct
        return totals

    # ---- subsetting
    def _subset(self, obs_mask=None, var_mask=None) -> "BioFrame":
        obs = self.obs if obs_mask is None else self.obs.filter(pl.Series(obs_mask))
        var = self.var if var_mask is None else self.var.filter(pl.Series(var_mask))
        # obsm/varm arrays are always new arrays (boolean indexing copies; an unfiltered axis is
        # copied explicitly), so editing the child's arrays in place never changes the parent
        obsm = {k: (v[obs_mask] if obs_mask is not None else np.array(v)) for k, v in self.obsm.items()}
        varm = {k: (v[var_mask] if var_mask is not None else np.array(v)) for k, v in self.varm.items()}
        uns = dict(self.uns)  # shallow: nested containers are shared with the parent (see filter_cells)
        if obs_mask is not None:
            # State derived from the old cell set. uns is a shallow copy, so replace these
            # entries, never mutate them (the parent keeps its own). The HVG scale cache
            # (var mean_log/std_log) is disowned so tl.pca recomputes mean/std on the kept
            # cells; obs-aligned arrays are sliced; uns["pca"] stays: it describes the PCs
            # that the kept obsm["X_pca"] rows hold (what ingest projects onto).
            if "ops" in uns.get("hvg", {}):
                uns["hvg"] = {k: v for k, v in uns["hvg"].items() if k != "ops"}
            if "ingest" in uns:
                uns["ingest"] = {**uns["ingest"], "indices": uns["ingest"]["indices"][obs_mask],
                                 "distances": uns["ingest"]["distances"][obs_mask]}
            # The graph is over the old cells, but its settings are still what ingest needs to search
            # a filtered reference with the same number of neighbours and PCs.
            nb_params = uns.pop("neighbors", {}).get("params")
            if nb_params is not None:
                uns["neighbors_params"] = {"params": dict(nb_params)}
        return BioFrame(self.store, obs, var, obsm, varm, uns, list(self.ops), self.chunk_nnz, self.cell_totals)

    def filter_cells(self, condition: Union[pl.Expr, np.ndarray]) -> "BioFrame":
        """Keep cells where ``condition`` (Polars expression on obs, or boolean mask) holds.

        Returns a new BioFrame over the same raw-count store. ``obs``/``var`` are new tables and
        every ``obsm`` / ``varm`` array is a new array, so editing those in place leaves the
        parent unchanged. ``uns`` is a shallow copy: the dictionary is new, but the nested
        dictionaries, arrays and tables inside it are the parent's objects (the neighbour graph
        and result tables are not duplicated). Do not edit them in place; use :meth:`copy`
        for a fully independent frame.
        """
        mask = condition if isinstance(condition, np.ndarray) else self.obs.select(condition).to_series().to_numpy()
        return self._subset(obs_mask=np.asarray(mask, dtype=bool))

    def filter_genes(self, condition: Union[pl.Expr, np.ndarray]) -> "BioFrame":
        """Keep genes where ``condition`` (Polars expression on var, or boolean mask) holds.

        Sharing as in :meth:`filter_cells`: new ``obs``/``var`` tables and new ``obsm`` / ``varm``
        arrays, a shallow copy of ``uns``.
        """
        mask = condition if isinstance(condition, np.ndarray) else self.var.select(condition).to_series().to_numpy()
        return self._subset(var_mask=np.asarray(mask, dtype=bool))

    def copy(self) -> "BioFrame":
        """Independent copy: ``obs``/``var``, every ``obsm`` / ``varm`` array, ``uns`` (deep copy, nested
        dictionaries and arrays included) and ``cell_totals`` are duplicated, so in-place edits of the
        copy never change this frame. Only the raw-count store is shared (it is never modified)."""
        return BioFrame(self.store, self.obs.clone(), self.var.clone(),
                        {k: np.array(v) for k, v in self.obsm.items()}, {k: np.array(v) for k, v in self.varm.items()},
                        _copy.deepcopy(self.uns), list(self.ops), self.chunk_nnz,
                        None if self.cell_totals is None else self.cell_totals.copy())

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
        """Convert to AnnData (requires ``anndata``). ``X`` holds raw counts unless ``transform``.

        Carries over obs/var (all columns except the store's own ``cell_id`` / ``gene_id``,
        which have no meaning outside this BioFrame), obsm, varm and uns (``neighbors``
        becomes ``obsp["connectivities"]``, scanpy's layout). The result can be written with
        ``AnnData.write_h5ad``: ``uns`` is converted to types the h5ad format stores. Polars
        result tables become pandas DataFrames, the recorded transforms in ``ops`` become
        ``"|"``-joined strings (``"normalize_total|10000.0|<fingerprint>"``) and entries that
        are ``None`` are left out. Use this to hand a ``BioFrame`` to ``sc.pl`` or anything
        else in the scverse ecosystem.
        """
        import anndata as ad
        obs, var = _to_pandas_safe(self.obs), _to_pandas_safe(self.var)
        obs.index = (obs["barcode"] if "barcode" in obs.columns else obs["cell_id"]).astype(str).values
        names = np.array(self.var_names, dtype=str)
        if len(set(names.tolist())) < len(names):
            n_dup = len(names) - len(set(names.tolist()))
            warnings.warn(f"to_anndata: {n_dup} gene names are not unique (Cell Ranger feature tables repeat some "
                          "symbols); the repeats get a '-1', '-2', ... suffix in the AnnData index so that it is "
                          "unique. The original names stay in the var column they came from.", UserWarning, stacklevel=2)
            import pandas as pd
            names = np.array(ad.utils.make_index_unique(pd.Index(names)), dtype=str)
        var.index = names
        obs = obs.drop(columns=["cell_id"])  # the store's ids; from_anndata/read_h5ad assign their own
        var = var.drop(columns=["gene_id"])
        a = ad.AnnData(X=self.to_scipy(transform), obs=obs, var=var)
        for k, v in self.obsm.items():
            a.obsm[k] = v
        for k, v in self.varm.items():
            a.varm[k] = v
        for k, v in self.uns.items():
            if k != "neighbors":
                c = _anndata_safe(v)
                if c is not _DROP:
                    a.uns[str(k)] = c
        if "neighbors" in self.uns:
            from .pp import connectivities_matrix
            a.obsp["connectivities"] = connectivities_matrix(self)
        return a

    @classmethod
    def from_anndata(cls, adata) -> "BioFrame":
        """From AnnData with a sparse (or dense) X of counts.

        This is the path for any format CREST doesn't read natively: Loom, Visium, a
        Zarr-backed AnnData store, CSV/Excel matrices, or anything else `scanpy`/`anndata`
        can load (``sc.read_loom``, ``sc.read_visium``, ``ad.read_zarr``, ...) — load it as
        an ``AnnData`` with the matching reader, then call this. Carries over ``obs``/``var``
        (every column, not just the index) and ``obsm``; ``varm``, ``uns`` and any ``layers``
        other than ``X`` are not. The store's own ``cell_id`` /
        ``gene_id`` columns are always assigned fresh (``0..n-1``); a column of that name in
        ``adata.obs`` / ``adata.var`` is kept as ``cell_id_orig`` / ``gene_id_orig``. Loads the whole matrix into memory, so for files too large
        for that, see :func:`crest.read_h5ad` / `convert_h5_to_parquet_stream` instead.
        """
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
        directory of whole-cell parts of ~``nnz_per_part`` non-zeros each.

        Raises ``ValueError``, before touching anything, if ``path`` is the directory this
        frame (or any other open frame) reads its counts from: that would replace the data
        being streamed. Write to a new directory instead. Existing ``part-*.parquet`` files in
        another target directory are replaced, but only after the new dataset is complete, so
        a failed write leaves the old one in place.
        """
        path = Path(path)
        _refuse_live_directory(path, "write_parquet")
        path.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".write-", dir=path))  # same file system as the target
        try:
            buf, n_buf, part = [], 0, 0

            def flush():
                nonlocal buf, n_buf, part
                if buf:
                    pl.concat(buf).write_parquet(staging / f"part-{part:05d}.parquet", compression="zstd")
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
            self.obs.with_columns(pl.Series("cell_id", np.arange(self.n_obs, dtype=np.uint32))).write_parquet(staging / "obs.parquet")
            self.var.with_columns(pl.Series("gene_id", np.arange(self.n_vars, dtype=np.uint32))).write_parquet(staging / "var.parquet")
            _publish_parquet_parts(staging, path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def totals_fingerprint(totals: np.ndarray) -> str:
    """Short content hash of stored ``cell_totals`` (recorded in the ``normalize_total`` op)."""
    return hashlib.blake2b(np.ascontiguousarray(totals, np.float64).tobytes(), digest_size=8).hexdigest()


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


def _with_store_ids(df, col: str, n: int, keep: bool) -> pl.DataFrame:
    """``df`` with ``col`` holding the store's ids ``0..n-1`` (see :meth:`BioFrame._with_store`)."""
    ids = np.arange(n if df is None else df.height, dtype=np.uint32)
    if df is None:
        return pl.DataFrame({col: ids})
    if col in df.columns:
        if keep:
            return df
        kept = f"{col}_orig"
        if kept in df.columns:
            raise ValueError(f"the table has both {col!r} and {kept!r}; {col!r} is the store's own id column "
                             f"and would be renamed {kept!r}: drop or rename one of them first")
        df = df.rename({col: kept})
    return df.with_columns(pl.Series(col, ids))


def _to_pandas_safe(df: pl.DataFrame):
    """polars -> pandas (no pyarrow) with every column in a type AnnData can write to h5ad:
    categorical, list and struct columns become strings (list items joined by ``"|"``), and
    missing strings become ``""``."""
    import pandas as pd
    cols = {}
    for c in df.columns:
        s, dt = df[c], df.schema[c]
        if isinstance(dt, pl.List):
            s = s.cast(pl.List(pl.Utf8)).list.join("|")
        elif isinstance(dt, (pl.Struct, pl.Object)):
            s = s.cast(pl.Utf8)
        elif dt == pl.Categorical:
            s = s.cast(pl.Utf8)
        if s.dtype == pl.Utf8:
            s = s.fill_null("")
        cols[c] = s.to_numpy()
    return pd.DataFrame(cols)


_DROP = object()  # "leave this entry out" marker of :func:`_anndata_safe`
_SCALARS = (str, bool, int, float, np.generic)


def _anndata_safe(v):
    """Convert one ``uns`` value to types the h5ad format can store, or ``_DROP``.

    Polars tables become pandas DataFrames; ``None`` is dropped; a list/tuple of scalars becomes
    a numpy array; a ragged list of tuples (the recorded ``ops``, of any tuple length) becomes a
    list of ``"|"``-joined strings; other sequences become a dict keyed by position; dict keys
    become ``str``; object arrays become string arrays. Numpy arrays, scipy sparse matrices,
    pandas objects and plain scalars pass through. Anything else is stored as ``str(value)``.
    """
    if v is None:
        return _DROP
    if isinstance(v, pl.DataFrame):
        return _to_pandas_safe(v)
    if isinstance(v, pl.Series):
        return _anndata_safe(v.to_numpy())
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            c = _anndata_safe(x)
            if c is not _DROP:
                out[str(k)] = c
        return out
    if isinstance(v, (set, frozenset)):
        return _anndata_safe(sorted(v, key=str))
    if isinstance(v, (list, tuple)):
        items = [x for x in v if x is not None]
        if all(isinstance(x, _SCALARS) for x in items):
            if any(isinstance(x, str) for x in items):
                return np.array([str(x) for x in items], dtype=str)
            return np.asarray(items)
        if all(isinstance(x, (list, tuple, *_SCALARS)) and
               (isinstance(x, _SCALARS) or all(isinstance(y, _SCALARS) for y in x if y is not None))
               for x in items):
            return np.array(["|".join(str(y) for y in x if y is not None) if isinstance(x, (list, tuple)) else str(x)
                             for x in items], dtype=str)
        return _anndata_safe({str(i): x for i, x in enumerate(items)})
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return np.array([str(x) for x in v.ravel()], dtype=str).reshape(v.shape)
        return v
    if isinstance(v, _SCALARS):
        return v
    if isinstance(v, Path):
        return str(v)
    mod = type(v).__module__
    if mod.startswith(("scipy.sparse", "pandas")):
        return v
    return str(v)


def read_parquet(path: Union[str, Path], chunk_nnz: int = DEFAULT_CHUNK_NNZ) -> BioFrame:
    """Open a Parquet dataset written by :meth:`BioFrame.write_parquet` (out-of-core)."""
    path = Path(path)
    obs = pl.read_parquet(path / "obs.parquet")
    var = pl.read_parquet(path / "var.parquet")
    bf = BioFrame._with_store(ParquetStore(path, obs.height, var.height), obs, var, keep_ids=True)
    bf.chunk_nnz = chunk_nnz
    return bf
