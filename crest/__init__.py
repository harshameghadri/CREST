"""CREST — Columnar Rust Engine for Single-cell Transcriptomics.

Scanpy-style API (``crest.pp`` / ``crest.tl``) over :class:`BioFrame`, backed by
native Rust kernels.
"""

__version__ = "0.2.0"

from .core import BioFrame, CSRStore, FrameStore, ParquetStore, read_parquet
from . import pp, tl, io
from .io import read_10x_h5, read_10x_mtx, read_h5ad

__all__ = ["BioFrame", "CSRStore", "FrameStore", "ParquetStore", "pp", "tl", "io",
           "read_10x_h5", "read_10x_mtx", "read_h5ad", "read_parquet", "__version__"]
