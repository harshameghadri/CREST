"""CREST — Columnar Rust Engine for Single-cell Transcriptomics.

Scanpy-style API (``crest.pp`` / ``crest.tl``) over :class:`BioFrame`, backed by
native Rust kernels, plus a ``.bio`` Polars expression namespace.
"""

__version__ = "0.2.0"

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

from .core import BioFrame, CSRStore, FrameStore, ParquetStore, read_parquet
from . import pp, tl, io
from .io import read_10x_h5, read_10x_mtx, read_h5ad

__all__ = ["BioFrame", "CSRStore", "FrameStore", "ParquetStore", "pp", "tl", "io",
           "read_10x_h5", "read_10x_mtx", "read_h5ad", "read_parquet", "__version__"]


def _get_lib_path() -> Path:
    """Locate the compiled extension (.so / .pyd / .dylib)."""
    parent = Path(__file__).parent
    for file in parent.iterdir():
        if file.name.startswith("crest") and file.suffix in [".so", ".pyd", ".dylib"]:
            return file
    return parent / "crest.abi3.so"


lib = _get_lib_path()


@pl.api.register_expr_namespace("bio")
class CrestExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def normalize_cpm(self, cell_id_col: pl.Expr, target_sum: float = 10_000.0) -> pl.Expr:
        """Normalize counts to a target sum per cell (default 10,000 / CP10k)."""
        return register_plugin_function(
            args=[self._expr, cell_id_col, pl.lit(target_sum).cast(pl.Float64)],
            plugin_path=lib,
            function_name="normalize_cpm",
            # Aggregates over all rows of a cell/gene: must see the whole column,
            # so Polars may not split it into batches.
            is_elementwise=False,
        )

    def log1p(self) -> pl.Expr:
        """Calculate ln(1+x) using the native Rust Polar extension."""
        return register_plugin_function(
            args=[self._expr],
            plugin_path=lib,
            function_name="log1p",
            is_elementwise=True,
        )

    def wilcoxon(self, other_expr: pl.Expr) -> pl.Expr:
        """Calculate Tie-Corrected Mann-Whitney U test natively in Rust."""
        return register_plugin_function(
            args=[self._expr.cast(pl.List(pl.Float32)), other_expr.cast(pl.List(pl.Float32))],
            plugin_path=lib,
            function_name="wilcoxon_rank_sum",
            is_elementwise=True
        )

    def nb_glm(self, size_factors: pl.Expr, design_matrix: pl.Expr, num_covariates: pl.Expr, dispersion: pl.Expr) -> pl.Expr:
        """
        Negative-binomial GLM fit (log link) by IRLS with a *given* dispersion per gene.

        Each row is one gene: counts and size factors are lists over samples, the
        design matrix is flattened row-major. Returns the coefficients, or null when
        IRLS does not converge. This is the GLM core of DESeq2-style pseudobulk
        tests; size factors and dispersions must be estimated separately.
        """
        return register_plugin_function(
            args=[
                self._expr.cast(pl.List(pl.Float32)), 
                size_factors.cast(pl.List(pl.Float32)),
                design_matrix.cast(pl.List(pl.Float32)),
                num_covariates.cast(pl.UInt32),
                dispersion.cast(pl.Float32)
            ],
            plugin_path=lib,
            function_name="deseq2_irls",
            is_elementwise=True
        )

    def deseq2(self, size_factors: pl.Expr, design_matrix: pl.Expr, num_covariates: pl.Expr, dispersion: pl.Expr) -> pl.Expr:
        """Deprecated alias of :meth:`nb_glm` (this is not a full DESeq2 implementation)."""
        import warnings
        warnings.warn("bio.deseq2() is renamed bio.nb_glm(); it fits an NB GLM with given dispersions, "
                      "not the full DESeq2 procedure", DeprecationWarning, stacklevel=2)
        return self.nb_glm(size_factors, design_matrix, num_covariates, dispersion)

    def svd(self, gene_id_col: pl.Expr, count_col: pl.Expr, n_cells: int, n_genes: int, n_comps: int = 50,
            n_iter: int = 7, seed: int = 42) -> pl.Expr:
        """
        Mean-centred PCA via randomized subspace iteration (Halko et al. 2011, Alg. 4.4),
        computed natively in Rust over sparse COO triplets without densifying.

        Accuracy matches ``sklearn.utils.extmath.randomized_svd`` at the same ``n_iter``;
        raise ``n_iter`` (e.g. 15) to converge trailing PCs towards the exact (ARPACK) solution.
        Expected usage: df.select(pl.all().implode()).select(
            pl.col("cell_id").bio.svd(pl.col("gene_id"), pl.col("count"), n_cells=..., n_genes=...))
        """
        return register_plugin_function(
            args=[
                self._expr.cast(pl.List(pl.UInt32)),   # cell_ids
                gene_id_col.cast(pl.List(pl.UInt32)),  # gene_ids
                count_col.cast(pl.List(pl.Float32)),   # counts
                pl.lit(n_cells).cast(pl.UInt32), # metadata needed to construct the CSR shape internally
                pl.lit(n_genes).cast(pl.UInt32),
                pl.lit(n_comps).cast(pl.UInt32),
                pl.lit(n_iter).cast(pl.UInt32),
                pl.lit(seed).cast(pl.UInt64),
            ],
            plugin_path=lib,
            function_name="sparse_randomized_svd",
            is_elementwise=False
        )

    def umap(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, spread: float = 1.0, n_epochs: int = 200, spectral_n_iter: int = 50) -> pl.Expr:
        """
        Calculates UMAP dimensionality reduction on the dense PCA coordinates.
        This must be called immediately after `.bio.svd()`.
        """
        return register_plugin_function(
            args=[
                self._expr, # PCA coords `List(Float32)`
                pl.lit(n_components).cast(pl.UInt32),
                pl.lit(n_neighbors).cast(pl.UInt32),
                pl.lit(min_dist).cast(pl.Float32),
                pl.lit(spread).cast(pl.Float32),
                pl.lit(n_epochs).cast(pl.UInt32),
                pl.lit(spectral_n_iter).cast(pl.UInt32)
            ],
            plugin_path=lib,
            function_name="native_umap",
            is_elementwise=False
        )

    def leiden(self, n_neighbors: int = 15, resolution: float = 1.0, n_iterations: int = 2, seed: int = 0) -> pl.Expr:
        """
        Leiden community detection from PCA coordinates (Traag et al. 2019).

        Builds the UMAP fuzzy-simplicial-set kNN graph (as scanpy's ``pp.neighbors``;
        ``n_neighbors`` counts the cell itself) and optimises modularity with the
        given resolution. Returns UInt32 cluster IDs.
        """
        return register_plugin_function(
            args=[
                self._expr, # PCA coords `List(Float32)`
                pl.lit(n_neighbors).cast(pl.UInt32),
                pl.lit(resolution).cast(pl.Float32),
                pl.lit(n_iterations).cast(pl.UInt32),
                pl.lit(seed).cast(pl.UInt64),
            ],
            plugin_path=lib,
            function_name="leiden_clustering",
            is_elementwise=False
        )

    def louvain(self, n_neighbors: int = 15) -> pl.Expr:
        """Deprecated: use .bio.leiden() (Leiden refines Louvain partitions)."""
        import warnings
        warnings.warn("louvain() is deprecated, use leiden() instead", DeprecationWarning, stacklevel=2)
        return self.leiden(n_neighbors=n_neighbors, resolution=1.0)

    def qc_total_counts(self, cell_id_col: pl.Expr) -> pl.Expr:
        """Per-cell total counts (sum of all gene counts per cell)."""
        return register_plugin_function(
            args=[self._expr, cell_id_col],
            plugin_path=lib,
            function_name="qc_total_counts",
            # Aggregates over all rows of a cell/gene: must see the whole column,
            # so Polars may not split it into batches.
            is_elementwise=False,
        )

    def qc_n_genes(self, cell_id_col: pl.Expr) -> pl.Expr:
        """Per-cell number of expressed genes (count > 0)."""
        return register_plugin_function(
            args=[self._expr, cell_id_col],
            plugin_path=lib,
            function_name="qc_n_genes",
            # Aggregates over all rows of a cell/gene: must see the whole column,
            # so Polars may not split it into batches.
            is_elementwise=False,
        )

    def filter_cells(self, cell_id_col: pl.Expr, min_genes: int = 200, min_counts: float = 0.0) -> pl.Expr:
        """Filter cells by minimum gene count and total counts thresholds. Returns boolean mask."""
        return register_plugin_function(
            args=[
                self._expr,
                cell_id_col,
                pl.lit(min_genes).cast(pl.UInt32),
                pl.lit(min_counts).cast(pl.Float32),
            ],
            plugin_path=lib,
            function_name="filter_cells",
            # Aggregates over all rows of a cell/gene: must see the whole column,
            # so Polars may not split it into batches.
            is_elementwise=False,
        )

    def scale(self, gene_id_col: pl.Expr, n_obs: int, max_value: float = 10.0) -> pl.Expr:
        """Scale the *stored* (non-zero) entries per gene: (x - mean) / std, clipped.

        Means/stds count implicit zeros (``n_obs`` cells), but implicit zeros are
        not materialised, so the output is not the dense scaled matrix. For PCA on
        scaled data use ``crest.pp.scale`` + ``crest.tl.pca``, which is exact.
        """
        return register_plugin_function(
            args=[
                self._expr,
                gene_id_col,
                pl.lit(n_obs).cast(pl.UInt32),
                pl.lit(max_value).cast(pl.Float32),
            ],
            plugin_path=lib,
            function_name="scale",
            # Aggregates over all rows of a cell/gene: must see the whole column,
            # so Polars may not split it into batches.
            is_elementwise=False,
        )

    def score_genes(self, cell_id_col: pl.Expr, gene_id_col: pl.Expr, gene_set: list) -> pl.Expr:
        """Per-cell mean of stored values in ``gene_set`` minus mean of the other stored values.

        A quick signature score. For the scanpy algorithm (expression-matched
        control genes, zeros included) use ``crest.tl.score_genes``.
        """
        return register_plugin_function(
            args=[
                self._expr,
                cell_id_col,
                gene_id_col,
                pl.lit(pl.Series("gene_set", gene_set, dtype=pl.List(pl.UInt32))),
            ],
            plugin_path=lib,
            function_name="score_genes",
            # Aggregates over all rows of a cell/gene: must see the whole column,
            # so Polars may not split it into batches.
            is_elementwise=False,
        )

    def rank_genes_groups(self, cell_id_col: pl.Expr, gene_id_col: pl.Expr, group_col: pl.Expr, target_group: int = 1) -> pl.Expr:
        """
        Differential expression (target group vs all other labelled cells) via Welch's
        t-test with Benjamini-Hochberg FDR, matching scanpy ``method="t-test"`` statistics.
        For all groups at once and Wilcoxon, use ``crest.tl.rank_genes_groups``.
        Returns List(List(Float32)): each inner list = [gene_id, t_stat, p_value, adj_p_value, log2_fc].
        Sorted by adjusted p-value (most significant first).
        """
        return register_plugin_function(
            args=[
                self._expr,
                cell_id_col,
                gene_id_col,
                group_col.cast(pl.UInt32),
                pl.lit(target_group).cast(pl.UInt32),
            ],
            plugin_path=lib,
            function_name="rank_genes_groups",
            is_elementwise=False,
        )

    def neighbors(self, n_neighbors: int = 15) -> pl.Expr:
        """
        Build K-nearest neighbor graph from PCA coordinates.
        Returns per-cell neighbor indices and distances as [idx0, dist0, idx1, dist1, ...].
        """
        return register_plugin_function(
            args=[
                self._expr,
                pl.lit(n_neighbors).cast(pl.UInt32),
            ],
            plugin_path=lib,
            function_name="compute_neighbors",
            is_elementwise=False,
        )

    def connectivities(self, n_neighbors: int = 15) -> pl.Expr:
        """
        Build UMAP-style fuzzy simplicial set connectivities from PCA coordinates.
        Returns List(Float32): flat COO triplets [cell_i, cell_j, weight, ...] (stride=3)
        with Gaussian kernel weights.
        """
        return register_plugin_function(
            args=[
                self._expr,
                pl.lit(n_neighbors).cast(pl.UInt32),
            ],
            plugin_path=lib,
            function_name="compute_connectivities",
            is_elementwise=False,
        )
