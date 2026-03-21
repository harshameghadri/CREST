import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path
# from .core import BioFrame # Removed as per diff

def _get_lib_path() -> Path:
    """Finds the compiled shared library (.so, .pyd, .dylib) dynamically"""
    parent = Path(__file__).parent
    
    for file in parent.iterdir():
        if file.name.startswith("biopolars") and file.suffix in [".so", ".pyd", ".dylib"]:
            return file
            
    # Fallback to standard name
    return parent / "biopolars.abi3.so"

lib = _get_lib_path()

@pl.api.register_expr_namespace("bio")
class BioPolarsExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def normalize_cpm(self, cell_id_col: pl.Expr, target_sum: float = 10_000.0) -> pl.Expr:
        """
        Normalize counts to a target sum per cell (default 10,000 / CP10k).
        Utilizes Polars native `.over()` syntax for maximum parallel performance.
        """
        return register_plugin_function(
            args=[self._expr, cell_id_col],
            plugin_path=lib,
            function_name="normalize_cpm",
            is_elementwise=True,
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

    def deseq2(self, size_factors: pl.Expr, design_matrix: pl.Expr, num_covariates: pl.Expr, dispersion: pl.Expr) -> pl.Expr:
        """
        Fit a Negative Binomial GLM using an Iteratively Reweighted Least Squares (IRLS) solver natively in Rust.
        Outputs a List of Beta coefficients for each gene.
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

    def svd(self, gene_id_col: pl.Expr, count_col: pl.Expr, n_cells: int, n_genes: int, n_comps: int = 50) -> pl.Expr:
        """
        Computes Truncated Randomized SVD directly in Rust natively over Sparse Arrow Matrices,
        bypassing Scipy and Python's GIL completely.
        Expected usage: df.group_by("cell_id").agg(pl.col("cell_id").bio.svd(pl.col("gene_id"), pl.col("count"), n_cells=..., n_genes=...))
        """
        # lib = Path(__file__).parent / "biopolars.abi3.so" # This line is removed as `lib` is now global
            
        return register_plugin_function(
            args=[
                self._expr.cast(pl.List(pl.UInt32)),   # cell_ids 
                gene_id_col.cast(pl.List(pl.UInt32)),  # gene_ids
                count_col.cast(pl.List(pl.Float32)),   # counts
                pl.lit(n_cells).cast(pl.UInt32), # metadata needed to construct the CSR shape internally
                pl.lit(n_genes).cast(pl.UInt32),
                pl.lit(n_comps).cast(pl.UInt32)
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

    def louvain(self, n_neighbors: int = 15) -> pl.Expr:
        """
        Calculates Louvain graph clustering community partition assignments directly from the Dense PCA coordinates.
        This must be called immediately after `.bio.svd()`. Returns UInt32 cluster IDs.
        """
        return register_plugin_function(
            args=[
                self._expr, # PCA coords `List(Float32)`
                pl.lit(n_neighbors).cast(pl.UInt32)
            ],
            plugin_path=lib,
            function_name="louvain_clustering",
            is_elementwise=False
        )

    def qc_total_counts(self, cell_id_col: pl.Expr) -> pl.Expr:
        """Per-cell total counts (sum of all gene counts per cell)."""
        return register_plugin_function(
            args=[self._expr, cell_id_col],
            plugin_path=lib,
            function_name="qc_total_counts",
            is_elementwise=True,
        )

    def qc_n_genes(self, cell_id_col: pl.Expr) -> pl.Expr:
        """Per-cell number of expressed genes (count > 0)."""
        return register_plugin_function(
            args=[self._expr, cell_id_col],
            plugin_path=lib,
            function_name="qc_n_genes",
            is_elementwise=True,
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
            is_elementwise=True,
        )

    def scale(self, gene_id_col: pl.Expr, n_obs: int, max_value: float = 10.0) -> pl.Expr:
        """Zero-center and scale to unit variance per gene, with clipping."""
        return register_plugin_function(
            args=[
                self._expr,
                gene_id_col,
                pl.lit(n_obs).cast(pl.UInt32),
                pl.lit(max_value).cast(pl.Float32),
            ],
            plugin_path=lib,
            function_name="scale",
            is_elementwise=True,
        )

    def score_genes(self, cell_id_col: pl.Expr, gene_id_col: pl.Expr, gene_set: list) -> pl.Expr:
        """Score a gene set per cell (mean_set - mean_background), like scanpy.tl.score_genes."""
        return register_plugin_function(
            args=[
                self._expr,
                cell_id_col,
                gene_id_col,
                pl.lit(pl.Series("gene_set", gene_set, dtype=pl.List(pl.UInt32))),
            ],
            plugin_path=lib,
            function_name="score_genes",
            is_elementwise=True,
        )
