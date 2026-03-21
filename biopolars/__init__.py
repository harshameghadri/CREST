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

    def sc_transform(self, cell_id_col: pl.Expr, size_factors_col: pl.Expr) -> pl.Expr:
        """
        Performs sctransform variance-stabilizing transformation.
        """
        return register_plugin_function(
            args=[self._expr, cell_id_col, size_factors_col.cast(pl.Float32)],
            plugin_path=lib,
            function_name="sc_transform",
            is_elementwise=True,
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
