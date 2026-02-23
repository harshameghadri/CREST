import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

lib = Path(__file__).parent

@pl.api.register_expr_namespace("bio")
class BioPolarsExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def normalize_cpm(self, cell_id_col: pl.Expr, target_sum: float = 10_000.0) -> pl.Expr:
        """
        Normalize counts to a target sum per cell (default 10,000 / CP10k).
        Utilizes Polars native `.over()` syntax for maximum parallel performance.
        """
        return (self._expr / self._expr.sum().over(cell_id_col)) * target_sum

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
