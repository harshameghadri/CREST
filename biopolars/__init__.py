import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

lib = Path(__file__).parent

@pl.api.register_expr_namespace("bio")
class BioPolarsExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

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
