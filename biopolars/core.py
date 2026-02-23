import polars as pl
from dataclasses import dataclass
from typing import Optional

@dataclass
class BioFrame:
    """
    A lightweight wrapper ensuring AnnData-like properties for streaming BioPolars pipelines.
    Designed to prevent 'COO Memory Bloat' by aggressively keeping metadata (`obs`, `var`) 
    mathematically distinct from the expression matrix (`X`).
    """
    X: pl.LazyFrame
    obs: Optional[pl.DataFrame] = None
    var: Optional[pl.DataFrame] = None

    def filter_cells(self, condition: pl.Expr) -> 'BioFrame':
        """
        Filters cells based on metadata conditions.
        Crucially, this performs an optimized `semi_join` rather than a memory-exploding `inner_join`.
        """
        if self.obs is None:
            raise ValueError("No `obs` metadata available to filter on.")
        
        # 1. Filter the metadata down to allowed cells
        valid_obs = self.obs.filter(condition)
        
        # 2. Semi-Join the massive streaming expression matrix to keep only the valid cells
        # Semi-joins prevent the metadata columns from physically attaching to X (saving RAM)
        filtered_X = self.X.join(
            valid_obs.lazy().select("cell_id"), 
            on="cell_id", 
            how="semi"
        )
        
        return BioFrame(X=filtered_X, obs=valid_obs, var=self.var)

    def filter_genes(self, condition: pl.Expr) -> 'BioFrame':
        """Filters genes based on metadata conditions via Zero-Copy SemiJoin."""
        if self.var is None:
            raise ValueError("No `var` metadata available to filter on.")
            
        valid_var = self.var.filter(condition)
        
        filtered_X = self.X.join(
            valid_var.lazy().select("gene_id"),
            on="gene_id",
            how="semi"
        )
        
        return BioFrame(X=filtered_X, obs=self.obs, var=valid_var)
