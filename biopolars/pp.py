import polars as pl
from typing import Union

def highly_variable_genes(
    lazy_df: pl.LazyFrame, 
    n_top_genes: int = 2000, 
    flavor: str = "seurat",
    total_cells: int = None
) -> pl.LazyFrame:
    """
    Identifies highly variable genes across a streaming sparse Triplet dataframe.
    
    Pain Point Solved: Instead of computing variances down dense/sparse columns
    (which destroys cache locality), we use Polars' Rust-native hash groupings.
    """
    if flavor != "seurat":
        raise NotImplementedError("Only 'seurat' flavor is currently supported.")
    
    # Calculate Mean and Variance for each gene natively in Polars
    # V(x) = E(x^2) - (E(x))^2  or we can just use Polars' built-in .var()
    # Note: .var() requires >1 observation. For sparse data, zeros are implicitly missing.
    # To compute accurate variance across ALL cells, we need the total number of cells.
    
    if total_cells is None:
        raise ValueError("total_cells must be provided to correctly calculate variance across implicit zeros.")
    
    # Calculate true mean and variance accounting for structural zeros
    # True Mean = (Sum of counts) / Total Cells
    # True Variance = Sum((x_i - mean)^2) / (Total Cells - 1)
    
    gene_stats = lazy_df.group_by("gene_id").agg([
        pl.col("count").sum().alias("sum_counts"),
        (pl.col("count") ** 2).sum().alias("sum_squares"),
        pl.len().alias("nnz_cells")  # number of non-zero expressions
    ]).with_columns([
        (pl.col("sum_counts") / total_cells).alias("mean"),
    ]).with_columns([
        # Variance formula expanding zeroes:
        # Var = ( sum_squares - 2*mean*sum_counts + total_cells * mean^2 ) / (total_cells - 1)
        ((pl.col("sum_squares") - 2 * pl.col("mean") * pl.col("sum_counts") + total_cells * (pl.col("mean") ** 2)) / (total_cells - 1)).alias("variance")
    ]).with_columns([
        # Seurat dispersion = variance / mean
        (pl.col("variance") / pl.col("mean")).alias("dispersion")
    ])
    
    # Sort by dispersion to get top genes
    top_genes = (
        gene_stats
        .sort("dispersion", descending=True)
        .limit(n_top_genes)
        .select("gene_id")
    )
    
    # Filter the original massive streaming dataframe using an exact inner join
    return lazy_df.join(top_genes, on="gene_id", how="inner")
