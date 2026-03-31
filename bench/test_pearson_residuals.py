import sys
import os
import time
import polars as pl
from biopolars.slaf_io import read_slaf_expression
from biopolars.pearson import select_highly_variable_genes
import biopolars  # Registers .bio

def main():
    slaf_path = "data/1M_neurons/1M_neurons.slaf"
    if not os.path.exists(slaf_path):
        print(f"Skipping: {slaf_path} not found")
        return

    print("📊 Testing Analytic Pearson Residuals Out-of-Core...")
    t0 = time.time()
    
    # Read subset to make test fast
    max_cells = 10000
    df, n_cells, n_genes = read_slaf_expression(slaf_path, max_cells=max_cells)
    
    # We want to do this lazily!
    lazy_df = df.lazy()
    
    # Apply Pearson Residuals
    residual_expr = pl.col("count").bio.pearson_residual(
        cell_id=pl.col("cell_id"),
        gene_id=pl.col("gene_id")
    )
    
    # Add it to the lazy frame
    lazy_df = lazy_df.with_columns(
        residual_expr.alias("pearson_residual")
    )
    
    # Find top 10 Highly Variable Genes directly using variance
    hvg_df = select_highly_variable_genes(
        lazy_df, 
        residual_col="pearson_residual",
        gene_col="gene_id",
        n_top_genes=10
    )
    
    # Execute the lazy graph
    result = hvg_df.collect()
    
    try:
        grouped = result.group_by("gene_id").agg(pl.col("pearson_residual").var().alias("variance"))
        top_genes = grouped.sort("variance", descending=True)["gene_id"].to_list()
        
        t1 = time.time()
        print(f"✅ Pearson Residuals Computed & Top 10 HVGs Found in {t1-t0:.2f}s!")
        print(f"🧬 Top 10 Gene IDs: {top_genes}")
    except Exception as e:
        print(f"❌ Failed processing HVG: {e}")

if __name__ == "__main__":
    main()
