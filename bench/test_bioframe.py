import polars as pl
from biopolars import BioFrame

def main():
    print("Initializing BioFrame Test...")

    # 1. Create a massive mock lazy expression matrix (X)
    X = pl.LazyFrame({
        "cell_id": [1, 1, 2, 2, 3, 3],
        "gene_id": [100, 200, 100, 300, 200, 300],
        "count": [5.0, 1.0, 3.0, 2.0, 4.0, 6.0]
    })

    # 2. Create Cell Metadata (obs)
    obs = pl.DataFrame({
        "cell_id": [1, 2, 3],
        "cluster": ["T-Cell", "B-Cell", "T-Cell"],
        "batch": ["A", "A", "B"]
    })

    # 3. Create Gene Metadata (var)
    var = pl.DataFrame({
        "gene_id": [100, 200, 300],
        "gene_name": ["CD4", "CD8A", "MS4A1"]
    })

    # 4. Instantiate BioFrame
    adata = BioFrame(X=X, obs=obs, var=var)
    
    print("\nOriginal X Matrix (Lazy Plan):")
    print(adata.X.explain())

    # 5. Filter for T-Cells ONLY using metadata
    # In Scanpy, this forces indexing the massive X matrix in-memory.
    # In BioPolars, this creates a Zero-Copy SemiJoin execution plan.
    t_cells = adata.filter_cells(pl.col("cluster") == "T-Cell")
    
    print("\nFiltered X Matrix (Optimized SemiJoin Plan):")
    print(t_cells.X.explain())
    
    # Execute the plan
    print("\nExecuted Filtered X Results:")
    print(t_cells.X.collect())
    
    # 6. Ensure metadata is properly sliced
    print("\nSliced obs Metadata:")
    print(t_cells.obs)

if __name__ == "__main__":
    main()
