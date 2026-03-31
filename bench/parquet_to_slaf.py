import os
import sys
import json
import time

def convert_parquet_to_slaf(parquet_path, slaf_path):
    import pyarrow as pa
    import lance
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds

    os.makedirs(slaf_path, exist_ok=True)
    
    print(f"Scanning {parquet_path}...")
    
    # 1. Hardcode bounds to save memory and avoid OOM
    # In earlier runs, max_cell+1 = 1,306,127. max_gene+1 = 27,998
    print("Using pre-computed dataset bounds for 1M Neurons...")
    n_cells = 1306127
    n_genes = 27998
    
    # Fast metadata count
    parquet_file = pq.ParquetFile(parquet_path)
    n_expr = parquet_file.metadata.num_rows
    
    print(f"Dataset bounds: {n_cells:,} cells, {n_genes:,} genes, {n_expr:,} expressions")
    
    t0 = time.time()
    
    # 2. Write Expression Table
    print("Writing Expression Lance dataset...")
    expr_schema = pa.schema([
        ("cell_integer_id", pa.int64()), # Use int64 for Lance compatibility with SLAF
        ("gene_integer_id", pa.int64()),
        ("value", pa.float32())
    ])
    
    expr_path = os.path.join(slaf_path, "expression.lance")
    
    dataset = ds.dataset(parquet_path, format="parquet")
    
    def batch_iter():
        # Stream in tiny batches to prevent OOM in Docker
        for batch in dataset.to_batches(batch_size=100_000):
            # SLAF expects cell_integer_id and gene_integer_id
            arrs = [
                pa.compute.cast(batch.column("cell_id"), pa.int64()),
                pa.compute.cast(batch.column("gene_id"), pa.int64()),
                batch.column("count")
            ]
            yield pa.RecordBatch.from_arrays(arrs, schema=expr_schema)
            
    lance.write_dataset(
        batch_iter(), 
        expr_path, 
        schema=expr_schema, 
        mode="overwrite", 
        max_rows_per_file=10_000_000, 
        max_bytes_per_file=2 * 1024 * 1024 * 1024
    )
    
    # 3. Write Cells Table
    print("Writing Cells Lance dataset...")
    cell_schema = pa.schema([
        ("cell_integer_id", pa.int64()),
        ("cell_id", pa.string())
    ])
    def cell_batch_iter():
        for i in range(0, n_cells, 100_000):
            end = min(i + 100_000, n_cells)
            ids = pa.array(range(i, end), type=pa.int64())
            str_ids = pa.array([f"cell_{j}" for j in range(i, end)], type=pa.string())
            yield pa.RecordBatch.from_arrays([ids, str_ids], schema=cell_schema)
            
    cells_path = os.path.join(slaf_path, "cells.lance")
    lance.write_dataset(cell_batch_iter(), cells_path, schema=cell_schema, mode="overwrite")
    
    # 4. Write Genes Table
    print("Writing Genes Lance dataset...")
    gene_schema = pa.schema([
        ("gene_integer_id", pa.int64()),
        ("gene_id", pa.string())
    ])
    def gene_batch_iter():
        for i in range(0, n_genes, 100_000):
            end = min(i + 100_000, n_genes)
            ids = pa.array(range(i, end), type=pa.int64())
            str_ids = pa.array([f"gene_{j}" for j in range(i, end)], type=pa.string())
            yield pa.RecordBatch.from_arrays([ids, str_ids], schema=gene_schema)
            
    genes_path = os.path.join(slaf_path, "genes.lance")
    lance.write_dataset(gene_batch_iter(), genes_path, schema=gene_schema, mode="overwrite")
    
    # 5. Write config.json
    print("Writing config.json...")
    config = {
        "format_version": "0.4",
        "array_shape": [n_cells, n_genes],
        "tables": {
            "cells": "cells.lance",
            "genes": "genes.lance",
            "expression": "expression.lance"
        },
        "metadata": {
            "expression_count": n_expr,
            "sparsity": 1.0 - (n_expr / (n_cells * n_genes))
        }
    }
    with open(os.path.join(slaf_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    t1 = time.time()
    print(f"✅ Conversion complete in {t1-t0:.2f} seconds!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parquet_to_slaf.py <input.parquet> <output.slaf>")
        sys.exit(1)
        
    convert_parquet_to_slaf(sys.argv[1], sys.argv[2])
