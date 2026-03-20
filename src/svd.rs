use nalgebra_sparse::coo::CooMatrix;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
// use single_svdlib::*; // We'll implement this iteratively once the CSR matrix is built.

#[polars_expr(output_type=Float32)]
fn sparse_randomized_svd(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 6 {
        return Err(PolarsError::ComputeError("sparse_randomized_svd requires exactly 6 inputs: [groups, genes, counts, n_cells, n_genes, n_comps]".into()));
    }

    let cell_ids = &inputs[0].list()?;
    let gene_ids = &inputs[1].list()?;
    let counts = &inputs[2].list()?;
    
    // Scalar variables passed via pl.lit().cast()
    let n_cells_series = &inputs[3].u32()?;
    let n_genes_series = &inputs[4].u32()?;
    let n_comps_series = &inputs[5].u32()?;

    let n_cells = n_cells_series.get(0).unwrap_or(0) as usize;
    let n_genes = n_genes_series.get(0).unwrap_or(0) as usize;
    let _n_comps = n_comps_series.get(0).unwrap_or(50) as usize;
    
    if n_cells == 0 || n_genes == 0 {
        return Err(PolarsError::ComputeError("Invalid Sparse array dimensions.".into()));
    }

    // 1. Reconstruct the CooMatrix natively in Rust
    let mut coo = CooMatrix::new(n_cells, n_genes);
    
    // We expect a single list per group containing all the data 
    // (since the user grouped by a dummy variable to pass the whole matrix)
    for ((opt_cells, opt_genes), opt_counts) in cell_ids.into_iter().zip(gene_ids.into_iter()).zip(counts.into_iter()) {
        if let (Some(c), Some(g), Some(v)) = (opt_cells, opt_genes, opt_counts) {
            let cells_ca = c.u32()?;
            let genes_ca = g.u32()?;
            let vals_ca = v.f32()?;
            
            for ((cell_idx, gene_idx), val) in cells_ca.into_iter().zip(genes_ca.into_iter()).zip(vals_ca.into_iter()) {
                if let (Some(row), Some(col), Some(count)) = (cell_idx, gene_idx, val) {
                    coo.push(row as usize, col as usize, count as f64);
                }
            }
        }
    }

    // Perform Truncated SVD natively in Rust
    use single_svdlib::legacy::svd_dim;
    let svd_result = svd_dim(&coo, _n_comps)
        .map_err(|e| PolarsError::ComputeError(format!("SVD failed: {:?}", e).into()))?;
    
    let k = svd_result.d;
    let ut = svd_result.ut; // Shape: (k, n_cells)
    let s = svd_result.s;   // Shape: (k,)
    
    // To properly return 2D matrices in Polars plugins, we return a `ListBuilder`
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        "pca",
        n_cells,
        k,
        DataType::Float32
    );
    
    for i in 0..n_cells {
        let mut row_vec = Vec::with_capacity(k);
        for c in 0..k {
            // PCA coordinate = U_{i,c} * S_c = U^T_{c,i} * S_c
            let val = ut[[c, i]] * s[c];
            row_vec.push(val as f32);
        }
        builder.append_slice(&row_vec);
    }
    
    Ok(builder.finish().into_series().implode()?.into_series())
}
