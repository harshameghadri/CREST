use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

pub fn umap_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "umap",
        DataType::List(Box::new(DataType::List(Box::new(DataType::Float32)))),
    ))
}

#[polars_expr(output_type_func=umap_output)]
fn native_umap(inputs: &[Series]) -> PolarsResult<Series> {
    // Expected inputs:
    // 0: A List(Float32) column representing the PCA coordinates for each cell
    // 1: n_components (UInt32)
    // 2: n_neighbors (UInt32)
    // 3: min_dist (Float32)
    // 4: spread (Float32)
    // 5: n_epochs (UInt32)
    // 6: spectral_n_iter (UInt32)
    
    if inputs.len() < 3 {
        return Err(PolarsError::ComputeError("native_umap requires at least 3 inputs: [pca_coords, n_components, n_neighbors]".into()));
    }

    let pca_coords = &inputs[0].list()?;
    let n_components = inputs[1].u32()?.get(0).unwrap_or(2) as usize;
    let n_neighbors = inputs[2].u32()?.get(0).unwrap_or(15) as usize;
    
    let min_dist = if inputs.len() > 3 { inputs[3].f32()?.get(0).unwrap_or(0.1) } else { 0.1 };
    let spread = if inputs.len() > 4 { inputs[4].f32()?.get(0).unwrap_or(1.0) } else { 1.0 };
    let n_epochs = if inputs.len() > 5 { inputs[5].u32()?.get(0).unwrap_or(200) as usize } else { 200 };
    let spectral_n_iter = if inputs.len() > 6 { inputs[6].u32()?.get(0).unwrap_or(50) as usize } else { 50 };
    
    // Validate parameter bounds
    if n_components == 0 || n_components > 100 {
        return Err(PolarsError::ComputeError(
            format!("n_components must be 1-100, got {}", n_components).into()
        ));
    }
    if n_neighbors == 0 || n_neighbors > 1000 {
        return Err(PolarsError::ComputeError(
            format!("n_neighbors must be 1-1000, got {}", n_neighbors).into()
        ));
    }
    if n_epochs == 0 || n_epochs > 10_000 {
        return Err(PolarsError::ComputeError(
            format!("n_epochs must be 1-10000, got {}", n_epochs).into()
        ));
    }
    if min_dist < 0.0 || !min_dist.is_finite() {
        return Err(PolarsError::ComputeError(
            format!("min_dist must be non-negative and finite, got {}", min_dist).into()
        ));
    }
    if spread <= 0.0 || !spread.is_finite() {
        return Err(PolarsError::ComputeError(
            format!("spread must be positive and finite, got {}", spread).into()
        ));
    }

    let n_cells = pca_coords.len();
    if n_cells == 0 {
        return Err(PolarsError::ComputeError("Cannot perform UMAP on empty DataFrame".into()));
    }

    // Check if the input is a grouped (imploded) doubly nested List(List(f32))
    // This happens if `.bio.svd(...)` imploded the matrix to fit inside a single DataFrame row group.
    let mut data: Vec<Vec<f32>> = Vec::with_capacity(n_cells);
    
    // Check the data type of the inner items
    if let Some(first_row) = pca_coords.get_as_series(0).map(|s| s.list().is_ok()) {
        if first_row {
            // Unpack the doubly nested list
            let inner_series = match pca_coords.get_as_series(0) {
                Some(s) => s,
                None => return Err(PolarsError::ComputeError("UMAP input list is empty".into())),
            };
            let inner_list = inner_series.list()?;
            for opt_row in inner_list.into_iter() {
                if let Some(row_series) = opt_row {
                    let float_ca = row_series.f32()?;
                    data.push(float_ca.into_no_null_iter().collect());
                } else {
                    data.push(vec![0.0f32; n_components]); // Fallback
                }
            }
        } else {
            // Unpack flattened rows
            for opt_row in pca_coords.into_iter() {
                if let Some(row_series) = opt_row {
                    let float_ca = row_series.f32()?;
                    data.push(float_ca.into_no_null_iter().collect());
                } else {
                    data.push(vec![0.0f32; n_components]);
                }
            }
        }
    }

    // Validate unpacked data
    if data.is_empty() {
        return Err(PolarsError::ComputeError("No PCA data could be extracted from input".into()));
    }
    let pca_dims = data[0].len();
    if pca_dims == 0 {
        return Err(PolarsError::ComputeError("PCA vectors have zero dimensions".into()));
    }
    // Warn if dimensions vary (shouldn't happen with valid SVD output)
    if data.iter().any(|row| row.len() != pca_dims) {
        return Err(PolarsError::ComputeError(
            "PCA vectors have inconsistent dimensions. Check SVD output.".into()
        ));
    }
    if n_components > pca_dims {
        return Err(PolarsError::ComputeError(
            format!(
                "n_components ({}) exceeds PCA dimensionality ({}). \
                 Reduce n_components or increase SVD n_comps.",
                n_components, pca_dims
            ).into()
        ));
    }
    if n_neighbors >= data.len() {
        return Err(PolarsError::ComputeError(
            format!(
                "n_neighbors ({}) must be less than n_cells ({})",
                n_neighbors, data.len()
            ).into()
        ));
    }

    // Execute our custom Native Rust UMAP core
    let embeddings = crate::umap::core::run_umap(
        &data,
        n_components,
        n_neighbors,
        min_dist,
        spread,
        n_epochs,
        spectral_n_iter
    );
    
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        "umap",
        n_cells,
        n_components,
        DataType::Float32
    );
    
    for row in embeddings {
        builder.append_slice(&row);
    }
    
    let s_orig = builder.finish().into_series();
    let s_wrapped = polars::prelude::Series::new("umap".into(), &[polars::prelude::AnyValue::List(s_orig)]);
    Ok(s_wrapped)
}
