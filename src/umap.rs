use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float32)]
fn native_umap(inputs: &[Series]) -> PolarsResult<Series> {
    // Expected inputs:
    // 0: A List(Float32) column representing the PCA coordinates for each cell
    // 1: The number of components to reduce to (e.g. 2 or 3) (UInt32)
    // 2: n_neighbors (UInt32)
    
    if inputs.len() != 3 {
        return Err(PolarsError::ComputeError("native_umap requires exactly 3 inputs: [pca_coords, n_components, n_neighbors]".into()));
    }

    let pca_coords = &inputs[0].list()?;
    let _n_components = inputs[1].u32()?.get(0).unwrap_or(2) as usize;
    let n_neighbors = inputs[2].u32()?.get(0).unwrap_or(15) as usize;
    
    let n_cells = pca_coords.len();
    if n_cells == 0 {
        return Err(PolarsError::ComputeError("Cannot perform UMAP on empty DataFrame".into()));
    }

    use rag_umap::{convert_to_2d, convert_to_3d};
    
    // Convert Arrow List arrays into a Vec<Vec<f32>> for `rag-umap`
    let mut data: Vec<Vec<f32>> = Vec::with_capacity(n_cells);
    
    for opt_row in pca_coords.into_iter() {
        if let Some(row_series) = opt_row {
            let float_ca = row_series.f32()?;
            let row_vec: Vec<f32> = float_ca.into_no_null_iter().collect();
            data.push(row_vec);
        } else {
            // Fill missing with 0.0s, though we shouldn't have them in PCA
            data.push(vec![0.0f32; 50]); // Assuming 50 PCs fallback
        }
    }
    
    // Execute UMAP using the high-level public functions
    let embeddings_f64 = if _n_components == 3 {
        convert_to_3d(data).map_err(|e| PolarsError::ComputeError(format!("UMAP 3D failed: {:?}", e).into()))?
    } else {
        convert_to_2d(data).map_err(|e| PolarsError::ComputeError(format!("UMAP 2D failed: {:?}", e).into()))?
    };
    
    // We get back Vec<Vec<f64>>, convert down to Float32 Series
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        "umap",
        n_cells,
        embeddings_f64.first().map(|r| r.len()).unwrap_or(2),
        DataType::Float32
    );
    
    for row in embeddings_f64 {
        let row_f32: Vec<f32> = row.into_iter().map(|v| v as f32).collect();
        builder.append_slice(&row_f32);
    }
    
    Ok(builder.finish().into_series())
}
