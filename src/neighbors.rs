use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use crate::knn::hnsw_knn;

/// Unpack PCA coordinates from either:
/// (a) a normal multi-row DataFrame (n_cells rows, each is List(f32))
/// (b) a 1-row aggregate plugin output (1 row, value is List(List(f32)))
fn unpack_pca(pca_coords: &ListChunked) -> PolarsResult<Vec<Vec<f32>>> {
    let is_nested = pca_coords.get_as_series(0)
        .map(|s| s.list().is_ok())
        .unwrap_or(false);

    if is_nested {
        let inner_series = pca_coords.get_as_series(0)
            .ok_or_else(|| PolarsError::ComputeError("Empty PCA input".into()))?;
        let inner_list = inner_series.list()?;
        let mut data = Vec::with_capacity(inner_list.len());
        for opt_row in inner_list.into_iter() {
            match opt_row {
                Some(row) => data.push(row.f32()?.into_no_null_iter().collect()),
                None => data.push(vec![]),
            }
        }
        Ok(data)
    } else {
        let mut data = Vec::with_capacity(pca_coords.len());
        for opt_row in pca_coords.into_iter() {
            match opt_row {
                Some(row) => data.push(row.f32()?.into_no_null_iter().collect()),
                None => data.push(vec![]),
            }
        }
        Ok(data)
    }
}

/// Output type: List(List(Float32)) — each cell gets a list of [neighbor_idx, distance] pairs
fn neighbors_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "neighbors".into(),
        DataType::List(Box::new(DataType::List(Box::new(DataType::Float32)))),
    ))
}

/// Output type: List(Float32) — flat COO triplets [cell_i, cell_j, weight, ...]
fn connectivities_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "connectivities".into(),
        DataType::List(Box::new(DataType::Float32)),
    ))
}

/// compute_neighbors: Build K-nearest neighbor graph from PCA coordinates using HNSW.
///
/// Returns List(List(Float32)): for each cell, a flattened list of
/// [neighbor_0_idx, neighbor_0_dist, neighbor_1_idx, neighbor_1_dist, ...]
#[polars_expr(output_type_func=neighbors_output)]
fn compute_neighbors(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "compute_neighbors requires 2 inputs: [pca_coords, n_neighbors]".into(),
        ));
    }

    let pca_coords = inputs[0].list()?;
    let k = inputs[1].u32()?.get(0).unwrap_or(15) as usize;
    let raw_data = unpack_pca(pca_coords)?;

    let n_cells = raw_data.len();
    if n_cells == 0 {
        let empty = Series::new("neighbors".into(), Vec::<Series>::new());
        return Ok(Series::new("neighbors".into(), vec![empty]));
    }

    // Query KNN for each cell (true row indices, self excluded)
    let knn = hnsw_knn(&raw_data, k);
    let mut cell_neighbors: Vec<Series> = Vec::with_capacity(n_cells);
    for (indices, dists) in knn {
        let mut flat: Vec<f32> = Vec::with_capacity(indices.len() * 2);
        for (j, d) in indices.into_iter().zip(dists) {
            flat.push(j as f32);
            flat.push(d);
        }
        cell_neighbors.push(Series::new("".into(), flat));
    }

    let outer = Series::new("neighbors".into(), cell_neighbors);
    let wrapper = Series::new("neighbors".into(), vec![outer]);
    Ok(wrapper)
}

/// compute_connectivities: Build UMAP-style fuzzy simplicial set from PCA coords using HNSW.
///
/// Returns List(Float32): sparse connectivities as flat COO triplets [cell_i, cell_j, weight, ...]
#[polars_expr(output_type_func=connectivities_output)]
fn compute_connectivities(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "compute_connectivities requires 2 inputs: [pca_coords, n_neighbors]".into(),
        ));
    }

    let pca_coords = inputs[0].list()?;
    let k = inputs[1].u32()?.get(0).unwrap_or(15) as usize;
    let raw_data = unpack_pca(pca_coords)?;

    let n_cells = raw_data.len();
    if n_cells == 0 {
        let empty = Series::new("connectivities".into(), Vec::<Series>::new());
        return Ok(Series::new("connectivities".into(), vec![empty]));
    }

    // Collect KNN for all cells (true row indices, self excluded)
    let mut all_kth_dists: Vec<f32> = Vec::with_capacity(n_cells);
    let mut all_neighbors: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_cells);
    for (indices, dists) in hnsw_knn(&raw_data, k) {
        let cell_nn: Vec<(usize, f32)> = indices.into_iter().zip(dists).collect();
        all_kth_dists.push(cell_nn.last().map(|x| x.1).unwrap_or(1.0));
        all_neighbors.push(cell_nn);
    }

    // Build COO triplets with Gaussian kernel weights
    let mut triplets: Vec<f32> = Vec::new();
    for (i, neighbors) in all_neighbors.iter().enumerate() {
        let sigma = all_kth_dists[i].max(1e-6);
        for &(j, dist) in neighbors {
            if i != j {
                let weight = (-dist * dist / (2.0 * sigma * sigma)).exp();
                triplets.push(i as f32);
                triplets.push(j as f32);
                triplets.push(weight);
            }
        }
    }

    let inner = Series::new("".into(), triplets);
    let wrapper = Series::new("connectivities".into(), vec![inner]);
    Ok(wrapper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_pca_flat() {
        // Simulate a flat List(Float32) input
        let s1 = Series::new("".into(), vec![1.0f32, 2.0, 3.0]);
        let s2 = Series::new("".into(), vec![4.0f32, 5.0, 6.0]);

        let chunked = Series::new("pca".into(), vec![s1, s2]);
        let list_ca = chunked.list().unwrap();

        let result = unpack_pca(list_ca).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1], vec![4.0, 5.0, 6.0]);
    }
}
