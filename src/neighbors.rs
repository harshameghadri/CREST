use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use instant_distance::{Builder, Hnsw, Point, Search};

/// PCA point for HNSW distance computation
#[derive(Clone, Debug)]
struct PcaPoint(Vec<f32>);

impl Point for PcaPoint {
    fn distance(&self, other: &Self) -> f32 {
        self.0.iter().zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

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

    let k = k.min(n_cells - 1);

    // Build HNSW index
    let points: Vec<PcaPoint> = raw_data.iter().map(|v| PcaPoint(v.clone())).collect();
    let hnsw: Hnsw<PcaPoint> = Builder::default()
        .ef_construction(200)
        .build(points.iter().cloned(), points.len());

    // Query KNN for each cell
    let mut cell_neighbors: Vec<Series> = Vec::with_capacity(n_cells);
    let mut search = Search::default();

    for (i, point) in points.iter().enumerate() {
        search = hnsw.search(point, &mut search);
        let mut flat: Vec<f32> = Vec::with_capacity(k * 2);
        for item in search.iter().take(k + 1) {
            let j = item.pid.into_inner();
            if j != i {
                flat.push(j as f32);
                flat.push(item.distance);
            }
            if flat.len() >= k * 2 {
                break;
            }
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

    let k = k.min(n_cells - 1);

    // Build HNSW index
    let points: Vec<PcaPoint> = raw_data.iter().map(|v| PcaPoint(v.clone())).collect();
    let hnsw: Hnsw<PcaPoint> = Builder::default()
        .ef_construction(200)
        .build(points.iter().cloned(), points.len());

    // Collect KNN for all cells
    let mut all_kth_dists: Vec<f32> = Vec::with_capacity(n_cells);
    let mut all_neighbors: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_cells);
    let mut search = Search::default();

    for (i, point) in points.iter().enumerate() {
        search = hnsw.search(point, &mut search);
        let mut cell_nn: Vec<(usize, f32)> = Vec::with_capacity(k);
        for item in search.iter().take(k + 1) {
            let j = item.pid.into_inner();
            if j != i {
                cell_nn.push((j, item.distance));
            }
            if cell_nn.len() >= k {
                break;
            }
        }
        let kth_dist = cell_nn.last().map(|x| x.1).unwrap_or(1.0);
        all_kth_dists.push(kth_dist);
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
    fn test_hnsw_basic() {
        let points = vec![
            PcaPoint(vec![0.0, 0.0]),
            PcaPoint(vec![1.0, 0.0]),
            PcaPoint(vec![10.0, 10.0]),
        ];

        let hnsw: Hnsw<PcaPoint> = Builder::default()
            .build(points.iter().cloned(), points.len());

        let mut search = Search::default();
        search = hnsw.search(&points[0], &mut search);
        let results: Vec<_> = search.iter().take(2).collect();

        // Most results should contain point 0 (self) and point 1 (nearest)
        assert!(!results.is_empty());
    }

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
