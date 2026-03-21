use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use kiddo::KdTree;

/// Output type: List(List(Float32)) — each cell gets a list of [neighbor_idx, distance] pairs flattened
fn neighbors_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "neighbors".into(),
        DataType::List(Box::new(DataType::List(Box::new(DataType::Float32)))),
    ))
}

/// compute_neighbors: Build K-nearest neighbor graph from PCA coordinates.
///
/// Inputs:
/// 0: pca_coords (List(List(Float32))) - PCA coordinates per cell (from SVD)
/// 1: n_neighbors (UInt32) - number of neighbors (default 15)
///
/// Returns List(List(Float32)): for each cell, a flattened list of
/// [neighbor_0_idx, neighbor_0_dist, neighbor_1_idx, neighbor_1_dist, ...]
///
/// This is the equivalent of scanpy's pp.neighbors, producing the connectivities
/// graph that feeds into UMAP and Leiden clustering.
#[polars_expr(output_type_func=neighbors_output)]
fn compute_neighbors(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "compute_neighbors requires 2 inputs: [pca_coords, n_neighbors]".into(),
        ));
    }

    let pca_coords = inputs[0].list()?;
    let k = inputs[1].u32()?.get(0).unwrap_or(15) as usize;

    let n_cells = pca_coords.len();
    if n_cells == 0 {
        let empty = Series::new("neighbors".into(), Vec::<Series>::new());
        return Ok(Series::new("neighbors".into(), vec![empty]));
    }

    // Build KD-Tree (max 50 dims, matching leiden.rs)
    const MAX_DIMS: usize = 50;
    let mut tree: KdTree<f32, MAX_DIMS> = KdTree::new();
    let mut points = Vec::with_capacity(n_cells);
    let mut warned = false;

    for (i, opt_row) in pca_coords.into_iter().enumerate() {
        let mut pt = [0.0f32; MAX_DIMS];
        if let Some(row_series) = opt_row {
            let float_ca = row_series.f32()?;
            let actual_dims = float_ca.len();
            if actual_dims > MAX_DIMS && !warned {
                eprintln!(
                    "crest warning: PCA has {} dimensions but KD-tree supports max {}. Truncating.",
                    actual_dims, MAX_DIMS
                );
                warned = true;
            }
            for (j, val) in float_ca.into_no_null_iter().enumerate() {
                if j < MAX_DIMS {
                    pt[j] = val;
                }
            }
        }
        tree.add(&pt, i as u64);
        points.push(pt);
    }

    // Query KNN for each cell
    let cell_neighbors: Vec<Series> = points
        .iter()
        .map(|pt| {
            let nn = tree.nearest_n::<kiddo::SquaredEuclidean>(pt, k);
            let mut flat: Vec<f32> = Vec::with_capacity(nn.len() * 2);
            for neighbor in &nn {
                flat.push(neighbor.item as f32);
                flat.push(neighbor.distance.sqrt()); // convert squared euclidean to euclidean
            }
            Series::new("".into(), flat)
        })
        .collect();

    let outer = Series::new("neighbors".into(), cell_neighbors);
    let wrapper = Series::new("neighbors".into(), vec![outer]);
    Ok(wrapper)
}

/// connectivities: Build a UMAP-style fuzzy simplicial set connectivities matrix.
///
/// Inputs:
/// 0: pca_coords (List(List(Float32))) - PCA coordinates per cell
/// 1: n_neighbors (UInt32) - number of neighbors
///
/// Returns List(List(Float32)): sparse connectivities as [cell_i, cell_j, weight, ...]
/// triplets, where weight is computed using a Gaussian kernel.
/// This matches scanpy's .obsp['connectivities'] output.
#[polars_expr(output_type_func=neighbors_output)]
fn compute_connectivities(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "compute_connectivities requires 2 inputs: [pca_coords, n_neighbors]".into(),
        ));
    }

    let pca_coords = inputs[0].list()?;
    let k = inputs[1].u32()?.get(0).unwrap_or(15) as usize;

    let n_cells = pca_coords.len();
    if n_cells == 0 {
        let empty = Series::new("connectivities".into(), Vec::<Series>::new());
        return Ok(Series::new("connectivities".into(), vec![empty]));
    }

    const MAX_DIMS: usize = 50;
    let mut tree: KdTree<f32, MAX_DIMS> = KdTree::new();
    let mut points = Vec::with_capacity(n_cells);
    let mut warned = false;

    for (i, opt_row) in pca_coords.into_iter().enumerate() {
        let mut pt = [0.0f32; MAX_DIMS];
        if let Some(row_series) = opt_row {
            let float_ca = row_series.f32()?;
            if float_ca.len() > MAX_DIMS && !warned {
                eprintln!(
                    "crest warning: PCA has {} dimensions but KD-tree supports max {}. Truncating.",
                    float_ca.len(), MAX_DIMS
                );
                warned = true;
            }
            for (j, val) in float_ca.into_no_null_iter().enumerate() {
                if j < MAX_DIMS {
                    pt[j] = val;
                }
            }
        }
        tree.add(&pt, i as u64);
        points.push(pt);
    }

    // Build connectivities as COO triplets [cell_i, cell_j, weight]
    // Use Gaussian kernel: weight = exp(-dist^2 / (2 * sigma^2))
    // sigma = median of k-th neighbor distances (local bandwidth)
    let mut all_kth_dists = Vec::with_capacity(n_cells);
    let mut all_neighbors: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_cells);

    for pt in &points {
        let nn = tree.nearest_n::<kiddo::SquaredEuclidean>(pt, k);
        let mut cell_nn = Vec::with_capacity(nn.len());
        let mut kth_dist = 0.0f32;
        for (idx, neighbor) in nn.iter().enumerate() {
            let dist = neighbor.distance.sqrt();
            cell_nn.push((neighbor.item as usize, dist));
            if idx == nn.len() - 1 {
                kth_dist = dist;
            }
        }
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
    fn test_knn_basic() {
        // Minimal KD-tree test: 3 points, k=2
        const D: usize = 50;
        let mut tree: KdTree<f32, D> = KdTree::new();

        let p0 = {
            let mut a = [0.0f32; D];
            a[0] = 0.0;
            a[1] = 0.0;
            a
        };
        let p1 = {
            let mut a = [0.0f32; D];
            a[0] = 1.0;
            a[1] = 0.0;
            a
        };
        let p2 = {
            let mut a = [0.0f32; D];
            a[0] = 10.0;
            a[1] = 10.0;
            a
        };

        tree.add(&p0, 0);
        tree.add(&p1, 1);
        tree.add(&p2, 2);

        let nn = tree.nearest_n::<kiddo::SquaredEuclidean>(&p0, 2);
        assert_eq!(nn.len(), 2);
        // Nearest to p0 should be p0 itself (dist=0) then p1 (dist=1)
        assert_eq!(nn[0].item, 0);
        assert_eq!(nn[1].item, 1);
        assert!((nn[1].distance - 1.0).abs() < 1e-6);
    }
}
