//! Shared approximate k-nearest-neighbour search (HNSW via `instant-distance`).
//!
//! `instant-distance` shuffles points internally while building the index, so the
//! `PointId` yielded by a search is NOT the caller's row index. `build_hnsw` returns
//! the mapping (input row -> PointId); we invert it once here so every consumer
//! (neighbors, connectivities, Leiden, UMAP) sees true row indices.

use instant_distance::{Builder, Point, Search};
use rayon::prelude::*;

/// PCA point for HNSW distance computation (Euclidean).
#[derive(Clone, Debug)]
pub struct PcaPoint(pub Vec<f32>);

impl Point for PcaPoint {
    fn distance(&self, other: &Self) -> f32 {
        self.0.iter().zip(other.0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }
}

/// For each row of `data`, the indices and distances of its `k` nearest neighbours
/// (self excluded), sorted by increasing distance.
pub fn hnsw_knn(data: &[Vec<f32>], k: usize) -> Vec<(Vec<usize>, Vec<f32>)> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.min(n - 1);

    let points: Vec<PcaPoint> = data.iter().map(|v| PcaPoint(v.clone())).collect();
    let (hnsw, ids) = Builder::default()
        .ef_construction(200)
        .ef_search((2 * k).max(100))
        .build_hnsw(points.clone());

    // ids[input_row] = internal PointId  ->  invert to internal -> input_row
    let mut internal_to_row = vec![0usize; n];
    for (row, pid) in ids.iter().enumerate() {
        internal_to_row[pid.into_inner() as usize] = row;
    }

    points
        .par_iter()
        .enumerate()
        .map_init(Search::default, |search, (i, point)| {
            let mut indices = Vec::with_capacity(k);
            let mut distances = Vec::with_capacity(k);
            for item in hnsw.search(point, search) {
                let j = internal_to_row[item.pid.into_inner() as usize];
                if j == i {
                    continue;
                }
                indices.push(j);
                distances.push(item.distance);
                if indices.len() == k {
                    break;
                }
            }
            (indices, distances)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: returned neighbours must be true row indices. Before the
    /// id-mapping fix, results were indices into HNSW's shuffled internal order.
    #[test]
    fn test_hnsw_knn_matches_brute_force() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let n = 500;
        let data: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..8).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
            .collect();
        let k = 10;
        let knn = hnsw_knn(&data, k);

        let mut hits = 0usize;
        for i in 0..n {
            let mut exact: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (PcaPoint(data[i].clone()).distance(&PcaPoint(data[j].clone())), j))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let truth: std::collections::HashSet<usize> = exact[..k].iter().map(|x| x.1).collect();
            assert_eq!(knn[i].0.len(), k);
            assert!(!knn[i].0.contains(&i), "self must be excluded");
            hits += knn[i].0.iter().filter(|j| truth.contains(j)).count();
        }
        let recall = hits as f64 / (n * k) as f64;
        assert!(recall > 0.95, "recall@{} = {:.3}", k, recall);
    }
}
