use instant_distance::{Builder, Point, Search};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub weight: f32,
}

pub struct UmapGraph {
    pub edges: Vec<Edge>,
}

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

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

pub fn build_fuzzy_simplicial_set(
    data: &[Vec<f32>],
    n_neighbors: usize,
) -> UmapGraph {
    let n = data.len();
    if n == 0 {
        return UmapGraph { edges: vec![] };
    }

    let k = n_neighbors.min(n - 1);
    if k == 0 {
        return UmapGraph { edges: vec![] };
    }
    let target_sum = (k as f32).log2();

    // 1. Build HNSW index for O(n log n) approximate KNN
    let points: Vec<PcaPoint> = data.iter().map(|v| PcaPoint(v.clone())).collect();
    let (hnsw, _ids) = Builder::default()
        .ef_construction(200)
        .build_hnsw(points.clone());

    // 2. Find k nearest neighbors for each point using HNSW
    // Note: instant-distance's search is NOT thread-safe with shared Search state,
    // so we build results sequentially (HNSW query is already fast: O(log n) per query)
    let mut knn_results: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(n);
    let mut search = Search::default();

    for (cell_idx, point) in points.iter().enumerate() {
        let results = hnsw.search(point, &mut search);
        let mut indices = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        for item in results.take(k + 1) {
            let idx = item.pid.into_inner() as usize;
            if idx != cell_idx {
                indices.push(idx);
                distances.push(item.distance);
            }
            if indices.len() >= k {
                break;
            }
        }
        knn_results.push((indices, distances));
    }

    // 3. Compute rho, sigma, and asymmetric weights (UMAP membership strengths)
    let asymmetric_edges: Vec<Vec<(usize, f32)>> = knn_results.into_par_iter().map(|(indices, dists)| {
        let rho = dists.first().copied().unwrap_or(0.0);

        // Binary search for sigma such that sum of membership strengths = log2(k)
        let mut lo = 0.0;
        let mut hi = 1e6;
        let mut sigma = 1.0;

        for _ in 0..64 {
            let mut sum: f32 = 0.0;
            for &d in &dists {
                let val = d - rho;
                if val > 0.0 {
                    sum += (-val / sigma).exp();
                } else {
                    sum += 1.0;
                }
            }

            if (sum - target_sum).abs() < 1e-5 {
                break;
            }

            if sum > target_sum {
                hi = sigma;
                sigma = (lo + hi) / 2.0;
            } else {
                lo = sigma;
                sigma = if hi > 1e5 { sigma * 2.0 } else { (lo + hi) / 2.0 };
            }
        }

        // Compute membership strengths
        indices.into_iter().zip(dists.into_iter()).map(|(j, d)| {
            let val = d - rho;
            let weight = if val > 0.0 {
                (-val / sigma).exp()
            } else {
                1.0
            };
            (j, weight)
        }).collect()
    }).collect();

    // 4. Symmetrize Graph: B = A + A^T - A * A^T
    let mut weight_map: HashMap<(usize, usize), f32> = HashMap::new();
    for (i, edges) in asymmetric_edges.iter().enumerate() {
        for &(j, w) in edges {
            weight_map.insert((i, j), w);
        }
    }

    let mut symmetric_graph: HashMap<(usize, usize), f32> = HashMap::new();

    for i in 0..n {
        for &(j, weight_ij) in &asymmetric_edges[i] {
            let weight_ji = weight_map.get(&(j, i)).copied().unwrap_or(0.0);
            let sym_weight = weight_ij + weight_ji - (weight_ij * weight_ji);

            if sym_weight > 0.0 {
                symmetric_graph.insert((i, j), sym_weight);
                symmetric_graph.insert((j, i), sym_weight);
            }
        }
    }

    // 5. Convert to Edge List
    let edges: Vec<Edge> = symmetric_graph.into_iter().map(|((src, tgt), w)| {
        Edge { source: src, target: tgt, weight: w }
    }).collect();

    UmapGraph { edges }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_simplicial_set() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![1.0, 1.0],
            vec![1.1, 1.1],
        ];

        let graph = build_fuzzy_simplicial_set(&data, 2);

        assert!(!graph.edges.is_empty(), "Graph should not be empty");

        let mut saw_0_1 = false;
        let mut saw_2_3 = false;

        for e in &graph.edges {
            if (e.source == 0 && e.target == 1) || (e.source == 1 && e.target == 0) {
                saw_0_1 = true;
            }
            if (e.source == 2 && e.target == 3) || (e.source == 3 && e.target == 2) {
                saw_2_3 = true;
            }
        }

        assert!(saw_0_1, "Expected edge between 0 and 1");
        assert!(saw_2_3, "Expected edge between 2 and 3");
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f32>> = vec![];
        let graph = build_fuzzy_simplicial_set(&data, 5);
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_single_point() {
        let data = vec![vec![1.0, 2.0, 3.0]];
        let graph = build_fuzzy_simplicial_set(&data, 5);
        assert!(graph.edges.is_empty());
    }
}
