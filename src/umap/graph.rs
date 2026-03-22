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

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

pub fn build_fuzzy_simplicial_set(
    data: &[Vec<f32>],
    n_neighbors: usize,
) -> UmapGraph {
    let n = data.len();
    if n <= 1 {
        // Cannot build a neighbor graph with 0 or 1 points
        return UmapGraph { edges: vec![] };
    }

    let k = n_neighbors.min(n - 1);
    if k == 0 {
        return UmapGraph { edges: vec![] };
    }
    let target_sum = (k as f32).log2();

    // 1. Find k nearest neighbors for each point (Exact via brute-force + Rayon for now)
    // Returns Vec of (knn_indices, knn_distances)
    let knn_results: Vec<(Vec<usize>, Vec<f32>)> = (0..n).into_par_iter().map(|i| {
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| i != j)
            .map(|j| (j, euclidean_distance(&data[i], &data[j])))
            .collect();
            
        // Sort by distance ascending
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top k
        let (indices, distances): (Vec<usize>, Vec<f32>) = dists.into_iter().take(k).unzip();
        (indices, distances)
    }).collect();

    // 2. Compute rho, sigma, and asymmetric weights
    let asymmetric_edges: Vec<Vec<(usize, f32)>> = knn_results.into_par_iter().map(|(indices, dists)| {
        let rho = dists.first().copied().unwrap_or(0.0);
        
        // Binary search for sigma
        let mut lo = 0.0;
        let mut hi = 1e6;
        let mut sigma = 1.0;
        
        for _ in 0..64 { // 64 iterations is enough for f32 precision
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
        
        // Compute weights
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

    // 3. Symmetrize Graph: B = A + A^T - A * A^T
    // Use a HashMap to accumulate B_{ij}
    let mut symmetric_graph: HashMap<(usize, usize), f32> = HashMap::new();
    
    for i in 0..n {
        for &(j, weight_ij) in &asymmetric_edges[i] {
            // Find weight_ji if it exists
            let weight_ji = asymmetric_edges[j].iter().find(|&&(idx, _)| idx == i).map(|&(_, w)| w).unwrap_or(0.0);
            
            // Symmetrize
            let sym_weight = weight_ij + weight_ji - (weight_ij * weight_ji);
            
            if sym_weight > 0.0 {
                // Ensure (i, j) has i < j to avoid duplicates, but we want directed edges for SGD, so add both.
                // UMAP optimization typically operates on undirected edges or uses directed edges twice.
                // We'll store both directions for simpler sampling, or just (src, tgt).
                symmetric_graph.insert((i, j), sym_weight);
                symmetric_graph.insert((j, i), sym_weight);
            }
        }
    }
    
    // 4. Convert to Edge List
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
        
        // n=4, each point has 2 neighbors => connected components.
        // We should have non-zero edges between 0-1 and 2-3.
        assert!(!graph.edges.is_empty(), "Graph should not be empty");
        
        let mut saw_0_1 = false;
        let mut saw_2_3 = false;
        
        for e in &graph.edges {
            if (e.source == 0 && e.target == 1) || (e.source == 1 && e.target == 0) {
                saw_0_1 = true;
                assert!(e.weight > 0.5, "Strong edge expected");
            }
            if (e.source == 2 && e.target == 3) || (e.source == 3 && e.target == 2) {
                saw_2_3 = true;
                assert!(e.weight > 0.5, "Strong edge expected");
            }
        }
        
        assert!(saw_0_1, "Expected edge between 0 and 1");
        assert!(saw_2_3, "Expected edge between 2 and 3");
    }
}
