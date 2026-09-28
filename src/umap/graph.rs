//! Fuzzy simplicial set (UMAP / scanpy `connectivities`) from a kNN graph.
//!
//! Follows umap-learn's `smooth_knn_dist` + `compute_membership_strengths` +
//! fuzzy union exactly, with the scanpy convention that `n_neighbors` counts
//! the cell itself (so the kNN graph holds `n_neighbors - 1` true neighbours).

use crate::knn::KnnGraph;
#[cfg(test)]
use crate::knn::{knn, Points};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub weight: f32,
}

/// Symmetric weighted graph; `edges` lists both (i, j) and (j, i).
pub struct UmapGraph {
    pub n: usize,
    pub edges: Vec<Edge>,
}

const SMOOTH_K_TOLERANCE: f64 = 1e-5;
const MIN_K_DIST_SCALE: f64 = 1e-3;

/// Per-cell (rho, sigma) of umap-learn's smooth_knn_dist (local_connectivity = 1).
fn smooth_knn_dist(dists: &[f32], k: usize, n_neighbors: usize, mean_all: f64) -> (f64, f64) {
    let target = (n_neighbors as f64).log2();
    let rho = dists.iter().copied().find(|&d| d > 0.0).map(|d| d as f64).unwrap_or(0.0);
    let (mut lo, mut hi, mut mid) = (0.0f64, f64::INFINITY, 1.0f64);
    for _ in 0..64 {
        let mut psum = 0.0;
        for &d in &dists[..k] {
            let v = d as f64 - rho;
            psum += if v > 0.0 { (-v / mid).exp() } else { 1.0 };
        }
        if (psum - target).abs() < SMOOTH_K_TOLERANCE {
            break;
        }
        if psum > target {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            if hi == f64::INFINITY {
                mid *= 2.0;
            } else {
                mid = (lo + hi) / 2.0;
            }
        }
    }
    // mean over the umap distance row, which includes self at distance 0
    let mean_i = dists[..k].iter().map(|&d| d as f64).sum::<f64>() / (k + 1) as f64;
    let floor = if rho > 0.0 { MIN_K_DIST_SCALE * mean_i } else { MIN_K_DIST_SCALE * mean_all };
    (rho, mid.max(floor))
}

/// Fuzzy simplicial set of a kNN graph whose rows hold `n_neighbors - 1` neighbours.
pub fn fuzzy_simplicial_set(g: &KnnGraph, n_neighbors: usize) -> UmapGraph {
    let (n, k) = (g.n, g.k);
    if n == 0 || k == 0 {
        return UmapGraph { n, edges: vec![] };
    }
    let mean_all = g.distances.iter().map(|&d| d as f64).sum::<f64>() / (n * (k + 1)) as f64;

    // Directed membership strengths keyed by the unordered pair.
    let mut entries: Vec<(u64, f32)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let d = &g.distances[i * k..(i + 1) * k];
            let idx = &g.indices[i * k..(i + 1) * k];
            let (rho, sigma) = smooth_knn_dist(d, k, n_neighbors, mean_all);
            idx.iter().zip(d).filter_map(move |(&j, &dist)| {
                if j == u32::MAX || j as usize == i {
                    return None;
                }
                let v = dist as f64 - rho;
                let w = if v <= 0.0 || sigma == 0.0 { 1.0 } else { (-v / sigma).exp() };
                let (a, b) = if (i as u32) < j { (i as u64, j as u64) } else { (j as u64, i as u64) };
                Some(((a << 32) | b, w as f32))
            })
        })
        .collect();
    entries.par_sort_unstable_by_key(|e| e.0);

    // Fuzzy union: w = a + b - a*b when both directions exist.
    let mut edges = Vec::with_capacity(entries.len() * 2);
    let mut p = 0;
    while p < entries.len() {
        let (key, mut w) = entries[p];
        let mut q = p + 1;
        while q < entries.len() && entries[q].0 == key {
            let b = entries[q].1;
            w = w + b - w * b;
            q += 1;
        }
        if w > 0.0 {
            let (a, b) = ((key >> 32) as usize, (key & 0xffff_ffff) as usize);
            edges.push(Edge { source: a, target: b, weight: w });
            edges.push(Edge { source: b, target: a, weight: w });
        }
        p = q;
    }
    UmapGraph { n, edges }
}

/// kNN (via `crate::knn`) + fuzzy simplicial set from dense rows.
/// `n_neighbors` includes the cell itself (scanpy convention).
#[cfg(test)]
pub fn build_fuzzy_simplicial_set(data: &[Vec<f32>], n_neighbors: usize) -> UmapGraph {
    let n = data.len();
    if n < 2 || n_neighbors < 2 {
        return UmapGraph { n, edges: vec![] };
    }
    let d = data[0].len();
    let flat: Vec<f32> = data.iter().flat_map(|r| r.iter().copied()).collect();
    let g = knn(Points::new(&flat, n, d), n_neighbors - 1, None, 0);
    fuzzy_simplicial_set(&g, n_neighbors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_simplicial_set() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![1.0, 1.0], vec![1.1, 1.1]];
        let graph = build_fuzzy_simplicial_set(&data, 2);
        assert_eq!(graph.n, 4);
        let has = |a: usize, b: usize| graph.edges.iter().any(|e| e.source == a && e.target == b);
        assert!(has(0, 1) && has(1, 0) && has(2, 3) && has(3, 2));
        // symmetric weights
        for e in &graph.edges {
            let back = graph.edges.iter().find(|f| f.source == e.target && f.target == e.source).unwrap();
            assert_eq!(e.weight, back.weight);
        }
    }

    #[test]
    fn test_sigma_hits_target() {
        // membership strengths of each row sum to log2(n_neighbors)
        let d = [1.0f32, 1.5, 2.0, 2.2, 3.0];
        let (rho, sigma) = smooth_knn_dist(&d, 5, 6, 1.0);
        let s: f64 = d.iter().map(|&x| { let v = x as f64 - rho; if v > 0.0 { (-v / sigma).exp() } else { 1.0 } }).sum();
        assert!((s - 6f64.log2()).abs() < 1e-4);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f32>> = vec![];
        assert!(build_fuzzy_simplicial_set(&data, 5).edges.is_empty());
    }

    #[test]
    fn test_single_point() {
        let data = vec![vec![1.0, 2.0, 3.0]];
        assert!(build_fuzzy_simplicial_set(&data, 5).edges.is_empty());
    }
}
