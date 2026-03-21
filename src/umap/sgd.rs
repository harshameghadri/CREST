//! Stochastic Gradient Descent for UMAP embeddings

use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

// Helper for Hogwild! atomic floats
struct AtomicF32<'a>(&'a AtomicU32);

impl<'a> AtomicF32<'a> {
    #[inline(always)]
    fn load(&self) -> f32 { f32::from_bits(self.0.load(Ordering::Relaxed)) }
    #[inline(always)]
    fn store(&self, val: f32) { self.0.store(val.to_bits(), Ordering::Relaxed) }
    #[inline(always)]
    fn add(&self, val: f32) { self.store(self.load() + val); }
    #[inline(always)]
    fn sub(&self, val: f32) { self.store(self.load() - val); }
}

/// Very fast LCG pseudo-random generator for negative sampling
#[inline(always)]
fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed as f32) / (u32::MAX as f32)
}

struct EdgeState {
    eps: f32,        // epochs per sample
    eps_neg: f32,    // epochs per negative sample
    next_sample: f32, // expected epoch for next positive sample
    next_neg: f32,    // expected epoch for next negative sample
    source: usize,
    target: usize,
}

/// Orchestrates the Stochastic Gradient Descent (SGD) layout optimization for UMAP.
/// 
/// This employs a lock-free Hogwild! approach to update the embedding positions. 
/// The cross-entropy loss gradients are derived from the layout continuous probability:
///     $ \Phi(x, y) = 1 / (1 + a \cdot dist(x,y)^{2b}) $
///
/// **Attractive force (positive edges):**
///     $ \nabla = \frac{-2 \cdot a \cdot b \cdot dist^{b-1}}{1 + a \cdot dist^b} \cdot (y_i - y_j) $
///
/// **Repulsive force (negative samples):**
///     $ \nabla = \frac{2 \cdot b}{(0.001 + dist) \cdot (1 + a \cdot dist^b)} \cdot (y_i - y_k) $
pub fn optimize_layout(
    graph: &super::graph::UmapGraph,
    initial_embedding: &mut [Vec<f32>],
    n_epochs: usize,
    min_dist: f32,
    spread: f32,
) {
    let n = initial_embedding.len();
    if n == 0 || graph.edges.is_empty() { return; }
    let dim = initial_embedding[0].len();
    
    let (a, b) = super::utils::find_ab_params(spread, min_dist);
    let gamma = 5.0; // UMAP's standard negative sample rate
    
    // Copy into atomic flat array for Hogwild!
    let mut atomics: Vec<AtomicU32> = Vec::with_capacity(n * dim);
    for row in initial_embedding.iter() {
        for &val in row {
            atomics.push(AtomicU32::new(val.to_bits()));
        }
    }
    
    let max_weight = graph.edges.iter()
        .map(|e| e.weight)
        .filter(|w| !w.is_nan())
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0);
        
    let mut states: Vec<EdgeState> = graph.edges.iter().map(|e| {
        let val = (e.weight / max_weight) * (n_epochs as f32);
        let eps = if val > 0.0 { n_epochs as f32 / val } else { f32::MAX };
        let eps_neg = eps / gamma;
        EdgeState {
            eps,
            eps_neg,
            next_sample: eps,
            next_neg: eps_neg,
            source: e.source,
            target: e.target,
        }
    }).collect();
    
    let initial_alpha = 1.0f32;
    
    for epoch in 1..=n_epochs {
        let alpha = initial_alpha * (1.0 - (epoch as f32 / n_epochs as f32));
        let e_f32 = epoch as f32;
        
        states.par_iter_mut().for_each(|state| {
            let mut rng_seed = (state.source.wrapping_mul(epoch).wrapping_add(state.target)) as u32;
            let i = state.source;
            let j = state.target;
            
            // 1. Positive edge updates
            while state.next_sample <= e_f32 {
                let mut dist_sq = 0.0f32;
                let mut y_i = vec![0.0f32; dim];
                let mut y_j = vec![0.0f32; dim];
                
                for d in 0..dim {
                    y_i[d] = AtomicF32(&atomics[i * dim + d]).load();
                    y_j[d] = AtomicF32(&atomics[j * dim + d]).load();
                    dist_sq += (y_i[d] - y_j[d]).powi(2);
                }
                
                let mut grad_coeff = 0.0f32;
                if dist_sq > 0.0 {
                    grad_coeff = (-2.0 * a * b * dist_sq.powf(b - 1.0)) / (1.0 + a * dist_sq.powf(b));
                }
                
                // apply gradient
                for d in 0..dim {
                    let grad = grad_coeff * (y_i[d] - y_j[d]);
                    // Clip gradients to prevent massive explosions in exact same points
                    let clipped = (grad * alpha).clamp(-4.0, 4.0);
                    AtomicF32(&atomics[i * dim + d]).add(clipped);
                    AtomicF32(&atomics[j * dim + d]).sub(clipped);
                }
                
                state.next_sample += state.eps;
            }
            
            // 2. Negative sample updates
            while state.next_neg <= e_f32 {
                let r = lcg(&mut rng_seed);
                let k = if n > 0 { ((r * n as f32) as usize) % n } else { break };
                
                if i != k {
                    let mut dist_sq = 0.0f32;
                    let mut y_i = vec![0.0f32; dim];
                    let mut y_k = vec![0.0f32; dim];
                    
                    for d in 0..dim {
                        y_i[d] = AtomicF32(&atomics[i * dim + d]).load();
                        y_k[d] = AtomicF32(&atomics[k * dim + d]).load();
                        dist_sq += (y_i[d] - y_k[d]).powi(2);
                    }
                    
                    let mut grad_coeff = 0.0f32;
                    if dist_sq > 0.0 {
                        grad_coeff = (2.0 * b) / ((0.001 + dist_sq) * (1.0 + a * dist_sq.powf(b)));
                    }
                    
                    for d in 0..dim {
                        let grad = grad_coeff * (y_i[d] - y_k[d]);
                        let clipped = (grad * alpha).clamp(-4.0, 4.0);
                        AtomicF32(&atomics[i * dim + d]).add(clipped);
                    }
                }
                
                state.next_neg += state.eps_neg;
            }
        });
    }
    
    // Write back results
    for i in 0..n {
        for d in 0..dim {
            initial_embedding[i][d] = AtomicF32(&atomics[i * dim + d]).load();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::{Edge, UmapGraph};

    #[test]
    fn test_optimize_layout() {
        let edges = vec![
            Edge { source: 0, target: 1, weight: 1.0 },
        ];
        let graph = UmapGraph { edges };
        
        let mut embedding: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];
        
        // Optimize should pull them closer together due to positive edge
        let initial_dist = (embedding[0][0] - embedding[1][0]).powi(2) + (embedding[0][1] - embedding[1][1]).powi(2);
        
        optimize_layout(&graph, &mut embedding, 50, 0.1, 1.0);
        
        let final_dist = (embedding[0][0] - embedding[1][0]).powi(2) + (embedding[0][1] - embedding[1][1]).powi(2);
        
        assert!(final_dist < initial_dist, "Points should move closer based on positive edge");
    }
}
