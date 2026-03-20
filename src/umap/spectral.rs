//! Spectral Initialization via Laplacian Eigenmaps
//!
//! Because UMAP requires initializing the embedding to preserve global topology,
//! we compute the eigenvectors corresponding to the largest eigenvalues of the
//! normalized adjacency matrix M = D^{-1/2} A D^{-1/2}.
//! Since we only need the top d+1 eigenvectors, Subspace Iteration (Simultaneous Iteration)
//! is exceptionally fast and bypasses the need for heavy ARPACK bindings.

use rand::Rng;
use rayon::prelude::*;

/// Initializes the embedding coordinates using Laplacian Eigenmaps (Spectral Embedding).
/// 
/// Calculates the eigenvectors of the Random Walk Normalized Graph Laplacian:
///     $ \mathbf{L}_{rw} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} $
/// 
/// Rather than using complex iterative solvers like ARPACK, this uses a robust, 
/// multi-threaded Sparse Subspace Iteration method:
/// 1. Initialize random orthornormal basis $\mathbf{V}$
/// 2. Iteratively multiply $\mathbf{V}_{new} = (\mathbf{I} + \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}) \mathbf{V}$
/// 3. Orthogonalize $\mathbf{V}_{new}$ using Modified Gram-Schmidt
pub fn spectral_layout(
    graph: &super::graph::UmapGraph,
    n_components: usize,
    max_iter: usize,
) -> Vec<Vec<f32>> {
    // Determine N (number of vertices)
    let n = graph.edges.iter()
        .map(|e| e.source.max(e.target))
        .max()
        .map(|max_idx| max_idx + 1)
        .unwrap_or(0);

    if n <= n_components {
        // Fallback or empty
        let mut rng = rand::thread_rng();
        let mut fallback = vec![vec![0.0; n_components]; n];
        for row in &mut fallback {
            for val in row.iter_mut() {
                *val = rng.gen_range(-10.0..10.0);
            }
        }
        return fallback;
    }

    // 1. Compute Degree Matrix D
    let mut degrees = vec![0.0f32; n];
    for e in &graph.edges {
        degrees[e.source] += e.weight;
    }

    // 2. Compute D^{-1/2} values
    let d_inv_sqrt: Vec<f32> = degrees.into_iter().map(|d| {
        if d > 1e-8 { d.powf(-0.5) } else { 0.0 }
    }).collect();

    // 3. Build CSR representation of M = D^{-1/2} A D^{-1/2} for fast SpMV
    // First, count edges per row
    let mut row_counts = vec![0usize; n];
    for e in &graph.edges {
        row_counts[e.source] += 1;
    }
    
    let mut row_ptrs = vec![0usize; n + 1];
    for i in 0..n {
        row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
    }
    
    let nnz = row_ptrs[n];
    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![0.0f32; nnz];
    
    // We reuse row_counts to keep track of insertion positions
    let mut current_pos = row_ptrs[..n].to_vec();
    for e in &graph.edges {
        let pos = current_pos[e.source];
        col_indices[pos] = e.target;
        // Apply D^{-1/2} scaling symmetrically
        values[pos] = e.weight * d_inv_sqrt[e.source] * d_inv_sqrt[e.target];
        current_pos[e.source] += 1;
    }

    // 4. Subspace Iteration to find largest `k` eigenvectors
    let k = n_components + 1;
    
    // Initialize dense matrix V of size (N x k) with random normal values
    // Using flat vector representation for V: V[i * k + j] is the j-th component of the i-th point
    let mut v = vec![0.0f32; n * k];
    let mut rng = rand::thread_rng();
    for val in &mut v {
        // Box-Muller transform for N(0, 1)
        let u1: f32 = rng.gen_range(0.0001..1.0);
        let u2: f32 = rng.gen_range(0.0001..1.0);
        *val = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
    }
    
    // Iteration parameters
    for _iter in 0..max_iter {
        // (a) V_new = M * V
        // Parallel SpMM (Sparse Matrix - Dense Matrix multiply)
        let mut v_new = vec![0.0f32; n * k];
        v_new.par_chunks_exact_mut(k).enumerate().for_each(|(i, row_out)| {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            
            for ptr in start..end {
                let j = col_indices[ptr];
                let val = values[ptr];
                let v_j_start = j * k;
                
                for c in 0..k {
                    row_out[c] += val * v[v_j_start + c];
                }
            }
            // Add Identity (M + I) to shift spectrum and avoid negative eigenvalue oscillations
            for c in 0..k {
                row_out[c] += v[i * k + c];
            }
        });
        
        // (b) Orthogonalize V_new using Modified Gram-Schmidt over columns
        // We transpose logic to operate on columns sequentially, since MGS depends on previous columns
        for c in 0..k {
            // Compute norm of column c
            let mut norm_sq = 0.0f32;
            for i in 0..n {
                let val = v_new[i * k + c];
                norm_sq += val * val;
            }
            let norm = if norm_sq > 1e-12 { norm_sq.sqrt() } else { 1.0 };
            
            // Normalize column c
            for i in 0..n {
                v_new[i * k + c] /= norm;
            }
            
            // Orthogonalize subsequent columns
            for next_c in (c + 1)..k {
                let mut dot = 0.0f32;
                for i in 0..n {
                    dot += v_new[i * k + c] * v_new[i * k + next_c];
                }
                for i in 0..n {
                    v_new[i * k + next_c] -= dot * v_new[i * k + c];
                }
            }
        }
        
        v = v_new;
    }

    // 5. Extract results (skip the first eigenvector which corresponds to constant eigenvalue 1)
    let mut embedding = vec![vec![0.0f32; n_components]; n];
    for i in 0..n {
        for c in 0..n_components {
            // Scale and map to components (c+1 corresponds to ignoring the 0th trivial eigenvector)
            // UMAP additionally scales by dividing by the second highest eigenvalue, 
            // but just returning the vectors is standard spectral embedding.
            embedding[i][c] = d_inv_sqrt[i] * v[i * k + (c + 1)];
        }
    }

    embedding
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::{Edge, UmapGraph};

    #[test]
    fn test_spectral_layout() {
        // A simple path graph 0 - 1 - 2 - 3
        let edges = vec![
            Edge { source: 0, target: 1, weight: 1.0 },
            Edge { source: 1, target: 0, weight: 1.0 },
            Edge { source: 1, target: 2, weight: 1.0 },
            Edge { source: 2, target: 1, weight: 1.0 },
            Edge { source: 2, target: 3, weight: 1.0 },
            Edge { source: 3, target: 2, weight: 1.0 },
        ];
        
        let graph = UmapGraph { edges };
        let embedding = spectral_layout(&graph, 2, 50);
        
        // Ensure the embedding has 4 nodes and 2 components
        assert_eq!(embedding.len(), 4);
        assert_eq!(embedding[0].len(), 2);
        
        // Spectral layout on a path graph should separate the endpoints (0 and 3) the most.
        let dist_0_3 = (embedding[0][0] - embedding[3][0]).powi(2) + (embedding[0][1] - embedding[3][1]).powi(2);
        let dist_0_1 = (embedding[0][0] - embedding[1][0]).powi(2) + (embedding[0][1] - embedding[1][1]).powi(2);
        
        assert!(dist_0_3 > dist_0_1, "Endpoints 0 and 3 should be further apart than connected nodes 0 and 1");
    }
}
