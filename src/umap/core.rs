//! Main entry point orchestrating UMAP phases

use super::{graph, sgd, spectral};

pub fn run_umap(
    data: &[Vec<f32>],
    n_components: usize,
    n_neighbors: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: usize,
    spectral_n_iter: usize,
) -> Vec<Vec<f32>> {
    // 1. Construct Fuzzy Simplicial Set
    let u_graph = graph::build_fuzzy_simplicial_set(data, n_neighbors);
    
    // 2. Spectral Initialization
    let mut embedding = spectral::spectral_layout(&u_graph, n_components, spectral_n_iter);
    
    // 3. SGD Optimization
    sgd::optimize_layout(&u_graph, &mut embedding, n_epochs, min_dist, spread);
    
    embedding
}
