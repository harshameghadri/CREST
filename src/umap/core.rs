//! Main entry point orchestrating UMAP phases

use super::{graph, sgd, spectral, utils};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// UMAP embedding of a precomputed fuzzy graph; returns n × n_components row-major.
///
/// `n_epochs = None` uses umap-learn's default (500 for n <= 10k, else 200).
/// Initialisation follows umap-learn: spectral layout scaled to max |x| = 10,
/// tiny Gaussian noise, then each dimension rescaled to [0, 10].
pub fn embed(
    g: &graph::UmapGraph,
    n_components: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    spectral_n_iter: usize,
    seed: u64,
) -> Vec<f32> {
    let n = g.n;
    let n_epochs = n_epochs.unwrap_or(if n <= 10_000 { 500 } else { 200 });
    let mut emb = spectral::spectral_layout(g, n_components, spectral_n_iter, seed);

    let maxabs = emb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let expansion = if maxabs > 0.0 { 10.0 / maxabs } else { 1.0 };
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(1));
    let noise = Normal::new(0.0f32, 1e-4).unwrap();
    for v in emb.iter_mut() {
        *v = *v * expansion + noise.sample(&mut rng);
    }
    for c in 0..n_components {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..n {
            lo = lo.min(emb[i * n_components + c]);
            hi = hi.max(emb[i * n_components + c]);
        }
        let range = if hi > lo { hi - lo } else { 1.0 };
        for i in 0..n {
            let v = &mut emb[i * n_components + c];
            *v = 10.0 * (*v - lo) / range;
        }
    }

    let (a, b) = utils::find_ab_params(spread, min_dist);
    sgd::optimize_layout(g, &mut emb, n_components, n_epochs, a, b, seed);
    emb
}

/// Legacy entry point used by the Polars plugin: rows in, rows out.
/// `n_neighbors` counts the cell itself (scanpy convention).
pub fn run_umap(
    data: &[Vec<f32>],
    n_components: usize,
    n_neighbors: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: usize,
    spectral_n_iter: usize,
) -> Vec<Vec<f32>> {
    let g = graph::build_fuzzy_simplicial_set(data, n_neighbors);
    let flat = embed(&g, n_components, min_dist, spread, Some(n_epochs), spectral_n_iter, 0);
    flat.chunks(n_components).map(|r| r.to_vec()).collect()
}
