//! Spectral initialisation (Laplacian eigenmaps), as umap-learn's `spectral_layout`.
//!
//! Top eigenvectors of M = D^{-1/2} A D^{-1/2} (equivalently the smallest of the
//! normalised Laplacian) by block subspace iteration on (M + I)/2 with the known
//! trivial eigenvector D^{1/2}1 deflated, followed by a Rayleigh-Ritz step.

use faer::Mat;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Returns an n × n_components row-major embedding (unscaled eigenvectors).
pub fn spectral_layout(
    graph: &super::graph::UmapGraph,
    n_components: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<f32> {
    let n = graph.n;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    if n <= n_components + 2 || graph.edges.is_empty() {
        return (0..n * n_components).map(|_| StandardNormal.sample(&mut rng)).collect();
    }

    // CSR of the symmetric graph
    let mut deg = vec![0.0f64; n];
    let mut cnt = vec![0usize; n + 1];
    for e in &graph.edges {
        deg[e.source] += e.weight as f64;
        cnt[e.source + 1] += 1;
    }
    for i in 0..n {
        cnt[i + 1] += cnt[i];
    }
    let mut pos = cnt[..n].to_vec();
    let mut col = vec![0usize; cnt[n]];
    let mut val = vec![0.0f64; cnt[n]];
    let dis: Vec<f64> = deg.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();
    for e in &graph.edges {
        let p = pos[e.source];
        col[p] = e.target;
        val[p] = e.weight as f64 * dis[e.source] * dis[e.target];
        pos[e.source] += 1;
    }

    // trivial eigenvector
    let mut v0: Vec<f64> = deg.iter().map(|d| d.sqrt()).collect();
    let nv0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    v0.iter_mut().for_each(|x| *x /= nv0);

    let m = n_components + 2; // guard vectors speed convergence
    // column-major n × m
    let mut v: Vec<f64> = (0..n * m).map(|_| StandardNormal.sample(&mut rng)).collect();

    let orthonormalize = |v: &mut Vec<f64>| {
        for c in 0..m {
            let (head, tail) = v.split_at_mut(c * n);
            let col_c = &mut tail[..n];
            // deflate trivial vector
            let dot0: f64 = col_c.iter().zip(&v0).map(|(a, b)| a * b).sum();
            col_c.iter_mut().zip(&v0).for_each(|(a, b)| *a -= dot0 * b);
            for p in 0..c {
                let col_p = &head[p * n..(p + 1) * n];
                let dot: f64 = col_c.iter().zip(col_p).map(|(a, b)| a * b).sum();
                col_c.iter_mut().zip(col_p).for_each(|(a, b)| *a -= dot * b);
            }
            let nrm = col_c.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nrm = if nrm > 1e-300 { nrm } else { 1.0 };
            col_c.iter_mut().for_each(|x| *x /= nrm);
        }
    };
    let apply = |v: &[f64]| -> Vec<f64> {
        // (M + I)/2 applied to every column
        let mut out = vec![0.0f64; n * m];
        out.par_chunks_mut(n).enumerate().for_each(|(c, oc)| {
            let vc = &v[c * n..(c + 1) * n];
            for i in 0..n {
                let mut s = vc[i];
                for p in cnt[i]..cnt[i + 1] {
                    s += val[p] * vc[col[p]];
                }
                oc[i] = 0.5 * s;
            }
        });
        out
    };

    orthonormalize(&mut v);
    for _ in 0..max_iter.max(1) {
        v = apply(&v);
        orthonormalize(&mut v);
    }

    // Rayleigh-Ritz: order the subspace by eigenvalue of (M + I)/2
    let mv = apply(&v);
    let h = Mat::<f64>::from_fn(m, m, |a, b| {
        v[a * n..(a + 1) * n].iter().zip(&mv[b * n..(b + 1) * n]).map(|(x, y)| x * y).sum()
    });
    let hs = Mat::<f64>::from_fn(m, m, |a, b| 0.5 * (h[(a, b)] + h[(b, a)]));
    let mut emb = vec![0.0f32; n * n_components];
    let eig = hs.self_adjoint_eigen(faer::Side::Lower);
    crate::simd::clean_simd_state();
    match eig {
        Ok(eig) => {
            let u = eig.U();
            // eigenvalues ascending: take the largest n_components
            for c in 0..n_components {
                let src = m - 1 - c;
                for i in 0..n {
                    let mut s = 0.0;
                    for a in 0..m {
                        s += v[a * n + i] * u[(a, src)];
                    }
                    emb[i * n_components + c] = s as f32;
                }
            }
        }
        Err(_) => {
            for c in 0..n_components {
                for i in 0..n {
                    emb[i * n_components + c] = v[c * n + i] as f32;
                }
            }
        }
    }
    emb
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::{Edge, UmapGraph};

    #[test]
    fn test_spectral_layout_path_graph() {
        // path graph 0-1-...-19: the random-walk Fiedler vector D^{-1/2} v is
        // monotone along the path (v itself is not, at the degree-1 endpoints)
        let n = 20;
        let mut edges = vec![];
        for i in 0..n - 1 {
            edges.push(Edge { source: i, target: i + 1, weight: 1.0 });
            edges.push(Edge { source: i + 1, target: i, weight: 1.0 });
        }
        let graph = UmapGraph { n, edges };
        let emb = spectral_layout(&graph, 2, 2000, 0);
        assert_eq!(emb.len(), n * 2);
        let first: Vec<f32> = (0..n)
            .map(|i| emb[i * 2] / if i == 0 || i == n - 1 { 1.0 } else { 2f32.sqrt() })
            .collect();
        let inc = first.windows(2).all(|w| w[0] < w[1]);
        let dec = first.windows(2).all(|w| w[0] > w[1]);
        assert!(inc || dec, "Fiedler vector not monotone: {:?}", first);
    }

    #[test]
    fn test_spectral_deterministic() {
        let mut edges = vec![];
        for i in 0..50usize {
            let j = (i * 7 + 3) % 50;
            if i != j {
                edges.push(Edge { source: i, target: j, weight: 1.0 });
                edges.push(Edge { source: j, target: i, weight: 1.0 });
            }
        }
        let g = UmapGraph { n: 50, edges };
        assert_eq!(spectral_layout(&g, 2, 30, 5), spectral_layout(&g, 2, 30, 5));
    }
}
