//! Spectral initialisation (Laplacian eigenmaps), as umap-learn's `spectral_layout`.
//!
//! The embedding of a connected graph is given by the eigenvectors of
//! M = D^{-1/2} A D^{-1/2} that belong to the 2nd .. (dim+1)-th largest
//! eigenvalues (equivalently the smallest non-trivial eigenvectors of the
//! normalised Laplacian). They are computed with a thick-restart Lanczos
//! iteration (full re-orthogonalisation, the known trivial eigenvector
//! D^{1/2}1 deflated) that stops on a residual test, `||M x - theta x|| <= TOL`
//! for every wanted Ritz pair, or at a cap on matrix-vector products.
//!
//! A graph with several connected components follows umap-learn's
//! multi-component layout: every component is embedded on its own and scaled to
//! a box around its own anchor point. For up to `2 * dim` components the anchors
//! are the points `+-e_i`; for more components they sit on a regular grid
//! (umap-learn uses the data centroids here, which are not available from a graph).
//!
//! All reductions are summed in a fixed chunk order, so the result depends on the
//! seed only and not on the number of threads.

use faer::Mat;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Residual tolerance of the eigensolver: every wanted Ritz pair satisfies
/// `||M x - theta x||_2 <= TOL` (the spectrum of M lies in [-1, 1]).
pub const SPECTRAL_TOL: f64 = 1e-7;
/// Components up to this size are solved with a dense eigendecomposition.
const DENSE_MAX: usize = 256;
/// Rows per chunk of the deterministic reductions.
const CH: usize = 4096;

/// Result of [`spectral_layout`].
pub struct Spectral {
    /// n × dim, row-major
    pub emb: Vec<f32>,
    /// false when any component stopped at the cap on matrix-vector products
    /// before the residual test was met
    pub converged: bool,
}

/// Default cap on matrix-vector products for a graph of `n` vertices.
pub fn default_max_matvec(n: usize) -> usize {
    n.saturating_mul(5).clamp(1000, 50_000)
}

struct Csr {
    n: usize,
    ptr: Vec<usize>,
    col: Vec<u32>,
    /// entries of D^{-1/2} A D^{-1/2}
    val: Vec<f64>,
}

fn pdot(a: &[f64], b: &[f64]) -> f64 {
    let parts: Vec<f64> = a
        .par_chunks(CH)
        .zip(b.par_chunks(CH))
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>())
        .collect();
    parts.iter().sum()
}

fn matvec(m: &Csr, x: &[f64], y: &mut [f64]) {
    y.par_chunks_mut(CH).enumerate().for_each(|(c, yc)| {
        let lo = c * CH;
        for (r, out) in yc.iter_mut().enumerate() {
            let i = lo + r;
            let mut s = 0.0;
            for p in m.ptr[i]..m.ptr[i + 1] {
                s += m.val[p] * x[m.col[p] as usize];
            }
            *out = s;
        }
    });
}

fn deflate(w: &mut [f64], v0: &[f64]) {
    let a = pdot(v0, w);
    w.par_chunks_mut(CH).zip(v0.par_chunks(CH)).for_each(|(x, y)| x.iter_mut().zip(y).for_each(|(p, q)| *p -= a * q));
}

/// c_j = <V_j, w> for the first `cols` columns of the column-major basis `v`.
fn coeffs(v: &[f64], n: usize, cols: usize, w: &[f64]) -> Vec<f64> {
    let nch = (n + CH - 1) / CH;
    let parts: Vec<Vec<f64>> = (0..nch)
        .into_par_iter()
        .map(|c| {
            let (lo, hi) = (c * CH, ((c + 1) * CH).min(n));
            (0..cols).map(|j| v[j * n + lo..j * n + hi].iter().zip(&w[lo..hi]).map(|(p, q)| p * q).sum::<f64>()).collect()
        })
        .collect();
    let mut out = vec![0.0; cols];
    for p in &parts {
        for (o, x) in out.iter_mut().zip(p) {
            *o += x;
        }
    }
    out
}

/// w -= V c
fn subtract(v: &[f64], n: usize, c: &[f64], w: &mut [f64]) {
    w.par_chunks_mut(CH).enumerate().for_each(|(k, wc)| {
        let lo = k * CH;
        for (j, &a) in c.iter().enumerate() {
            let col = &v[j * n + lo..j * n + lo + wc.len()];
            wc.iter_mut().zip(col).for_each(|(x, y)| *x -= a * y);
        }
    });
}

fn normalise(w: &mut [f64]) -> f64 {
    let nrm = pdot(w, w).sqrt();
    if nrm > 0.0 {
        w.par_chunks_mut(CH).for_each(|c| c.iter_mut().for_each(|x| *x /= nrm));
    }
    nrm
}

/// Eigenvalues (ascending) and eigenvectors (columns of the returned matrix) of a
/// symmetric matrix given row-major.
fn sym_eigen(a: &[f64], m: usize) -> Option<(Vec<f64>, Mat<f64>)> {
    let h = Mat::<f64>::from_fn(m, m, |i, j| 0.5 * (a[i * m + j] + a[j * m + i]));
    let eig = h.self_adjoint_eigen(faer::Side::Lower);
    crate::simd::clean_simd_state();
    let eig = eig.ok()?;
    let s = eig.S().column_vector();
    let vals: Vec<f64> = (0..m).map(|i| s[i]).collect();
    // ascending order is not part of the contract: sort explicitly
    let mut ord: Vec<usize> = (0..m).collect();
    ord.sort_by(|&x, &y| vals[x].partial_cmp(&vals[y]).unwrap_or(std::cmp::Ordering::Equal));
    let u = eig.U();
    let sorted_vals: Vec<f64> = ord.iter().map(|&i| vals[i]).collect();
    let sorted_u = Mat::<f64>::from_fn(m, m, |i, j| u[(i, ord[j])]);
    Some((sorted_vals, sorted_u))
}

/// Top `nev` eigenpairs of M restricted to the complement of `v0` (a unit
/// eigenvector of M). Returns (eigenvalues descending, n × nev column-major
/// eigenvectors, converged).
fn lanczos_top(m: &Csr, v0: &[f64], nev: usize, max_matvec: usize, tol: f64, rng: &mut StdRng) -> (Vec<f64>, Vec<f64>, bool) {
    let n = m.n;
    let ncv_target = if n > 400_000 { (2 * nev + 10).max(24) } else { 48usize.max(4 * nev + 8) };
    let ncv = ncv_target.min(n - 1);
    let keep = (nev + 8).min(ncv.saturating_sub(4)).max(nev);
    let mut v = vec![0.0f64; n * (ncv + 1)];
    let mut hm = vec![0.0f64; ncv * ncv];
    let mut w = vec![0.0f64; n];
    let randn = |rng: &mut StdRng, out: &mut [f64]| out.iter_mut().for_each(|x| *x = StandardNormal.sample(rng));

    randn(rng, &mut v[..n]);
    deflate(&mut v[..n], v0);
    normalise(&mut v[..n]);

    let (mut j0, mut nmv) = (0usize, 0usize);
    loop {
        let mut beta_last = 0.0;
        for j in j0..ncv {
            let (basis, _) = v.split_at(n * (j + 1));
            matvec(m, &basis[j * n..], &mut w);
            nmv += 1;
            deflate(&mut w, v0);
            let mut hcol = vec![0.0; j + 1];
            for _ in 0..2 {
                let c = coeffs(&v, n, j + 1, &w);
                subtract(&v, n, &c, &mut w);
                for (h, x) in hcol.iter_mut().zip(&c) {
                    *h += x;
                }
                deflate(&mut w, v0);
            }
            for i in 0..=j {
                hm[i * ncv + j] = hcol[i];
                hm[j * ncv + i] = hcol[i];
            }
            let beta = pdot(&w, &w).sqrt();
            if beta < 1e-10 {
                // invariant subspace: continue with a fresh direction
                randn(rng, &mut w);
                deflate(&mut w, v0);
                for _ in 0..2 {
                    let c = coeffs(&v, n, j + 1, &w);
                    subtract(&v, n, &c, &mut w);
                    deflate(&mut w, v0);
                }
                normalise(&mut w);
                beta_last = 0.0;
            } else {
                w.par_chunks_mut(CH).for_each(|c| c.iter_mut().for_each(|x| *x /= beta));
                beta_last = beta;
            }
            v[(j + 1) * n..(j + 2) * n].copy_from_slice(&w);
        }

        let (vals, u) = match sym_eigen(&hm, ncv) {
            Some(x) => x,
            None => return (vec![0.0; nev], v[..n * nev].to_vec(), false),
        };
        let resid = |i: usize| (beta_last * u[(ncv - 1, i)]).abs();
        let wanted: Vec<usize> = (0..nev).map(|c| ncv - 1 - c).collect();
        let converged = wanted.iter().all(|&i| resid(i) <= tol);
        if converged || nmv >= max_matvec {
            let mut out = vec![0.0f64; n * nev];
            for (c, &i) in wanted.iter().enumerate() {
                let coef: Vec<f64> = (0..ncv).map(|k| u[(k, i)]).collect();
                out[c * n..(c + 1) * n].par_chunks_mut(CH).enumerate().for_each(|(k, oc)| {
                    let lo = k * CH;
                    for (kk, &a) in coef.iter().enumerate() {
                        let col = &v[kk * n + lo..kk * n + lo + oc.len()];
                        oc.iter_mut().zip(col).for_each(|(x, y)| *x += a * y);
                    }
                });
            }
            return (wanted.iter().map(|&i| vals[i]).collect(), out, converged);
        }

        // thick restart: keep the `keep` largest Ritz vectors plus the residual vector
        let mut tmp = vec![0.0f64; n * keep];
        for c in 0..keep {
            let i = ncv - 1 - c;
            let coef: Vec<f64> = (0..ncv).map(|k| u[(k, i)]).collect();
            tmp[c * n..(c + 1) * n].par_chunks_mut(CH).enumerate().for_each(|(k, oc)| {
                let lo = k * CH;
                for (kk, &a) in coef.iter().enumerate() {
                    let col = &v[kk * n + lo..kk * n + lo + oc.len()];
                    oc.iter_mut().zip(col).for_each(|(x, y)| *x += a * y);
                }
            });
        }
        v.copy_within(ncv * n..(ncv + 1) * n, keep * n);
        v[..keep * n].copy_from_slice(&tmp);
        hm.iter_mut().for_each(|x| *x = 0.0);
        for c in 0..keep {
            hm[c * ncv + c] = vals[ncv - 1 - c];
        }
        j0 = keep;
    }
}

/// Flip each column of an n × cols column-major block so that its entry of largest
/// magnitude is positive (the eigenvector sign is arbitrary; this makes it reproducible).
fn fix_signs(v: &mut [f64], n: usize, cols: usize) {
    for c in 0..cols {
        let col = &mut v[c * n..(c + 1) * n];
        let mut best = 0usize;
        for (i, x) in col.iter().enumerate() {
            if x.abs() > col[best].abs() * (1.0 + 1e-12) {
                best = i;
            }
        }
        if col[best] < 0.0 {
            col.iter_mut().for_each(|x| *x = -*x);
        }
    }
}

/// Eigenvector embedding (n × dim row-major, unscaled) of one connected component.
fn embed_component(m: &Csr, deg: &[f64], dim: usize, max_matvec: usize, rng: &mut StdRng) -> (Vec<f32>, bool) {
    let n = m.n;
    let mut cols = vec![0.0f64; n * dim];
    let mut converged = true;
    if n <= DENSE_MAX {
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for p in m.ptr[i]..m.ptr[i + 1] {
                a[i * n + m.col[p] as usize] += m.val[p];
            }
        }
        match sym_eigen(&a, n) {
            Some((_, u)) => {
                for c in 0..dim {
                    for i in 0..n {
                        cols[c * n + i] = u[(i, n - 2 - c)];
                    }
                }
            }
            None => {
                cols.iter_mut().for_each(|x| *x = StandardNormal.sample(rng));
                converged = false;
            }
        }
    } else {
        let mut v0: Vec<f64> = deg.iter().map(|d| d.sqrt()).collect();
        normalise(&mut v0);
        let (_, vecs, ok) = lanczos_top(m, &v0, dim, max_matvec, SPECTRAL_TOL, rng);
        cols.copy_from_slice(&vecs);
        converged = ok;
    }
    fix_signs(&mut cols, n, dim);
    let mut out = vec![0.0f32; n * dim];
    for c in 0..dim {
        for i in 0..n {
            out[i * dim + c] = cols[c * n + i] as f32;
        }
    }
    (out, converged)
}

/// Anchor point of each of `c` components (row-major c × dim) and the half-width
/// of the box each component is scaled into.
fn anchors(c: usize, dim: usize, order_by_size: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let mut meta = vec![0.0f64; c * dim];
    if c <= 2 * dim {
        let k = (c + 1) / 2;
        for i in 0..c {
            let (axis, sign) = if i < k { (i, 1.0) } else { (i - k, -1.0) };
            meta[i * dim + axis] = sign;
        }
        let range: Vec<f64> = (0..c)
            .map(|i| {
                let mut best = f64::INFINITY;
                for j in 0..c {
                    if i != j {
                        let d2: f64 = (0..dim).map(|a| (meta[i * dim + a] - meta[j * dim + a]).powi(2)).sum();
                        if d2 > 0.0 {
                            best = best.min(d2.sqrt());
                        }
                    }
                }
                best / 2.0
            })
            .collect();
        (meta, range)
    } else {
        // regular grid with unit spacing, largest components first
        let mut g = 1usize;
        while g.pow(dim as u32) < c {
            g += 1;
        }
        for (rank, &comp) in order_by_size.iter().enumerate() {
            let mut r = rank;
            for a in 0..dim {
                meta[comp * dim + a] = (r % g) as f64 - (g as f64 - 1.0) / 2.0;
                r /= g;
            }
        }
        (meta, vec![0.5; c])
    }
}

/// n × dim row-major embedding from the eigenvectors of the normalised graph.
/// `max_matvec` caps the matrix-vector products per component.
pub fn spectral_layout(graph: &super::graph::UmapGraph, dim: usize, max_matvec: usize, seed: u64) -> Spectral {
    let n = graph.n;
    let mut rng = StdRng::seed_from_u64(seed);
    if n <= dim + 2 || graph.edges.is_empty() {
        let emb = (0..n * dim).map(|_| StandardNormal.sample(&mut rng)).collect();
        return Spectral { emb, converged: true };
    }
    let max_matvec = max_matvec.max(1);

    // CSR of the symmetric graph (weights of D^{-1/2} A D^{-1/2})
    let mut deg = vec![0.0f64; n];
    let mut ptr = vec![0usize; n + 1];
    for e in &graph.edges {
        deg[e.source] += e.weight as f64;
        ptr[e.source + 1] += 1;
    }
    for i in 0..n {
        ptr[i + 1] += ptr[i];
    }
    let dis: Vec<f64> = deg.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();
    let mut pos = ptr[..n].to_vec();
    let mut col = vec![0u32; ptr[n]];
    let mut val = vec![0.0f64; ptr[n]];
    for e in &graph.edges {
        let p = pos[e.source];
        col[p] = e.target as u32;
        val[p] = e.weight as f64 * dis[e.source] * dis[e.target];
        pos[e.source] += 1;
    }
    let whole = Csr { n, ptr, col, val };

    // connected components (union-find), labels in order of the smallest vertex
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for e in &graph.edges {
        if e.weight > 0.0 {
            let (a, b) = (find(&mut parent, e.source as u32), find(&mut parent, e.target as u32));
            if a != b {
                parent[a.max(b) as usize] = a.min(b);
            }
        }
    }
    let mut label = vec![u32::MAX; n];
    let mut root_label = vec![u32::MAX; n];
    let mut c = 0usize;
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        if root_label[r] == u32::MAX {
            root_label[r] = c as u32;
            c += 1;
        }
        label[i] = root_label[r];
    }

    if c == 1 {
        let (emb, converged) = embed_component(&whole, &deg, dim, max_matvec, &mut rng);
        return Spectral { emb, converged };
    }

    // several components: umap-learn's multi-component layout
    let mut members: Vec<Vec<u32>> = vec![vec![]; c];
    for i in 0..n {
        members[label[i] as usize].push(i as u32);
    }
    let mut order: Vec<usize> = (0..c).collect();
    order.sort_by(|&a, &b| members[b].len().cmp(&members[a].len()).then(a.cmp(&b)));
    let (meta, range) = anchors(c, dim, &order);
    let mut emb = vec![0.0f32; n * dim];
    let mut converged = true;
    let mut local = vec![0u32; n];
    for (comp, vs) in members.iter().enumerate() {
        let m = vs.len();
        let dr = range[comp];
        if m < 2 * dim || m <= dim + 1 {
            for &v in vs {
                for a in 0..dim {
                    emb[v as usize * dim + a] = (rng.gen_range(-dr..dr) + meta[comp * dim + a]) as f32;
                }
            }
            continue;
        }
        for (k, &v) in vs.iter().enumerate() {
            local[v as usize] = k as u32;
        }
        let mut ptr = vec![0usize; m + 1];
        let mut col = Vec::new();
        let mut val = Vec::new();
        for (k, &v) in vs.iter().enumerate() {
            for p in whole.ptr[v as usize]..whole.ptr[v as usize + 1] {
                col.push(local[whole.col[p] as usize]);
                val.push(whole.val[p]);
            }
            ptr[k + 1] = col.len();
        }
        let sub = Csr { n: m, ptr, col, val };
        let sdeg: Vec<f64> = vs.iter().map(|&v| deg[v as usize]).collect();
        let (e, ok) = embed_component(&sub, &sdeg, dim, max_matvec, &mut rng);
        converged &= ok;
        let mx = e.iter().fold(0.0f32, |a, x| a.max(x.abs())) as f64;
        let expansion = if mx > 0.0 { dr / mx } else { 1.0 };
        for (k, &v) in vs.iter().enumerate() {
            for a in 0..dim {
                emb[v as usize * dim + a] = (e[k * dim + a] as f64 * expansion + meta[comp * dim + a]) as f32;
            }
        }
    }
    Spectral { emb, converged }
}

#[cfg(test)]
mod tests {
    use super::super::graph::{Edge, UmapGraph};
    use super::*;

    fn path(n: usize) -> UmapGraph {
        let mut edges = vec![];
        for i in 0..n - 1 {
            edges.push(Edge { source: i, target: i + 1, weight: 1.0 });
            edges.push(Edge { source: i + 1, target: i, weight: 1.0 });
        }
        UmapGraph { n, edges }
    }

    #[test]
    fn test_spectral_layout_path_graph() {
        // path graph 0-1-...-19: the random-walk Fiedler vector D^{-1/2} v is
        // monotone along the path (v itself is not, at the degree-1 endpoints)
        let n = 20;
        let emb = spectral_layout(&path(n), 2, 2000, 0).emb;
        assert_eq!(emb.len(), n * 2);
        let first: Vec<f32> = (0..n).map(|i| emb[i * 2] / if i == 0 || i == n - 1 { 1.0 } else { 2f32.sqrt() }).collect();
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
        assert_eq!(spectral_layout(&g, 2, 30, 5).emb, spectral_layout(&g, 2, 30, 5).emb);
    }

    /// Normalised path graph: eigenvalues are cos(pi k / (n - 1)).
    #[test]
    fn test_lanczos_matches_closed_form_on_a_long_path() {
        let n = 1500usize;
        let g = path(n);
        let mut deg = vec![0.0f64; n];
        for e in &g.edges {
            deg[e.source] += 1.0;
        }
        let mut ptr = vec![0usize; n + 1];
        let (mut col, mut val) = (vec![], vec![]);
        for i in 0..n {
            for e in g.edges.iter().filter(|e| e.source == i) {
                col.push(e.target as u32);
                val.push(1.0 / (deg[i] * deg[e.target]).sqrt());
            }
            ptr[i + 1] = col.len();
        }
        let m = Csr { n, ptr, col, val };
        let mut v0: Vec<f64> = deg.iter().map(|d| d.sqrt()).collect();
        normalise(&mut v0);
        let mut rng = StdRng::seed_from_u64(3);
        let (vals, vecs, ok) = lanczos_top(&m, &v0, 3, 200_000, SPECTRAL_TOL, &mut rng);
        assert!(ok);
        for (k, v) in vals.iter().enumerate() {
            let exact = (std::f64::consts::PI * (k + 1) as f64 / (n - 1) as f64).cos();
            assert!((v - exact).abs() < 1e-6, "eigenvalue {k}: {v} vs {exact}");
        }
        // residual of the returned vectors
        let mut y = vec![0.0; n];
        for k in 0..3 {
            matvec(&m, &vecs[k * n..(k + 1) * n], &mut y);
            let r: f64 = y.iter().zip(&vecs[k * n..(k + 1) * n]).map(|(a, b)| (a - vals[k] * b).powi(2)).sum::<f64>().sqrt();
            assert!(r < 5e-7, "residual {r}");
        }
    }

    #[test]
    fn test_cap_reports_non_convergence() {
        let n = 1500usize;
        let s = spectral_layout(&path(n), 2, 50, 0);
        assert!(!s.converged);
        assert_eq!(s.emb.len(), n * 2);
    }

    /// Two disjoint paths: the components are laid out around separate anchors.
    #[test]
    fn test_two_components_are_separated() {
        let n = 600usize;
        let mut g = path(n);
        let h = path(n);
        for e in h.edges {
            g.edges.push(Edge { source: e.source + n, target: e.target + n, weight: e.weight });
        }
        g.n = 2 * n;
        let s = spectral_layout(&g, 2, 100_000, 0);
        assert!(s.converged);
        let centroid = |lo: usize| -> (f32, f32) {
            let (mut a, mut b) = (0.0, 0.0);
            for i in lo..lo + n {
                a += s.emb[2 * i];
                b += s.emb[2 * i + 1];
            }
            (a / n as f32, b / n as f32)
        };
        let (c0, c1) = (centroid(0), centroid(n));
        let dist = ((c0.0 - c1.0).powi(2) + (c0.1 - c1.1).powi(2)).sqrt();
        // anchors are (+1, 0) and (-1, 0); each component fills a box of half-width 1
        let spread = |lo: usize, ax: f32| {
            (lo..lo + n).map(|i| (s.emb[2 * i] - ax).abs().max(s.emb[2 * i + 1].abs())).fold(0.0f32, f32::max)
        };
        assert!(dist > 0.1, "centroids coincide: {dist}");
        assert!(spread(0, 1.0) <= 1.0 + 1e-4 && spread(n, -1.0) <= 1.0 + 1e-4);
    }
}
