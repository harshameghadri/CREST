//! k-nearest-neighbour search on dense low-dimensional embeddings (e.g. PCA).
//!
//! Two regimes, both built on dense GEMM tiles (the FAISS "flat" trick
//! ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q.x, evaluated with a SIMD matrix multiply):
//!
//! * small n: exact brute force, tiled so memory stays O(tile^2).
//! * large n: an inverted-file index (IVF) in the spirit of cell lists in
//!   molecular dynamics: points are bucketed by k-means, and each bucket is
//!   searched only against its `nprobe` spatially nearest buckets. The
//!   approximate graph is then polished with one round of NN-descent
//!   ("a neighbour of a neighbour is likely a neighbour", Dong et al. 2011).
//!
//! Final distances are recomputed exactly (no GEMM cancellation error).

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Row-major n × d matrix view.
#[derive(Clone, Copy)]
pub struct Points<'a> {
    pub data: &'a [f32],
    pub n: usize,
    pub d: usize,
}

impl<'a> Points<'a> {
    pub fn new(data: &'a [f32], n: usize, d: usize) -> Self {
        assert_eq!(data.len(), n * d, "data length must equal n * d");
        Points { data, n, d }
    }
    #[inline]
    pub fn row(&self, i: usize) -> &'a [f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

/// k nearest neighbours of every point (self excluded), ascending distance.
pub struct KnnGraph {
    pub n: usize,
    pub k: usize,
    pub indices: Vec<u32>,   // n × k
    pub distances: Vec<f32>, // n × k, Euclidean
}

#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    // chunked so LLVM vectorises it
    let mut acc = [0.0f32; 8];
    let (ca, ra) = a.split_at(a.len() - a.len() % 8);
    let (cb, rb) = b.split_at(ca.len());
    for (x, y) in ca.chunks_exact(8).zip(cb.chunks_exact(8)) {
        for l in 0..8 {
            let t = x[l] - y[l];
            acc[l] += t * t;
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        let t = x - y;
        s += t * t;
    }
    s
}

/// Insert (d, j) into an ascending top-k list if it beats the current worst.
#[inline]
fn push_topk(dist: &mut [f32], idx: &mut [u32], d: f32, j: u32) {
    let k = dist.len();
    if d >= dist[k - 1] {
        return;
    }
    // reject duplicates (possible in refinement)
    if idx.contains(&j) {
        return;
    }
    let mut p = k - 1;
    while p > 0 && dist[p - 1] > d {
        dist[p] = dist[p - 1];
        idx[p] = idx[p - 1];
        p -= 1;
    }
    dist[p] = d;
    idx[p] = j;
}

const TILE: usize = 1024;

/// Search `queries` against `cands` (both lists of row ids) and merge into the
/// queries' top-k lists (`topd`/`topi`, laid out queries.len() × k).
fn search_block(
    pts: Points,
    norms: &[f32],
    queries: &[u32],
    cands: &[u32],
    k: usize,
    topd: &mut [f32],
    topi: &mut [u32],
) {
    let d = pts.d;
    let mut qbuf = vec![0.0f32; queries.len() * d];
    for (r, &q) in queries.iter().enumerate() {
        qbuf[r * d..(r + 1) * d].copy_from_slice(pts.row(q as usize));
    }
    let mut cbuf = vec![0.0f32; TILE.min(cands.len()) * d];
    let mut prod = vec![0.0f32; queries.len() * TILE.min(cands.len())];
    for tile in cands.chunks(TILE) {
        let m = tile.len();
        for (r, &c) in tile.iter().enumerate() {
            cbuf[r * d..(r + 1) * d].copy_from_slice(pts.row(c as usize));
        }
        let lhs = MatRef::from_row_major_slice(&qbuf, queries.len(), d);
        let rhs = MatRef::from_row_major_slice(&cbuf[..m * d], m, d);
        let dst = MatMut::from_row_major_slice_mut(&mut prod[..queries.len() * m], queries.len(), m);
        matmul(dst, Accum::Replace, lhs, rhs.transpose(), 1.0f32, Par::Seq);
        for (r, &q) in queries.iter().enumerate() {
            let qn = norms[q as usize];
            let row = &prod[r * m..(r + 1) * m];
            let (td, ti) = (&mut topd[r * k..(r + 1) * k], &mut topi[r * k..(r + 1) * k]);
            for (t, &c) in tile.iter().enumerate() {
                if c == q {
                    continue;
                }
                let dist = (qn + norms[c as usize] - 2.0 * row[t]).max(0.0);
                if dist < td[k - 1] {
                    push_topk(td, ti, dist, c);
                }
            }
        }
    }
}

/// Exact k-NN by tiled brute force.
fn knn_brute(pts: Points, norms: &[f32], k: usize) -> (Vec<f32>, Vec<u32>) {
    let n = pts.n;
    let all: Vec<u32> = (0..n as u32).collect();
    let mut topd = vec![f32::INFINITY; n * k];
    let mut topi = vec![u32::MAX; n * k];
    let qb = 256;
    topd.par_chunks_mut(qb * k)
        .zip(topi.par_chunks_mut(qb * k))
        .enumerate()
        .for_each(|(b, (td, ti))| {
            let lo = b * qb;
            let hi = (lo + qb).min(n);
            search_block(pts, norms, &all[lo..hi], &all, k, td, ti);
        });
    (topd, topi)
}

/// Assign each point to its nearest centroid (GEMM tiles). Returns labels.
fn assign(pts: Points, cent: &[f32], nc: usize) -> Vec<u32> {
    let d = pts.d;
    let cnorm: Vec<f32> = cent.chunks_exact(d).map(|c| c.iter().map(|x| x * x).sum()).collect();
    let block = 1024;
    let mut labels = vec![0u32; pts.n];
    labels.par_chunks_mut(block).enumerate().for_each(|(b, lab)| {
        let lo = b * block;
        let rows = lab.len();
        let lhs = MatRef::from_row_major_slice(&pts.data[lo * d..(lo + rows) * d], rows, d);
        let rhs = MatRef::from_row_major_slice(cent, nc, d);
        let mut prod = vec![0.0f32; rows * nc];
        matmul(MatMut::from_row_major_slice_mut(&mut prod, rows, nc), Accum::Replace, lhs, rhs.transpose(), 1.0f32, Par::Seq);
        for r in 0..rows {
            let row = &prod[r * nc..(r + 1) * nc];
            let mut best = (f32::INFINITY, 0u32);
            for c in 0..nc {
                let v = cnorm[c] - 2.0 * row[c];
                if v < best.0 {
                    best = (v, c as u32);
                }
            }
            lab[r] = best.1;
        }
    });
    labels
}

/// Lloyd k-means on a random training subsample; returns centroids (nc × d).
fn kmeans(pts: Points, nc: usize, iters: usize, seed: u64) -> Vec<f32> {
    let d = pts.d;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..pts.n).collect();
    order.shuffle(&mut rng);
    let n_train = (nc * 64).min(pts.n);
    let train_idx = &order[..n_train];
    let mut train = Vec::with_capacity(n_train * d);
    for &i in train_idx {
        train.extend_from_slice(pts.row(i));
    }
    let tp = Points::new(&train, n_train, d);
    let mut cent: Vec<f32> = Vec::with_capacity(nc * d);
    for &i in &order[..nc] {
        cent.extend_from_slice(pts.row(i));
    }
    for _ in 0..iters {
        let lab = assign(tp, &cent, nc);
        let mut sums = vec![0.0f64; nc * d];
        let mut cnt = vec![0usize; nc];
        for (r, &c) in lab.iter().enumerate() {
            cnt[c as usize] += 1;
            for (s, &x) in sums[c as usize * d..(c as usize + 1) * d].iter_mut().zip(tp.row(r)) {
                *s += x as f64;
            }
        }
        for c in 0..nc {
            if cnt[c] == 0 {
                // re-seed empty cluster with a random training point
                let r = order[(c * 7919 + 13) % pts.n];
                cent[c * d..(c + 1) * d].copy_from_slice(pts.row(r));
            } else {
                for j in 0..d {
                    cent[c * d + j] = (sums[c * d + j] / cnt[c] as f64) as f32;
                }
            }
        }
    }
    cent
}

/// Approximate k-NN via IVF buckets + NN-descent refinement.
fn knn_ivf(pts: Points, norms: &[f32], k: usize, nprobe: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let (n, d) = (pts.n, pts.d);
    let nc = (n / 512).clamp(8, 8192);
    let cent = kmeans(pts, nc, 8, seed);
    let labels = assign(pts, &cent, nc);
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for (i, &c) in labels.iter().enumerate() {
        lists[c as usize].push(i as u32);
    }

    // For each bucket, its nprobe nearest buckets by centroid distance.
    let cp = Points::new(&cent, nc, d);
    let probes: Vec<Vec<u32>> = (0..nc)
        .into_par_iter()
        .map(|c| {
            let mut ds: Vec<(f32, u32)> = (0..nc).map(|o| (sq_dist(cp.row(c), cp.row(o)), o as u32)).collect();
            let p = nprobe.min(nc);
            ds.select_nth_unstable_by(p - 1, |a, b| a.0.total_cmp(&b.0));
            ds[..p].iter().map(|x| x.1).collect()
        })
        .collect();

    let per_list: Vec<(Vec<f32>, Vec<u32>)> = (0..nc)
        .into_par_iter()
        .map(|c| {
            let queries = &lists[c];
            let mut td = vec![f32::INFINITY; queries.len() * k];
            let mut ti = vec![u32::MAX; queries.len() * k];
            if !queries.is_empty() {
                let cands: Vec<u32> = probes[c].iter().flat_map(|&o| lists[o as usize].iter().copied()).collect();
                // large buckets: split queries so the product tile stays cache-sized
                for (qs, (dd, ii)) in queries.chunks(256).zip(td.chunks_mut(256 * k).zip(ti.chunks_mut(256 * k))) {
                    search_block(pts, norms, qs, &cands, k, dd, ii);
                }
            }
            (td, ti)
        })
        .collect();

    let mut topd = vec![f32::INFINITY; n * k];
    let mut topi = vec![u32::MAX; n * k];
    for (c, (td, ti)) in per_list.into_iter().enumerate() {
        for (r, &q) in lists[c].iter().enumerate() {
            let q = q as usize;
            topd[q * k..(q + 1) * k].copy_from_slice(&td[r * k..(r + 1) * k]);
            topi[q * k..(q + 1) * k].copy_from_slice(&ti[r * k..(r + 1) * k]);
        }
    }

    // NN-descent polish: candidates = neighbours and reverse neighbours of
    // current neighbours (Dong et al. 2011 local join, simplified).
    for _round in 0..3 {
        let old_i = topi.clone();
        // reverse lists (capped at k per node to bound work)
        let mut rev_cnt = vec![0u32; n];
        for &j in &old_i {
            if j != u32::MAX {
                rev_cnt[j as usize] += 1;
            }
        }
        let mut rev_ptr = vec![0usize; n + 1];
        for i in 0..n {
            rev_ptr[i + 1] = rev_ptr[i] + (rev_cnt[i] as usize).min(k);
        }
        let mut rev = vec![0u32; rev_ptr[n]];
        let mut fill = vec![0usize; n];
        for (e, &j) in old_i.iter().enumerate() {
            if j != u32::MAX {
                let j = j as usize;
                if fill[j] < rev_ptr[j + 1] - rev_ptr[j] {
                    rev[rev_ptr[j] + fill[j]] = (e / k) as u32;
                    fill[j] += 1;
                }
            }
        }
        let changed: usize = topd
            .par_chunks_mut(k)
            .zip(topi.par_chunks_mut(k))
            .enumerate()
            .map(|(i, (td, ti))| {
                let mut updates = 0;
                let q = pts.row(i);
                let fwd = &old_i[i * k..(i + 1) * k];
                let bwd = &rev[rev_ptr[i]..rev_ptr[i + 1]];
                for &j in fwd.iter().chain(bwd) {
                    if j == u32::MAX {
                        continue;
                    }
                    let j = j as usize;
                    let second = old_i[j * k..(j + 1) * k].iter().chain(&rev[rev_ptr[j]..rev_ptr[j + 1]]);
                    for &l in std::iter::once(&(j as u32)).chain(second) {
                        if l == u32::MAX || l as usize == i {
                            continue;
                        }
                        let dist = sq_dist(q, pts.row(l as usize));
                        if dist < td[k - 1] && !ti.contains(&l) {
                            push_topk(td, ti, dist, l);
                            updates += 1;
                        }
                    }
                }
                updates
            })
            .sum();
        if (changed as f64) < 0.001 * (n * k) as f64 {
            break;
        }
    }
    (topd, topi)
}

/// k nearest neighbours (self excluded). `exact=None` picks brute force for
/// n <= 20_000 and IVF + NN-descent above.
pub fn knn(pts: Points, k: usize, exact: Option<bool>, seed: u64) -> KnnGraph {
    let n = pts.n;
    if n == 0 {
        return KnnGraph { n, k: 0, indices: vec![], distances: vec![] };
    }
    let k = k.min(n - 1).max(if n > 1 { 1 } else { 0 });
    if k == 0 {
        return KnnGraph { n, k: 0, indices: vec![], distances: vec![] };
    }
    let norms: Vec<f32> = (0..n).into_par_iter().map(|i| pts.row(i).iter().map(|x| x * x).sum()).collect();
    let use_exact = exact.unwrap_or(n <= 20_000);
    let (_, mut topi) = if use_exact {
        knn_brute(pts, &norms, k)
    } else {
        knn_ivf(pts, &norms, k, 20, seed)
    };

    // exact distances, re-sorted
    let mut topd = vec![0.0f32; n * k];
    topd.par_chunks_mut(k).zip(topi.par_chunks_mut(k)).enumerate().for_each(|(i, (td, ti))| {
        let mut pairs: Vec<(f32, u32)> = ti
            .iter()
            .map(|&j| if j == u32::MAX { (f32::INFINITY, j) } else { (sq_dist(pts.row(i), pts.row(j as usize)).sqrt(), j) })
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (s, (dd, jj)) in pairs.into_iter().enumerate() {
            td[s] = dd;
            ti[s] = jj;
        }
    });
    KnnGraph { n, k, indices: topi, distances: topd }
}

/// Convenience wrapper for callers holding `Vec<Vec<f32>>` rows.
pub fn hnsw_knn(data: &[Vec<f32>], k: usize) -> Vec<(Vec<usize>, Vec<f32>)> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let d = data[0].len();
    let flat: Vec<f32> = data.iter().flat_map(|r| r.iter().copied()).collect();
    let g = knn(Points::new(&flat, n, d), k, None, 0);
    (0..n)
        .map(|i| {
            (
                g.indices[i * g.k..(i + 1) * g.k].iter().map(|&j| j as usize).collect(),
                g.distances[i * g.k..(i + 1) * g.k].to_vec(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn clustered(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..20).map(|_| (0..d).map(|_| rng.gen_range(-10.0f32..10.0)).collect()).collect();
        (0..n)
            .flat_map(|i| {
                let c = &centers[i % 20];
                let noise: Vec<f32> = (0..d).map(|j| c[j] + rng.gen_range(-1.5f32..1.5)).collect();
                noise
            })
            .collect()
    }

    fn recall(pts: Points, g: &KnnGraph) -> f64 {
        let exact = knn(pts, g.k, Some(true), 0);
        let mut hits = 0;
        for i in 0..pts.n {
            let t = &exact.indices[i * g.k..(i + 1) * g.k];
            hits += g.indices[i * g.k..(i + 1) * g.k].iter().filter(|j| t.contains(j)).count();
        }
        hits as f64 / (pts.n * g.k) as f64
    }

    #[test]
    fn test_brute_matches_naive() {
        let data = clustered(600, 7, 1);
        let pts = Points::new(&data, 600, 7);
        let g = knn(pts, 10, Some(true), 0);
        for i in 0..600 {
            let mut all: Vec<(f32, u32)> = (0..600u32)
                .filter(|&j| j as usize != i)
                .map(|j| (sq_dist(pts.row(i), pts.row(j as usize)), j))
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert!(!g.indices[i * 10..(i + 1) * 10].contains(&(i as u32)));
            // k-th distance must match exactly (ties may permute indices)
            assert!((g.distances[i * 10 + 9] - all[9].0.sqrt()).abs() < 1e-4);
        }
    }

    #[test]
    fn test_ivf_recall() {
        let (n, d) = (30_000, 30);
        let data = clustered(n, d, 2);
        let pts = Points::new(&data, n, d);
        let g = knn(pts, 15, Some(false), 0);
        let r = recall(pts, &g);
        assert!(r > 0.97, "IVF recall {}", r);
        for w in g.distances.chunks(15) {
            assert!(w.windows(2).all(|p| p[0] <= p[1]));
        }
    }
}
