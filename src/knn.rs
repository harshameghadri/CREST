//! k-nearest-neighbour search on dense low-dimensional embeddings (e.g. PCA).
//!
//! Two regimes, both built on dense GEMM tiles (the FAISS "flat" trick
//! ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q.x, evaluated with a SIMD matrix multiply):
//!
//! * small n: exact brute force, tiled so memory stays O(tile^2).
//! * large n: an inverted-file index (IVF) in the spirit of cell lists in
//!   molecular dynamics: points are bucketed by k-means, and every point is
//!   searched against the `nprobe` buckets whose centroids are nearest to the
//!   point itself (a point near the edge of its bucket has most neighbours in other buckets). The
//!   approximate graph is then polished with one round of NN-descent
//!   ("a neighbour of a neighbour is likely a neighbour", Dong et al. 2011).
//!
//! The GEMM only *screens* candidates. Coordinates are first centred on the
//! reference mean (accumulated in f64), which makes the expansion translation
//! invariant, and a candidate is admitted to a top-k list only if its screened
//! distance, lowered by a rounding-error bound, could beat the current k-th
//! exact distance; admitted candidates are compared by their exact squared
//! distance. Final distances are recomputed exactly (no GEMM cancellation error).

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

/// Index of the first row of `data` (row length `d`) holding a NaN or infinite
/// value, if any.
pub fn first_non_finite_row(data: &[f32], d: usize) -> Option<usize> {
    if d == 0 || data.is_empty() {
        return None;
    }
    data.par_chunks(d).position_first(|row| row.iter().any(|x| !x.is_finite()))
}

/// Per-column means in f64. Rows are summed in fixed blocks, so the result does
/// not depend on the number of threads.
fn column_means(pts: Points) -> Vec<f64> {
    const BLOCK: usize = 4096;
    let d = pts.d;
    let parts: Vec<Vec<f64>> = pts
        .data
        .par_chunks(BLOCK * d)
        .map(|blk| {
            let mut s = vec![0.0f64; d];
            for row in blk.chunks_exact(d) {
                for (a, &x) in s.iter_mut().zip(row) {
                    *a += x as f64;
                }
            }
            s
        })
        .collect();
    let mut m = vec![0.0f64; d];
    for p in &parts {
        for (a, b) in m.iter_mut().zip(p) {
            *a += b;
        }
    }
    for a in m.iter_mut() {
        *a /= pts.n as f64;
    }
    m
}

/// `pts - shift`, computed in f64 and rounded once to f32.
fn centred(pts: Points, shift: &[f64]) -> Vec<f32> {
    let d = pts.d;
    let mut out = vec![0.0f32; pts.n * d];
    out.par_chunks_mut(d).zip(pts.data.par_chunks(d)).for_each(|(o, r)| {
        for j in 0..d {
            o[j] = (r[j] as f64 - shift[j]) as f32;
        }
    });
    out
}

fn row_norms(pts: Points) -> Vec<f32> {
    (0..pts.n).into_par_iter().map(|i| pts.row(i).iter().map(|x| x * x).sum()).collect()
}

/// Relative bound on the f32 rounding error of `||q||^2 + ||c||^2 - 2 q.c`
/// (multiplied by `||q||^2 + ||c||^2`), with a safety factor of about 2.
#[inline]
fn gemm_slack_coef(d: usize) -> f32 {
    (d as f32 + 4.0) * 1.2e-7
}

/// Exact searches are used up to this many points when `exact` is not given.
pub const EXACT_MAX_POINTS: usize = 20_000;

/// Which search `knn` runs for a given input size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnnPlan {
    pub exact: bool,
    /// inverted lists probed per list (0 for the exact search)
    pub nprobe: usize,
    /// inverted lists of the index (0 for the exact search)
    pub n_lists: usize,
}

/// Number of inverted lists used for `n` points.
fn ivf_lists(n: usize) -> usize {
    (n / IVF_LIST_SIZE).clamp(IVF_MIN_LISTS, 8192)
}

/// Target number of points per inverted list of the IVF search.
const IVF_LIST_SIZE: usize = 512;

/// Smallest number of lists a cell probes in the IVF search.
const IVF_MIN_NPROBE: usize = 16;

/// Cells of one list are searched together against the union of the lists they probe; a group is closed
/// when that union would exceed this many times `nprobe` lists (larger groups make larger GEMMs).
const IVF_JOB_LISTS_PER_PROBE: usize = 6;

/// Lists probed per cell: the `nprobe` lists whose centroids are nearest to the cell. The number of lists
/// grows with n, so a fixed count would search a shrinking fraction of the data; the rule is
/// `max(16, ceil(0.5 sqrt(n_lists)))`.
fn ivf_nprobe(nc: usize) -> usize {
    ((0.5 * (nc as f64).sqrt()).ceil() as usize).max(IVF_MIN_NPROBE)
}

/// The search `knn(.., n points, exact, ..)` will run.
#[cfg(test)]
pub fn knn_plan(n: usize, exact: Option<bool>) -> KnnPlan {
    knn_plan_with(n, exact, None)
}

/// [`knn_plan`] with the number of lists probed per cell set by the caller (clamped to the number of lists;
/// `None` uses the rule of [`ivf_nprobe`]).
pub fn knn_plan_with(n: usize, exact: Option<bool>, nprobe: Option<usize>) -> KnnPlan {
    if exact.unwrap_or(n <= EXACT_MAX_POINTS) || n < IVF_MIN_LISTS {
        KnnPlan { exact: true, nprobe: 0, n_lists: 0 }
    } else {
        let nc = ivf_lists(n);
        KnnPlan { exact: false, nprobe: nprobe.unwrap_or_else(|| ivf_nprobe(nc)).clamp(1, nc), n_lists: nc }
    }
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
/// queries' top-k lists (`topd`/`topi`, laid out queries.len() × k; `topd` holds
/// exact squared distances). `cpts`/`cnorms` are the centred coordinates and
/// their squared norms used for the GEMM screen.
#[allow(clippy::too_many_arguments)]
fn search_block(
    pts: Points,
    cpts: Points,
    cnorms: &[f32],
    queries: &[u32],
    cands: &[u32],
    k: usize,
    topd: &mut [f32],
    topi: &mut [u32],
) {
    let d = pts.d;
    let coef = gemm_slack_coef(d);
    let mut qbuf = vec![0.0f32; queries.len() * d];
    for (r, &q) in queries.iter().enumerate() {
        qbuf[r * d..(r + 1) * d].copy_from_slice(cpts.row(q as usize));
    }
    let mut cbuf = vec![0.0f32; TILE.min(cands.len()) * d];
    let mut prod = vec![0.0f32; queries.len() * TILE.min(cands.len())];
    for tile in cands.chunks(TILE) {
        let m = tile.len();
        for (r, &c) in tile.iter().enumerate() {
            cbuf[r * d..(r + 1) * d].copy_from_slice(cpts.row(c as usize));
        }
        let lhs = MatRef::from_row_major_slice(&qbuf, queries.len(), d);
        let rhs = MatRef::from_row_major_slice(&cbuf[..m * d], m, d);
        let dst = MatMut::from_row_major_slice_mut(&mut prod[..queries.len() * m], queries.len(), m);
        matmul(dst, Accum::Replace, lhs, rhs.transpose(), 1.0f32, Par::Seq);
        for (r, &q) in queries.iter().enumerate() {
            let qn = cnorms[q as usize];
            let row = &prod[r * m..(r + 1) * m];
            let (td, ti) = (&mut topd[r * k..(r + 1) * k], &mut topi[r * k..(r + 1) * k]);
            for (t, &c) in tile.iter().enumerate() {
                if c == q {
                    continue;
                }
                let cn = cnorms[c as usize];
                let screened = qn + cn - 2.0 * row[t];
                // a NaN screen compares false and is never admitted
                if screened - coef * (qn + cn) < td[k - 1] {
                    let dist = sq_dist(pts.row(q as usize), pts.row(c as usize));
                    if dist < td[k - 1] {
                        push_topk(td, ti, dist, c);
                    }
                }
            }
        }
    }
}

/// Points reordered list by list, so a list is a contiguous block of rows for the GEMM (no gather per job).
struct IvfLayout {
    off: Vec<usize>,
    ids: Vec<u32>,
    data: Vec<f32>,
    norms: Vec<f32>,
}

impl IvfLayout {
    fn new(cpts: Points, cnorms: &[f32], lists: &[Vec<u32>]) -> Self {
        let d = cpts.d;
        let mut off = vec![0usize; lists.len() + 1];
        for (c, l) in lists.iter().enumerate() {
            off[c + 1] = off[c] + l.len();
        }
        let ids: Vec<u32> = lists.iter().flat_map(|l| l.iter().copied()).collect();
        let mut data = vec![0.0f32; ids.len() * d];
        data.par_chunks_mut(d).zip(ids.par_iter()).for_each(|(o, &i)| o.copy_from_slice(cpts.row(i as usize)));
        let norms: Vec<f32> = ids.par_iter().map(|&i| cnorms[i as usize]).collect();
        IvfLayout { off, ids, data, norms }
    }
}

/// [`search_block`] against whole inverted lists of an [`IvfLayout`].
#[allow(clippy::too_many_arguments)]
fn search_lists(
    pts: Points,
    cpts: Points,
    cnorms: &[f32],
    lay: &IvfLayout,
    queries: &[u32],
    lists: &[u32],
    k: usize,
    topd: &mut [f32],
    topi: &mut [u32],
) {
    let d = pts.d;
    let coef = gemm_slack_coef(d);
    let mut qbuf = vec![0.0f32; queries.len() * d];
    for (r, &q) in queries.iter().enumerate() {
        qbuf[r * d..(r + 1) * d].copy_from_slice(cpts.row(q as usize));
    }
    let mut prod = vec![0.0f32; queries.len() * TILE];
    for &o in lists {
        let (lo, hi) = (lay.off[o as usize], lay.off[o as usize + 1]);
        let mut a = lo;
        while a < hi {
            let b = (a + TILE).min(hi);
            let m = b - a;
            let lhs = MatRef::from_row_major_slice(&qbuf, queries.len(), d);
            let rhs = MatRef::from_row_major_slice(&lay.data[a * d..b * d], m, d);
            let dst = MatMut::from_row_major_slice_mut(&mut prod[..queries.len() * m], queries.len(), m);
            matmul(dst, Accum::Replace, lhs, rhs.transpose(), 1.0f32, Par::Seq);
            for (r, &q) in queries.iter().enumerate() {
                let qn = cnorms[q as usize];
                let row = &prod[r * m..(r + 1) * m];
                let (td, ti) = (&mut topd[r * k..(r + 1) * k], &mut topi[r * k..(r + 1) * k]);
                for t in 0..m {
                    let c = lay.ids[a + t];
                    if c == q {
                        continue;
                    }
                    let cn = lay.norms[a + t];
                    let screened = qn + cn - 2.0 * row[t];
                    if screened - coef * (qn + cn) < td[k - 1] {
                        let dist = sq_dist(pts.row(q as usize), pts.row(c as usize));
                        if dist < td[k - 1] {
                            push_topk(td, ti, dist, c);
                        }
                    }
                }
            }
            a = b;
        }
    }
}

/// Exact k-NN by tiled brute force.
fn knn_brute(pts: Points, cpts: Points, cnorms: &[f32], k: usize) -> (Vec<f32>, Vec<u32>) {
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
            search_block(pts, cpts, cnorms, &all[lo..hi], &all, k, td, ti);
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

/// Smallest number of IVF lists (k-means centroids) used by [`knn_ivf`].
const IVF_MIN_LISTS: usize = 8;

/// Approximate k-NN via IVF buckets + NN-descent refinement.
fn knn_ivf(pts: Points, cpts: Points, cnorms: &[f32], k: usize, nprobe: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let (n, d) = (pts.n, pts.d);
    let nc = ivf_lists(n);
    debug_assert!(nc <= n);
    let cent = kmeans(cpts, nc, 8, seed);
    let labels = assign(cpts, &cent, nc);
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for (i, &c) in labels.iter().enumerate() {
        lists[c as usize].push(i as u32);
    }

    // Every cell probes the lists nearest to the cell itself, not the lists around the centroid of its own
    // list: a cell near the boundary of its list has most of its neighbours elsewhere. Cells of one list
    // are searched together against the union of their lists (closed at `cap` lists or 256 cells).
    let cnorm: Vec<f32> = cent.chunks_exact(d).map(|c| c.iter().map(|x| x * x).sum()).collect();
    let lay = IvfLayout::new(cpts, cnorms, &lists);
    let p = nprobe.clamp(1, nc);
    let cap = IVF_JOB_LISTS_PER_PROBE * p;
    let per_list: Vec<(Vec<f32>, Vec<u32>)> = (0..nc)
        .into_par_iter()
        .map_init(
            || vec![false; nc],
            |member, c| {
                let queries = &lists[c];
                let m = queries.len();
                let mut td = vec![f32::INFINITY; m * k];
                let mut ti = vec![u32::MAX; m * k];
                if m > 0 {
                    let mut near = vec![0u32; m * p];
                    nearest_lists_of_rows(cpts, queries, &cent, &cnorm, nc, p, &mut near);
                    let local: Vec<u32> = (0..m as u32).collect();
                    for job in split_into_jobs(&local, &near, p, cap, member) {
                        let gq: Vec<u32> = job.queries.iter().map(|&l| queries[l as usize]).collect();
                        let mut jd = vec![f32::INFINITY; gq.len() * k];
                        let mut ji = vec![u32::MAX; gq.len() * k];
                        search_lists(pts, cpts, cnorms, &lay, &gq, &job.lists, k, &mut jd, &mut ji);
                        for (r, &l) in job.queries.iter().enumerate() {
                            let l = l as usize;
                            td[l * k..(l + 1) * k].copy_from_slice(&jd[r * k..(r + 1) * k]);
                            ti[l * k..(l + 1) * k].copy_from_slice(&ji[r * k..(r + 1) * k]);
                        }
                    }
                }
                (td, ti)
            },
        )
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
/// n <= 20_000 and IVF + NN-descent above. IVF needs at least IVF_MIN_LISTS
/// points (one per inverted list); tinier inputs use the exact search.
#[cfg(test)]
pub fn knn(pts: Points, k: usize, exact: Option<bool>, seed: u64) -> KnnGraph {
    knn_with(pts, k, exact, seed, None)
}

/// [`knn`] with the number of lists probed per cell set by the caller (see [`knn_plan_with`]).
pub fn knn_with(pts: Points, k: usize, exact: Option<bool>, seed: u64, nprobe: Option<usize>) -> KnnGraph {
    let n = pts.n;
    if n == 0 {
        return KnnGraph { n, k: 0, indices: vec![], distances: vec![] };
    }
    let k = k.min(n - 1).max(if n > 1 { 1 } else { 0 });
    if k == 0 {
        return KnnGraph { n, k: 0, indices: vec![], distances: vec![] };
    }
    // centre on the column means (f64) so the expanded distance is translation invariant
    let shift = column_means(pts);
    let cdata = centred(pts, &shift);
    let cpts = Points::new(&cdata, n, pts.d);
    let cnorms = row_norms(cpts);
    let plan = knn_plan_with(n, exact, nprobe);
    let (_, mut topi) = if plan.exact {
        knn_brute(pts, cpts, &cnorms, k)
    } else {
        knn_ivf(pts, cpts, &cnorms, k, plan.nprobe, seed)
    };
    crate::simd::clean_simd_state(); // after faer GEMM tiles

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

/// Top-k search of query rows against candidate rows of a second point set.
/// `qp`/`rp` are the original coordinates (exact distances), `cqp`/`crp` the
/// reference-centred copies used for the GEMM screen.
#[allow(clippy::too_many_arguments)]
fn search_block_ref(
    qp: Points,
    cqp: Points,
    qnorms: &[f32],
    rp: Points,
    crp: Points,
    rnorms: &[f32],
    queries: &[u32],
    cands: &[u32],
    k: usize,
    topd: &mut [f32],
    topi: &mut [u32],
) {
    let d = qp.d;
    let coef = gemm_slack_coef(d);
    let mut qbuf = vec![0.0f32; queries.len() * d];
    for (r, &q) in queries.iter().enumerate() {
        qbuf[r * d..(r + 1) * d].copy_from_slice(cqp.row(q as usize));
    }
    let mut cbuf = vec![0.0f32; TILE.min(cands.len()) * d];
    let mut prod = vec![0.0f32; queries.len() * TILE.min(cands.len())];
    for tile in cands.chunks(TILE) {
        let m = tile.len();
        for (r, &c) in tile.iter().enumerate() {
            cbuf[r * d..(r + 1) * d].copy_from_slice(crp.row(c as usize));
        }
        let lhs = MatRef::from_row_major_slice(&qbuf, queries.len(), d);
        let rhs = MatRef::from_row_major_slice(&cbuf[..m * d], m, d);
        let dst = MatMut::from_row_major_slice_mut(&mut prod[..queries.len() * m], queries.len(), m);
        matmul(dst, Accum::Replace, lhs, rhs.transpose(), 1.0f32, Par::Seq);
        for (r, &q) in queries.iter().enumerate() {
            let qn = qnorms[q as usize];
            let row = &prod[r * m..(r + 1) * m];
            let (td, ti) = (&mut topd[r * k..(r + 1) * k], &mut topi[r * k..(r + 1) * k]);
            for (t, &c) in tile.iter().enumerate() {
                let cn = rnorms[c as usize];
                let screened = qn + cn - 2.0 * row[t];
                if screened - coef * (qn + cn) < td[k - 1] {
                    let dist = sq_dist(qp.row(q as usize), rp.row(c as usize));
                    if dist < td[k - 1] {
                        push_topk(td, ti, dist, c);
                    }
                }
            }
        }
    }
}

/// Lists probed per query by the IVF path of [`knn_query`]. Queries need not lie on
/// the reference manifold, so they probe more lists than the points of the index
/// do: `max(32, ceil(2.5 sqrt(nc)))`.
fn query_nprobe(nc: usize) -> usize {
    ((2.5 * (nc as f64).sqrt()).ceil() as usize).max(32).min(nc)
}

/// The query rows of one search job and the inverted lists they are searched in.
struct QueryJob {
    queries: Vec<u32>,
    lists: Vec<u32>,
}

/// For the rows `rows` of `cq`: the `p` inverted lists nearest to each row (nearest first), written to `out`
/// (`rows.len() × p`).
fn nearest_lists_of_rows(cq: Points, rows: &[u32], cent: &[f32], cnorm: &[f32], nc: usize, p: usize, out: &mut [u32]) {
    let d = cq.d;
    let block = 512;
    let mut buf = vec![0.0f32; block.min(rows.len()) * d];
    let mut prod = vec![0.0f32; block.min(rows.len()) * nc];
    let mut ds: Vec<(f32, u32)> = Vec::with_capacity(nc);
    let cmp = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    for (b, o) in out.chunks_mut(block * p).enumerate() {
        let lo = b * block;
        let r = o.len() / p;
        for (i, &q) in rows[lo..lo + r].iter().enumerate() {
            buf[i * d..(i + 1) * d].copy_from_slice(cq.row(q as usize));
        }
        let lhs = MatRef::from_row_major_slice(&buf[..r * d], r, d);
        let rhs = MatRef::from_row_major_slice(cent, nc, d);
        matmul(MatMut::from_row_major_slice_mut(&mut prod[..r * nc], r, nc), Accum::Replace, lhs, rhs.transpose(), 1.0f32, Par::Seq);
        for i in 0..r {
            ds.clear();
            // ||q||^2 is the same for every list, so it does not change the order
            ds.extend((0..nc).map(|c| (cnorm[c] - 2.0 * prod[i * nc + c], c as u32)));
            if p < nc {
                ds.select_nth_unstable_by(p - 1, cmp);
            }
            ds[..p].sort_by(cmp);
            for (slot, e) in o[i * p..(i + 1) * p].iter_mut().zip(&ds[..p]) {
                *slot = e.1;
            }
        }
    }
}

/// The `p` inverted lists nearest to every query row (nearest first), `nq × p`.
fn nearest_lists(cq: Points, cent: &[f32], nc: usize, p: usize) -> Vec<u32> {
    let d = cq.d;
    let cnorm: Vec<f32> = cent.chunks_exact(d).map(|c| c.iter().map(|x| x * x).sum()).collect();
    let rows: Vec<u32> = (0..cq.n as u32).collect();
    let mut out = vec![0u32; cq.n * p];
    out.par_chunks_mut(512 * p).enumerate().for_each(|(b, o)| {
        let lo = b * 512;
        nearest_lists_of_rows(cq, &rows[lo..lo + o.len() / p], cent, &cnorm, nc, p, o);
    });
    out
}

/// Split the queries whose nearest list is the same into search jobs. A job
/// searches the union of its queries' own nearest lists (so every query sees at
/// least its `p` nearest lists) and is closed before that union exceeds `cap`
/// lists or the job holds 256 queries.
fn split_into_jobs(qs: &[u32], nearest: &[u32], p: usize, cap: usize, member: &mut [bool]) -> Vec<QueryJob> {
    let mut order: Vec<u32> = qs.to_vec();
    order.sort_by(|&a, &b| nearest[a as usize * p..(a as usize + 1) * p].cmp(&nearest[b as usize * p..(b as usize + 1) * p]).then(a.cmp(&b)));
    let mut jobs = Vec::new();
    let mut cur = QueryJob { queries: Vec::new(), lists: Vec::new() };
    for &q in &order {
        let probe = &nearest[q as usize * p..(q as usize + 1) * p];
        let fresh = probe.iter().filter(|&&l| !member[l as usize]).count();
        if !cur.queries.is_empty() && (cur.queries.len() >= 256 || cur.lists.len() + fresh > cap) {
            for &l in &cur.lists {
                member[l as usize] = false;
            }
            jobs.push(std::mem::replace(&mut cur, QueryJob { queries: Vec::new(), lists: Vec::new() }));
        }
        for &l in probe {
            if !member[l as usize] {
                member[l as usize] = true;
                cur.lists.push(l);
            }
        }
        cur.queries.push(q);
    }
    for &l in &cur.lists {
        member[l as usize] = false;
    }
    if !cur.queries.is_empty() {
        jobs.push(cur);
    }
    jobs
}

/// k nearest reference rows of each query row (reference-to-query mapping).
/// `exact=None` is brute force when n_ref × n_query <= 2e8, otherwise an IVF
/// index over the reference in which every query probes the lists nearest to
/// the query itself.
pub fn knn_query(refp: Points, query: Points, k: usize, exact: Option<bool>, seed: u64) -> KnnGraph {
    let (nr, nq) = (refp.n, query.n);
    let k = k.min(nr);
    if nq == 0 || k == 0 {
        return KnnGraph { n: nq, k: 0, indices: vec![], distances: vec![] };
    }
    let d = refp.d;
    // centre both sets on the reference mean (f64): translation invariant screening
    let shift = column_means(refp);
    let crdata = centred(refp, &shift);
    let cqdata = centred(query, &shift);
    let (crp, cqp) = (Points::new(&crdata, nr, d), Points::new(&cqdata, nq, d));
    let rnorms = row_norms(crp);
    let qnorms = row_norms(cqp);
    let use_exact = exact.unwrap_or((nr as f64) * (nq as f64) <= 2e8);
    let (lists, jobs): (Vec<Vec<u32>>, Vec<QueryJob>) = if use_exact {
        let jobs = (0..nq as u32)
            .collect::<Vec<u32>>()
            .chunks(256)
            .map(|c| QueryJob { queries: c.to_vec(), lists: vec![0] })
            .collect();
        (vec![(0..nr as u32).collect()], jobs)
    } else {
        let nc = (nr / 512).clamp(8, 8192).min(nr);
        let p = query_nprobe(nc);
        let cent = kmeans(crp, nc, 8, seed);
        let mut lists = vec![Vec::new(); nc];
        for (i, &c) in assign(crp, &cent, nc).iter().enumerate() {
            lists[c as usize].push(i as u32);
        }
        let nearest = nearest_lists(cqp, &cent, nc, p);
        let mut groups: Vec<Vec<u32>> = vec![Vec::new(); nc];
        for q in 0..nq {
            groups[nearest[q * p] as usize].push(q as u32);
        }
        let jobs: Vec<QueryJob> = groups
            .par_iter()
            .map_init(|| vec![false; nc], |member, qs| split_into_jobs(qs, &nearest, p, 2 * p, member))
            .collect::<Vec<Vec<QueryJob>>>()
            .into_iter()
            .flatten()
            .collect();
        (lists, jobs)
    };
    let per: Vec<(Vec<u32>, Vec<u32>)> = jobs
        .par_iter()
        .map(|job| {
            let cands: Vec<u32> = job.lists.iter().flat_map(|&o| lists[o as usize].iter().copied()).collect();
            let mut td = vec![f32::INFINITY; job.queries.len() * k];
            let mut ti = vec![u32::MAX; job.queries.len() * k];
            for (qs, (dd, ii)) in job.queries.chunks(256).zip(td.chunks_mut(256 * k).zip(ti.chunks_mut(256 * k))) {
                search_block_ref(query, cqp, &qnorms, refp, crp, &rnorms, qs, &cands, k, dd, ii);
            }
            (job.queries.clone(), ti)
        })
        .collect();
    crate::simd::clean_simd_state();
    let mut topi = vec![u32::MAX; nq * k];
    for (qs, ti) in &per {
        for (r, &q) in qs.iter().enumerate() {
            let q = q as usize;
            topi[q * k..(q + 1) * k].copy_from_slice(&ti[r * k..(r + 1) * k]);
        }
    }
    let mut topd = vec![0.0f32; nq * k];
    topd.par_chunks_mut(k).zip(topi.par_chunks_mut(k)).enumerate().for_each(|(i, (td, ti))| {
        let mut pairs: Vec<(f32, u32)> = ti
            .iter()
            .map(|&j| if j == u32::MAX { (f32::INFINITY, j) } else { (sq_dist(query.row(i), refp.row(j as usize)).sqrt(), j) })
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (s, (dd, jj)) in pairs.into_iter().enumerate() {
            td[s] = dd;
            ti[s] = jj;
        }
    });
    KnnGraph { n: nq, k, indices: topi, distances: topd }
}

#[cfg(test)]
mod query_tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn query_ivf_matches_brute() {
        let (nr, nq, d, k) = (20_000usize, 2_000usize, 8usize, 10usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let r: Vec<f32> = (0..nr * d).map(|_| rng.gen::<f32>()).collect();
        let q: Vec<f32> = (0..nq * d).map(|_| rng.gen::<f32>()).collect();
        let (rp, qp) = (Points::new(&r, nr, d), Points::new(&q, nq, d));
        let ex = knn_query(rp, qp, k, Some(true), 0);
        let ap = knn_query(rp, qp, k, Some(false), 0);
        // brute force against a direct scan for the first query
        let mut all: Vec<(f32, u32)> = (0..nr).map(|j| (sq_dist(qp.row(0), rp.row(j)), j as u32)).collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        let want: Vec<u32> = all[..k].iter().map(|x| x.1).collect();
        assert_eq!(&ex.indices[..k], &want[..]);
        let mut hit = 0;
        for i in 0..nq {
            let e = &ex.indices[i * k..(i + 1) * k];
            hit += ap.indices[i * k..(i + 1) * k].iter().filter(|j| e.contains(j)).count();
        }
        let recall = hit as f64 / (nq * k) as f64;
        assert!(recall > 0.97, "recall {}", recall);
    }

    #[test]
    fn test_knn_approximate_tiny_inputs() {
        // exact=false used to panic for n < 8 (more IVF lists than points)
        for n in 2..40usize {
            let mut rng = rand::rngs::StdRng::seed_from_u64(n as u64);
            let data: Vec<f32> = (0..n * 4).map(|_| rng.gen_range(-5.0f32..5.0)).collect();
            let pts = Points::new(&data, n, 4);
            let g = knn(pts, 3, Some(false), 0);
            let ex = knn(pts, 3, Some(true), 0);
            assert_eq!(g.k, 3.min(n - 1));
            assert!(g.indices.iter().all(|&j| (j as usize) < n), "placeholder neighbour at n={}", n);
            for i in 0..n {
                assert!(g.indices[i * g.k..(i + 1) * g.k].iter().all(|&j| j as usize != i));
            }
            if n < IVF_MIN_LISTS {
                assert_eq!(g.indices, ex.indices);
            }
        }
    }
}

#[cfg(test)]
mod precision_tests {
    use super::*;
    use rand::Rng;

    fn embedding(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n * d).map(|i| rng.gen_range(-1.0f32..1.0) * (3.0 / (1.0 + (i % d) as f32).sqrt())).collect()
    }

    /// float64 brute-force neighbour sets of the float32 array actually passed.
    fn truth(data: &[f32], n: usize, d: usize, k: usize, i: usize) -> Vec<u32> {
        let mut all: Vec<(f64, u32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let s: f64 = (0..d).map(|c| (data[i * d + c] as f64 - data[j * d + c] as f64).powi(2)).sum();
                (s, j as u32)
            })
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        all[..k].iter().map(|x| x.1).collect()
    }

    #[test]
    fn exact_search_ignores_the_offset_of_the_data() {
        let (n, d, k) = (1500usize, 40usize, 10usize);
        let base = embedding(n, d, 7);
        for offset in [0.0f32, 100.0, 1.0e3, 1.0e4] {
            let data: Vec<f32> = base.iter().map(|x| x + offset).collect();
            let g = knn(Points::new(&data, n, d), k, Some(true), 0);
            let mut hit = 0;
            for i in (0..n).step_by(5) {
                let t = truth(&data, n, d, k, i);
                hit += g.indices[i * k..(i + 1) * k].iter().filter(|j| t.contains(j)).count();
            }
            let recall = hit as f64 / ((n / 5) * k) as f64;
            assert!(recall >= 0.999, "offset {}: recall {}", offset, recall);
        }
    }

    #[test]
    fn small_local_scale_against_a_large_offset_is_exact() {
        let (n, d, k) = (1200usize, 30usize, 8usize);
        let data: Vec<f32> = embedding(n, d, 9).iter().map(|x| x * 0.01 + 1.0e3).collect();
        let g = knn(Points::new(&data, n, d), k, Some(true), 0);
        let mut hit = 0;
        for i in 0..n {
            let t = truth(&data, n, d, k, i);
            hit += g.indices[i * k..(i + 1) * k].iter().filter(|j| t.contains(j)).count();
        }
        assert!(hit as f64 / (n * k) as f64 >= 0.999);
    }

    #[test]
    fn non_finite_rows_are_located() {
        let mut data = vec![1.0f32; 5 * 3];
        assert_eq!(first_non_finite_row(&data, 3), None);
        data[4 * 3 + 1] = f32::INFINITY;
        data[2 * 3] = f32::NAN;
        assert_eq!(first_non_finite_row(&data, 3), Some(2));
        assert_eq!(first_non_finite_row(&[], 3), None);
    }

    #[test]
    fn plan_matches_the_search_that_runs() {
        assert!(knn_plan(5, Some(false)).exact, "fewer points than lists falls back to exact");
        assert!(knn_plan(20_000, None).exact);
        assert!(!knn_plan(20_001, None).exact);
        assert!(knn_plan(300_000, Some(true)).exact);
        let small = knn_plan(30_000, None);
        let large = knn_plan(1_000_000, None);
        assert_eq!((small.n_lists, small.nprobe), (58, IVF_MIN_NPROBE));
        assert!(large.nprobe > small.nprobe, "nprobe must grow with the number of lists");
        assert!(large.nprobe <= large.n_lists);
        assert_eq!(knn_plan_with(30_000, None, Some(7)).nprobe, 7);
        assert_eq!(knn_plan_with(30_000, None, Some(1_000)).nprobe, 58, "clamped to the number of lists");
        assert!(knn_plan_with(30_000, Some(true), Some(7)).exact);
    }

    #[test]
    fn query_ivf_probes_by_the_query_not_its_bucket() {
        // far-away queries must still find their neighbours
        let (nr, nq, d, k) = (40_000usize, 300usize, 6usize, 8usize);
        let r = embedding(nr, d, 21);
        let mut rng = rand::rngs::StdRng::seed_from_u64(22);
        let q: Vec<f32> = (0..nq * d).map(|_| rng.gen_range(-6.0f32..6.0)).collect();
        let (rp, qp) = (Points::new(&r, nr, d), Points::new(&q, nq, d));
        let ex = knn_query(rp, qp, k, Some(true), 0);
        let ap = knn_query(rp, qp, k, Some(false), 0);
        let hit: usize = (0..nq).map(|i| ap.indices[i * k..(i + 1) * k].iter().filter(|j| ex.indices[i * k..(i + 1) * k].contains(j)).count()).sum();
        assert!(hit as f64 / (nq * k) as f64 > 0.98);
        let again = knn_query(rp, qp, k, Some(false), 0);
        assert_eq!(ap.indices, again.indices, "IVF query search must be deterministic");
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use rand::Rng;

    /// A continuum (no clusters) with PCA-like column scales: most points sit near the edge of their list.
    fn continuum(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n * d)
            .map(|i| {
                let g: f32 = (0..6).map(|_| rng.gen_range(-1.0f32..1.0)).sum::<f32>();
                g / (1.0 + (i % d) as f32).sqrt()
            })
            .collect()
    }

    #[test]
    fn ivf_finds_neighbours_that_lie_in_other_lists() {
        let (n, d, k) = (30_000usize, 30usize, 14usize);
        let data = continuum(n, d, 5);
        let pts = Points::new(&data, n, d);
        let ap = knn(pts, k, Some(false), 1);
        let mut hit = 0usize;
        let mut total = 0usize;
        for i in (0..n).step_by(30) {
            let mut all: Vec<(f32, u32)> = (0..n as u32).filter(|&j| j as usize != i).map(|j| (sq_dist(pts.row(i), pts.row(j as usize)), j)).collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            let t: Vec<u32> = all[..k].iter().map(|x| x.1).collect();
            hit += ap.indices[i * k..(i + 1) * k].iter().filter(|j| t.contains(j)).count();
            total += k;
        }
        let recall = hit as f64 / total as f64;
        assert!(recall > 0.99, "recall {}", recall);
    }

    #[test]
    fn explicit_nprobe_changes_the_search_and_is_deterministic() {
        let (n, d, k) = (20_500usize, 12usize, 10usize);
        let data = continuum(n, d, 8);
        let pts = Points::new(&data, n, d);
        let a = knn_with(pts, k, None, 3, Some(4));
        let b = knn_with(pts, k, None, 3, Some(4));
        assert_eq!(a.indices, b.indices);
        let wide = knn_with(pts, k, None, 3, Some(30));
        assert_ne!(a.indices, wide.indices);
    }
}
