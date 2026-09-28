//! Harmony batch integration (Korsunsky et al. 2019), harmony2 variant.
//!
//! A port of the algorithm in R `harmony` >= 1.2 / `harmonypy` 2.0 (the C++
//! backend both share): cosine-normalised soft k-means with a diversity
//! penalty, block-wise assignment updates, and a per-cluster ridge regression
//! ("mixture of experts") that removes the batch terms. Defaults, objective,
//! convergence tests, the automatic ridge penalty (`lambda = alpha * E`) and the
//! multi-covariate handling follow that code.
//!
//! Layout: cells are rows (row-major N × d, N × K), and cells are internally
//! sorted by the first covariate so that each of its levels is a contiguous
//! block. The two O(N·K·d) steps (per-batch weighted sums and the correction)
//! are then GEMMs per batch level; everything per cell runs in parallel.

use crate::deseq::linalg::Lu;
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

pub struct Params {
    pub n_clusters: usize,
    pub sigma: f64,
    /// diversity penalty per batch level (all covariates, concatenated)
    pub theta: Vec<f64>,
    /// fixed ridge penalty per level; `None` = estimate as `alpha * E`
    pub lambda: Option<Vec<f64>>,
    pub alpha: f64,
    pub block_size: f64,
    pub max_iter_harmony: usize,
    pub max_iter_kmeans: usize,
    pub epsilon_cluster: f64,
    pub epsilon_harmony: f64,
    pub batch_prop_cutoff: f64,
    pub seed: u64,
}

pub struct Output {
    /// corrected embedding, N × d, input cell order
    pub z_corr: Vec<f32>,
    /// soft cluster assignments, N × K, input cell order
    pub r: Vec<f32>,
    /// centroids, K × d (unit length)
    pub y: Vec<f32>,
    pub objective_harmony: Vec<f64>,
    pub objective_kmeans: Vec<f64>,
    pub kmeans_rounds: Vec<usize>,
    pub converged: bool,
}

const WINDOW: usize = 3;

fn normalize_rows(src: &[f32], dst: &mut [f32], d: usize) {
    dst.par_chunks_mut(d).zip(src.par_chunks(d)).for_each(|(o, x)| {
        let nrm = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        let s = if nrm > 0.0 { (1.0 / nrm) as f32 } else { 0.0 };
        for (a, &b) in o.iter_mut().zip(x) {
            *a = b * s;
        }
    });
}

/// out (a_rows × b_rows) = A (a_rows × d) · Bᵀ, both row-major.
fn gemm_abt(out: &mut [f32], a: &[f32], b: &[f32], a_rows: usize, b_rows: usize, d: usize) {
    matmul(
        MatMut::from_row_major_slice_mut(out, a_rows, b_rows),
        Accum::Replace,
        MatRef::from_row_major_slice(a, a_rows, d),
        MatRef::from_row_major_slice(b, b_rows, d).transpose(),
        1.0f32,
        Par::rayon(0),
    );
}

/// dist = 2 (1 − Zc · Yᵀ)
fn distances(dist: &mut [f32], zc: &[f32], y: &[f32], n: usize, k: usize, d: usize) {
    gemm_abt(dist, zc, y, n, k, d);
    crate::simd::clean_simd_state();
    dist.par_iter_mut().for_each(|v| *v = 2.0 * (1.0 - *v));
}

/// Row-wise softmax of `logits` written into `out` (f32).
#[inline]
fn softmax_into(logits: &mut [f64], out: &mut [f32]) {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0;
    for v in logits.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for (o, &v) in out.iter_mut().zip(logits.iter()) {
        *o = (v / s) as f32;
    }
}

struct State<'a> {
    n: usize,
    d: usize,
    k: usize,
    nb: usize,
    ncov: usize,
    p: &'a Params,
    z_orig: Vec<f32>,
    z_corr: Vec<f32>,
    zc: Vec<f32>,
    y: Vec<f32>,
    r: Vec<f32>,
    dist: Vec<f32>,
    /// ncov × n, global level ids, in sorted cell order
    batch: Vec<u32>,
    cov_of_level: Vec<usize>,
    /// cells of each level (sorted order); covariate-0 levels are contiguous ranges
    level_cells: Vec<Vec<u32>>,
    level_range: Vec<Option<(usize, usize)>>,
    batch_sizes: Vec<f64>,
    pr_b: Vec<f64>,
    sigma: Vec<f64>,
    o: Vec<f64>,
    e: Vec<f64>,
    /// multi-covariate: pair id per (cell, covariate pair); pair → (level, level)
    cell_pairs: Vec<u32>,
    pairs: Vec<(u32, u32)>,
    obj_kmeans: Vec<f64>,
    obj_harmony: Vec<f64>,
    rounds: Vec<usize>,
    rng: StdRng,
}

impl<'a> State<'a> {
    fn e_o_from_r(&mut self) {
        let (k, nb, n) = (self.k, self.nb, self.n);
        let (o, colsum) = self.scatter(&(0..n as u32).collect::<Vec<_>>());
        self.o = o;
        self.e = vec![0.0; k * nb];
        for kk in 0..k {
            for b in 0..nb {
                self.e[kk * nb + b] = colsum[kk] * self.pr_b[b];
            }
        }
    }

    /// (O contribution K × B, column sums K) of the given cells' current R rows.
    fn scatter(&self, cells: &[u32]) -> (Vec<f64>, Vec<f64>) {
        let (k, nb, ncov, n) = (self.k, self.nb, self.ncov, self.n);
        cells
            .par_chunks(4096)
            .fold(
                || (vec![0.0f64; k * nb], vec![0.0f64; k]),
                |(mut o, mut cs), chunk| {
                    for &c in chunk {
                        let c = c as usize;
                        let row = &self.r[c * k..(c + 1) * k];
                        for (kk, &v) in row.iter().enumerate() {
                            cs[kk] += v as f64;
                        }
                        for cv in 0..ncov {
                            let b = self.batch[cv * n + c] as usize;
                            for (kk, &v) in row.iter().enumerate() {
                                o[kk * nb + b] += v as f64;
                            }
                        }
                    }
                    (o, cs)
                },
            )
            .reduce(
                || (vec![0.0f64; k * nb], vec![0.0f64; k]),
                |(mut a, mut b), (x, y)| {
                    a.iter_mut().zip(&x).for_each(|(p, q)| *p += q);
                    b.iter_mut().zip(&y).for_each(|(p, q)| *p += q);
                    (a, b)
                },
            )
    }

    fn r_from_dist(&mut self) {
        let (k, sigma) = (self.k, &self.sigma);
        self.r.par_chunks_mut(k).zip(self.dist.par_chunks(k)).for_each(|(r, dd)| {
            let mut lg: Vec<f64> = dd.iter().zip(sigma).map(|(&x, &s)| -(x as f64) / s).collect();
            softmax_into(&mut lg, r);
        });
    }

    /// Recompute Zc, distances, R, E, O from the current Z_corr and Y.
    fn reset_assignments(&mut self) {
        normalize_rows(&self.z_corr, &mut self.zc, self.d);
        distances(&mut self.dist, &self.zc, &self.y, self.n, self.k, self.d);
        self.r_from_dist();
        self.e_o_from_r();
    }

    fn compute_objective(&mut self) {
        let (k, nb, sigma) = (self.k, self.nb, &self.sigma);
        let (km, ent) = self
            .r
            .par_chunks(k)
            .zip(self.dist.par_chunks(k))
            .map(|(r, dd)| {
                let mut a = 0.0;
                let mut b = 0.0;
                for kk in 0..k {
                    let v = r[kk] as f64;
                    a += v * dd[kk] as f64;
                    if v > 0.0 {
                        b += sigma[kk] * v * v.ln();
                    }
                }
                (a, b)
            })
            .reduce(|| (0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));
        let mut cross = 0.0;
        for kk in 0..k {
            for b in 0..nb {
                let (o, e) = (self.o[kk * nb + b], self.e[kk * nb + b]);
                cross += sigma[kk] * self.p.theta[b] * o * ((o + e + 1.0) / (2.0 * e + 1.0)).ln();
            }
        }
        self.obj_kmeans.push((km + ent + cross) * 2000.0 / self.n as f64);
    }

    fn update_r(&mut self) {
        let (n, k, nb, ncov) = (self.n, self.k, self.nb, self.ncov);
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.shuffle(&mut self.rng);
        let n_blocks = (1.0 / self.p.block_size).ceil() as usize;
        let per = ((n as f64 * self.p.block_size) as usize).max(1);
        for i in 0..n_blocks {
            let lo = i * per;
            if lo >= n {
                break;
            }
            let hi = if i == n_blocks - 1 { n } else { ((i + 1) * per).min(n) };
            let cells = &order[lo..hi];
            let (o_blk, cs) = self.scatter(cells);
            for kk in 0..k {
                for b in 0..nb {
                    self.e[kk * nb + b] -= cs[kk] * self.pr_b[b];
                    self.o[kk * nb + b] -= o_blk[kk * nb + b];
                }
            }
            // log diversity penalty with E, O frozen for the block
            let mut ld = vec![0.0f64; k * nb];
            for kk in 0..k {
                for b in 0..nb {
                    let (o, e) = (self.o[kk * nb + b], self.e[kk * nb + b]);
                    ld[kk * nb + b] = self.p.theta[b] * ((2.0 * e + 1.0).ln() - (o + e + 1.0).ln());
                }
            }
            let new_rows: Vec<f32> = cells
                .par_iter()
                .flat_map_iter(|&c| {
                    let c = c as usize;
                    let dd = &self.dist[c * k..(c + 1) * k];
                    let mut lg: Vec<f64> = (0..k).map(|kk| -(dd[kk] as f64) / self.sigma[kk]).collect();
                    for cv in 0..ncov {
                        let b = self.batch[cv * n + c] as usize;
                        for kk in 0..k {
                            lg[kk] += ld[kk * nb + b];
                        }
                    }
                    let mut out = vec![0.0f32; k];
                    softmax_into(&mut lg, &mut out);
                    out.into_iter()
                })
                .collect();
            for (j, &c) in cells.iter().enumerate() {
                let c = c as usize;
                self.r[c * k..(c + 1) * k].copy_from_slice(&new_rows[j * k..(j + 1) * k]);
            }
            let (o_blk, cs) = self.scatter(cells);
            for kk in 0..k {
                for b in 0..nb {
                    self.e[kk * nb + b] += cs[kk] * self.pr_b[b];
                    self.o[kk * nb + b] += o_blk[kk * nb + b];
                }
            }
        }
    }

    fn kmeans_converged(&self) -> bool {
        let v = &self.obj_kmeans;
        if v.len() <= WINDOW + 1 {
            return false;
        }
        let m = v.len();
        let (mut old, mut new) = (0.0, 0.0);
        for i in 0..WINDOW {
            old += v[m - 2 - i];
            new += v[m - 1 - i];
        }
        (old - new).abs() / old.abs() < self.p.epsilon_cluster
    }

    fn harmony_converged(&self) -> bool {
        let v = &self.obj_harmony;
        if v.len() < 2 {
            return false;
        }
        let (old, new) = (v[v.len() - 2], v[v.len() - 1]);
        if !old.is_finite() || !new.is_finite() {
            return false;
        }
        if old == 0.0 {
            return new == 0.0;
        }
        let delta = (old - new) / old.abs();
        delta >= 0.0 && delta < self.p.epsilon_harmony
    }

    fn cluster(&mut self) {
        if self.obj_harmony.len() > 1 {
            self.reset_assignments();
        }
        let mut rounds = 0;
        for i in 0..self.p.max_iter_kmeans {
            self.update_r();
            self.compute_objective();
            rounds = i + 1;
            if i > WINDOW && self.kmeans_converged() {
                break;
            }
        }
        self.rounds.push(rounds);
        self.obj_harmony.push(*self.obj_kmeans.last().unwrap());
    }

    /// Rows of `m` (width w) for a level: borrowed if contiguous, gathered otherwise.
    fn level_rows<'s>(&self, m: &'s [f32], w: usize, lvl: usize) -> std::borrow::Cow<'s, [f32]> {
        if let Some((a, b)) = self.level_range[lvl] {
            std::borrow::Cow::Borrowed(&m[a * w..b * w])
        } else {
            let cells = &self.level_cells[lvl];
            let mut out = vec![0.0f32; cells.len() * w];
            out.par_chunks_mut(w).zip(cells.par_iter()).for_each(|(o, &c)| {
                o.copy_from_slice(&m[c as usize * w..(c as usize + 1) * w]);
            });
            std::borrow::Cow::Owned(out)
        }
    }

    fn moe_correct_ridge(&mut self) {
        let (n, d, k, nb, ncov) = (self.n, self.d, self.k, self.nb, self.ncov);
        let multi = ncov > 1;
        // zsum[b] (K × d) = R_bᵀ Z_b
        let mut zsum = vec![0.0f32; nb * k * d];
        for b in 0..nb {
            let nbc = self.level_cells[b].len();
            if nbc == 0 {
                continue;
            }
            let rb = self.level_rows(&self.r, k, b);
            let zb = self.level_rows(&self.z_orig, d, b);
            matmul(
                MatMut::from_row_major_slice_mut(&mut zsum[b * k * d..(b + 1) * k * d], k, d),
                Accum::Replace,
                MatRef::from_row_major_slice(&rb, nbc, k).transpose(),
                MatRef::from_row_major_slice(&zb, nbc, d),
                1.0f32,
                Par::rayon(0),
            );
        }
        crate::simd::clean_simd_state();
        // multi-covariate overlaps: pairsum[pair][k]
        let npairs_cell = ncov * (ncov - 1) / 2;
        let pairsum: Vec<f64> = if multi {
            let np = self.pairs.len();
            (0..n)
                .into_par_iter()
                .fold(
                    || vec![0.0f64; np * k],
                    |mut acc, c| {
                        let row = &self.r[c * k..(c + 1) * k];
                        for q in 0..npairs_cell {
                            let pid = self.cell_pairs[c * npairs_cell + q] as usize;
                            for kk in 0..k {
                                acc[pid * k + kk] += row[kk] as f64;
                            }
                        }
                        acc
                    },
                )
                .reduce(|| vec![0.0f64; np * k], |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    a
                })
        } else {
            vec![]
        };

        // W[b] (K × d): the correction for cells of level b
        let mut wlev = vec![0.0f32; nb * k * d];
        let n_cov_levels: Vec<usize> = (0..ncov).map(|c| self.cov_of_level.iter().filter(|&&x| x == c).count()).collect();
        for kk in 0..k {
            let avg: Vec<f64> = (0..nb).map(|b| self.o[kk * nb + b] / self.batch_sizes[b]).collect();
            let mut cov_levels = vec![0usize; ncov];
            for b in 0..nb {
                if avg[b] > self.p.batch_prop_cutoff {
                    cov_levels[self.cov_of_level[b]] += 1;
                }
            }
            if cov_levels.iter().all(|&l| l <= 1) {
                continue;
            }
            let keep: Vec<usize> =
                (0..nb).filter(|&b| avg[b] > self.p.batch_prop_cutoff && cov_levels[self.cov_of_level[b]] > 1).collect();
            let m = keep.len() + 1;
            let mut row_of = vec![0usize; nb];
            for (i, &b) in keep.iter().enumerate() {
                row_of[b] = i + 1;
            }
            let mut cov = vec![0.0f64; m * m];
            let ok: Vec<f64> = keep.iter().map(|&b| self.o[kk * nb + b]).collect();
            cov[0] = ok.iter().sum();
            for (i, &v) in ok.iter().enumerate() {
                cov[i + 1] = v;
                cov[(i + 1) * m] = v;
                cov[(i + 1) * m + i + 1] = v;
            }
            let mut z_all = vec![0.0f64; d];
            if multi {
                for (pid, &(b1, b2)) in self.pairs.iter().enumerate() {
                    let (r1, r2) = (row_of[b1 as usize], row_of[b2 as usize]);
                    if r1 > 0 && r2 > 0 {
                        let v = pairsum[pid * k + kk];
                        cov[r1 * m + r2] += v;
                        cov[r2 * m + r1] += v;
                    }
                }
                // cells counted once, if in at least one retained level
                // a covariate with every level retained covers all cells
                let full_cov =
                    (0..ncov).find(|&c| n_cov_levels[c] == keep.iter().filter(|&&b| self.cov_of_level[b] == c).count());
                if let Some(fc) = full_cov {
                    cov[0] = (0..n).into_par_iter().map(|c| self.r[c * k + kk] as f64).sum();
                    for b in 0..nb {
                        if self.cov_of_level[b] == fc {
                            for j in 0..d {
                                z_all[j] += zsum[(b * k + kk) * d + j] as f64;
                            }
                        }
                    }
                } else {
                    let (w0, za) = (0..n)
                        .into_par_iter()
                        .fold(
                            || (0.0f64, vec![0.0f64; d]),
                            |(mut w0, mut za), c| {
                                if (0..ncov).any(|cv| row_of[self.batch[cv * n + c] as usize] > 0) {
                                    let w = self.r[c * k + kk] as f64;
                                    w0 += w;
                                    for (j, &z) in self.z_orig[c * d..(c + 1) * d].iter().enumerate() {
                                        za[j] += w * z as f64;
                                    }
                                }
                                (w0, za)
                            },
                        )
                        .reduce(|| (0.0, vec![0.0; d]), |(a, mut x), (b, y)| {
                            x.iter_mut().zip(&y).for_each(|(p, q)| *p += q);
                            (a + b, x)
                        });
                    cov[0] = w0;
                    z_all = za;
                }
            } else {
                for &b in &keep {
                    for j in 0..d {
                        z_all[j] += zsum[(b * k + kk) * d + j] as f64;
                    }
                }
            }
            for (i, &b) in keep.iter().enumerate() {
                let lam = match &self.p.lambda {
                    None => self.p.alpha * self.e[kk * nb + b],
                    Some(l) => l[b],
                };
                cov[(i + 1) * m + i + 1] += lam;
            }
            let inv = match Lu::new(cov, m) {
                Some(lu) => lu.inverse(),
                None => continue,
            };
            // W = inv[:,0] z_allᵀ + Σ_i inv[:,i+1] z_sum_iᵀ   (m × d)
            let mut w = vec![0.0f64; m * d];
            for r in 0..m {
                let a = inv[r * m];
                for j in 0..d {
                    w[r * d + j] = a * z_all[j];
                }
                for (i, &b) in keep.iter().enumerate() {
                    let a = inv[r * m + i + 1];
                    let zs = &zsum[(b * k + kk) * d..(b * k + kk + 1) * d];
                    for j in 0..d {
                        w[r * d + j] += a * zs[j] as f64;
                    }
                }
            }
            for j in 0..d {
                self.y[kk * d + j] = w[j] as f32;
            }
            for (i, &b) in keep.iter().enumerate() {
                for j in 0..d {
                    wlev[(b * k + kk) * d + j] = w[(i + 1) * d + j] as f32;
                }
            }
        }

        // Z_corr = Z_orig − Σ_levels R_b W_b
        self.z_corr.copy_from_slice(&self.z_orig);
        for b in 0..nb {
            let nbc = self.level_cells[b].len();
            if nbc == 0 {
                continue;
            }
            let wb = &wlev[b * k * d..(b + 1) * k * d];
            if let Some((lo, hi)) = self.level_range[b] {
                let (r, z) = (&self.r[lo * k..hi * k], &mut self.z_corr[lo * d..hi * d]);
                matmul(
                    MatMut::from_row_major_slice_mut(z, nbc, d),
                    Accum::Add,
                    MatRef::from_row_major_slice(r, nbc, k),
                    MatRef::from_row_major_slice(wb, k, d),
                    -1.0f32,
                    Par::rayon(0),
                );
            } else {
                let rb = self.level_rows(&self.r, k, b).into_owned();
                let mut delta = vec![0.0f32; nbc * d];
                matmul(
                    MatMut::from_row_major_slice_mut(&mut delta, nbc, d),
                    Accum::Replace,
                    MatRef::from_row_major_slice(&rb, nbc, k),
                    MatRef::from_row_major_slice(wb, k, d),
                    1.0f32,
                    Par::rayon(0),
                );
                for (i, &c) in self.level_cells[b].iter().enumerate() {
                    let c = c as usize;
                    for j in 0..d {
                        self.z_corr[c * d + j] -= delta[i * d + j];
                    }
                }
            }
        }
        crate::simd::clean_simd_state();
        let yc = self.y.clone();
        normalize_rows(&yc, &mut self.y, d);
    }
}

/// k-means initialisation of the centroids, as harmony2: K random cells, then
/// for each centroid a draw proportional to the distance from it (exponential
/// race, without replacement), then 10 Lloyd iterations. Rows of `x` are unit length.
fn kmeans_init(x: &[f32], n: usize, d: usize, k: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut y = vec![0.0f32; k * d];
    for i in 0..k {
        let idx = ((rng.gen::<f64>() * n as f64).round() as usize).min(n - 1);
        y[i * d..(i + 1) * d].copy_from_slice(&x[idx * d..(idx + 1) * d]);
    }
    let mut chosen = std::collections::HashSet::new();
    for i in 0..k {
        let yi = y[i * d..(i + 1) * d].to_vec();
        let seed: u64 = rng.gen();
        let best = (0..n)
            .into_par_iter()
            .with_min_len(4096)
            .filter(|c| !chosen.contains(c))
            .map(|c| {
                let dot: f32 = x[c * d..(c + 1) * d].iter().zip(&yi).map(|(a, b)| a * b).sum();
                let dist = (2.0 * (1.0 - dot as f64)).abs();
                let mut r = StdRng::seed_from_u64(seed ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let u: f64 = r.gen::<f64>().max(f64::MIN_POSITIVE);
                (-u.ln() / (dist + 1e-10), c)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        if let Some((_, c)) = best {
            chosen.insert(c);
            y[i * d..(i + 1) * d].copy_from_slice(&x[c * d..(c + 1) * d]);
        }
    }
    let mut prod = vec![0.0f32; n * k];
    for _ in 0..10 {
        gemm_abt(&mut prod, x, &y, n, k, d);
        crate::simd::clean_simd_state();
        let ynorm: Vec<f32> = y.chunks(d).map(|c| c.iter().map(|v| v * v).sum()).collect();
        let (sums, counts) = prod
            .par_chunks(k)
            .enumerate()
            .fold(
                || (vec![0.0f64; k * d], vec![0usize; k]),
                |(mut s, mut cnt), (c, row)| {
                    let mut best = (f32::INFINITY, 0usize);
                    for kk in 0..k {
                        let v = ynorm[kk] - 2.0 * row[kk];
                        if v < best.0 {
                            best = (v, kk);
                        }
                    }
                    cnt[best.1] += 1;
                    for (j, &z) in x[c * d..(c + 1) * d].iter().enumerate() {
                        s[best.1 * d + j] += z as f64;
                    }
                    (s, cnt)
                },
            )
            .reduce(
                || (vec![0.0f64; k * d], vec![0usize; k]),
                |(mut a, mut b), (x2, y2)| {
                    a.iter_mut().zip(&x2).for_each(|(p, q)| *p += q);
                    b.iter_mut().zip(&y2).for_each(|(p, q)| *p += q);
                    (a, b)
                },
            );
        for kk in 0..k {
            if counts[kk] > 0 {
                for j in 0..d {
                    y[kk * d + j] = (sums[kk * d + j] / counts[kk] as f64) as f32;
                }
            }
        }
    }
    y
}

/// Run Harmony. `z` is N × d (row-major, e.g. PCA coordinates), `batch` holds
/// `levels.len()` rows of N global level ids (covariate c uses ids
/// `sum(levels[..c]) .. sum(levels[..=c])`).
pub fn harmony(z: &[f32], n: usize, d: usize, batch: &[u32], levels: &[usize], p: &Params) -> Result<Output, String> {
    let ncov = levels.len();
    let nb: usize = levels.iter().sum();
    let k = p.n_clusters;
    if n < 2 || d == 0 || z.len() != n * d {
        return Err("z must be a non-empty N × d matrix".into());
    }
    if ncov == 0 || batch.len() != ncov * n {
        return Err("batch must hold one row of N level ids per covariate".into());
    }
    if k < 2 || k > n {
        return Err("n_clusters must be in [2, N]".into());
    }
    if p.theta.len() != nb || p.lambda.as_ref().map_or(false, |l| l.len() != nb) {
        return Err("theta/lambda must have one value per batch level".into());
    }
    if !(p.block_size > 0.0 && p.block_size <= 1.0) || !(p.sigma > 0.0) {
        return Err("block_size must be in (0, 1] and sigma positive".into());
    }
    if p.max_iter_harmony == 0 {
        return Err("max_iter_harmony must be >= 1".into());
    }
    let mut cov_of_level = Vec::with_capacity(nb);
    for (c, &l) in levels.iter().enumerate() {
        cov_of_level.extend(std::iter::repeat(c).take(l));
    }
    let mut off = 0usize;
    for (c, &l) in levels.iter().enumerate() {
        if batch[c * n..(c + 1) * n].iter().any(|&b| (b as usize) < off || (b as usize) >= off + l) {
            return Err(format!("batch ids of covariate {} out of range", c));
        }
        off += l;
    }

    // sort cells by the first covariate so its levels are contiguous
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.sort_by_key(|&c| batch[c as usize]);
    let mut z_orig = vec![0.0f32; n * d];
    z_orig.par_chunks_mut(d).zip(perm.par_iter()).for_each(|(o, &c)| o.copy_from_slice(&z[c as usize * d..(c as usize + 1) * d]));
    let mut bsort = vec![0u32; ncov * n];
    for cv in 0..ncov {
        for (i, &c) in perm.iter().enumerate() {
            bsort[cv * n + i] = batch[cv * n + c as usize];
        }
    }
    let mut level_cells: Vec<Vec<u32>> = vec![vec![]; nb];
    for cv in 0..ncov {
        for i in 0..n {
            level_cells[bsort[cv * n + i] as usize].push(i as u32);
        }
    }
    let level_range: Vec<Option<(usize, usize)>> = (0..nb)
        .map(|b| {
            let cs = &level_cells[b];
            if cov_of_level[b] == 0 && !cs.is_empty() {
                Some((cs[0] as usize, *cs.last().unwrap() as usize + 1))
            } else {
                None
            }
        })
        .collect();
    let batch_sizes: Vec<f64> = level_cells.iter().map(|c| c.len() as f64).collect();
    let pr_b: Vec<f64> = batch_sizes.iter().map(|&s| s / n as f64).collect();

    let (mut pairs, mut cell_pairs) = (vec![], vec![]);
    if ncov > 1 {
        let mut ids: HashMap<(u32, u32), u32> = HashMap::new();
        for i in 0..n {
            for c1 in 0..ncov {
                for c2 in c1 + 1..ncov {
                    let key = (bsort[c1 * n + i], bsort[c2 * n + i]);
                    let next = ids.len() as u32;
                    let id = *ids.entry(key).or_insert_with(|| {
                        pairs.push(key);
                        next
                    });
                    cell_pairs.push(id);
                }
            }
        }
    }

    let mut zc = vec![0.0f32; n * d];
    normalize_rows(&z_orig, &mut zc, d);
    let mut rng = StdRng::seed_from_u64(p.seed);
    let y0 = kmeans_init(&zc, n, d, k, &mut rng);
    let mut y = vec![0.0f32; k * d];
    normalize_rows(&y0, &mut y, d);

    let mut st = State {
        n,
        d,
        k,
        nb,
        ncov,
        p,
        z_corr: zc.clone(),
        z_orig,
        zc,
        y,
        r: vec![0.0; n * k],
        dist: vec![0.0; n * k],
        batch: bsort,
        cov_of_level,
        level_cells,
        level_range,
        batch_sizes,
        pr_b,
        sigma: vec![p.sigma; k],
        o: vec![],
        e: vec![],
        cell_pairs,
        pairs,
        obj_kmeans: vec![],
        obj_harmony: vec![],
        rounds: vec![],
        rng,
    };
    distances(&mut st.dist, &st.zc, &st.y, n, k, d);
    st.r_from_dist();
    st.e_o_from_r();
    st.compute_objective();
    st.obj_harmony.push(*st.obj_kmeans.last().unwrap());

    let mut converged = false;
    for _ in 0..p.max_iter_harmony {
        st.cluster();
        st.moe_correct_ridge();
        if st.harmony_converged() {
            converged = true;
            break;
        }
    }
    if st.z_corr.iter().any(|v| !v.is_finite()) {
        return Err("Harmony produced non-finite coordinates".into());
    }

    // back to input order
    let mut z_out = vec![0.0f32; n * d];
    let mut r_out = vec![0.0f32; n * k];
    for (i, &c) in perm.iter().enumerate() {
        let c = c as usize;
        z_out[c * d..(c + 1) * d].copy_from_slice(&st.z_corr[i * d..(i + 1) * d]);
        r_out[c * k..(c + 1) * k].copy_from_slice(&st.r[i * k..(i + 1) * k]);
    }
    Ok(Output {
        z_corr: z_out,
        r: r_out,
        y: st.y,
        objective_harmony: st.obj_harmony,
        objective_kmeans: st.obj_kmeans,
        kmeans_rounds: st.rounds,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(nb: usize, k: usize) -> Params {
        Params {
            n_clusters: k,
            sigma: 0.1,
            theta: vec![2.0; nb],
            lambda: None,
            alpha: 0.2,
            block_size: 0.05,
            max_iter_harmony: 10,
            max_iter_kmeans: 20,
            epsilon_cluster: 1e-3,
            epsilon_harmony: 1e-2,
            batch_prop_cutoff: 1e-5,
            seed: 0,
        }
    }

    /// Two cell types, two batches with a strong additive shift: after Harmony the
    /// batch means within each type should nearly coincide, the types stay apart.
    #[test]
    fn removes_batch_shift() {
        let (n, d) = (2000usize, 10usize);
        let mut rng = StdRng::seed_from_u64(1);
        let mut z = vec![0.0f32; n * d];
        let mut batch = vec![0u32; n];
        for i in 0..n {
            let t = i % 2;
            let b = (i / 2) % 2;
            batch[i] = b as u32;
            for j in 0..d {
                let mu = if j == 0 { 5.0 * t as f32 } else { 0.0 } + if j == 1 { 3.0 * b as f32 } else { 0.0 };
                z[i * d + j] = mu + rng.gen::<f32>() - 0.5 + 1.0;
            }
        }
        let out = harmony(&z, n, d, &batch, &[2], &params(2, 20)).unwrap();
        let mean = |t: usize, b: usize, j: usize| {
            let v: Vec<f32> = (0..n).filter(|&i| i % 2 == t && (i / 2) % 2 == b).map(|i| out.z_corr[i * d + j]).collect();
            v.iter().sum::<f32>() / v.len() as f32
        };
        for t in 0..2 {
            assert!((mean(t, 0, 1) - mean(t, 1, 1)).abs() < 0.6, "batch shift not removed");
        }
        assert!((mean(0, 0, 0) - mean(1, 0, 0)).abs() > 3.0, "cell types merged");
        let h = &out.objective_harmony;
        assert!(h.iter().all(|v| v.is_finite()));
    }
}
