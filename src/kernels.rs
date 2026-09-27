//! Streaming kernels over cell-sorted COO chunks.
//!
//! Every function takes a chunk of (row, gene, value) triplets whose rows are
//! non-decreasing, and either returns a transformed chunk or accumulates into
//! caller-owned buffers. A dataset of any size is processed by feeding its
//! chunks in turn, so peak memory is one chunk plus the (small) accumulators.

use faer::{Mat, Side};
use rayon::prelude::*;

/// Split `rows` (non-decreasing) into ~`parts` segments that never split a row.
fn row_segments(rows: &[u32], parts: usize) -> Vec<(usize, usize)> {
    let n = rows.len();
    if n == 0 {
        return vec![];
    }
    let parts = parts.max(1).min(n);
    let mut cuts = vec![0usize];
    for p in 1..parts {
        let mut c = n * p / parts;
        while c < n && c > 0 && rows[c] == rows[c - 1] {
            c += 1;
        }
        if c > *cuts.last().unwrap() && c < n {
            cuts.push(c);
        }
    }
    cuts.push(n);
    cuts.windows(2).map(|w| (w[0], w[1])).collect()
}

fn n_parts(len: usize) -> usize {
    (rayon::current_num_threads() * 4).min(len / 16_384 + 1).max(1)
}

/// Filter + remap + normalise + log1p one chunk.
///
/// * `cells` are original cell ids (non-decreasing); `cell_map[id]` is the output
///   row or -1 to drop the cell.
/// * `gene_map[g]` is the output gene index or -1 to drop the gene.
/// * `target_sum > 0` scales each cell to that total (over kept genes), as
///   scanpy `normalize_total`; `log1p` then applies ln(1 + x).
pub fn preprocess_chunk(
    cells: &[u32],
    genes: &[u32],
    values: &[f32],
    cell_map: &[i64],
    gene_map: &[i32],
    target_sum: f64,
    log1p: bool,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let segs = row_segments(cells, n_parts(cells.len()));
    let parts: Vec<(Vec<u32>, Vec<u32>, Vec<f32>)> = segs
        .par_iter()
        .map(|&(lo, hi)| {
            let (mut r, mut g, mut v) = (Vec::with_capacity(hi - lo), Vec::with_capacity(hi - lo), Vec::with_capacity(hi - lo));
            let mut p = lo;
            while p < hi {
                let c = cells[p];
                let mut q = p;
                while q < hi && cells[q] == c {
                    q += 1;
                }
                let row = cell_map.get(c as usize).copied().unwrap_or(-1);
                if row >= 0 {
                    let start = v.len();
                    let mut total = 0.0f64;
                    for e in p..q {
                        let gm = gene_map.get(genes[e] as usize).copied().unwrap_or(-1);
                        if gm >= 0 {
                            r.push(row as u32);
                            g.push(gm as u32);
                            v.push(values[e]);
                            total += values[e] as f64;
                        }
                    }
                    let scale = if target_sum > 0.0 && total > 0.0 { target_sum / total } else { 1.0 };
                    for x in &mut v[start..] {
                        let y = if target_sum > 0.0 { (*x as f64 * scale) as f32 } else { *x };
                        *x = if log1p { y.ln_1p() } else { y };
                    }
                }
                p = q;
            }
            (r, g, v)
        })
        .collect();
    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let (mut r, mut g, mut v) = (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total));
    for (a, b, c) in parts {
        r.extend(a);
        g.extend(b);
        v.extend(c);
    }
    (r, g, v)
}

/// Per-gene sums for mean/variance on two scales: the values as given (log
/// domain) and expm1(values) (normalised-count domain, used by seurat HVG).
/// `out` is n_genes × 5: [Σx, Σx², Σexpm1(x), Σexpm1(x)², nnz].
pub fn gene_stats(genes: &[u32], values: &[f32], n_genes: usize, out: &mut [f64]) {
    let chunk = 1 << 16;
    let partial = genes
        .par_chunks(chunk)
        .zip(values.par_chunks(chunk))
        .fold(
            || vec![0.0f64; n_genes * 5],
            |mut acc, (gs, vs)| {
                for (&g, &x) in gs.iter().zip(vs) {
                    let (x, e) = (x as f64, (x as f64).exp_m1());
                    let o = &mut acc[g as usize * 5..g as usize * 5 + 5];
                    o[0] += x;
                    o[1] += x * x;
                    o[2] += e;
                    o[3] += e * e;
                    o[4] += 1.0;
                }
                acc
            },
        )
        .reduce(|| vec![0.0f64; n_genes * 5], |mut a, b| {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
            a
        });
    out.iter_mut().zip(partial).for_each(|(o, p)| *o += p);
}

/// Value transform used by PCA. With `scale`, reproduces scanpy
/// `pp.scale(max_value)`: z = clip((x - mu) / sd). Because zero entries map to
/// the per-gene constant clip(-mu / sd), the scaled matrix is Z = 1 b^T + S where
/// S is sparse with entries z - b; after centring, PCA of Z equals PCA of S.
#[derive(Clone, Copy)]
pub struct Transform<'a> {
    pub scale: bool,
    pub mean: &'a [f64],
    pub std: &'a [f64],
    pub max_value: f64, // <= 0: no clipping
}

impl<'a> Transform<'a> {
    #[inline]
    fn apply(&self, g: usize, x: f32) -> f64 {
        if !self.scale {
            return x as f64;
        }
        let (mu, sd) = (self.mean[g], self.std[g]);
        let clip = |z: f64| if self.max_value > 0.0 { z.clamp(-self.max_value, self.max_value) } else { z };
        clip((x as f64 - mu) / sd) - clip(-mu / sd)
    }
}

/// Accumulate G += S^T S and colsum += S^T 1 over one chunk. Only genes with
/// `gene_map[g] >= 0` take part (mapped to 0..d). `gram` is d × d row-major
/// (upper triangle filled; call `finish_gram` to symmetrise).
///
/// Threads own disjoint row blocks of G (balanced by work), so there is no
/// reduction step and memory stays at one d × d matrix.
pub fn gram_accumulate(
    rows: &[u32],
    genes: &[u32],
    values: &[f32],
    gene_map: &[i32],
    t: Transform,
    gram: &mut [f64],
    colsum: &mut [f64],
) {
    let d = colsum.len();
    // compact to (row-start, mapped gene, transformed value), sorted by gene within row
    let mut starts = vec![0usize];
    let mut eg: Vec<u32> = Vec::with_capacity(values.len());
    let mut ev: Vec<f64> = Vec::with_capacity(values.len());
    let mut p = 0;
    while p < rows.len() {
        let r = rows[p];
        let s = eg.len();
        while p < rows.len() && rows[p] == r {
            let m = gene_map.get(genes[p] as usize).copied().unwrap_or(-1);
            if m >= 0 {
                eg.push(m as u32);
                ev.push(t.apply(m as usize, values[p]));
            }
            p += 1;
        }
        if eg.len() > s {
            // sort this row by gene if needed
            if eg[s..].windows(2).any(|w| w[0] > w[1]) {
                let mut idx: Vec<usize> = (s..eg.len()).collect();
                idx.sort_by_key(|&i| eg[i]);
                let (g2, v2): (Vec<u32>, Vec<f64>) = idx.iter().map(|&i| (eg[i], ev[i])).unzip();
                eg[s..].copy_from_slice(&g2);
                ev[s..].copy_from_slice(&v2);
            }
            starts.push(eg.len());
        }
    }
    for (&g, &v) in eg.iter().zip(&ev) {
        colsum[g as usize] += v;
    }

    // work per gene a: number of entries at or after it in its row (upper triangle)
    let mut work = vec![0u64; d];
    for w in starts.windows(2) {
        let len = w[1] - w[0];
        for (k, &g) in eg[w[0]..w[1]].iter().enumerate() {
            work[g as usize] += (len - k) as u64;
        }
    }
    let total: u64 = work.iter().sum();
    let parts = (rayon::current_num_threads() * 2).max(1);
    let mut bounds = vec![0usize];
    let mut acc = 0u64;
    for (g, &w) in work.iter().enumerate() {
        acc += w;
        if acc * parts as u64 >= total * bounds.len() as u64 && bounds.len() < parts && g + 1 < d {
            bounds.push(g + 1);
        }
    }
    bounds.push(d);
    bounds.dedup();

    // split G into disjoint row blocks
    let mut blocks: Vec<(usize, usize, &mut [f64])> = Vec::new();
    let mut rest: &mut [f64] = gram;
    for w in bounds.windows(2) {
        let (a, b) = rest.split_at_mut((w[1] - w[0]) * d);
        blocks.push((w[0], w[1], a));
        rest = b;
    }
    blocks.into_par_iter().for_each(|(lo, hi, block)| {
        for w in starts.windows(2) {
            let (rg, rv) = (&eg[w[0]..w[1]], &ev[w[0]..w[1]]);
            // entries of this row whose gene is in [lo, hi)
            let first = rg.partition_point(|&g| (g as usize) < lo);
            for k in first..rg.len() {
                let a = rg[k] as usize;
                if a >= hi {
                    break;
                }
                let va = rv[k];
                let out = &mut block[(a - lo) * d..(a - lo + 1) * d];
                for m in k..rg.len() {
                    out[rg[m] as usize] += va * rv[m];
                }
            }
        }
    });
}

/// Covariance from accumulated Gram + column sums, then its top-k eigenpairs.
/// Returns (eigenvalues desc, loadings d × k row-major, total variance).
pub fn pca_from_gram(gram: &[f64], colsum: &[f64], n: usize, k: usize) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let d = colsum.len();
    if n < 2 || d == 0 {
        return Err("need at least 2 cells and 1 gene".into());
    }
    let nf = n as f64;
    let cov = Mat::<f64>::from_fn(d, d, |i, j| {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        (gram[a * d + b] - colsum[a] * colsum[b] / nf) / (nf - 1.0)
    });
    let total: f64 = (0..d).map(|i| cov[(i, i)]).sum();
    let eig = cov.self_adjoint_eigen(Side::Upper).map_err(|e| format!("{:?}", e))?;
    let (u, s) = (eig.U(), eig.S().column_vector());
    let k = k.min(d);
    let mut vals = Vec::with_capacity(k);
    let mut vecs = vec![0.0f64; d * k];
    for c in 0..k {
        let src = d - 1 - c; // ascending order
        vals.push(s[src].max(0.0));
        // sign: largest |loading| positive (sklearn svd_flip, u_based_decision=False)
        let (mut best, mut sign) = (0.0f64, 1.0f64);
        for i in 0..d {
            let x = u[(i, src)];
            if x.abs() > best {
                best = x.abs();
                sign = x.signum();
            }
        }
        for i in 0..d {
            vecs[i * k + c] = u[(i, src)] * sign;
        }
    }
    Ok((vals, vecs, total))
}

/// out[row] += Σ_j s_j V[j, :] for one chunk (caller pre-fills out with the
/// centring shift -mean(S)·V). `v` is d × k row-major.
pub fn project(
    rows: &[u32],
    genes: &[u32],
    values: &[f32],
    gene_map: &[i32],
    t: Transform,
    v: &[f64],
    k: usize,
    out: &mut [f32],
) {
    let segs = row_segments(rows, n_parts(rows.len()));
    let results: Vec<Vec<(u32, Vec<f64>)>> = segs
        .par_iter()
        .map(|&(lo, hi)| {
            let mut res = Vec::new();
            let mut p = lo;
            while p < hi {
                let r = rows[p];
                let mut acc = vec![0.0f64; k];
                while p < hi && rows[p] == r {
                    let m = gene_map.get(genes[p] as usize).copied().unwrap_or(-1);
                    if m >= 0 {
                        let s = t.apply(m as usize, values[p]);
                        let vr = &v[m as usize * k..(m as usize + 1) * k];
                        acc.iter_mut().zip(vr).for_each(|(a, &x)| *a += s * x);
                    }
                    p += 1;
                }
                res.push((r, acc));
            }
            res
        })
        .collect();
    for seg in results {
        for (r, acc) in seg {
            let o = &mut out[r as usize * k..(r as usize + 1) * k];
            o.iter_mut().zip(acc).for_each(|(a, b)| *a += b as f32);
        }
    }
}

/// Per (group, gene) Σx, Σx², nnz for one chunk; `group_of_row[row]` < n_groups
/// or u32::MAX to ignore the cell. Buffers are n_groups × n_genes.
pub fn group_gene_sums(
    rows: &[u32],
    genes: &[u32],
    values: &[f32],
    group_of_row: &[u32],
    n_genes: usize,
    sum: &mut [f64],
    sumsq: &mut [f64],
    nnz: &mut [f64],
) {
    let n_groups = sum.len() / n_genes.max(1);
    let chunk = 1 << 16;
    let size = n_groups * n_genes;
    let (s, q, z) = rows
        .par_chunks(chunk)
        .zip(genes.par_chunks(chunk))
        .zip(values.par_chunks(chunk))
        .fold(
            || (vec![0.0f64; size], vec![0.0f64; size], vec![0.0f64; size]),
            |(mut s, mut q, mut z), ((rs, gs), vs)| {
                for ((&r, &g), &x) in rs.iter().zip(gs).zip(vs) {
                    let grp = group_of_row[r as usize];
                    if grp == u32::MAX {
                        continue;
                    }
                    let i = grp as usize * n_genes + g as usize;
                    let x = x as f64;
                    s[i] += x;
                    q[i] += x * x;
                    z[i] += 1.0;
                }
                (s, q, z)
            },
        )
        .reduce(
            || (vec![0.0f64; size], vec![0.0f64; size], vec![0.0f64; size]),
            |mut a, b| {
                a.0.iter_mut().zip(b.0).for_each(|(x, y)| *x += y);
                a.1.iter_mut().zip(b.1).for_each(|(x, y)| *x += y);
                a.2.iter_mut().zip(b.2).for_each(|(x, y)| *x += y);
                a
            },
        );
    sum.iter_mut().zip(s).for_each(|(a, b)| *a += b);
    sumsq.iter_mut().zip(q).for_each(|(a, b)| *a += b);
    nnz.iter_mut().zip(z).for_each(|(a, b)| *a += b);
}

/// Wilcoxon rank sums of every group vs the rest, per gene, exploiting sparsity:
/// all implicit zeros form one tie block, so only non-zero values are sorted.
///
/// Input: entries (gene_local, value, group) for genes 0..n_genes (any order);
/// `group_sizes[g]` = number of cells in group g. Returns (rank_sum n_groups ×
/// n_genes, tie term Σ(t³ - t) per gene).
pub fn wilcoxon_rank_sums(
    genes: &[u32],
    values: &[f32],
    groups: &[u32],
    n_genes: usize,
    group_sizes: &[u64],
) -> (Vec<f64>, Vec<f64>) {
    let n_groups = group_sizes.len();
    let n_total: u64 = group_sizes.iter().sum();
    // bucket entries by gene
    let mut cnt = vec![0usize; n_genes + 1];
    for &g in genes {
        cnt[g as usize + 1] += 1;
    }
    for i in 0..n_genes {
        cnt[i + 1] += cnt[i];
    }
    let mut pos = cnt[..n_genes].to_vec();
    let mut ent: Vec<(f32, u32)> = vec![(0.0, 0); genes.len()];
    for ((&g, &v), &grp) in genes.iter().zip(values).zip(groups) {
        let p = &mut pos[g as usize];
        ent[*p] = (v, grp);
        *p += 1;
    }
    let per_gene: Vec<(Vec<f64>, f64)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut e: Vec<(f32, u32)> = ent[cnt[g]..cnt[g + 1]].iter().copied().filter(|x| x.0 != 0.0).collect();
            e.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let mut nnz_grp = vec![0u64; n_groups];
            for x in &e {
                nnz_grp[x.1 as usize] += 1;
            }
            let n_nonzero = e.len() as u64;
            let n_zero = n_total - n_nonzero;
            let mut rs = vec![0.0f64; n_groups];
            let zero_rank = (n_zero as f64 + 1.0) / 2.0;
            for grp in 0..n_groups {
                rs[grp] += (group_sizes[grp] - nnz_grp[grp]) as f64 * zero_rank;
            }
            let mut tie = if n_zero > 1 { (n_zero as f64).powi(3) - n_zero as f64 } else { 0.0 };
            let mut i = 0;
            while i < e.len() {
                let mut j = i + 1;
                while j < e.len() && e[j].0 == e[i].0 {
                    j += 1;
                }
                // ranks n_zero + i + 1 ..= n_zero + j, averaged
                let avg = n_zero as f64 + (i + 1 + j) as f64 / 2.0;
                for x in &e[i..j] {
                    rs[x.1 as usize] += avg;
                }
                let t = (j - i) as f64;
                if t > 1.0 {
                    tie += t * t * t - t;
                }
                i = j;
            }
            (rs, tie)
        })
        .collect();
    let mut rank_sum = vec![0.0f64; n_groups * n_genes];
    let mut ties = vec![0.0f64; n_genes];
    for (g, (rs, t)) in per_gene.into_iter().enumerate() {
        for grp in 0..n_groups {
            rank_sum[grp * n_genes + g] = rs[grp];
        }
        ties[g] = t;
    }
    (rank_sum, ties)
}

/// out[row] += Σ_gene w[gene] * value (used for gene-set scores); w indexed by gene.
pub fn weighted_row_sums(rows: &[u32], genes: &[u32], values: &[f32], w: &[f64], out: &mut [f64]) {
    for ((&r, &g), &x) in rows.iter().zip(genes).zip(values) {
        let wg = w[g as usize];
        if wg != 0.0 {
            out[r as usize] += wg * x as f64;
        }
    }
}

/// QC per chunk on raw counts. Per-cell (indexed by original cell id): total
/// counts, genes detected, counts in `flag` genes (e.g. mitochondrial).
/// Per gene: cells detected, total counts.
pub fn qc_chunk(
    cells: &[u32],
    genes: &[u32],
    values: &[f32],
    flag: &[bool],
    cell_total: &mut [f64],
    cell_ngenes: &mut [u32],
    cell_flag: &mut [f64],
    gene_ncells: &mut [u32],
    gene_total: &mut [f64],
) {
    for ((&c, &g), &x) in cells.iter().zip(genes).zip(values) {
        if x == 0.0 {
            continue;
        }
        let (c, g) = (c as usize, g as usize);
        cell_total[c] += x as f64;
        cell_ngenes[c] += 1;
        if flag[g] {
            cell_flag[c] += x as f64;
        }
        gene_ncells[g] += 1;
        gene_total[g] += x as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_chunk() {
        // cell 0: genes 0,1 (2, 6); cell 1 dropped; cell 2: genes 1,2 (1, 3), gene 2 dropped
        let (r, g, v) = preprocess_chunk(&[0, 0, 1, 2, 2], &[0, 1, 0, 1, 2], &[2.0, 6.0, 5.0, 1.0, 3.0], &[0, -1, 1], &[0, 1, -1], 100.0, false);
        assert_eq!(r, vec![0, 0, 1]);
        assert_eq!(g, vec![0, 1, 1]);
        assert!((v[0] - 25.0).abs() < 1e-5 && (v[1] - 75.0).abs() < 1e-5 && (v[2] - 100.0).abs() < 1e-5);
    }

    /// Gram-based scaled PCA must equal PCA of the dense, scaled, clipped matrix.
    #[test]
    fn test_scaled_gram_matches_dense() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let (n, d) = (200usize, 12usize);
        let mut dense = vec![vec![0.0f64; d]; n];
        let (mut rows, mut genes, mut vals) = (vec![], vec![], vec![]);
        for i in 0..n {
            for j in 0..d {
                if rng.gen::<f64>() < 0.35 {
                    let x = if j == 3 { 50.0 } else { rng.gen_range(0.1f32..4.0) };
                    dense[i][j] = x as f64;
                    rows.push(i as u32);
                    genes.push(j as u32);
                    vals.push(x);
                }
            }
        }
        // gene stats (ddof = 1)
        let mean: Vec<f64> = (0..d).map(|j| dense.iter().map(|r| r[j]).sum::<f64>() / n as f64).collect();
        let std: Vec<f64> = (0..d)
            .map(|j| (dense.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt())
            .collect();
        let mv = 2.0;
        let t = Transform { scale: true, mean: &mean, std: &std, max_value: mv };
        let gmap: Vec<i32> = (0..d as i32).collect();
        let mut gram = vec![0.0f64; d * d];
        let mut colsum = vec![0.0f64; d];
        // feed in two chunks
        let cut = rows.iter().position(|&r| r >= 100).unwrap();
        gram_accumulate(&rows[..cut], &genes[..cut], &vals[..cut], &gmap, t, &mut gram, &mut colsum);
        gram_accumulate(&rows[cut..], &genes[cut..], &vals[cut..], &gmap, t, &mut gram, &mut colsum);
        let (ev, _, _) = pca_from_gram(&gram, &colsum, n, 5).unwrap();

        // dense reference: scale, clip, centre, covariance eigenvalues
        let z: Vec<Vec<f64>> = dense
            .iter()
            .map(|r| (0..d).map(|j| ((r[j] - mean[j]) / std[j]).clamp(-mv, mv)).collect())
            .collect();
        let zm: Vec<f64> = (0..d).map(|j| z.iter().map(|r| r[j]).sum::<f64>() / n as f64).collect();
        let cov = Mat::<f64>::from_fn(d, d, |a, b| {
            z.iter().map(|r| (r[a] - zm[a]) * (r[b] - zm[b])).sum::<f64>() / (n - 1) as f64
        });
        let ref_vals = cov.self_adjoint_eigenvalues(Side::Lower).unwrap();
        for c in 0..5 {
            let r = ref_vals[d - 1 - c];
            assert!((ev[c] - r).abs() < 1e-9 * r.max(1.0), "eig {}: {} vs {}", c, ev[c], r);
        }
    }

    #[test]
    fn test_wilcoxon_matches_naive() {
        // 3 groups, one gene, values with ties and zeros
        let vals = [0.0f32, 1.0, 1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0, 5.0];
        let grps = [0u32, 0, 1, 1, 2, 2, 0, 1, 2, 2];
        let sizes = [3u64, 3, 4];
        let nz: Vec<usize> = (0..10).filter(|&i| vals[i] != 0.0).collect();
        let (rs, tie) = wilcoxon_rank_sums(&vec![0; nz.len()], &nz.iter().map(|&i| vals[i]).collect::<Vec<_>>(), &nz.iter().map(|&i| grps[i]).collect::<Vec<_>>(), 1, &sizes);
        // naive average ranks
        let mut idx: Vec<usize> = (0..10).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        let mut ranks = [0.0f64; 10];
        let mut i = 0;
        while i < 10 {
            let mut j = i;
            while j < 10 && vals[idx[j]] == vals[idx[i]] { j += 1; }
            for &k in &idx[i..j] { ranks[k] = (i + 1 + j) as f64 / 2.0; }
            i = j;
        }
        for g in 0..3 {
            let expect: f64 = (0..10).filter(|&k| grps[k] == g as u32).map(|k| ranks[k]).sum();
            assert!((rs[g] - expect).abs() < 1e-12, "group {}: {} vs {}", g, rs[g], expect);
        }
        // ties: zeros x3, ones x3, twos x2
        assert_eq!(tie[0], 24.0 + 24.0 + 6.0);
    }
}
