//! Fused streaming kernels over cell-sorted chunks of the raw count matrix.
//!
//! A chunk is read-only raw data (CSR via `indptr`, or COO via per-entry cell
//! ids) plus the current cell/gene filters and the lazy `normalize_total` /
//! `log1p` transform. Every kernel applies filters and transform inline, one
//! cell at a time into a small per-thread buffer, so no filtered or normalised
//! copy of the chunk is ever materialised. Feeding the chunks of a dataset in
//! turn processes any size with memory = one raw chunk + small accumulators.
//!
//! `normalize_total` divides by each cell's total over the genes kept when it
//! was called (as scanpy): Python computes those totals once and passes them as
//! `cell_totals`, so filtering genes later never renormalises the cells.
//! Without `cell_totals` the kernels use the total over the genes kept now.

use faer::{Mat, Side};
use rayon::prelude::*;

/// How entries map to cells.
#[derive(Clone, Copy)]
pub enum Cells<'a> {
    /// per-entry cell id (non-decreasing)
    Ids(&'a [u32]),
    /// CSR row pointer for cells `first .. first + indptr.len() - 1`; offsets
    /// are absolute, `indptr[0]` is the offset of `genes[0]`.
    Ptr { indptr: &'a [i64], first: u32 },
}

#[derive(Clone, Copy)]
pub struct ChunkView<'a> {
    pub cells: Cells<'a>,
    pub genes: &'a [u32],
    pub values: &'a [f32],
    pub cell_map: &'a [i64],
    pub gene_map: &'a [i32],
    pub target_sum: f64,
    pub log1p: bool,
    /// Per-cell totals for `normalize_total`, indexed by store cell id (as
    /// `cell_map`). `None`: the total over the currently kept genes.
    pub cell_totals: Option<&'a [f64]>,
}

impl<'a> ChunkView<'a> {
    /// (cell id, entry lo, entry hi) for each cell present in the chunk.
    fn cell_spans(&self) -> Vec<(u32, usize, usize)> {
        match self.cells {
            Cells::Ptr { indptr, first } => {
                let base = indptr.first().copied().unwrap_or(0);
                (0..indptr.len().saturating_sub(1))
                    .map(|i| (first + i as u32, (indptr[i] - base) as usize, (indptr[i + 1] - base) as usize))
                    .collect()
            }
            Cells::Ids(ids) => {
                let mut out = Vec::new();
                let mut p = 0;
                while p < ids.len() {
                    let c = ids[p];
                    let mut q = p + 1;
                    while q < ids.len() && ids[q] == c {
                        q += 1;
                    }
                    out.push((c, p, q));
                    p = q;
                }
                out
            }
        }
    }

    /// Filtered + transformed entries of one cell into `buf`; returns its row.
    #[inline]
    fn cell(&self, c: u32, lo: usize, hi: usize, buf: &mut Vec<(u32, f32)>) -> Option<u32> {
        buf.clear();
        let row = *self.cell_map.get(c as usize)?;
        if row < 0 {
            return None;
        }
        let mut total = 0.0f64;
        for e in lo..hi {
            let m = self.gene_map.get(self.genes[e] as usize).copied().unwrap_or(-1);
            if m >= 0 {
                buf.push((m as u32, self.values[e]));
                total += self.values[e] as f64;
            }
        }
        if let Some(t) = self.cell_totals {
            total = t[c as usize];
        }
        if self.target_sum > 0.0 || self.log1p {
            let scale = if self.target_sum > 0.0 && total > 0.0 { self.target_sum / total } else { 1.0 };
            for x in buf.iter_mut() {
                let y = if self.target_sum > 0.0 { (x.1 as f64 * scale) as f32 } else { x.1 };
                x.1 = if self.log1p { y.ln_1p() } else { y };
            }
        }
        Some(row as u32)
    }

    /// Group cell spans into ~4 × threads parts of similar entry counts.
    fn parts(&self) -> Vec<Vec<(u32, usize, usize)>> {
        let spans = self.cell_spans();
        let total = self.genes.len().max(1);
        let n_parts = (rayon::current_num_threads() * 4).min(total / 8192 + 1).max(1);
        let mut parts: Vec<Vec<(u32, usize, usize)>> = vec![Vec::new()];
        let mut acc = 0usize;
        for s in spans {
            if acc * n_parts >= total * parts.len() && !parts.last().unwrap().is_empty() {
                parts.push(Vec::new());
            }
            acc += s.2 - s.1;
            parts.last_mut().unwrap().push(s);
        }
        parts
    }

    /// Run `f(row, entries)` over every kept cell in parallel, folding per part.
    fn fold_cells<T: Send, I: Fn() -> T + Sync, F: Fn(&mut T, u32, &[(u32, f32)]) + Sync>(&self, init: I, f: F) -> Vec<T> {
        self.parts()
            .into_par_iter()
            .map(|part| {
                let mut acc = init();
                let mut buf = Vec::new();
                for (c, lo, hi) in part {
                    if let Some(row) = self.cell(c, lo, hi, &mut buf) {
                        f(&mut acc, row, &buf);
                    }
                }
                acc
            })
            .collect()
    }
}

/// Materialise the filtered + transformed chunk as (row, gene, value) triplets.
pub fn materialize(v: ChunkView) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let parts = v.fold_cells(|| (Vec::new(), Vec::new(), Vec::new()), |acc, row, e| {
        for &(g, x) in e {
            acc.0.push(row);
            acc.1.push(g);
            acc.2.push(x);
        }
    });
    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let (mut r, mut g, mut x) = (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total));
    for (a, b, c) in parts {
        r.extend(a);
        g.extend(b);
        x.extend(c);
    }
    (r, g, x)
}

/// Per-gene sums on two scales: the values as given (log domain) and expm1
/// (normalised-count domain, used by seurat HVG). `out` is n_genes × 5:
/// [Σx, Σx², Σexpm1(x), Σexpm1(x)², nnz]. Stored zeros are skipped (they are not
/// non-zeros, as in `qc`).
pub fn gene_stats(v: ChunkView, out: &mut [f64]) {
    let n = out.len();
    let parts = v.fold_cells(|| vec![0.0f64; n], |acc, _row, e| {
        for &(g, x) in e {
            if x == 0.0 {
                continue;
            }
            let (x, ex) = (x as f64, (x as f64).exp_m1());
            let o = &mut acc[g as usize * 5..g as usize * 5 + 5];
            o[0] += x;
            o[1] += x * x;
            o[2] += ex;
            o[3] += ex * ex;
            o[4] += 1.0;
        }
    });
    for p in parts {
        out.iter_mut().zip(p).for_each(|(o, x)| *o += x);
    }
}

/// QC on raw counts (call with an un-transformed view). Per-row: total counts,
/// genes detected, counts in `flag` genes. Per gene: cells detected, total.
pub fn qc(v: ChunkView, flag: &[bool], cell_total: &mut [f64], cell_ngenes: &mut [u32], cell_flag: &mut [f64],
          gene_ncells: &mut [u32], gene_total: &mut [f64]) {
    let ng = gene_total.len();
    let parts = v.fold_cells(
        || (Vec::new(), vec![0u32; ng], vec![0.0f64; ng]),
        |acc, row, e| {
            let (mut t, mut n, mut f) = (0.0f64, 0u32, 0.0f64);
            for &(g, x) in e {
                if x == 0.0 {
                    continue;
                }
                t += x as f64;
                n += 1;
                if flag[g as usize] {
                    f += x as f64;
                }
                acc.1[g as usize] += 1;
                acc.2[g as usize] += x as f64;
            }
            acc.0.push((row, t, n, f));
        },
    );
    for (cells, gn, gt) in parts {
        for (row, t, n, f) in cells {
            cell_total[row as usize] += t;
            cell_ngenes[row as usize] += n;
            cell_flag[row as usize] += f;
        }
        gene_ncells.iter_mut().zip(gn).for_each(|(a, b)| *a += b);
        gene_total.iter_mut().zip(gt).for_each(|(a, b)| *a += b);
    }
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
/// `sub_map[var] >= 0` take part (mapped to 0..d). `gram` is d × d row-major
/// (upper triangle filled; `pca_from_gram` reads only that).
///
/// Threads own disjoint row blocks of G (balanced by work), so there is no
/// reduction step and memory stays at one d × d matrix.
pub fn gram_accumulate(v: ChunkView, sub_map: &[i32], t: Transform, gram: &mut [f64], colsum: &mut [f64]) {
    let d = colsum.len();
    // compact to rows of (mapped gene, transformed value), sorted by gene within row
    let parts = v.fold_cells(|| (Vec::<usize>::new(), Vec::<u32>::new(), Vec::<f64>::new()), |acc, _row, e| {
        let s = acc.1.len();
        for &(g, x) in e {
            let m = sub_map[g as usize];
            if m >= 0 {
                acc.1.push(m as u32);
                acc.2.push(t.apply(m as usize, x));
            }
        }
        if acc.1.len() > s {
            if acc.1[s..].windows(2).any(|w| w[0] > w[1]) {
                let mut idx: Vec<usize> = (s..acc.1.len()).collect();
                idx.sort_by_key(|&i| acc.1[i]);
                let (g2, v2): (Vec<u32>, Vec<f64>) = idx.iter().map(|&i| (acc.1[i], acc.2[i])).unzip();
                acc.1[s..].copy_from_slice(&g2);
                acc.2[s..].copy_from_slice(&v2);
            }
            acc.0.push(acc.1.len());
        }
    });
    let mut starts = vec![0usize];
    let (mut eg, mut ev) = (Vec::new(), Vec::new());
    for (ends, g, x) in parts {
        let off = eg.len();
        starts.extend(ends.into_iter().map(|e| e + off));
        eg.extend(g);
        ev.extend(x);
    }
    for (&g, &x) in eg.iter().zip(&ev) {
        colsum[g as usize] += x;
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
    let eig = cov.self_adjoint_eigen(Side::Upper).map_err(|e| format!("{:?}", e));
    crate::simd::clean_simd_state();
    let eig = eig?;
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
pub fn project(view: ChunkView, sub_map: &[i32], t: Transform, v: &[f64], k: usize, out: &mut [f32]) {
    let parts = view.fold_cells(Vec::new, |acc: &mut Vec<(u32, Vec<f64>)>, row, e| {
        let mut a = vec![0.0f64; k];
        for &(g, x) in e {
            let m = sub_map[g as usize];
            if m >= 0 {
                let s = t.apply(m as usize, x);
                let vr = &v[m as usize * k..(m as usize + 1) * k];
                a.iter_mut().zip(vr).for_each(|(o, &y)| *o += s * y);
            }
        }
        acc.push((row, a));
    });
    for part in parts {
        for (r, a) in part {
            let o = &mut out[r as usize * k..(r as usize + 1) * k];
            o.iter_mut().zip(a).for_each(|(x, y)| *x += y as f32);
        }
    }
}

/// Raw pointer with the length of the buffer it points into, shared across threads
/// that write disjoint elements. Every write goes through `add`, which checks the index.
#[derive(Clone, Copy)]
struct SyncPtr(*mut f64, usize);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

impl SyncPtr {
    fn null() -> Self {
        SyncPtr(std::ptr::null_mut(), 0)
    }
    fn of(buf: &mut [f64]) -> Self {
        SyncPtr(buf.as_mut_ptr(), buf.len())
    }
    fn is_null(&self) -> bool {
        self.0.is_null()
    }
    /// `*self[i] += x`; false (and no write) when `i` is out of bounds.
    ///
    /// # Safety
    /// No other thread may access element `i` concurrently.
    #[inline]
    unsafe fn add(&self, i: usize, x: f64) -> bool {
        if i >= self.1 {
            return false;
        }
        *self.0.add(i) += x;
        true
    }
}

/// Per (group, gene) Σx, Σx², nnz for one chunk; `group_of_row[row]` < n_groups
/// or u32::MAX to ignore the cell. Buffers are n_groups × n_genes.
///
/// The chunk is transformed once, then each thread owns a contiguous block of
/// genes and accumulates directly into the shared outputs (disjoint columns),
/// so memory does not grow with the number of groups × threads.
/// With `clip` ((n_groups, n_genes)), each value is first capped at its group's
/// gene limit (seurat_v3 HVG). Stored zeros are not counted in `nnz`.
///
/// A group id `>= n_groups` (with `n_groups = sum.len() / n_genes`) is reported as an
/// error, never written: the kernel checks every output index before writing, so a
/// caller whose `group_of_row` changes during the call cannot cause an out-of-bounds write.
#[allow(clippy::too_many_arguments)]
pub fn group_gene_sums(v: ChunkView, group_of_row: &[u32], n_genes: usize, sum: &mut [f64], sumsq: Option<&mut [f64]>,
                       nnz: Option<&mut [f64]>, clip: Option<&[f64]>) -> Result<(), String> {
    if sumsq.as_ref().map_or(false, |q| q.len() != sum.len()) || nnz.as_ref().map_or(false, |z| z.len() != sum.len()) {
        return Err("sum, sumsq and nnz must have the same length".into());
    }
    if clip.map_or(false, |c| c.len() != sum.len()) {
        return Err("clip must have the same length as sum".into());
    }
    let (rows, genes, vals) = materialize(v);
    // entries per gene -> balanced gene blocks
    let mut per_gene = vec![0usize; n_genes];
    for &g in &genes {
        per_gene[g as usize] += 1;
    }
    let total = genes.len().max(1);
    let parts = rayon::current_num_threads().max(1);
    let mut bounds = vec![0u32];
    let mut acc = 0usize;
    for (g, &c) in per_gene.iter().enumerate() {
        acc += c;
        if acc * parts >= total * bounds.len() && bounds.len() < parts && g + 1 < n_genes {
            bounds.push(g as u32 + 1);
        }
    }
    bounds.push(n_genes as u32);
    bounds.dedup();
    let (ps, pq, pz) = (
        SyncPtr::of(sum),
        sumsq.map_or_else(SyncPtr::null, SyncPtr::of),
        nnz.map_or_else(SyncPtr::null, SyncPtr::of),
    );
    let out_of_range = std::sync::atomic::AtomicBool::new(false);
    bounds.par_windows(2).for_each(|w| {
        let (lo, hi) = (w[0], w[1]);
        let (ps, pq, pz) = (ps, pq, pz);
        for ((&r, &g), &x) in rows.iter().zip(&genes).zip(&vals) {
            if g < lo || g >= hi {
                continue;
            }
            let grp = group_of_row[r as usize];
            if grp == u32::MAX {
                continue;
            }
            // checked: an id that is not a valid group index lands at or beyond `ps.1`
            let i = match (grp as usize).checked_mul(n_genes).and_then(|b| b.checked_add(g as usize)) {
                Some(i) if i < ps.1 => i,
                _ => {
                    out_of_range.store(true, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };
            if x == 0.0 {
                continue; // a stored zero adds nothing and is not a non-zero
            }
            let x = match clip {
                Some(c) => (x as f64).min(c[i]),
                None => x as f64,
            };
            // SAFETY: gene blocks are disjoint, so index i is written by one thread only;
            // `add` re-checks the bound of each buffer before writing.
            let ok = unsafe { ps.add(i, x) && (pq.is_null() || pq.add(i, x * x)) && (pz.is_null() || pz.add(i, 1.0)) };
            if !ok {
                out_of_range.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        }
    });
    if out_of_range.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("group id out of range".into());
    }
    Ok(())
}

/// Entries of genes in [lo, hi) as (gene - lo, value, group) for cells with a group.
pub fn collect_gene_block(v: ChunkView, group_of_row: &[u32], lo: u32, hi: u32) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
    let parts = v.fold_cells(|| (Vec::new(), Vec::new(), Vec::new()), |acc, row, e| {
        let grp = group_of_row[row as usize];
        if grp == u32::MAX {
            return;
        }
        for &(g, x) in e {
            if g >= lo && g < hi && x != 0.0 {
                acc.0.push(g - lo);
                acc.1.push(x);
                acc.2.push(grp);
            }
        }
    });
    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let (mut a, mut b, mut c) = (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total));
    for (x, y, z) in parts {
        a.extend(x);
        b.extend(y);
        c.extend(z);
    }
    (a, b, c)
}

/// Checks the input of `wilcoxon_rank_sums`: ids in range, and no more non-zero
/// entries of a gene in a group than the group has cells (a repeated (cell, gene)
/// pair would otherwise make the unsigned zero counts wrap around).
pub fn check_rank_sum_counts(genes: &[u32], values: &[f32], groups: &[u32], n_genes: usize, group_sizes: &[u64]) -> Result<(), String> {
    let n_groups = group_sizes.len();
    let mut nnz = vec![0u64; n_groups * n_genes];
    for ((&g, &v), &grp) in genes.iter().zip(values).zip(groups) {
        if g as usize >= n_genes || grp as usize >= n_groups {
            return Err("gene or group index out of range".into());
        }
        if v != 0.0 {
            nnz[grp as usize * n_genes + g as usize] += 1;
        }
    }
    for (i, &c) in nnz.iter().enumerate() {
        if c > group_sizes[i / n_genes.max(1)] {
            return Err(format!(
                "gene {} has {} non-zero entries in group {}, which has {} cells: more entries than cells (is a (cell, gene) pair repeated?)",
                i % n_genes.max(1), c, i / n_genes.max(1), group_sizes[i / n_genes.max(1)]
            ));
        }
    }
    Ok(())
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

/// out[row] += Σ_gene w[gene] * value (used for gene-set scores).
pub fn weighted_row_sums(v: ChunkView, w: &[f64], out: &mut [f64]) {
    let parts = v.fold_cells(Vec::new, |acc: &mut Vec<(u32, f64)>, row, e| {
        let s: f64 = e.iter().map(|&(g, x)| w[g as usize] * x as f64).sum();
        acc.push((row, s));
    });
    for part in parts {
        for (r, s) in part {
            out[r as usize] += s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_materialize_filters_and_normalises() {
        // cell 0: genes 0,1 (2, 6); cell 1 dropped; cell 2: genes 1,2 (1, 3), gene 2 dropped
        let (cells, genes, vals) = ([0u32, 0, 1, 2, 2], [0u32, 1, 0, 1, 2], [2.0f32, 6.0, 5.0, 1.0, 3.0]);
        let (cm, gm) = ([0i64, -1, 1], [0i32, 1, -1]);
        let v = ChunkView { cells: Cells::Ids(&cells), genes: &genes, values: &vals, cell_map: &cm, gene_map: &gm, target_sum: 100.0, log1p: false, cell_totals: None };
        let (r, g, x) = materialize(v);
        assert_eq!(r, vec![0, 0, 1]);
        assert_eq!(g, vec![0, 1, 1]);
        assert!((x[0] - 25.0).abs() < 1e-5 && (x[1] - 75.0).abs() < 1e-5 && (x[2] - 100.0).abs() < 1e-5);
        // same data as CSR
        let indptr = [0i64, 2, 3, 5];
        let v2 = ChunkView { cells: Cells::Ptr { indptr: &indptr, first: 0 }, ..v };
        assert_eq!(materialize(v2), (r, g, x));
    }

    #[test]
    fn test_stored_cell_totals_override_kept_gene_totals() {
        // cell 0 had total 16 when normalised (gene 2, now dropped, held 8 counts)
        let (cells, genes, vals) = ([0u32, 0, 0, 2, 2], [0u32, 1, 2, 1, 2], [2.0f32, 6.0, 8.0, 1.0, 3.0]);
        let (cm, gm) = ([0i64, -1, 1], [0i32, 1, -1]);
        let totals = [16.0f64, f64::NAN, 4.0];
        let v = ChunkView { cells: Cells::Ids(&cells), genes: &genes, values: &vals, cell_map: &cm, gene_map: &gm,
                            target_sum: 100.0, log1p: false, cell_totals: Some(&totals) };
        let (r, g, x) = materialize(v);
        assert_eq!((r, g), (vec![0, 0, 1], vec![0, 1, 1]));
        assert!((x[0] - 12.5).abs() < 1e-5 && (x[1] - 37.5).abs() < 1e-5 && (x[2] - 25.0).abs() < 1e-5);
        // without stored totals: renormalised over the kept genes
        let (_, _, y) = materialize(ChunkView { cell_totals: None, ..v });
        assert!((y[0] - 25.0).abs() < 1e-5 && (y[2] - 100.0).abs() < 1e-5);
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
        let cmap: Vec<i64> = (0..n as i64).collect();
        let mut gram = vec![0.0f64; d * d];
        let mut colsum = vec![0.0f64; d];
        // feed in two chunks
        let cut = rows.iter().position(|&r| r >= 100).unwrap();
        for (lo, hi) in [(0, cut), (cut, rows.len())] {
            let v = ChunkView { cells: Cells::Ids(&rows[lo..hi]), genes: &genes[lo..hi], values: &vals[lo..hi],
                                cell_map: &cmap, gene_map: &gmap, target_sum: 0.0, log1p: false, cell_totals: None };
            gram_accumulate(v, &gmap, t, &mut gram, &mut colsum);
        }
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


#[cfg(test)]
mod rank_sum_count_tests {
    use super::*;

    #[test]
    fn test_more_entries_than_cells_is_an_error_not_a_wrap() {
        // three non-zero entries of one gene in a group of two cells
        let r = check_rank_sum_counts(&[0, 0, 0], &[1.0, 2.0, 3.0], &[0, 0, 0], 1, &[2]);
        assert!(r.unwrap_err().contains("more entries than cells"));
        // per group: group 1 has one cell but two entries
        let r = check_rank_sum_counts(&[0; 4], &[1.0, 2.0, 3.0, 4.0], &[0, 0, 1, 1], 1, &[3, 1]);
        assert!(r.is_err());
        // explicit zeros are not entries; counts equal to the group size are fine
        assert!(check_rank_sum_counts(&[0, 0, 0], &[1.0, 0.0, 3.0], &[0, 0, 0], 1, &[2]).is_ok());
        assert!(check_rank_sum_counts(&[0, 0], &[1.0, 3.0], &[0, 0], 1, &[2]).is_ok());
        // ids out of range
        assert!(check_rank_sum_counts(&[1], &[1.0], &[0], 1, &[2]).is_err());
        assert!(check_rank_sum_counts(&[0], &[1.0], &[1], 1, &[2]).is_err());
    }
}

#[cfg(test)]
mod group_sums_tests {
    use super::*;

    fn view<'a>(genes: &'a [u32], vals: &'a [f32], indptr: &'a [i64], cm: &'a [i64], gm: &'a [i32]) -> ChunkView<'a> {
        ChunkView { cells: Cells::Ptr { indptr, first: 0 }, genes, values: vals, cell_map: cm, gene_map: gm, target_sum: 0.0, log1p: false, cell_totals: None }
    }

    #[test]
    fn test_group_id_beyond_n_groups_is_an_error_and_writes_nothing() {
        // 3 cells, 2 genes; group ids 0, 1 and 7 with only 2 groups
        let (genes, vals, indptr) = ([0u32, 1, 0, 1, 0], [1.0f32, 2.0, 3.0, 4.0, 5.0], [0i64, 2, 4, 5]);
        let (cm, gm) = ([0i64, 1, 2], [0i32, 1]);
        for bad in [2u32, 7, u32::MAX - 1] {
            let grp = [0u32, 1, bad];
            let (mut s, mut q, mut z) = (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
            let r = group_gene_sums(view(&genes, &vals, &indptr, &cm, &gm), &grp, 2, &mut s, Some(&mut q), Some(&mut z), None);
            assert_eq!(r, Err("group id out of range".to_string()));
            // the valid cells may or may not have been added before the error, nothing outside the buffers was touched
            assert_eq!(s.len(), 4);
        }
    }

    #[test]
    fn test_valid_group_ids_and_ignored_cells() {
        let (genes, vals, indptr) = ([0u32, 1, 0, 1, 0], [1.0f32, 2.0, 3.0, 4.0, 5.0], [0i64, 2, 4, 5]);
        let (cm, gm) = ([0i64, 1, 2], [0i32, 1]);
        let grp = [0u32, 1, u32::MAX];
        let (mut s, mut q, mut z) = (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
        group_gene_sums(view(&genes, &vals, &indptr, &cm, &gm), &grp, 2, &mut s, Some(&mut q), Some(&mut z), None).unwrap();
        assert_eq!(s, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(q, vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(z, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_stored_zeros_are_not_counted_as_non_zeros() {
        // cell 0 stores gene 1 as an explicit zero
        let (genes, vals, indptr) = ([0u32, 1, 0, 1], [2.0f32, 0.0, 3.0, 4.0], [0i64, 2, 4]);
        let (cm, gm) = ([0i64, 1], [0i32, 1]);
        let grp = [0u32, 0];
        let (mut s, mut q, mut z) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
        group_gene_sums(view(&genes, &vals, &indptr, &cm, &gm), &grp, 2, &mut s, Some(&mut q), Some(&mut z), None).unwrap();
        assert_eq!((s, q, z), (vec![5.0, 4.0], vec![13.0, 16.0], vec![2.0, 1.0]));
        let mut out = vec![0.0; 10];
        gene_stats(view(&genes, &vals, &indptr, &cm, &gm), &mut out);
        assert_eq!((out[4], out[9]), (2.0, 1.0));
    }

    #[test]
    fn test_mismatched_output_lengths_are_an_error() {
        let (genes, vals, indptr) = ([0u32], [1.0f32], [0i64, 1]);
        let (cm, gm) = ([0i64], [0i32]);
        let (mut s, mut q) = (vec![0.0; 2], vec![0.0; 1]);
        assert!(group_gene_sums(view(&genes, &vals, &indptr, &cm, &gm), &[0], 1, &mut s, Some(&mut q), None, None).is_err());
    }
}
