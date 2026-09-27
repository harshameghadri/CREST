use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use faer::Mat;
use rayon::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

pub fn svd_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "pca",
        DataType::List(Box::new(DataType::List(Box::new(DataType::Float32)))),
    ))
}

/// Sparse matrix held in both CSR (row) and CSC (column) layout so that both
/// X @ M and X^T @ M are parallel, cache-friendly row-wise accumulations.
pub(crate) struct SparseMatrix {
    n_rows: usize,
    n_cols: usize,
    row_ptr: Vec<usize>,
    row_idx: Vec<u32>, // column index of each CSR entry
    row_val: Vec<f32>,
    col_ptr: Vec<usize>,
    col_idx: Vec<u32>, // row index of each CSC entry
    col_val: Vec<f32>,
}

impl SparseMatrix {
    pub(crate) fn from_triplets(n_rows: usize, n_cols: usize, rows: &[u32], cols: &[u32], vals: &[f32]) -> Self {
        fn compress(n: usize, major: &[u32], minor: &[u32], vals: &[f32]) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
            let mut ptr = vec![0usize; n + 1];
            for &m in major {
                ptr[m as usize + 1] += 1;
            }
            for i in 0..n {
                ptr[i + 1] += ptr[i];
            }
            let mut pos = ptr[..n].to_vec();
            let mut idx = vec![0u32; major.len()];
            let mut val = vec![0f32; major.len()];
            for ((&m, &o), &v) in major.iter().zip(minor).zip(vals) {
                let p = &mut pos[m as usize];
                idx[*p] = o;
                val[*p] = v;
                *p += 1;
            }
            (ptr, idx, val)
        }
        let (row_ptr, row_idx, row_val) = compress(n_rows, rows, cols, vals);
        let (col_ptr, col_idx, col_val) = compress(n_cols, cols, rows, vals);
        SparseMatrix { n_rows, n_cols, row_ptr, row_idx, row_val, col_ptr, col_idx, col_val }
    }

    /// Column means (for implicit centering).
    fn col_means(&self) -> Vec<f64> {
        (0..self.n_cols)
            .map(|j| {
                let r = self.col_ptr[j]..self.col_ptr[j + 1];
                self.col_val[r].iter().map(|&v| v as f64).sum::<f64>() / self.n_rows as f64
            })
            .collect()
    }
}

/// out (major × l, row-major) = S @ m (minor × l, row-major) for one compressed layout.
fn spmm(ptr: &[usize], idx: &[u32], val: &[f32], m: &[f64], l: usize, out: &mut [f64]) {
    out.par_chunks_mut(l).enumerate().for_each(|(i, row)| {
        row.iter_mut().for_each(|x| *x = 0.0);
        for p in ptr[i]..ptr[i + 1] {
            let v = val[p] as f64;
            let src = &m[idx[p] as usize * l..(idx[p] as usize + 1) * l];
            for (o, &x) in row.iter_mut().zip(src) {
                *o += v * x;
            }
        }
    });
}

/// Subtract the rank-one centering term: out -= a ⊗ (b^T m), i.e. out[i,:] -= a[i] * (Σ_r b[r] m[r,:]).
fn center_correction(out: &mut [f64], a: Option<&[f64]>, b: Option<&[f64]>, m: &[f64], l: usize) {
    let mut bm = vec![0.0f64; l];
    for (r, row) in m.chunks_exact(l).enumerate() {
        let w = b.map_or(1.0, |b| b[r]);
        if w != 0.0 {
            for (acc, &x) in bm.iter_mut().zip(row) {
                *acc += w * x;
            }
        }
    }
    out.par_chunks_mut(l).enumerate().for_each(|(i, row)| {
        let w = a.map_or(1.0, |a| a[i]);
        if w != 0.0 {
            for (o, &x) in row.iter_mut().zip(&bm) {
                *o -= w * x;
            }
        }
    });
}

/// Y = X_c @ M, where X_c = X - 1 μ^T is never materialised.
fn centered_mul(x: &SparseMatrix, mu: &[f64], m: &[f64], l: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; x.n_rows * l];
    spmm(&x.row_ptr, &x.row_idx, &x.row_val, m, l, &mut y);
    center_correction(&mut y, None, Some(mu), m, l); // - 1 (μ^T M)
    y
}

/// Z = X_c^T @ M.
fn centered_tmul(x: &SparseMatrix, mu: &[f64], m: &[f64], l: usize) -> Vec<f64> {
    let mut z = vec![0.0f64; x.n_cols * l];
    spmm(&x.col_ptr, &x.col_idx, &x.col_val, m, l, &mut z);
    center_correction(&mut z, Some(mu), None, m, l); // - μ (1^T M)
    z
}

/// Orthonormalise the columns of a row-major (rows × l) matrix via thin QR.
fn orthonormalize(y: &[f64], rows: usize, l: usize) -> Vec<f64> {
    let mat = Mat::<f64>::from_fn(rows, l, |i, j| y[i * l + j]);
    let q = mat.qr().compute_thin_Q();
    let mut out = vec![0.0f64; rows * l];
    out.par_chunks_mut(l).enumerate().for_each(|(i, row)| {
        for j in 0..l {
            row[j] = q[(i, j)];
        }
    });
    out
}

/// Randomized PCA of a sparse matrix with implicit mean-centering.
///
/// Halko, Martinsson & Tropp (2011), Algorithm 4.4 (randomized subspace
/// iteration): the sketch is re-orthonormalised after every multiplication so
/// that trailing components are not lost to round-off, as in scikit-learn's
/// `randomized_svd`.
///
/// Returns (PCA coordinates U·S as n_rows × k row-major rows, singular values, k).
pub(crate) fn randomized_svd(
    x: &SparseMatrix,
    n_comps: usize,
    n_oversampling: usize,
    n_power_iter: usize,
    seed: u64,
) -> Result<(Vec<Vec<f32>>, Vec<f64>, usize), String> {
    let (n_rows, n_cols) = (x.n_rows, x.n_cols);
    if n_rows < 2 || n_cols < 2 {
        return Err("Matrix must have at least 2 rows and 2 columns".to_string());
    }
    let k = n_comps.min(n_rows.min(n_cols) - 1);
    let l = (k + n_oversampling).min(n_rows.min(n_cols));
    let mu = x.col_means();

    // Gaussian test matrix Ω (n_cols × l)
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let omega: Vec<f64> = (0..n_cols * l).map(|_| StandardNormal.sample(&mut rng)).collect();

    let mut q = orthonormalize(&centered_mul(x, &mu, &omega, l), n_rows, l);
    for _ in 0..n_power_iter {
        let z = orthonormalize(&centered_tmul(x, &mu, &q, l), n_cols, l);
        q = orthonormalize(&centered_mul(x, &mu, &z, l), n_rows, l);
    }

    // B^T = X_c^T Q (n_cols × l);  B = U_B S V^T  with  B^T = V S U_B^T
    let bt = centered_tmul(x, &mu, &q, l);
    let bt_mat = Mat::<f64>::from_fn(n_cols, l, |i, j| bt[i * l + j]);
    let svd = bt_mat.thin_svd().map_err(|e| format!("SVD failed: {:?}", e))?;
    let u_b = svd.V(); // l × l  (left singular vectors of B)
    let s_col = svd.S().column_vector();
    let k = k.min(s_col.nrows());
    let sing: Vec<f64> = (0..k).map(|c| s_col[c]).collect();

    // U = Q U_B ; deterministic signs: largest-|value| entry of each U column positive
    let mut u = vec![0.0f64; n_rows * k];
    u.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        for c in 0..k {
            row[c] = (0..l).map(|s| q[i * l + s] * u_b[(s, c)]).sum();
        }
    });
    let mut sign = vec![1.0f64; k];
    for c in 0..k {
        let (mut best, mut arg) = (0.0f64, 0usize);
        for i in 0..n_rows {
            let a = u[i * k + c].abs();
            if a > best {
                best = a;
                arg = i;
            }
        }
        if u[arg * k + c] < 0.0 {
            sign[c] = -1.0;
        }
    }
    let coords: Vec<Vec<f32>> = u
        .par_chunks(k)
        .map(|row| (0..k).map(|c| (row[c] * sign[c] * sing[c]) as f32).collect())
        .collect();
    Ok((coords, sing, k))
}

#[polars_expr(output_type_func=svd_output)]
fn sparse_randomized_svd(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 6 {
        return Err(PolarsError::ComputeError(
            "sparse_randomized_svd requires at least 6 inputs: \
             [groups, genes, counts, n_cells, n_genes, n_comps, (n_iter), (seed)]".into()
        ));
    }

    let cell_ids = &inputs[0].list()?;
    let gene_ids = &inputs[1].list()?;
    let counts = &inputs[2].list()?;

    let n_cells_series = &inputs[3].u32()?;
    let n_genes_series = &inputs[4].u32()?;
    let n_comps_series = &inputs[5].u32()?;

    let n_cells = n_cells_series.get(0).unwrap_or(0) as usize;
    let n_genes = n_genes_series.get(0).unwrap_or(0) as usize;
    let n_comps = n_comps_series.get(0).unwrap_or(50) as usize;

    if n_cells == 0 || n_genes == 0 {
        return Err(PolarsError::ComputeError("Invalid sparse array dimensions.".into()));
    }

    // Guard against unreasonable dimensions
    const MAX_DIM: usize = 10_000_000;
    if n_cells > MAX_DIM || n_genes > MAX_DIM {
        return Err(PolarsError::ComputeError(
            format!("Dimensions too large: {}x{} (max {})", n_cells, n_genes, MAX_DIM).into()
        ));
    }

    // 1. Gather the COO triplets and build CSR/CSC natively in Rust
    let mut rows: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();

    for ((opt_cells, opt_genes), opt_counts) in cell_ids.into_iter()
        .zip(gene_ids.into_iter())
        .zip(counts.into_iter())
    {
        if let (Some(c), Some(g), Some(v)) = (opt_cells, opt_genes, opt_counts) {
            let cells_ca = c.u32()?;
            let genes_ca = g.u32()?;
            let vals_ca = v.f32()?;

            for ((cell_idx, gene_idx), val) in cells_ca.into_iter()
                .zip(genes_ca.into_iter())
                .zip(vals_ca.into_iter())
            {
                if let (Some(row), Some(col), Some(count)) = (cell_idx, gene_idx, val) {
                    if row as usize >= n_cells || col as usize >= n_genes {
                        return Err(PolarsError::ComputeError(
                            format!("COO index out of bounds: ({}, {}) for {}x{}", row, col, n_cells, n_genes).into()
                        ));
                    }
                    rows.push(row);
                    cols.push(col);
                    vals.push(count);
                }
            }
        }
    }
    let x = SparseMatrix::from_triplets(n_cells, n_genes, &rows, &cols, &vals);
    drop((rows, cols, vals));

    // 2. Randomized PCA (subspace iteration; sklearn-like defaults)
    let n_oversampling = 20;
    let n_power_iter = if inputs.len() > 6 { inputs[6].u32()?.get(0).unwrap_or(7) as usize } else { 7 };
    let seed = if inputs.len() > 7 { inputs[7].u64()?.get(0).unwrap_or(42) } else { 42 };

    let (pca_coords, _sing, k) = randomized_svd(&x, n_comps, n_oversampling, n_power_iter, seed)
        .map_err(|e| PolarsError::ComputeError(format!("SVD failed: {}", e).into()))?;

    // 3. Build output as List(List(Float32))
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        "pca", n_cells, k, DataType::Float32,
    );

    for row in &pca_coords {
        builder.append_slice(row);
    }

    let s_orig = builder.finish().into_series();
    let s_wrapped = Series::new("pca".into(), &[AnyValue::List(s_orig)]);
    Ok(s_wrapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_pca_singular_values(dense: &[Vec<f64>]) -> Vec<f64> {
        let (n, g) = (dense.len(), dense[0].len());
        let means: Vec<f64> = (0..g).map(|j| dense.iter().map(|r| r[j]).sum::<f64>() / n as f64).collect();
        let m = Mat::<f64>::from_fn(n, g, |i, j| dense[i][j] - means[j]);
        let svd = m.thin_svd().unwrap();
        let s = svd.S().column_vector();
        (0..s.nrows()).map(|i| s[i]).collect()
    }

    /// Randomized PCA must reproduce the exact singular values and principal
    /// subspace of the dense, mean-centred matrix (including trailing components).
    #[test]
    fn test_randomized_svd_matches_dense() {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let (n, g, k) = (300usize, 60usize, 20usize);
        let mut dense = vec![vec![0.0f64; g]; n];
        let (mut r, mut c, mut v) = (vec![], vec![], vec![]);
        for i in 0..n {
            for j in 0..g {
                // sparse, low-rank-plus-noise structure with a slowly decaying spectrum
                if rng.gen::<f64>() < 0.3 {
                    let val = ((i % 7) as f64 + 1.0) * ((j % 5) as f64 + 1.0) * 0.1 + rng.gen::<f64>();
                    dense[i][j] = val as f32 as f64;
                    r.push(i as u32); c.push(j as u32); v.push(val as f32);
                }
            }
        }
        let x = SparseMatrix::from_triplets(n, g, &r, &c, &v);
        let (coords, sing, kk) = randomized_svd(&x, k, 20, 7, 42).unwrap();
        assert_eq!(kk, k);
        assert_eq!(coords.len(), n);
        let exact = dense_pca_singular_values(&dense);
        for i in 0..k {
            let rel = (sing[i] - exact[i]).abs() / exact[i];
            assert!(rel < 1e-6, "singular value {} rel err {}", i, rel);
        }
        // coordinate column norms equal singular values
        for c in 0..k {
            let norm: f64 = coords.iter().map(|row| (row[c] as f64).powi(2)).sum::<f64>().sqrt();
            assert!((norm - sing[c]).abs() / sing[c] < 1e-4);
        }
    }

    #[test]
    fn test_randomized_svd_empty() {
        let x = SparseMatrix::from_triplets(0, 0, &[], &[], &[]);
        assert!(randomized_svd(&x, 2, 5, 2, 42).is_err());
    }
}
