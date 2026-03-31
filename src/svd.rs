use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use nalgebra_sparse::coo::CooMatrix;
use faer::Mat;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

pub fn svd_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "pca",
        DataType::List(Box::new(DataType::List(Box::new(DataType::Float32)))),
    ))
}

/// Halko Randomized SVD (Halko, Martinsson & Tropp 2011)
///
/// Algorithm:
/// 1. Generate random Gaussian matrix Ω of shape (n_genes, k + oversampling)
/// 2. Form sketch Y = X @ Ω  (sparse matrix × dense — O(nnz × (k+p)))
/// 3. Power iteration: for q iterations, Y = X @ (X^T @ Y) to improve range approx
/// 4. QR decompose Y → Q
/// 5. B = Q^T @ X → small dense matrix
/// 6. SVD of B → U_B, S, V^T
/// 7. U = Q @ U_B
/// 8. PCA coordinates = U[:, :k] * S[:k]
///
/// This avoids materializing the dense mean-centered matrix.
fn randomized_svd(
    coo: &CooMatrix<f64>,
    n_comps: usize,
    n_oversampling: usize,
    n_power_iter: usize,
    seed: u64,
) -> Result<(Vec<Vec<f32>>, usize), String> {
    let n_rows = coo.nrows();
    let n_cols = coo.ncols();

    if n_rows == 0 || n_cols == 0 {
        return Err("Empty matrix".to_string());
    }

    let k = n_comps.min(n_rows.min(n_cols) - 1);
    let sketch_dim = (k + n_oversampling).min(n_cols);

    // 1. Compute column means for implicit centering (one pass over sparse data)
    let mut col_sums = vec![0.0f64; n_cols];
    for (&_row, &col, &val) in coo.triplet_iter().map(|(r, c, v)| (r, c, v)) {
        col_sums[col] += val;
    }
    let col_means: Vec<f64> = col_sums.iter().map(|s| s / n_rows as f64).collect();

    // 2. Generate random Gaussian sketch matrix Ω (n_cols × sketch_dim)
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = StandardNormal;
    let mut omega = Mat::<f64>::zeros(n_cols, sketch_dim);
    for j in 0..sketch_dim {
        for i in 0..n_cols {
            omega[(i, j)] = normal.sample(&mut rng);
        }
    }

    // 3. Compute Y = X_centered @ Ω via sparse multiply + mean correction
    // Y = X @ Ω - ones * (means @ Ω)
    // where means @ Ω is a 1×sketch_dim row vector broadcast to all rows
    let mut y = Mat::<f64>::zeros(n_rows, sketch_dim);

    // Sparse X @ Ω: for each non-zero (row, col, val), add val * Ω[col, :]
    for (&row, &col, &val) in coo.triplet_iter().map(|(r, c, v)| (r, c, v)) {
        for s in 0..sketch_dim {
            y[(row, s)] += val * omega[(col, s)];
        }
    }

    // Mean correction: subtract means @ Ω broadcast across all rows
    let mut means_omega = vec![0.0f64; sketch_dim];
    for s in 0..sketch_dim {
        for c in 0..n_cols {
            means_omega[s] += col_means[c] * omega[(c, s)];
        }
    }
    for i in 0..n_rows {
        for s in 0..sketch_dim {
            y[(i, s)] -= means_omega[s];
        }
    }

    // 4. Power iterations to sharpen the range approximation
    for _ in 0..n_power_iter {
        // Compute X_centered^T @ Y = X^T @ Y - means * (ones^T @ Y)
        let mut xty = Mat::<f64>::zeros(n_cols, sketch_dim);

        // Sparse X^T @ Y
        for (&row, &col, &val) in coo.triplet_iter().map(|(r, c, v)| (r, c, v)) {
            for s in 0..sketch_dim {
                xty[(col, s)] += val * y[(row, s)];
            }
        }

        // Mean correction: subtract means * sum_rows(Y)
        let mut col_sums_y = vec![0.0f64; sketch_dim];
        for s in 0..sketch_dim {
            for i in 0..n_rows {
                col_sums_y[s] += y[(i, s)];
            }
        }
        for c in 0..n_cols {
            for s in 0..sketch_dim {
                xty[(c, s)] -= col_means[c] * col_sums_y[s];
            }
        }

        // Y = X_centered @ (X_centered^T @ Y)
        y = Mat::<f64>::zeros(n_rows, sketch_dim);
        for (&row, &col, &val) in coo.triplet_iter().map(|(r, c, v)| (r, c, v)) {
            for s in 0..sketch_dim {
                y[(row, s)] += val * xty[(col, s)];
            }
        }
        // Mean correction
        let mut means_xty = vec![0.0f64; sketch_dim];
        for s in 0..sketch_dim {
            for c in 0..n_cols {
                means_xty[s] += col_means[c] * xty[(c, s)];
            }
        }
        for i in 0..n_rows {
            for s in 0..sketch_dim {
                y[(i, s)] -= means_xty[s];
            }
        }
    }

    // 5. QR decomposition of Y → Q (thin QR: Q is n_rows × sketch_dim)
    let qr = y.qr();
    let q = qr.compute_thin_q();

    // 6. B = Q^T @ X_centered → (sketch_dim × n_cols)
    let mut b = Mat::<f64>::zeros(sketch_dim, n_cols);

    // B = Q^T @ X (sparse multiply)
    for (&row, &col, &val) in coo.triplet_iter().map(|(r, c, v)| (r, c, v)) {
        for s in 0..sketch_dim {
            b[(s, col)] += q[(row, s)] * val;
        }
    }

    // Mean correction: B -= (Q^T @ ones) * means^T
    let mut qt_ones = vec![0.0f64; sketch_dim];
    for s in 0..sketch_dim {
        for i in 0..n_rows {
            qt_ones[s] += q[(i, s)];
        }
    }
    for s in 0..sketch_dim {
        for c in 0..n_cols {
            b[(s, c)] -= qt_ones[s] * col_means[c];
        }
    }

    // 7. SVD of small matrix B → thin SVD
    let b_svd = b.thin_svd();
    let u_b = b_svd.u();
    let s_diag = b_svd.s_diagonal();

    // 8. U = Q @ U_B, PCA coords = U * S
    let actual_k = k.min(s_diag.nrows());
    let mut pca_coords: Vec<Vec<f32>> = Vec::with_capacity(n_rows);

    for i in 0..n_rows {
        let mut row = Vec::with_capacity(actual_k);
        for c in 0..actual_k {
            // U[i, c] = sum_s Q[i, s] * U_B[s, c]
            let mut u_ic = 0.0f64;
            for s in 0..sketch_dim {
                u_ic += q[(i, s)] * u_b[(s, c)];
            }
            // PCA coordinate = U[i,c] * S[c]
            row.push((u_ic * s_diag[(c, 0)]) as f32);
        }
        pca_coords.push(row);
    }

    Ok((pca_coords, actual_k))
}

#[polars_expr(output_type_func=svd_output)]
fn sparse_randomized_svd(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 6 {
        return Err(PolarsError::ComputeError(
            "sparse_randomized_svd requires exactly 6 inputs: \
             [groups, genes, counts, n_cells, n_genes, n_comps]".into()
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

    // 1. Reconstruct the CooMatrix natively in Rust
    let mut coo = CooMatrix::new(n_cells, n_genes);

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
                    let r = row as usize;
                    let c = col as usize;
                    if r >= n_cells || c >= n_genes {
                        return Err(PolarsError::ComputeError(
                            format!("COO index out of bounds: ({}, {}) for {}x{}", r, c, n_cells, n_genes).into()
                        ));
                    }
                    coo.push(r, c, count as f64);
                }
            }
        }
    }

    // 2. Perform Halko Randomized SVD
    let n_oversampling = n_comps.max(10) / 4 + 10;
    let n_power_iter = 3;

    let (pca_coords, k) = randomized_svd(&coo, n_comps, n_oversampling, n_power_iter, 42)
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

    #[test]
    fn test_randomized_svd_basic() {
        // Simple 4x3 matrix with known rank-2 structure
        let mut coo = CooMatrix::new(4, 3);
        coo.push(0, 0, 1.0);
        coo.push(0, 1, 2.0);
        coo.push(1, 0, 3.0);
        coo.push(1, 1, 4.0);
        coo.push(2, 0, 5.0);
        coo.push(2, 1, 6.0);
        coo.push(2, 2, 1.0);
        coo.push(3, 0, 7.0);
        coo.push(3, 2, 2.0);

        let (coords, k) = randomized_svd(&coo, 2, 5, 2, 42).unwrap();
        assert_eq!(coords.len(), 4);
        assert_eq!(k, 2);
        // Each row should have 2 components
        for row in &coords {
            assert_eq!(row.len(), 2);
        }
    }

    #[test]
    fn test_randomized_svd_empty() {
        let coo = CooMatrix::new(0, 0);
        let result = randomized_svd(&coo, 2, 5, 2, 42);
        assert!(result.is_err());
    }
}
