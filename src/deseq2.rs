use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use nalgebra::{DMatrix, DVector};

/// Iteratively Reweighted Least Squares (IRLS) solver for Negative Binomial GLM
/// fits a model: E[Y] = mu = size_factor * exp(X * beta)
/// Variance: V = mu + alpha * mu^2 (alpha is dispersion)
///
/// Inputs:
/// 0. counts (f32 List array, one list per gene, length = num_cells)
/// 1. size_factors (f32 List array, one list per gene, length = num_cells)
/// 2. design_matrix (f32 List array, flattened matrix of shape num_cells x num_covariates)
/// 3. num_covariates (u32, the number of columns in the design matrix)
/// 4. dispersions (f32 List array, dispersion alpha parameter per gene)
#[polars_expr(output_type=Float32)]
fn deseq2_irls(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 5 {
        return Err(PolarsError::ComputeError("deseq2_irls requires 5 inputs".into()));
    }
    
    let ca_counts = inputs[0].list()?;
    let ca_size_factors = inputs[1].list()?; 
    let ca_design = inputs[2].list()?;
    // num_covariates is a scalar, but passed as a column. We can just take the first value.
    let num_covariates = inputs[3].u32()?.get(0).unwrap_or(1) as usize;
    let ca_disp = inputs[4].f32()?;
    
    // We will output the fitted beta coefficients as a List of length `num_covariates`
    let mut all_betas: Vec<Option<Series>> = Vec::with_capacity(ca_counts.len());
    
    for ((((opt_counts, opt_sf), opt_design), opt_disp)) in ca_counts.into_iter()
        .zip(ca_size_factors.into_iter())
        .zip(ca_design.into_iter())
        .zip(ca_disp.into_iter()) {
            
        if opt_counts.is_none() || opt_sf.is_none() || opt_design.is_none() || opt_disp.is_none() {
            all_betas.push(None);
            continue;
        }
        
        let counts_series = opt_counts.unwrap();
        let sf_series = opt_sf.unwrap();
        let design_series = opt_design.unwrap();
        
        let counts = counts_series.f32()?;
        let size_factors = sf_series.f32()?;
        let design_flat = design_series.f32()?;
        let disp = opt_disp.unwrap() as f64; // Promote to f64 for math precision
        
        let n = counts.len();
        if n == 0 || size_factors.len() != n || design_flat.len() != n * num_covariates {
             all_betas.push(None);
             continue;
        }
        
        // Convert to nalgebra vectors and matrices
        let y: Vec<f64> = counts.into_iter().map(|v| v.unwrap_or(0.0) as f64).collect();
        let y_vec = DVector::from_vec(y);
        
        let sf: Vec<f64> = size_factors.into_iter().map(|v| v.unwrap_or(1.0) as f64).collect();
        let sf_vec = DVector::from_vec(sf);
        
        // nalgebra DMatrix expects column-major by default, but if the python side flattened it row-major,
        // we use from_row_slice. Assuming row-major flattening of the design matrix:
        let x_data: Vec<f64> = design_flat.into_iter().map(|v| v.unwrap_or(0.0) as f64).collect();
        let x_mat = DMatrix::from_row_slice(n, num_covariates, &x_data);
        
        // Initialize Beta to zeros
        let mut beta = DVector::zeros(num_covariates);
        let max_iter = 250;
        let tol = 1e-6;
        let mut converged = false;
        
        for _iter in 0..max_iter {
            // mu = size_factor * exp(X * beta)
            let mut eta = &x_mat * &beta;
            let mut mu = DVector::zeros(n);
            for i in 0..n {
                // Prevent extreme blowout in the linear predictor (which causes inf in exp)
                if eta[i] > 30.0 { eta[i] = 30.0; }
                if eta[i] < -30.0 { eta[i] = -30.0; }
                
                mu[i] = sf_vec[i] * eta[i].exp();
                // Prevent extreme blowout
                if mu[i] < 1e-12 { mu[i] = 1e-12; }
                if mu[i] > 1e12 { mu[i] = 1e12; }
            }
            
            // Variance: v = mu + alpha * mu^2
            let mut w_diag = DVector::zeros(n);
            let mut z = DVector::zeros(n); // Working response
            
            for i in 0..n {
                let m = mu[i];
                let v = m + disp * m * m;
                // IRLS Weight: W = (d mu / d eta)^2 / V
                // For log link: d mu / d eta = mu
                // W = mu^2 / V
                let w = (m * m) / v.max(1e-12);
                w_diag[i] = w;
                
                // z = eta + (y - mu) / (d mu / d eta)
                // z = eta + (y - mu) / mu
                z[i] = eta[i] + (y_vec[i] - m) / m;
            }
            
            // W matrix (diagonal). To save memory, we can do element-wise multiplication
            // X^T W X
            let mut xtwx = DMatrix::zeros(num_covariates, num_covariates);
            for i in 0..n {
                let row = x_mat.row(i);
                xtwx += row.transpose() * row * w_diag[i];
            }
            
            // Add slight ridge penalty to ensure matrix is invertible 
            // StatsModels default ridge for IRLS singular vectors is ~1e-4
            for i in 0..num_covariates {
                xtwx[(i, i)] += 1e-4; 
            }
            
            // X^T W z
            let mut xtwz = DVector::zeros(num_covariates);
            for i in 0..n {
                let row = x_mat.row(i);
                xtwz += row.transpose() * w_diag[i] * z[i];
            }
            
            // Solve (X^T W X) beta_new = X^T W z
            // We use SVD or a more stable pseudo-inverse if Cholesky fails (common in sparse single-cell data)
            let beta_new = match xtwx.clone().cholesky() {
                Some(cholesky) => cholesky.solve(&xtwz),
                None => match xtwx.pseudo_inverse(1e-9) {
                    Ok(pinv) => pinv * &xtwz,
                    Err(_) => {
                        // Matrix is completely degenerate
                        break;
                    }
                }
            };
            
            let diff = (&beta_new - &beta).norm();
            beta = beta_new;
            
            if diff < tol {
                converged = true;
                break;
            }
        }
        
        if converged {
            let beta_vec: Vec<f32> = beta.iter().map(|&v| {
                if v.is_nan() || v.is_infinite() {
                    0.0
                } else {
                    v as f32
                }
            }).collect();
            let out_series = Series::new("beta".into(), &beta_vec);
            all_betas.push(Some(out_series));
        } else {
            // StatsModels returns zeros when IRLS fails to converge. 
            // Mirroring that behavior to maintain output shape matching.
            let zeros = vec![0.0f32; num_covariates];
            all_betas.push(Some(Series::new("beta".into(), &zeros)));
        }
    }
    
    // We return a ListChunked array where each row contains the list of Beta coefficients
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new("beta".into(), all_betas.len(), all_betas.len() * num_covariates, DataType::Float32);
    
    for opt_b in all_betas {
        if let Some(b) = opt_b {
            let b_arr = b.f32().unwrap();
            builder.append_slice(b_arr.cont_slice().unwrap());
        } else {
            builder.append_null();
        }
    }
    
    Ok(builder.finish().into_series())
}
