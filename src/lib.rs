use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use polars::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

// Custom structure to hold values for sorting and ranking
#[derive(Clone, Copy, Debug)]
struct RankItem {
    value: f32,
    group: u8,
}
use polars::prelude::*;

mod deseq2;
mod svd;

#[polars_expr(output_type=Float32)]
fn log1p(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.f32()?;
    
    // Apply ln(1+x) vectorized over the ChunkedArray
    // f32 is critical for single cell data to preserve memory
    let ca_log = ca.apply_values(|v| v.ln_1p());
    
    Ok(ca_log.into_series())
}


#[polars_expr(output_type=Float64)]
fn wilcoxon_rank_sum(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError("wilcoxon requires 2 input series".into()));
    }

    let s1 = &inputs[0];
    let s2 = &inputs[1];
    
    let ca1 = s1.list()?;
    let ca2 = s2.list()?;
    
    let mut pvalues: Vec<Option<f64>> = Vec::with_capacity(ca1.len());
    
    let normal_dist = Normal::new(0.0, 1.0)
        .map_err(|e| PolarsError::ComputeError(format!("Normal dist error: {}", e).into()))?;

    for (opt_s1, opt_s2) in ca1.into_iter().zip(ca2.into_iter()) {
        if opt_s1.is_none() || opt_s2.is_none() {
            pvalues.push(None);
            continue;
        }
        
        let series1 = opt_s1.unwrap();
        let series2 = opt_s2.unwrap();
        
        let arr1 = series1.f32()?;
        let arr2 = series2.f32()?;
        
        let n1 = arr1.len() as f64;
        let n2 = arr2.len() as f64;
        
        if n1 == 0.0 || n2 == 0.0 {
            pvalues.push(Some(1.0)); // No data for one group
            continue;
        }

        let mut combined: Vec<RankItem> = Vec::with_capacity((n1 + n2) as usize);
        
        for v in arr1.into_iter().flatten() {
            combined.push(RankItem { value: v, group: 1 });
        }
        for v in arr2.into_iter().flatten() {
            combined.push(RankItem { value: v, group: 2 });
        }
        
        combined.sort_by(|a, b| a.value.total_cmp(&b.value));
        
        let n = combined.len();
        let mut ranks = vec![0.0; n];
        let mut r1_sum = 0.0;
        let mut tie_correction = 0.0;
        
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && combined[j].value == combined[i].value { j += 1; }
            let t = (j - i) as f64;
            let rank_sum = (t * ((i as f64 + 1.0) + (j as f64))) / 2.0;
            let avg_rank = rank_sum / t;
            
            for k in i..j {
                ranks[k] = avg_rank;
                if combined[k].group == 1 {
                    r1_sum += avg_rank;
                }
            }
            if t > 1.0 { tie_correction += (t * t * t) - t; }
            i = j;
        }
        
        let n_obs = n as f64;
        let u1 = r1_sum - (n1 * (n1 + 1.0)) / 2.0;
        let expected_u = (n1 * n2) / 2.0;
        let var_u = (n1 * n2 / 12.0) * ((n_obs + 1.0) - tie_correction / (n_obs * (n_obs - 1.0)));
        
        if var_u == 0.0 {
            pvalues.push(Some(1.0));
            continue;
        }
        
        let std_u = var_u.sqrt().max(1e-12);
        
        let mut z = 0.0;
        let limit = (u1 - expected_u).abs();
        if limit > 0.0 {
            z = (limit - 0.5) / std_u;
        }
        
        let p_val: f64 = 2.0 * (1.0 - normal_dist.cdf(z));
        pvalues.push(Some(p_val));
    }
    
    let ca_pvalues: Float64Chunked = pvalues.into_iter().collect();
    Ok(ca_pvalues.into_series())
}
