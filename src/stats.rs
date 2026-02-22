use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float64)]
fn wilcoxon_rank_sum(inputs: &[Series]) -> PolarsResult<Series> {
    println!("DEBUG: Rust function entered successfully. Array size: {}", inputs.len());
    
    // Create a dummy output of length matching input 0's length
    let len = inputs[0].len();
    let ca = Float64Chunked::full("p_value".into(), 0.0, len);
    
    Ok(ca.into_series())
}
