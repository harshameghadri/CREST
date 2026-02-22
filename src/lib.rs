use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use polars::prelude::*;

#[polars_expr(output_type=Float32)]
fn log1p(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.f32()?;
    
    // Apply ln(1+x) vectorized over the ChunkedArray
    // f32 is critical for single cell data to preserve memory
    let ca_log = ca.apply_values(|v| v.ln_1p());
    
    Ok(ca_log.into_series())
}

#[pymodule]
fn biopolars(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
