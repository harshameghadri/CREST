use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// normalize_cpm: Counts Per Million (CP10k) normalization.
///
/// Inputs:
/// 0: count column (Float32) - raw expression counts
/// 1: cell_id column (UInt32) - cell identifiers
///
/// For each cell, computes: count / sum(counts_in_cell) * target_sum
/// Target sum defaults to 10,000 (CP10k), matching scanpy's normalize_total.
///
/// This is an elementwise operation that uses the cell_id to group.
#[polars_expr(output_type=Float32)]
fn normalize_cpm(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "normalize_cpm requires 2 inputs: [counts, cell_ids]".into(),
        ));
    }

    let counts = inputs[0].f32()?;
    let cell_ids = inputs[1].u32()?;
    let n = counts.len();

    if n == 0 {
        return Ok(Float32Chunked::full("normalized".into(), 0.0, 0).into_series());
    }

    // Find max cell_id to size the accumulator
    let max_cell = cell_ids
        .into_no_null_iter()
        .max()
        .unwrap_or(0) as usize;

    // Pass 1: accumulate per-cell sums
    let mut cell_sums = vec![0.0f64; max_cell + 1];
    for (opt_count, opt_cell) in counts.into_iter().zip(cell_ids.into_iter()) {
        if let (Some(c), Some(cell)) = (opt_count, opt_cell) {
            cell_sums[cell as usize] += c as f64;
        }
    }

    // Pass 2: normalize each count by its cell's total
    let target_sum = 10_000.0f64;
    let normalized: Float32Chunked = counts
        .into_iter()
        .zip(cell_ids.into_iter())
        .map(|(opt_count, opt_cell)| {
            match (opt_count, opt_cell) {
                (Some(c), Some(cell)) => {
                    let sum = cell_sums[cell as usize];
                    if sum > 0.0 {
                        Some((c as f64 / sum * target_sum) as f32)
                    } else {
                        Some(0.0f32)
                    }
                }
                _ => None,
            }
        })
        .collect();

    Ok(normalized.into_series())
}

/// calculate_qc_metrics: Compute per-cell and per-gene quality control metrics.
///
/// Inputs:
/// 0: count column (Float32)
/// 1: cell_id column (UInt32)
///
/// Returns a struct with: total_counts, n_genes_by_counts (per cell)
/// This is used for QC filtering before normalization.
///
/// Output: List of [total_counts_f32, n_genes_u32] per row, aligned to input.
#[polars_expr(output_type=Float32)]
fn qc_total_counts(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "qc_total_counts requires 2 inputs: [counts, cell_ids]".into(),
        ));
    }

    let counts = inputs[0].f32()?;
    let cell_ids = inputs[1].u32()?;
    let n = counts.len();

    if n == 0 {
        return Ok(Float32Chunked::full("total_counts".into(), 0.0, 0).into_series());
    }

    let max_cell = cell_ids.into_no_null_iter().max().unwrap_or(0) as usize;

    // Accumulate per-cell total counts
    let mut cell_sums = vec![0.0f32; max_cell + 1];
    for (opt_count, opt_cell) in counts.into_iter().zip(cell_ids.into_iter()) {
        if let (Some(c), Some(cell)) = (opt_count, opt_cell) {
            cell_sums[cell as usize] += c;
        }
    }

    // Map back: each row gets its cell's total count
    let result: Float32Chunked = cell_ids
        .into_iter()
        .map(|opt_cell| opt_cell.map(|cell| cell_sums[cell as usize]))
        .collect();

    Ok(result.into_series())
}

/// qc_n_genes: Count number of expressed genes per cell.
///
/// Inputs:
/// 0: count column (Float32)
/// 1: cell_id column (UInt32)
///
/// Returns the number of genes with count > 0 for each cell.
#[polars_expr(output_type=UInt32)]
fn qc_n_genes(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "qc_n_genes requires 2 inputs: [counts, cell_ids]".into(),
        ));
    }

    let counts = inputs[0].f32()?;
    let cell_ids = inputs[1].u32()?;
    let n = counts.len();

    if n == 0 {
        return Ok(UInt32Chunked::full("n_genes".into(), 0, 0).into_series());
    }

    let max_cell = cell_ids.into_no_null_iter().max().unwrap_or(0) as usize;

    let mut n_genes = vec![0u32; max_cell + 1];
    for (opt_count, opt_cell) in counts.into_iter().zip(cell_ids.into_iter()) {
        if let (Some(c), Some(cell)) = (opt_count, opt_cell) {
            if c > 0.0 {
                n_genes[cell as usize] += 1;
            }
        }
    }

    let result: UInt32Chunked = cell_ids
        .into_iter()
        .map(|opt_cell| opt_cell.map(|cell| n_genes[cell as usize]))
        .collect();

    Ok(result.into_series())
}

/// filter_cells: Mark cells that pass min_genes / min_counts thresholds.
///
/// Inputs:
/// 0: count column (Float32)
/// 1: cell_id column (UInt32)
/// 2: min_genes (UInt32) - minimum number of expressed genes
/// 3: min_counts (Float32) - minimum total counts
///
/// Returns Boolean mask (true = cell passes filter).
#[polars_expr(output_type=Boolean)]
fn filter_cells(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 4 {
        return Err(PolarsError::ComputeError(
            "filter_cells requires 4 inputs: [counts, cell_ids, min_genes, min_counts]".into(),
        ));
    }

    let counts = inputs[0].f32()?;
    let cell_ids = inputs[1].u32()?;
    let min_genes = inputs[2].u32()?.get(0).unwrap_or(0);
    let min_counts = inputs[3].f32()?.get(0).unwrap_or(0.0);
    let n = counts.len();

    if n == 0 {
        return Ok(BooleanChunked::full("filter".into(), true, 0).into_series());
    }

    let max_cell = cell_ids.into_no_null_iter().max().unwrap_or(0) as usize;

    let mut cell_sums = vec![0.0f32; max_cell + 1];
    let mut cell_n_genes = vec![0u32; max_cell + 1];

    for (opt_count, opt_cell) in counts.into_iter().zip(cell_ids.into_iter()) {
        if let (Some(c), Some(cell)) = (opt_count, opt_cell) {
            cell_sums[cell as usize] += c;
            if c > 0.0 {
                cell_n_genes[cell as usize] += 1;
            }
        }
    }

    let result: BooleanChunked = cell_ids
        .into_iter()
        .map(|opt_cell| {
            opt_cell.map(|cell| {
                let idx = cell as usize;
                cell_n_genes[idx] >= min_genes && cell_sums[idx] >= min_counts
            })
        })
        .collect();

    Ok(result.into_series())
}

/// scale: Zero-center and optionally scale to unit variance.
///
/// Inputs:
/// 0: value column (Float32) - expression values (log-normalized)
/// 1: gene_id column (UInt32)
/// 2: n_obs (UInt32) - total number of observations (cells) for proper mean/var
/// 3: max_value (Float32) - clip scaled values to this maximum (default 10.0)
///
/// Computes per-gene: (x - mean) / std, then clips to [-max_value, max_value].
/// Accounts for structural zeros in sparse data.
#[polars_expr(output_type=Float32)]
fn scale(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 4 {
        return Err(PolarsError::ComputeError(
            "scale requires 4 inputs: [values, gene_ids, n_obs, max_value]".into(),
        ));
    }

    let values = inputs[0].f32()?;
    let gene_ids = inputs[1].u32()?;
    let n_obs = inputs[2].u32()?.get(0).unwrap_or(1) as f64;
    let max_value = inputs[3].f32()?.get(0).unwrap_or(10.0);
    let n = values.len();

    if n == 0 || n_obs < 2.0 {
        return Ok(Float32Chunked::full("scaled".into(), 0.0, n).into_series());
    }

    let max_gene = gene_ids.into_no_null_iter().max().unwrap_or(0) as usize;

    // Pass 1: accumulate per-gene sum and sum-of-squares (for observed nonzeros)
    let mut gene_sum = vec![0.0f64; max_gene + 1];
    let mut gene_sum_sq = vec![0.0f64; max_gene + 1];

    for (opt_val, opt_gene) in values.into_iter().zip(gene_ids.into_iter()) {
        if let (Some(v), Some(g)) = (opt_val, opt_gene) {
            let idx = g as usize;
            let vf = v as f64;
            gene_sum[idx] += vf;
            gene_sum_sq[idx] += vf * vf;
        }
    }

    // Compute mean and std accounting for structural zeros
    // mean = sum / n_obs  (zeros contribute 0 to sum)
    // var = (sum_sq - 2*mean*sum + n_obs*mean^2) / (n_obs - 1)
    let mut gene_mean = vec![0.0f64; max_gene + 1];
    let mut gene_std = vec![1.0f64; max_gene + 1];

    for g in 0..=max_gene {
        let mean = gene_sum[g] / n_obs;
        let var = (gene_sum_sq[g] - 2.0 * mean * gene_sum[g] + n_obs * mean * mean) / (n_obs - 1.0);
        gene_mean[g] = mean;
        gene_std[g] = if var > 1e-12 { var.sqrt() } else { 1.0 };
    }

    // Pass 2: scale each value
    let result: Float32Chunked = values
        .into_iter()
        .zip(gene_ids.into_iter())
        .map(|(opt_val, opt_gene)| {
            match (opt_val, opt_gene) {
                (Some(v), Some(g)) => {
                    let idx = g as usize;
                    let scaled = ((v as f64 - gene_mean[idx]) / gene_std[idx]) as f32;
                    Some(scaled.clamp(-max_value, max_value))
                }
                _ => None,
            }
        })
        .collect();

    Ok(result.into_series())
}

/// score_genes: Score a set of genes (gene signature) per cell.
///
/// Inputs:
/// 0: count/expression column (Float32)
/// 1: cell_id column (UInt32)
/// 2: gene_id column (UInt32)
/// 3: gene_set (List(UInt32)) - list of gene IDs in the signature
///
/// For each cell, computes the mean expression of genes in the gene set,
/// minus the mean expression of a reference set (all other genes).
/// This mirrors scanpy's `tl.score_genes`.
#[polars_expr(output_type=Float32)]
fn score_genes(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 4 {
        return Err(PolarsError::ComputeError(
            "score_genes requires 4 inputs: [values, cell_ids, gene_ids, gene_set_list]".into(),
        ));
    }

    let values = inputs[0].f32()?;
    let cell_ids = inputs[1].u32()?;
    let gene_ids = inputs[2].u32()?;
    let gene_set_series = &inputs[3].list()?;
    let n = values.len();

    if n == 0 {
        return Ok(Float32Chunked::full("score".into(), 0.0, 0).into_series());
    }

    // Extract gene set from the first list element
    let gene_set_inner = match gene_set_series.get_as_series(0) {
        Some(s) => s,
        None => return Err(PolarsError::ComputeError("Empty gene set".into())),
    };
    let gene_set_ca = gene_set_inner.u32()?;
    let mut gene_set = std::collections::HashSet::new();
    for opt_g in gene_set_ca.into_iter() {
        if let Some(g) = opt_g {
            gene_set.insert(g);
        }
    }

    if gene_set.is_empty() {
        return Err(PolarsError::ComputeError("Gene set is empty".into()));
    }

    let max_cell = cell_ids.into_no_null_iter().max().unwrap_or(0) as usize;

    // Accumulate per-cell sums for gene_set and background
    let mut set_sum = vec![0.0f64; max_cell + 1];
    let mut set_count = vec![0u32; max_cell + 1];
    let mut bg_sum = vec![0.0f64; max_cell + 1];
    let mut bg_count = vec![0u32; max_cell + 1];

    for ((opt_val, opt_cell), opt_gene) in values.into_iter().zip(cell_ids.into_iter()).zip(gene_ids.into_iter()) {
        if let (Some(v), Some(cell), Some(gene)) = (opt_val, opt_cell, opt_gene) {
            let idx = cell as usize;
            if gene_set.contains(&gene) {
                set_sum[idx] += v as f64;
                set_count[idx] += 1;
            } else {
                bg_sum[idx] += v as f64;
                bg_count[idx] += 1;
            }
        }
    }

    // Score = mean(gene_set) - mean(background)
    let result: Float32Chunked = cell_ids
        .into_iter()
        .map(|opt_cell| {
            opt_cell.map(|cell| {
                let idx = cell as usize;
                let set_mean = if set_count[idx] > 0 {
                    set_sum[idx] / set_count[idx] as f64
                } else {
                    0.0
                };
                let bg_mean = if bg_count[idx] > 0 {
                    bg_sum[idx] / bg_count[idx] as f64
                } else {
                    0.0
                };
                (set_mean - bg_mean) as f32
            })
        })
        .collect();

    Ok(result.into_series())
}

#[cfg(test)]
mod tests {
    /// Test normalization logic: given counts and cell sums, verify CP10k math
    #[test]
    fn test_cpm_math() {
        // Cell with counts [2, 3, 5], sum = 10
        // CP10k: 2/10*10000 = 2000, 3/10*10000 = 3000, 5/10*10000 = 5000
        let counts = [2.0f32, 3.0, 5.0];
        let cell_sum = 10.0f64;
        let target = 10_000.0f64;
        let normalized: Vec<f32> = counts.iter().map(|&c| (c as f64 / cell_sum * target) as f32).collect();
        assert!((normalized[0] - 2000.0).abs() < 1.0);
        assert!((normalized[1] - 3000.0).abs() < 1.0);
        assert!((normalized[2] - 5000.0).abs() < 1.0);
    }

    /// Test scale math: z-score with sparse-aware variance
    #[test]
    fn test_scale_math() {
        // Values [1, 2, 3], n_obs = 3
        // mean = (1+2+3)/3 = 2
        // var = (sum_sq - 2*mean*sum + n*mean^2) / (n-1)
        //     = (14 - 2*2*6 + 3*4) / 2 = (14 - 24 + 12) / 2 = 1.0
        // std = 1.0
        // scaled = [-1, 0, 1]
        let values = [1.0f64, 2.0, 3.0];
        let n_obs = 3.0f64;
        let sum: f64 = values.iter().sum();
        let sum_sq: f64 = values.iter().map(|v| v * v).sum();
        let mean = sum / n_obs;
        let var = (sum_sq - 2.0 * mean * sum + n_obs * mean * mean) / (n_obs - 1.0);
        let std = var.sqrt();
        let scaled: Vec<f64> = values.iter().map(|v| (v - mean) / std).collect();
        assert!((scaled[0] - (-1.0)).abs() < 1e-10);
        assert!(scaled[1].abs() < 1e-10);
        assert!((scaled[2] - 1.0).abs() < 1e-10);
    }

    /// Test scale clipping
    #[test]
    fn test_scale_clipping() {
        let scaled = 15.0f32;
        let max_value = 5.0f32;
        let clipped = scaled.clamp(-max_value, max_value);
        assert_eq!(clipped, 5.0);
    }

    /// Test filter logic: min_genes threshold
    #[test]
    fn test_filter_logic() {
        // Cell 0: 3 expressed genes, Cell 1: 1 expressed gene
        let cell_n_genes = [3u32, 1u32];
        let cell_sums = [10.0f32, 1.0f32];
        let min_genes = 2u32;
        let min_counts = 0.0f32;

        assert!(cell_n_genes[0] >= min_genes && cell_sums[0] >= min_counts);
        assert!(!(cell_n_genes[1] >= min_genes && cell_sums[1] >= min_counts));
    }
}
