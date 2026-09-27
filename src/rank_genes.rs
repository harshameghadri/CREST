use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Output type: List(List(Float32)) — each inner list is [gene_id, t_stat, p_value, adj_p_value, log2_fc]
fn rank_genes_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "rank_genes_result".into(),
        DataType::List(Box::new(DataType::List(Box::new(DataType::Float32)))),
    ))
}

/// rank_genes_groups: Welch's t-test for differential expression between two groups
/// (target group vs all other labelled cells), matching scanpy's `method="t-test"`.
///
/// Inputs:
/// 0: count/expression column (Float32) - log-normalized values
/// 1: cell_id column (UInt32)
/// 2: gene_id column (UInt32)
/// 3: group_labels column (UInt32) - group label for each row's cell
/// 4: target_group (UInt32) - which group label is the "test" group
///
/// Returns List(List(Float32)): each inner list = [gene_id, t_stat, p_value, adj_p_value, log2_fc]
/// Sorted by adjusted p-value (most significant first).
#[polars_expr(output_type_func=rank_genes_output)]
fn rank_genes_groups(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 5 {
        return Err(PolarsError::ComputeError(
            "rank_genes_groups requires 5 inputs: [values, cell_ids, gene_ids, group_labels, target_group]".into(),
        ));
    }

    let values = inputs[0].f32()?;
    let cell_ids = inputs[1].u32()?;
    let gene_ids = inputs[2].u32()?;
    let group_labels = inputs[3].u32()?;
    let target_group = inputs[4].u32()?.get(0).unwrap_or(1);

    let n = values.len();
    if n == 0 {
        let empty_inner = Series::new("".into(), Vec::<f32>::new());
        let builder_out = Series::new("rank_genes_result".into(), vec![empty_inner]);
        return Ok(builder_out);
    }

    // Build cell->group lookup
    let max_cell = cell_ids.into_no_null_iter().max().unwrap_or(0) as usize;
    let max_gene = gene_ids.into_no_null_iter().max().unwrap_or(0) as usize;

    // Guard against OOM from corrupted IDs
    const MAX_DIM: usize = 10_000_000;
    if max_cell >= MAX_DIM {
        return Err(PolarsError::ComputeError(
            format!("max cell_id too large: {} (max {}). Check for corrupted IDs.", max_cell, MAX_DIM).into()
        ));
    }
    if max_gene >= MAX_DIM {
        return Err(PolarsError::ComputeError(
            format!("max gene_id too large: {} (max {}). Check for corrupted IDs.", max_gene, MAX_DIM).into()
        ));
    }

    let mut cell_group = vec![u32::MAX; max_cell + 1];
    for (opt_cell, opt_group) in cell_ids.into_iter().zip(group_labels.into_iter()) {
        if let (Some(cell), Some(group)) = (opt_cell, opt_group) {
            cell_group[cell as usize] = group;
        }
    }

    // Group sizes in CELLS. Sparse input omits zeros, so means and variances
    // must divide by the number of cells in each group, not by the number of
    // non-zero entries observed for a gene.
    let mut n_tgt_cells = 0u64;
    let mut n_ref_cells = 0u64;
    for &grp in &cell_group {
        if grp == target_group {
            n_tgt_cells += 1;
        } else if grp != u32::MAX {
            n_ref_cells += 1;
        }
    }

    // Per-gene, per-group accumulators (over non-zero entries; zeros add nothing)
    let n_genes = max_gene + 1;
    let mut sum_tgt = vec![0.0f64; n_genes];
    let mut sum_sq_tgt = vec![0.0f64; n_genes];
    let mut sum_ref = vec![0.0f64; n_genes];
    let mut sum_sq_ref = vec![0.0f64; n_genes];

    for ((opt_val, opt_cell), opt_gene) in values.into_iter().zip(cell_ids.into_iter()).zip(gene_ids.into_iter()) {
        if let (Some(v), Some(cell), Some(gene)) = (opt_val, opt_cell, opt_gene) {
            let g_idx = gene as usize;
            let vf = v as f64;
            let cell_grp = cell_group[cell as usize];
            if cell_grp == target_group {
                sum_tgt[g_idx] += vf;
                sum_sq_tgt[g_idx] += vf * vf;
            } else if cell_grp != u32::MAX {
                sum_ref[g_idx] += vf;
                sum_sq_ref[g_idx] += vf * vf;
            }
        }
    }

    // Compute per-gene Welch t-statistics and p-values
    let mut results: Vec<GeneResult> = Vec::with_capacity(n_genes);
    let n1 = n_tgt_cells as f64;
    let n2 = n_ref_cells as f64;

    for g in 0..n_genes {
        if n1 < 2.0 || n2 < 2.0 {
            results.push(GeneResult {
                gene_id: g as u32,
                t_stat: 0.0,
                p_value: 1.0,
                adj_p_value: 1.0,
                log2_fc: 0.0,
            });
            continue;
        }

        let mean1 = sum_tgt[g] / n1;
        let mean2 = sum_ref[g] / n2;
        let var1 = ((sum_sq_tgt[g] - n1 * mean1 * mean1) / (n1 - 1.0)).max(0.0);
        let var2 = ((sum_sq_ref[g] - n2 * mean2 * mean2) / (n2 - 1.0)).max(0.0);

        let s1n = var1 / n1;
        let s2n = var2 / n2;
        let se = (s1n + s2n).sqrt();

        let (t_stat, p_value) = if se > 1e-12 {
            let t = (mean1 - mean2) / se;
            // Welch-Satterthwaite degrees of freedom
            let df = (s1n + s2n).powi(2) / (s1n * s1n / (n1 - 1.0) + s2n * s2n / (n2 - 1.0));
            (t, two_sided_t_p(t, df))
        } else {
            (0.0, 1.0)
        };

        // Fold change on the linear scale of log1p data, as scanpy does.
        let log2_fc = ((mean1.exp_m1() + 1e-9) / (mean2.exp_m1() + 1e-9)).log2();

        results.push(GeneResult {
            gene_id: g as u32,
            t_stat,
            p_value,
            adj_p_value: 1.0,
            log2_fc,
        });
    }

    // Benjamini-Hochberg FDR correction
    benjamini_hochberg(&mut results);

    // Sort by adjusted p-value (most significant first)
    results.sort_by(|a, b| a.adj_p_value.partial_cmp(&b.adj_p_value).unwrap_or(std::cmp::Ordering::Equal));

    // Build output as List(List(Float32))
    // Each inner list: [gene_id, t_stat, p_value, adj_p_value, log2_fc]
    let inner_lists: Vec<Series> = results
        .iter()
        .map(|r| {
            Series::new(
                "".into(),
                vec![r.gene_id as f32, r.t_stat as f32, r.p_value as f32, r.adj_p_value as f32, r.log2_fc as f32],
            )
        })
        .collect();

    let outer = Series::new("rank_genes_result".into(), inner_lists);
    // Wrap in a length-1 list for aggregate plugin output
    let wrapper = Series::new("rank_genes_result".into(), vec![outer]);
    Ok(wrapper)
}

struct GeneResult {
    gene_id: u32,
    t_stat: f64,
    p_value: f64,
    adj_p_value: f64,
    log2_fc: f64,
}

/// Benjamini-Hochberg FDR correction
fn benjamini_hochberg(results: &mut [GeneResult]) {
    let n = results.len();
    if n == 0 {
        return;
    }

    // Sort by p-value
    results.sort_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap_or(std::cmp::Ordering::Equal));

    let nf = n as f64;
    let mut cummin = 1.0f64;

    // Walk backwards to compute adjusted p-values
    for i in (0..n).rev() {
        let rank = (i + 1) as f64;
        let adjusted = (results[i].p_value * nf / rank).min(1.0);
        cummin = cummin.min(adjusted);
        results[i].adj_p_value = cummin;
    }
}

/// Exact two-sided p-value of Student's t distribution.
fn two_sided_t_p(t: f64, df: f64) -> f64 {
    if !t.is_finite() || !df.is_finite() || df <= 0.0 {
        return 1.0;
    }
    match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => (2.0 * dist.sf(t.abs())).min(1.0),
        Err(_) => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benjamini_hochberg() {
        let mut results = vec![
            GeneResult { gene_id: 0, t_stat: 3.0, p_value: 0.01, adj_p_value: 0.0, log2_fc: 1.0 },
            GeneResult { gene_id: 1, t_stat: 2.0, p_value: 0.05, adj_p_value: 0.0, log2_fc: 0.5 },
            GeneResult { gene_id: 2, t_stat: 1.0, p_value: 0.10, adj_p_value: 0.0, log2_fc: 0.2 },
            GeneResult { gene_id: 3, t_stat: 0.5, p_value: 0.50, adj_p_value: 0.0, log2_fc: 0.1 },
        ];

        benjamini_hochberg(&mut results);

        assert!((results[0].adj_p_value - 0.04).abs() < 1e-10);
        assert!((results[1].adj_p_value - 0.10).abs() < 1e-10);
        assert!(results[2].adj_p_value > 0.13 && results[2].adj_p_value < 0.14);
        assert!((results[3].adj_p_value - 0.50).abs() < 1e-10);
    }

    #[test]
    fn test_two_sided_t_p() {
        // Reference values from scipy.stats.t.sf
        assert!((two_sided_t_p(0.0, 10.0) - 1.0).abs() < 1e-12);
        assert!((two_sided_t_p(2.228138851986273, 10.0) - 0.05).abs() < 1e-9);
        assert!((two_sided_t_p(-1.959963984540054, 1e9) - 0.05).abs() < 1e-6);
        assert!(two_sided_t_p(40.0, 1000.0) > 0.0);
    }
}
