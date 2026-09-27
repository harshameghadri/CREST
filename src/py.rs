//! Native Python API (numpy in, numpy out; the GIL is released during compute).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::kernels::{self, Transform};
use crate::knn::{knn, KnnGraph, Points};
use numpy::{PyReadwriteArrayDyn, PyReadonlyArrayDyn};

fn sl<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("{} must be C-contiguous", name)))
}
fn sl_mut<'a, T: numpy::Element>(a: &'a mut PyReadwriteArrayDyn<'_, T>, name: &str) -> PyResult<&'a mut [T]> {
    a.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("{} must be C-contiguous", name)))
}
fn check_len(names: &str, lens: &[usize]) -> PyResult<()> {
    if lens.windows(2).any(|w| w[0] != w[1]) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("{} must have equal length", names)));
    }
    Ok(())
}
fn check_rows(rows: &[u32], n: usize, what: &str) -> PyResult<()> {
    if rows.windows(2).any(|w| w[0] > w[1]) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("{} must be sorted (non-decreasing)", what)));
    }
    if rows.last().map_or(false, |&r| r as usize >= n) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("{} index out of range", what)));
    }
    Ok(())
}
fn check_ids(ids: &[u32], n: usize, what: &str) -> PyResult<()> {
    if ids.iter().any(|&g| g as usize >= n) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("{} index out of range", what)));
    }
    Ok(())
}
use crate::umap::graph::{fuzzy_simplicial_set, Edge, UmapGraph};

fn contiguous<'a, T: numpy::Element>(a: &'a numpy::PyReadonlyArrayDyn<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("{} must be C-contiguous", name)))
}

fn undirected(n: usize, rows: &[u32], cols: &[u32], w: &[f32]) -> PyResult<Vec<(usize, usize, f32)>> {
    if rows.len() != cols.len() || rows.len() != w.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("rows, cols and weights must have equal length"));
    }
    let mut out = Vec::with_capacity(rows.len());
    for ((&r, &c), &x) in rows.iter().zip(cols).zip(w) {
        let (r, c) = (r as usize, c as usize);
        if r >= n || c >= n {
            return Err(pyo3::exceptions::PyValueError::new_err("edge index out of range"));
        }
        if r != c {
            out.push((r.min(c), r.max(c), x));
        }
    }
    Ok(out)
}

/// k nearest neighbours of each row of a float32 (n, d) array, self excluded.
/// Returns (indices uint32 (n, k), distances float32 (n, k)).
#[pyfunction]
#[pyo3(signature = (data, k, exact=None, seed=0))]
fn knn_graph<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    k: usize,
    exact: Option<bool>,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    let slice = data.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("data must be C-contiguous"))?;
    let g = py.allow_threads(|| knn(Points::new(slice, n, d), k, exact, seed));
    let kk = g.k;
    let idx = PyArray1::from_vec_bound(py, g.indices).reshape([n, kk])?;
    let dist = PyArray1::from_vec_bound(py, g.distances).reshape([n, kk])?;
    Ok((idx, dist))
}

/// UMAP fuzzy simplicial set (scanpy `connectivities`) from a kNN graph whose rows
/// hold `n_neighbors - 1` neighbours. Returns undirected edges (rows < cols, weights).
#[pyfunction]
fn connectivities<'py>(
    py: Python<'py>,
    indices: numpy::PyReadonlyArrayDyn<'py, u32>,
    distances: numpy::PyReadonlyArrayDyn<'py, f32>,
    n_neighbors: usize,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
    let shape = indices.shape().to_vec();
    if shape.len() != 2 || distances.shape() != shape.as_slice() {
        return Err(pyo3::exceptions::PyValueError::new_err("indices and distances must be (n, k) arrays of equal shape"));
    }
    let g = KnnGraph {
        n: shape[0],
        k: shape[1],
        indices: contiguous(&indices, "indices")?.to_vec(),
        distances: contiguous(&distances, "distances")?.to_vec(),
    };
    let ug = py.allow_threads(|| fuzzy_simplicial_set(&g, n_neighbors));
    let (mut r, mut c, mut w) = (Vec::new(), Vec::new(), Vec::new());
    for e in ug.edges.iter().filter(|e| e.source < e.target) {
        r.push(e.source as u32);
        c.push(e.target as u32);
        w.push(e.weight);
    }
    Ok((PyArray1::from_vec_bound(py, r), PyArray1::from_vec_bound(py, c), PyArray1::from_vec_bound(py, w)))
}

/// Leiden community detection (modularity, RB configuration) on an undirected weighted graph.
#[pyfunction]
#[pyo3(signature = (n, rows, cols, weights, resolution=1.0, n_iterations=2, seed=0))]
fn leiden<'py>(
    py: Python<'py>,
    n: usize,
    rows: numpy::PyReadonlyArrayDyn<'py, u32>,
    cols: numpy::PyReadonlyArrayDyn<'py, u32>,
    weights: numpy::PyReadonlyArrayDyn<'py, f32>,
    resolution: f64,
    n_iterations: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    if !(resolution > 0.0 && resolution.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err("resolution must be positive and finite"));
    }
    let e = undirected(n, contiguous(&rows, "rows")?, contiguous(&cols, "cols")?, contiguous(&weights, "weights")?)?;
    let labels = py.allow_threads(|| {
        let edges: Vec<(usize, usize, f64)> = e.iter().map(|&(a, b, w)| (a, b, w as f64)).collect();
        let g = crate::leiden::Graph::from_edges(n, &edges);
        crate::leiden::leiden_partition(&g, resolution, n_iterations, seed)
    });
    Ok(PyArray1::from_vec_bound(py, labels.into_iter().map(|c| c as u32).collect()))
}

/// UMAP layout of an undirected weighted graph (umap-learn semantics, parallel SGD).
#[pyfunction]
#[pyo3(signature = (n, rows, cols, weights, n_components=2, min_dist=0.5, spread=1.0, n_epochs=None, spectral_n_iter=100, seed=0))]
fn umap<'py>(
    py: Python<'py>,
    n: usize,
    rows: numpy::PyReadonlyArrayDyn<'py, u32>,
    cols: numpy::PyReadonlyArrayDyn<'py, u32>,
    weights: numpy::PyReadonlyArrayDyn<'py, f32>,
    n_components: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    spectral_n_iter: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if n_components == 0 || !(spread > 0.0) || !(min_dist >= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("invalid UMAP parameters"));
    }
    let e = undirected(n, contiguous(&rows, "rows")?, contiguous(&cols, "cols")?, contiguous(&weights, "weights")?)?;
    let emb = py.allow_threads(|| {
        let mut edges = Vec::with_capacity(e.len() * 2);
        for &(a, b, w) in &e {
            edges.push(Edge { source: a, target: b, weight: w });
            edges.push(Edge { source: b, target: a, weight: w });
        }
        let g = UmapGraph { n, edges };
        crate::umap::core::embed(&g, n_components, min_dist, spread, n_epochs, spectral_n_iter, seed)
    });
    Ok(PyArray1::from_vec_bound(py, emb).reshape([n, n_components])?)
}

/// Filter/remap cells and genes, normalise to target_sum (<= 0: skip) and log1p.
#[pyfunction]
fn preprocess_chunk<'py>(
    py: Python<'py>,
    cells: PyReadonlyArrayDyn<'py, u32>,
    genes: PyReadonlyArrayDyn<'py, u32>,
    values: PyReadonlyArrayDyn<'py, f32>,
    cell_map: PyReadonlyArrayDyn<'py, i64>,
    gene_map: PyReadonlyArrayDyn<'py, i32>,
    target_sum: f64,
    log1p: bool,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
    let (c, g, v) = (sl(&cells, "cells")?, sl(&genes, "genes")?, sl(&values, "values")?);
    check_len("cells, genes, values", &[c.len(), g.len(), v.len()])?;
    if c.windows(2).any(|w| w[0] > w[1]) {
        return Err(pyo3::exceptions::PyValueError::new_err("chunk must be sorted by cell"));
    }
    let (cm, gm) = (sl(&cell_map, "cell_map")?, sl(&gene_map, "gene_map")?);
    let (r, g, v) = py.allow_threads(|| kernels::preprocess_chunk(c, g, v, cm, gm, target_sum, log1p));
    Ok((PyArray1::from_vec_bound(py, r), PyArray1::from_vec_bound(py, g), PyArray1::from_vec_bound(py, v)))
}

/// Accumulate per-gene [sum x, sum x^2, sum expm1 x, sum expm1(x)^2, nnz] into out (n_genes, 5).
#[pyfunction]
fn gene_stats<'py>(py: Python<'py>, genes: PyReadonlyArrayDyn<'py, u32>, values: PyReadonlyArrayDyn<'py, f32>, mut out: PyReadwriteArrayDyn<'py, f64>) -> PyResult<()> {
    let (g, v) = (sl(&genes, "genes")?, sl(&values, "values")?);
    check_len("genes, values", &[g.len(), v.len()])?;
    let o = sl_mut(&mut out, "out")?;
    let n_genes = o.len() / 5;
    check_ids(g, n_genes, "gene")?;
    py.allow_threads(|| kernels::gene_stats(g, v, n_genes, o));
    Ok(())
}

/// Accumulate the Gram matrix (upper triangle) and column sums of the (optionally scaled) matrix.
#[pyfunction]
#[pyo3(signature = (rows, genes, values, gene_map, gram, colsum, scale=false, mean=None, std=None, max_value=0.0))]
fn gram_accumulate<'py>(
    py: Python<'py>,
    rows: PyReadonlyArrayDyn<'py, u32>,
    genes: PyReadonlyArrayDyn<'py, u32>,
    values: PyReadonlyArrayDyn<'py, f32>,
    gene_map: PyReadonlyArrayDyn<'py, i32>,
    mut gram: PyReadwriteArrayDyn<'py, f64>,
    mut colsum: PyReadwriteArrayDyn<'py, f64>,
    scale: bool,
    mean: Option<PyReadonlyArrayDyn<'py, f64>>,
    std: Option<PyReadonlyArrayDyn<'py, f64>>,
    max_value: f64,
) -> PyResult<()> {
    let (r, g, v, gm) = (sl(&rows, "rows")?, sl(&genes, "genes")?, sl(&values, "values")?, sl(&gene_map, "gene_map")?);
    check_len("rows, genes, values", &[r.len(), g.len(), v.len()])?;
    check_rows(r, u32::MAX as usize, "rows")?;
    let cs = sl_mut(&mut colsum, "colsum")?;
    let d = cs.len();
    let gr = sl_mut(&mut gram, "gram")?;
    if gr.len() != d * d || gm.iter().any(|&m| m >= d as i32) {
        return Err(pyo3::exceptions::PyValueError::new_err("gram must be (d, d) and gene_map values < d"));
    }
    let empty: Vec<f64> = vec![];
    let (mu, sd) = match (&mean, &std) {
        (Some(m), Some(s)) => (sl(m, "mean")?, sl(s, "std")?),
        _ if scale => return Err(pyo3::exceptions::PyValueError::new_err("scale=True needs mean and std")),
        _ => (&empty[..], &empty[..]),
    };
    if scale && (mu.len() != d || sd.len() != d) {
        return Err(pyo3::exceptions::PyValueError::new_err("mean/std must have length d"));
    }
    let t = Transform { scale, mean: mu, std: sd, max_value };
    py.allow_threads(|| kernels::gram_accumulate(r, g, v, gm, t, gr, cs));
    Ok(())
}

/// Top-k eigenpairs of the covariance implied by (gram, colsum, n).
/// Returns (variance (k,), loadings (d, k), total_variance).
#[pyfunction]
fn pca_from_gram<'py>(py: Python<'py>, gram: PyReadonlyArrayDyn<'py, f64>, colsum: PyReadonlyArrayDyn<'py, f64>, n: usize, k: usize)
    -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, f64)> {
    let (g, c) = (sl(&gram, "gram")?, sl(&colsum, "colsum")?);
    let d = c.len();
    if g.len() != d * d {
        return Err(pyo3::exceptions::PyValueError::new_err("gram must be (d, d)"));
    }
    let (vals, vecs, total) = py.allow_threads(|| kernels::pca_from_gram(g, c, n, k)).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let kk = vals.len();
    Ok((PyArray1::from_vec_bound(py, vals), PyArray1::from_vec_bound(py, vecs).reshape([d, kk])?, total))
}

/// out[row] += scaled row · loadings for one chunk (out: (n_rows, k) float32).
#[pyfunction]
#[pyo3(signature = (rows, genes, values, gene_map, loadings, out, scale=false, mean=None, std=None, max_value=0.0))]
fn project<'py>(
    py: Python<'py>,
    rows: PyReadonlyArrayDyn<'py, u32>,
    genes: PyReadonlyArrayDyn<'py, u32>,
    values: PyReadonlyArrayDyn<'py, f32>,
    gene_map: PyReadonlyArrayDyn<'py, i32>,
    loadings: PyReadonlyArrayDyn<'py, f64>,
    mut out: PyReadwriteArrayDyn<'py, f32>,
    scale: bool,
    mean: Option<PyReadonlyArrayDyn<'py, f64>>,
    std: Option<PyReadonlyArrayDyn<'py, f64>>,
    max_value: f64,
) -> PyResult<()> {
    let (r, g, v, gm, l) = (sl(&rows, "rows")?, sl(&genes, "genes")?, sl(&values, "values")?, sl(&gene_map, "gene_map")?, sl(&loadings, "loadings")?);
    check_len("rows, genes, values", &[r.len(), g.len(), v.len()])?;
    let shape = loadings.shape().to_vec();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("loadings must be (d, k)"));
    }
    let (d, k) = (shape[0], shape[1]);
    let o = sl_mut(&mut out, "out")?;
    check_rows(r, o.len() / k.max(1), "rows")?;
    if gm.iter().any(|&m| m >= d as i32) {
        return Err(pyo3::exceptions::PyValueError::new_err("gene_map values must be < d"));
    }
    let empty: Vec<f64> = vec![];
    let (mu, sd) = match (&mean, &std) {
        (Some(m), Some(s)) => (sl(m, "mean")?, sl(s, "std")?),
        _ if scale => return Err(pyo3::exceptions::PyValueError::new_err("scale=True needs mean and std")),
        _ => (&empty[..], &empty[..]),
    };
    let t = Transform { scale, mean: mu, std: sd, max_value };
    py.allow_threads(|| kernels::project(r, g, v, gm, t, l, k, o));
    Ok(())
}

/// Accumulate per (group, gene) sums, sums of squares and nnz (each (n_groups, n_genes)).
#[pyfunction]
fn group_gene_sums<'py>(
    py: Python<'py>,
    rows: PyReadonlyArrayDyn<'py, u32>,
    genes: PyReadonlyArrayDyn<'py, u32>,
    values: PyReadonlyArrayDyn<'py, f32>,
    group_of_row: PyReadonlyArrayDyn<'py, u32>,
    mut sum: PyReadwriteArrayDyn<'py, f64>,
    mut sumsq: PyReadwriteArrayDyn<'py, f64>,
    mut nnz: PyReadwriteArrayDyn<'py, f64>,
) -> PyResult<()> {
    let (r, g, v, gr) = (sl(&rows, "rows")?, sl(&genes, "genes")?, sl(&values, "values")?, sl(&group_of_row, "group_of_row")?);
    check_len("rows, genes, values", &[r.len(), g.len(), v.len()])?;
    let shape = sum.shape().to_vec();
    if shape.len() != 2 || sumsq.shape() != shape.as_slice() || nnz.shape() != shape.as_slice() {
        return Err(pyo3::exceptions::PyValueError::new_err("sum, sumsq, nnz must be (n_groups, n_genes)"));
    }
    let (n_groups, n_genes) = (shape[0], shape[1]);
    check_ids(r, gr.len(), "row")?;
    check_ids(g, n_genes, "gene")?;
    if gr.iter().any(|&x| x != u32::MAX && x as usize >= n_groups) {
        return Err(pyo3::exceptions::PyValueError::new_err("group id out of range"));
    }
    let (s, q, z) = (sl_mut(&mut sum, "sum")?, sl_mut(&mut sumsq, "sumsq")?, sl_mut(&mut nnz, "nnz")?);
    py.allow_threads(|| kernels::group_gene_sums(r, g, v, gr, n_genes, s, q, z));
    Ok(())
}

/// Wilcoxon rank sums per (group, gene) and tie terms per gene (see kernels::wilcoxon_rank_sums).
#[pyfunction]
fn wilcoxon_rank_sums<'py>(
    py: Python<'py>,
    genes: PyReadonlyArrayDyn<'py, u32>,
    values: PyReadonlyArrayDyn<'py, f32>,
    groups: PyReadonlyArrayDyn<'py, u32>,
    n_genes: usize,
    group_sizes: PyReadonlyArrayDyn<'py, u64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (g, v, gr, gs) = (sl(&genes, "genes")?, sl(&values, "values")?, sl(&groups, "groups")?, sl(&group_sizes, "group_sizes")?);
    check_len("genes, values, groups", &[g.len(), v.len(), gr.len()])?;
    check_ids(g, n_genes, "gene")?;
    check_ids(gr, gs.len(), "group")?;
    let (rs, ties) = py.allow_threads(|| kernels::wilcoxon_rank_sums(g, v, gr, n_genes, gs));
    Ok((PyArray1::from_vec_bound(py, rs).reshape([gs.len(), n_genes])?, PyArray1::from_vec_bound(py, ties)))
}

/// out[row] += sum_gene w[gene] * value.
#[pyfunction]
fn weighted_row_sums<'py>(py: Python<'py>, rows: PyReadonlyArrayDyn<'py, u32>, genes: PyReadonlyArrayDyn<'py, u32>, values: PyReadonlyArrayDyn<'py, f32>, w: PyReadonlyArrayDyn<'py, f64>, mut out: PyReadwriteArrayDyn<'py, f64>) -> PyResult<()> {
    let (r, g, v, wv) = (sl(&rows, "rows")?, sl(&genes, "genes")?, sl(&values, "values")?, sl(&w, "w")?);
    check_len("rows, genes, values", &[r.len(), g.len(), v.len()])?;
    let o = sl_mut(&mut out, "out")?;
    check_ids(r, o.len(), "row")?;
    check_ids(g, wv.len(), "gene")?;
    py.allow_threads(|| kernels::weighted_row_sums(r, g, v, wv, o));
    Ok(())
}

/// QC accumulation on raw counts (all outputs indexed by original cell / gene id).
#[pyfunction]
fn qc_chunk<'py>(
    py: Python<'py>,
    cells: PyReadonlyArrayDyn<'py, u32>,
    genes: PyReadonlyArrayDyn<'py, u32>,
    values: PyReadonlyArrayDyn<'py, f32>,
    flag: PyReadonlyArrayDyn<'py, bool>,
    mut cell_total: PyReadwriteArrayDyn<'py, f64>,
    mut cell_ngenes: PyReadwriteArrayDyn<'py, u32>,
    mut cell_flag: PyReadwriteArrayDyn<'py, f64>,
    mut gene_ncells: PyReadwriteArrayDyn<'py, u32>,
    mut gene_total: PyReadwriteArrayDyn<'py, f64>,
) -> PyResult<()> {
    let (c, g, v, f) = (sl(&cells, "cells")?, sl(&genes, "genes")?, sl(&values, "values")?, sl(&flag, "flag")?);
    check_len("cells, genes, values", &[c.len(), g.len(), v.len()])?;
    let (ct, cn, cf) = (sl_mut(&mut cell_total, "cell_total")?, sl_mut(&mut cell_ngenes, "cell_ngenes")?, sl_mut(&mut cell_flag, "cell_flag")?);
    let (gn, gt) = (sl_mut(&mut gene_ncells, "gene_ncells")?, sl_mut(&mut gene_total, "gene_total")?);
    check_ids(c, ct.len().min(cn.len()).min(cf.len()), "cell")?;
    check_ids(g, f.len().min(gn.len()).min(gt.len()), "gene")?;
    py.allow_threads(|| kernels::qc_chunk(c, g, v, f, ct, cn, cf, gn, gt));
    Ok(())
}

/// Two-sided p-values of Student's t (NaN/invalid df -> NaN), as scipy's 2 * t.sf(|t|, df).
#[pyfunction]
fn t_pvalues<'py>(py: Python<'py>, t: PyReadonlyArrayDyn<'py, f64>, df: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    let (t, df) = (sl(&t, "t")?, sl(&df, "df")?);
    check_len("t, df", &[t.len(), df.len()])?;
    let p: Vec<f64> = py.allow_threads(|| {
        t.par_iter().zip(df.par_iter()).map(|(&x, &d)| {
            if !x.is_finite() || !(d > 0.0) { return f64::NAN; }
            match StudentsT::new(0.0, 1.0, d) { Ok(dist) => (2.0 * dist.sf(x.abs())).min(1.0), Err(_) => f64::NAN }
        }).collect()
    });
    Ok(PyArray1::from_vec_bound(py, p))
}

/// Two-sided normal p-values, 2 * norm.sf(|z|).
#[pyfunction]
fn normal_pvalues<'py>(py: Python<'py>, z: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use statrs::distribution::{ContinuousCDF, Normal};
    let z = sl(&z, "z")?;
    let nd = Normal::new(0.0, 1.0).unwrap();
    let p: Vec<f64> = py.allow_threads(|| z.par_iter().map(|&x| if x.is_finite() { (2.0 * nd.sf(x.abs())).min(1.0) } else { f64::NAN }).collect());
    Ok(PyArray1::from_vec_bound(py, p))
}

#[pymodule]
fn crest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_graph, m)?)?;
    m.add_function(wrap_pyfunction!(connectivities, m)?)?;
    m.add_function(wrap_pyfunction!(leiden, m)?)?;
    m.add_function(wrap_pyfunction!(umap, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(gene_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gram_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(pca_from_gram, m)?)?;
    m.add_function(wrap_pyfunction!(project, m)?)?;
    m.add_function(wrap_pyfunction!(group_gene_sums, m)?)?;
    m.add_function(wrap_pyfunction!(wilcoxon_rank_sums, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_row_sums, m)?)?;
    m.add_function(wrap_pyfunction!(qc_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(t_pvalues, m)?)?;
    m.add_function(wrap_pyfunction!(normal_pvalues, m)?)?;
    Ok(())
}
