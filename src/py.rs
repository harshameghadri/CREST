//! Native Python API (numpy in, numpy out; the GIL is released during compute).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::kernels::{self, Cells, ChunkView, Transform};
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
    let labels = py.allow_threads(move || {
        let g = crate::leiden::Graph::from_edges(n, &e);
        drop(e);
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

type Arr<'py, T> = PyReadonlyArrayDyn<'py, T>;

/// Common chunk arguments: raw genes/values, filters, transform, and either
/// per-entry `cells` or a CSR `indptr` for cells `first_cell ..`.
fn view<'a>(
    genes: &'a Arr<'_, u32>, values: &'a Arr<'_, f32>, cell_map: &'a Arr<'_, i64>, gene_map: &'a Arr<'_, i32>,
    target_sum: f64, log1p: bool, cells: &'a Option<Arr<'_, u32>>, indptr: &'a Option<Arr<'_, i64>>, first_cell: u32,
) -> PyResult<(ChunkView<'a>, usize)> {
    let (g, v) = (sl(genes, "genes")?, sl(values, "values")?);
    check_len("genes, values", &[g.len(), v.len()])?;
    let (cm, gm) = (sl(cell_map, "cell_map")?, sl(gene_map, "gene_map")?);
    let cells = match (cells, indptr) {
        (Some(c), None) => {
            let c = sl(c, "cells")?;
            check_len("cells, genes", &[c.len(), g.len()])?;
            if c.windows(2).any(|w| w[0] > w[1]) {
                return Err(pyo3::exceptions::PyValueError::new_err("chunk must be sorted by cell"));
            }
            Cells::Ids(c)
        }
        (None, Some(p)) => {
            let p = sl(p, "indptr")?;
            if p.is_empty() || p.windows(2).any(|w| w[0] > w[1]) || (p[p.len() - 1] - p[0]) as usize != g.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("indptr must be non-decreasing and span the chunk"));
            }
            Cells::Ptr { indptr: p, first: first_cell }
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err("pass exactly one of cells / indptr")),
    };
    let n_rows = cm.iter().copied().max().map_or(0, |m| (m + 1).max(0) as usize);
    Ok((ChunkView { cells, genes: g, values: v, cell_map: cm, gene_map: gm, target_sum, log1p }, n_rows))
}

fn n_var(gene_map: &[i32]) -> usize {
    gene_map.iter().copied().max().map_or(0, |m| (m + 1).max(0) as usize)
}

macro_rules! chunk_fn {
    ($(#[$m:meta])* fn $name:ident<$py:lifetime>($pyv:ident, $v:ident, $nrows:ident, $nvar:ident $(, $a:ident : $t:ty)*) -> $ret:ty $body:block) => {
        $(#[$m])*
        #[pyfunction]
        #[pyo3(signature = (genes, values, cell_map, gene_map, target_sum, log1p, cells, indptr, first_cell $(, $a)*))]
        #[allow(clippy::too_many_arguments)]
        fn $name<$py>(
            $pyv: Python<$py>, genes: Arr<$py, u32>, values: Arr<$py, f32>, cell_map: Arr<$py, i64>, gene_map: Arr<$py, i32>,
            target_sum: f64, log1p: bool, cells: Option<Arr<$py, u32>>, indptr: Option<Arr<$py, i64>>, first_cell: u32
            $(, $a: $t)*
        ) -> $ret {
            let ($v, $nrows) = view(&genes, &values, &cell_map, &gene_map, target_sum, log1p, &cells, &indptr, first_cell)?;
            let $nvar = n_var(sl(&gene_map, "gene_map")?);
            $body
        }
    };
}

chunk_fn! {
    /// Materialise the filtered/transformed chunk as (row, var index, value).
    fn materialize<'py>(py, v, _nr, _nv) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
        let (r, g, x) = py.allow_threads(|| kernels::materialize(v));
        Ok((PyArray1::from_vec_bound(py, r), PyArray1::from_vec_bound(py, g), PyArray1::from_vec_bound(py, x)))
    }
}

chunk_fn! {
    /// Accumulate per-gene [Σx, Σx², Σexpm1 x, Σexpm1(x)², nnz] into out (n_vars, 5).
    fn gene_stats<'py>(py, v, _nr, nv, out: PyReadwriteArrayDyn<'py, f64>) -> PyResult<()> {
        let mut out = out;
        let o = sl_mut(&mut out, "out")?;
        if o.len() != nv * 5 {
            return Err(pyo3::exceptions::PyValueError::new_err("out must be (n_vars, 5)"));
        }
        py.allow_threads(|| kernels::gene_stats(v, o));
        Ok(())
    }
}

chunk_fn! {
    /// QC accumulation on raw counts (pass target_sum=0, log1p=False).
    fn qc<'py>(py, v, nr, nv, flag: Arr<'py, bool>, cell_total: PyReadwriteArrayDyn<'py, f64>, cell_ngenes: PyReadwriteArrayDyn<'py, u32>,
               cell_flag: PyReadwriteArrayDyn<'py, f64>, gene_ncells: PyReadwriteArrayDyn<'py, u32>, gene_total: PyReadwriteArrayDyn<'py, f64>) -> PyResult<()> {
        let (mut ct, mut cn, mut cf, mut gn, mut gt) = (cell_total, cell_ngenes, cell_flag, gene_ncells, gene_total);
        let f = sl(&flag, "flag")?;
        let (ct, cn, cf) = (sl_mut(&mut ct, "cell_total")?, sl_mut(&mut cn, "cell_ngenes")?, sl_mut(&mut cf, "cell_flag")?);
        let (gn, gt) = (sl_mut(&mut gn, "gene_ncells")?, sl_mut(&mut gt, "gene_total")?);
        if ct.len() < nr || cn.len() < nr || cf.len() < nr || gn.len() != nv || gt.len() != nv || f.len() != nv {
            return Err(pyo3::exceptions::PyValueError::new_err("output sizes do not match the maps"));
        }
        py.allow_threads(|| kernels::qc(v, f, ct, cn, cf, gn, gt));
        Ok(())
    }
}

fn transform<'a>(scale: bool, mean: &'a Option<Arr<'_, f64>>, std: &'a Option<Arr<'_, f64>>, max_value: f64, d: usize) -> PyResult<Transform<'a>> {
    static EMPTY: [f64; 0] = [];
    let (mu, sd): (&[f64], &[f64]) = match (mean, std) {
        (Some(m), Some(s)) => (sl(m, "mean")?, sl(s, "std")?),
        _ if scale => return Err(pyo3::exceptions::PyValueError::new_err("scale=True needs mean and std")),
        _ => (&EMPTY, &EMPTY),
    };
    if scale && (mu.len() != d || sd.len() != d) {
        return Err(pyo3::exceptions::PyValueError::new_err("mean/std must have length d"));
    }
    Ok(Transform { scale, mean: mu, std: sd, max_value })
}

fn check_sub(sub: &[i32], nv: usize, d: usize) -> PyResult<()> {
    if sub.len() != nv || sub.iter().any(|&m| m >= d as i32) {
        return Err(pyo3::exceptions::PyValueError::new_err("sub_map must have length n_vars and values < d"));
    }
    Ok(())
}

chunk_fn! {
    /// Accumulate the Gram matrix (upper triangle) and column sums of the (optionally scaled) submatrix.
    fn gram_accumulate<'py>(py, v, _nr, nv, sub_map: Arr<'py, i32>, gram: PyReadwriteArrayDyn<'py, f64>, colsum: PyReadwriteArrayDyn<'py, f64>,
                            scale: bool, mean: Option<Arr<'py, f64>>, std: Option<Arr<'py, f64>>, max_value: f64) -> PyResult<()> {
        let (mut gram, mut colsum) = (gram, colsum);
        let sm = sl(&sub_map, "sub_map")?;
        let cs = sl_mut(&mut colsum, "colsum")?;
        let d = cs.len();
        let gr = sl_mut(&mut gram, "gram")?;
        if gr.len() != d * d {
            return Err(pyo3::exceptions::PyValueError::new_err("gram must be (d, d)"));
        }
        check_sub(sm, nv, d)?;
        let t = transform(scale, &mean, &std, max_value, d)?;
        py.allow_threads(|| kernels::gram_accumulate(v, sm, t, gr, cs));
        Ok(())
    }
}

chunk_fn! {
    /// out[row] += (scaled) row · loadings for one chunk (out: (n_rows, k) float32).
    fn project<'py>(py, v, nr, nv, sub_map: Arr<'py, i32>, loadings: Arr<'py, f64>, out: PyReadwriteArrayDyn<'py, f32>,
                    scale: bool, mean: Option<Arr<'py, f64>>, std: Option<Arr<'py, f64>>, max_value: f64) -> PyResult<()> {
        let mut out = out;
        let shape = loadings.shape().to_vec();
        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("loadings must be (d, k)"));
        }
        let (d, k) = (shape[0], shape[1]);
        let (sm, l) = (sl(&sub_map, "sub_map")?, sl(&loadings, "loadings")?);
        check_sub(sm, nv, d)?;
        let o = sl_mut(&mut out, "out")?;
        if o.len() < nr * k {
            return Err(pyo3::exceptions::PyValueError::new_err("out must be (n_rows, k)"));
        }
        let t = transform(scale, &mean, &std, max_value, d)?;
        py.allow_threads(|| kernels::project(v, sm, t, l, k, o));
        Ok(())
    }
}

chunk_fn! {
    /// Accumulate per (group, var) sums, sums of squares and nnz (each (n_groups, n_vars)).
    fn group_gene_sums<'py>(py, v, nr, nv, group_of_row: Arr<'py, u32>, sum: PyReadwriteArrayDyn<'py, f64>,
                            sumsq: PyReadwriteArrayDyn<'py, f64>, nnz: PyReadwriteArrayDyn<'py, f64>,
                            clip: Option<Arr<'py, f64>>) -> PyResult<()> {
        let (mut sum, mut sumsq, mut nnz) = (sum, sumsq, nnz);
        let gr = sl(&group_of_row, "group_of_row")?;
        let shape = sum.shape().to_vec();
        if shape.len() != 2 || shape[1] != nv || sumsq.shape() != shape.as_slice() || nnz.shape() != shape.as_slice() || gr.len() < nr {
            return Err(pyo3::exceptions::PyValueError::new_err("sum, sumsq, nnz must be (n_groups, n_vars); group_of_row (n_rows,)"));
        }
        if gr.iter().any(|&x| x != u32::MAX && x as usize >= shape[0]) {
            return Err(pyo3::exceptions::PyValueError::new_err("group id out of range"));
        }
        let c = match &clip {
            Some(c) => {
                let c = sl(c, "clip")?;
                if c.len() != shape[0] * shape[1] {
                    return Err(pyo3::exceptions::PyValueError::new_err("clip must be (n_groups, n_vars)"));
                }
                Some(c)
            }
            None => None,
        };
        let (s, q, z) = (sl_mut(&mut sum, "sum")?, sl_mut(&mut sumsq, "sumsq")?, sl_mut(&mut nnz, "nnz")?);
        py.allow_threads(|| kernels::group_gene_sums(v, gr, nv, s, Some(q), Some(z), c));
        Ok(())
    }
}

chunk_fn! {
    /// Accumulate per (group, var) sums only, (n_groups, n_vars) — pseudobulk aggregation.
    fn group_gene_totals<'py>(py, v, nr, nv, group_of_row: Arr<'py, u32>, sum: PyReadwriteArrayDyn<'py, f64>) -> PyResult<()> {
        let mut sum = sum;
        let gr = sl(&group_of_row, "group_of_row")?;
        let shape = sum.shape().to_vec();
        if shape.len() != 2 || shape[1] != nv || gr.len() < nr {
            return Err(pyo3::exceptions::PyValueError::new_err("sum must be (n_groups, n_vars); group_of_row (n_rows,)"));
        }
        if gr.iter().any(|&x| x != u32::MAX && x as usize >= shape[0]) {
            return Err(pyo3::exceptions::PyValueError::new_err("group id out of range"));
        }
        let s = sl_mut(&mut sum, "sum")?;
        py.allow_threads(|| kernels::group_gene_sums(v, gr, nv, s, None, None, None));
        Ok(())
    }
}

chunk_fn! {
    /// Non-zero entries of vars in [lo, hi) as (var - lo, value, group) for cells with a group.
    fn collect_gene_block<'py>(py, v, nr, _nv, group_of_row: Arr<'py, u32>, lo: u32, hi: u32)
        -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<u32>>)> {
        let gr = sl(&group_of_row, "group_of_row")?;
        if gr.len() < nr {
            return Err(pyo3::exceptions::PyValueError::new_err("group_of_row too short"));
        }
        let (a, b, c) = py.allow_threads(|| kernels::collect_gene_block(v, gr, lo, hi));
        Ok((PyArray1::from_vec_bound(py, a), PyArray1::from_vec_bound(py, b), PyArray1::from_vec_bound(py, c)))
    }
}

chunk_fn! {
    /// out[row] += Σ_var w[var] * value.
    fn weighted_row_sums<'py>(py, v, nr, nv, w: Arr<'py, f64>, out: PyReadwriteArrayDyn<'py, f64>) -> PyResult<()> {
        let mut out = out;
        let wv = sl(&w, "w")?;
        let o = sl_mut(&mut out, "out")?;
        if wv.len() != nv || o.len() < nr {
            return Err(pyo3::exceptions::PyValueError::new_err("w must be (n_vars,), out (n_rows,)"));
        }
        py.allow_threads(|| kernels::weighted_row_sums(v, wv, o));
        Ok(())
    }
}

/// Top-k eigenpairs of the covariance implied by (gram, colsum, n).
/// Returns (variance (k,), loadings (d, k), total_variance).
#[pyfunction]
fn pca_from_gram<'py>(py: Python<'py>, gram: Arr<'py, f64>, colsum: Arr<'py, f64>, n: usize, k: usize)
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

/// Wilcoxon rank sums per (group, gene) and tie terms per gene (see kernels::wilcoxon_rank_sums).
#[pyfunction]
fn wilcoxon_rank_sums<'py>(py: Python<'py>, genes: Arr<'py, u32>, values: Arr<'py, f32>, groups: Arr<'py, u32>, n_genes: usize,
                           group_sizes: Arr<'py, u64>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (g, v, gr, gs) = (sl(&genes, "genes")?, sl(&values, "values")?, sl(&groups, "groups")?, sl(&group_sizes, "group_sizes")?);
    check_len("genes, values, groups", &[g.len(), v.len(), gr.len()])?;
    check_ids(g, n_genes, "gene")?;
    check_ids(gr, gs.len(), "group")?;
    let (rs, ties) = py.allow_threads(|| kernels::wilcoxon_rank_sums(g, v, gr, n_genes, gs));
    Ok((PyArray1::from_vec_bound(py, rs).reshape([gs.len(), n_genes])?, PyArray1::from_vec_bound(py, ties)))
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

/// DESeq2 on a genes × samples integer count matrix with a (samples × coefs)
/// model matrix. Returns a dict of numpy arrays (see `crest.tl.deseq2`).
#[pyfunction]
#[pyo3(signature = (counts, design, size_factors=None, sf_type="ratio", fit_type="parametric",
                    min_replicates_for_replace=7.0, min_mu=0.5, reduced=None))]
#[allow(clippy::too_many_arguments)]
fn deseq2_fit<'py>(
    py: Python<'py>, counts: PyReadonlyArray2<'py, f64>, design: PyReadonlyArray2<'py, f64>,
    size_factors: Option<PyReadonlyArrayDyn<'py, f64>>, sf_type: &str, fit_type: &str,
    min_replicates_for_replace: f64, min_mu: f64, reduced: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use crate::deseq::{self, Design, FitType, Params};
    let err = pyo3::exceptions::PyValueError::new_err;
    let (g, m) = (counts.shape()[0], counts.shape()[1]);
    let (m2, p) = (design.shape()[0], design.shape()[1]);
    if m != m2 {
        return Err(err("design must have one row per sample (column of counts)"));
    }
    let c = counts.as_slice().map_err(|_| err("counts must be C-contiguous"))?;
    let x = design.as_slice().map_err(|_| err("design must be C-contiguous"))?.to_vec();
    let sf = match &size_factors {
        Some(s) => Some(sl(s, "size_factors")?.to_vec()),
        None => None,
    };
    let poscounts = match sf_type {
        "ratio" => false,
        "poscounts" => true,
        _ => return Err(err("sf_type must be \"ratio\" or \"poscounts\"")),
    };
    let ft = match fit_type {
        "parametric" => FitType::Parametric,
        "mean" => FitType::Mean,
        _ => return Err(err("fit_type must be \"parametric\" or \"mean\"")),
    };
    let pa = Params { min_replicates_for_replace, min_mu, fit_type: ft, ..Params::default() };
    let d = Design::new(x, m, p).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let dr = match &reduced {
        Some(r) => {
            if r.shape()[0] != m {
                return Err(err("reduced design must have one row per sample"));
            }
            let xr = r.as_slice().map_err(|_| err("reduced must be C-contiguous"))?.to_vec();
            Some(Design::new(xr, m, r.shape()[1]).map_err(pyo3::exceptions::PyValueError::new_err)?)
        }
        None => None,
    };
    let fit = py
        .allow_threads(|| deseq::deseq(c, g, &d, sf, poscounts, &pa, dr.as_ref()))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new_bound(py);
    let v = |x: Vec<f64>| PyArray1::from_vec_bound(py, x);
    let b = |x: Vec<bool>| PyArray1::from_vec_bound(py, x);
    out.set_item("size_factors", v(fit.size_factors))?;
    out.set_item("baseMean", v(fit.base_mean))?;
    out.set_item("baseVar", v(fit.base_var))?;
    out.set_item("allZero", b(fit.all_zero))?;
    out.set_item("dispGeneEst", v(fit.disp_gene_est))?;
    out.set_item("dispGeneIter", v(fit.disp_gene_iter))?;
    out.set_item("dispFit", v(fit.disp_fit))?;
    out.set_item("dispMAP", v(fit.disp_map))?;
    out.set_item("dispersion", v(fit.dispersion))?;
    out.set_item("dispOutlier", b(fit.disp_outlier))?;
    out.set_item("beta", v(fit.beta).reshape([g, p])?)?;
    out.set_item("beta_cov", v(fit.beta_cov).reshape([g, p, p])?)?;
    out.set_item("betaConv", b(fit.beta_conv))?;
    out.set_item("betaIter", v(fit.beta_iter))?;
    out.set_item("deviance", v(fit.deviance))?;
    out.set_item("cooks", v(fit.cooks).reshape([g, m])?)?;
    out.set_item("maxCooks", v(fit.max_cooks))?;
    out.set_item("replace", b(fit.replace))?;
    if let Some(rc) = fit.replace_counts {
        out.set_item("replaceCounts", v(rc).reshape([g, m])?)?;
    }
    out.set_item("fitType", match fit.fit_type { FitType::Parametric => "parametric", FitType::Mean => "mean" })?;
    out.set_item("trend", v(fit.trend))?;
    out.set_item("dispPriorVar", fit.disp_prior_var)?;
    out.set_item("varLogDispEsts", fit.var_log_disp_ests)?;
    out.set_item("messages", fit.messages)?;
    if let (Some(dv), Some(st), Some(pv), Some(bc)) = (fit.deviance_reduced, fit.lrt_stat, fit.lrt_pvalue, fit.beta_conv_reduced) {
        out.set_item("devianceReduced", v(dv))?;
        out.set_item("LRTStatistic", v(st))?;
        out.set_item("LRTPvalue", v(pv))?;
        out.set_item("betaConvReduced", b(bc))?;
    }
    let cutoff = if m > p {
        use statrs::distribution::{ContinuousCDF, FisherSnedecor};
        FisherSnedecor::new(p as f64, (m - p) as f64).map(|f| f.inverse_cdf(0.99)).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    out.set_item("cooksCutoff", cutoff)?;
    Ok(out)
}

/// R's `lowess(x, y, f, iter, delta)`; `x` sorted ascending.
#[pyfunction]
#[pyo3(signature = (x, y, f=2.0/3.0, iter=3, delta=None))]
fn lowess<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f64>, y: PyReadonlyArrayDyn<'py, f64>, f: f64, iter: usize,
               delta: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (x, y) = (sl(&x, "x")?, sl(&y, "y")?);
    check_len("x, y", &[x.len(), y.len()])?;
    if x.windows(2).any(|w| w[0] > w[1]) {
        return Err(pyo3::exceptions::PyValueError::new_err("x must be sorted"));
    }
    let delta = delta.unwrap_or_else(|| if x.is_empty() { 0.0 } else { 0.01 * (x[x.len() - 1] - x[0]) });
    Ok(PyArray1::from_vec_bound(py, crate::deseq::lowess::lowess(x, y, f, iter, delta)))
}

/// Harmony batch integration of an (N, d) float32 embedding. `batch` is (n_cov, N)
/// uint32 global level ids; `levels` the number of levels per covariate.
#[pyfunction]
#[pyo3(signature = (z, batch, levels, n_clusters, theta, lamb=None, sigma=0.1, alpha=0.2, block_size=0.05,
                    max_iter_harmony=10, max_iter_kmeans=20, epsilon_cluster=1e-3, epsilon_harmony=1e-2,
                    batch_prop_cutoff=1e-5, seed=0))]
#[allow(clippy::too_many_arguments)]
fn harmony<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f32>,
    batch: PyReadonlyArray2<'py, u32>,
    levels: Vec<usize>,
    n_clusters: usize,
    theta: Vec<f64>,
    lamb: Option<Vec<f64>>,
    sigma: f64,
    alpha: f64,
    block_size: f64,
    max_iter_harmony: usize,
    max_iter_kmeans: usize,
    epsilon_cluster: f64,
    epsilon_harmony: f64,
    batch_prop_cutoff: f64,
    seed: u64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let (n, d) = (z.shape()[0], z.shape()[1]);
    let zs = z.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("z must be C-contiguous"))?;
    let bs = batch.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("batch must be C-contiguous"))?;
    if batch.shape()[1] != n || batch.shape()[0] != levels.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("batch must have shape (len(levels), N)"));
    }
    let p = crate::harmony::Params {
        n_clusters, sigma, theta, lambda: lamb, alpha, block_size, max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, batch_prop_cutoff, seed,
    };
    let out = py
        .allow_threads(|| crate::harmony::harmony(zs, n, d, bs, &levels, &p))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("Z_corr", PyArray1::from_vec_bound(py, out.z_corr).reshape([n, d])?)?;
    dict.set_item("R", PyArray1::from_vec_bound(py, out.r).reshape([n, n_clusters])?)?;
    dict.set_item("Y", PyArray1::from_vec_bound(py, out.y).reshape([n_clusters, d])?)?;
    dict.set_item("objective_harmony", out.objective_harmony)?;
    dict.set_item("objective_kmeans", out.objective_kmeans)?;
    dict.set_item("kmeans_rounds", out.kmeans_rounds)?;
    dict.set_item("converged", out.converged)?;
    Ok(dict)
}

/// Leiden for many (resolution, seed) pairs on one graph, the runs in parallel.
/// Returns an (n_runs, n) uint32 array of labels.
#[pyfunction]
#[pyo3(signature = (n, rows, cols, weights, resolutions, seeds, n_iterations=2))]
#[allow(clippy::too_many_arguments)]
fn leiden_sweep<'py>(
    py: Python<'py>,
    n: usize,
    rows: numpy::PyReadonlyArrayDyn<'py, u32>,
    cols: numpy::PyReadonlyArrayDyn<'py, u32>,
    weights: numpy::PyReadonlyArrayDyn<'py, f32>,
    resolutions: Vec<f64>,
    seeds: Vec<u64>,
    n_iterations: usize,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    if resolutions.len() != seeds.len() || resolutions.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("resolutions and seeds must be non-empty and of equal length"));
    }
    if resolutions.iter().any(|r| !(*r > 0.0 && r.is_finite())) {
        return Err(pyo3::exceptions::PyValueError::new_err("resolutions must be positive and finite"));
    }
    let e = undirected(n, contiguous(&rows, "rows")?, contiguous(&cols, "cols")?, contiguous(&weights, "weights")?)?;
    let runs = resolutions.len();
    let flat = py.allow_threads(move || {
        let g = crate::leiden::Graph::from_edges(n, &e);
        drop(e);
        let labels: Vec<Vec<usize>> = resolutions
            .par_iter()
            .zip(seeds.par_iter())
            .map(|(&r, &s)| crate::leiden::leiden_partition(&g, r, n_iterations, s))
            .collect();
        labels.into_iter().flatten().map(|c| c as u32).collect::<Vec<u32>>()
    });
    Ok(PyArray1::from_vec_bound(py, flat).reshape([runs, n])?)
}

/// k nearest rows of `reference` for each row of `query` (float32, same width).
/// Returns (indices uint32 (n_query, k), distances float32 (n_query, k)).
#[pyfunction]
#[pyo3(signature = (reference, query, k, exact=None, seed=0))]
fn knn_query<'py>(
    py: Python<'py>,
    reference: PyReadonlyArray2<'py, f32>,
    query: PyReadonlyArray2<'py, f32>,
    k: usize,
    exact: Option<bool>,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)> {
    let (nr, d) = (reference.shape()[0], reference.shape()[1]);
    let nq = query.shape()[0];
    if query.shape()[1] != d {
        return Err(pyo3::exceptions::PyValueError::new_err("reference and query must have the same number of columns"));
    }
    let rs = reference.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("reference must be C-contiguous"))?;
    let qs = query.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("query must be C-contiguous"))?;
    let g = py.allow_threads(|| crate::knn::knn_query(Points::new(rs, nr, d), Points::new(qs, nq, d), k, exact, seed));
    let kk = g.k;
    Ok((PyArray1::from_vec_bound(py, g.indices).reshape([nq, kk])?, PyArray1::from_vec_bound(py, g.distances).reshape([nq, kk])?))
}

#[pymodule]
fn crest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_graph, m)?)?;
    m.add_function(wrap_pyfunction!(connectivities, m)?)?;
    m.add_function(wrap_pyfunction!(leiden, m)?)?;
    m.add_function(wrap_pyfunction!(umap, m)?)?;
    m.add_function(wrap_pyfunction!(materialize, m)?)?;
    m.add_function(wrap_pyfunction!(gene_stats, m)?)?;
    m.add_function(wrap_pyfunction!(qc, m)?)?;
    m.add_function(wrap_pyfunction!(collect_gene_block, m)?)?;
    m.add_function(wrap_pyfunction!(gram_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(pca_from_gram, m)?)?;
    m.add_function(wrap_pyfunction!(project, m)?)?;
    m.add_function(wrap_pyfunction!(group_gene_sums, m)?)?;
    m.add_function(wrap_pyfunction!(wilcoxon_rank_sums, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_row_sums, m)?)?;
    m.add_function(wrap_pyfunction!(t_pvalues, m)?)?;
    m.add_function(wrap_pyfunction!(normal_pvalues, m)?)?;
    m.add_function(wrap_pyfunction!(group_gene_totals, m)?)?;
    m.add_function(wrap_pyfunction!(deseq2_fit, m)?)?;
    m.add_function(wrap_pyfunction!(lowess, m)?)?;
    m.add_function(wrap_pyfunction!(harmony, m)?)?;
    m.add_function(wrap_pyfunction!(leiden_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(knn_query, m)?)?;
    Ok(())
}
