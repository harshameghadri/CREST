//! Native Python API (numpy in, numpy out; the GIL is released during compute).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::knn::{knn, KnnGraph, Points};
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

#[pymodule]
fn crest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_graph, m)?)?;
    m.add_function(wrap_pyfunction!(connectivities, m)?)?;
    m.add_function(wrap_pyfunction!(leiden, m)?)?;
    m.add_function(wrap_pyfunction!(umap, m)?)?;
    Ok(())
}
