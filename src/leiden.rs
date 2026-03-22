use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use graphrs::{Edge, Graph, GraphSpecs};
use graphrs::algorithms::community::louvain::louvain_communities;
use kiddo::KdTree;

#[polars_expr(output_type=UInt32)]
fn louvain_clustering(inputs: &[Series]) -> PolarsResult<Series> {
    // Expected inputs:
    // 0: A List(Float32) column representing the PCA coordinates for each cell
    // 1: The number of neighbors (UInt32) to construct the KNN graph

    if inputs.len() != 2 {
        return Err(PolarsError::ComputeError("louvain_clustering requires exactly 2 inputs: [pca_coords, n_neighbors]".into()));
    }

    let pca_coords = &inputs[0].list()?;
    let k_neighbors = inputs[1].u32()?.get(0).unwrap_or(15) as usize;

    // Validate parameter bounds
    if k_neighbors == 0 || k_neighbors > 1000 {
        return Err(PolarsError::ComputeError(
            format!("n_neighbors must be 1-1000, got {}", k_neighbors).into()
        ));
    }

    let n_cells = pca_coords.len();
    if n_cells == 0 {
        return Err(PolarsError::ComputeError("Cannot perform clustering on empty DataFrame".into()));
    }

    // 1. Build the Kiddo KD-Tree for exact fast nearest neighbors
    // KdTree is parameterized with max 50 dimensions at compile time.
    // Input dimensions > 50 will be truncated with a warning.
    const MAX_DIMS: usize = 50;
    let mut tree: KdTree<f32, MAX_DIMS> = KdTree::new();
    let mut pcas = Vec::with_capacity(n_cells);
    let mut warned_truncation = false;

    for (i, opt_row) in pca_coords.into_iter().enumerate() {
        if let Some(row_series) = opt_row {
            let float_ca = row_series.f32()?;
            let actual_dims = float_ca.len();
            if actual_dims > MAX_DIMS && !warned_truncation {
                eprintln!("crest warning: PCA has {} dimensions but Leiden KD-tree supports max {}. Truncating.", actual_dims, MAX_DIMS);
                warned_truncation = true;
            }
            let mut pt = [0.0f32; MAX_DIMS];

            for (j, val) in float_ca.into_no_null_iter().enumerate() {
                if j < MAX_DIMS {
                    pt[j] = val;
                }
            }
            tree.add(&pt, i as u64);
            pcas.push(pt);
        } else {
            pcas.push([0.0f32; MAX_DIMS]);
        }
    }

    // 2. Query KNN and build the graphrs Graph
    let mut graph = Graph::<usize, ()>::new(GraphSpecs::undirected_create_missing());
    let mut edges = Vec::with_capacity(n_cells * k_neighbors);

    for (i, pt) in pcas.iter().enumerate() {
        // Find nearest `k` neighbors
        let neighbors = tree.nearest_n::<kiddo::SquaredEuclidean>(pt, k_neighbors);
        
        for neighbor in neighbors {
            let neighbor_idx = neighbor.item as usize;
            
            // Only add edges in one direction to avoid duplication in undirected graph
            if i < neighbor_idx {
                // Calculate weight using a simple Gaussian kernel over squared distance
                let sq_dist = neighbor.distance;
                let weight = (-sq_dist as f64).exp(); // Standard heuristic
                
                edges.push(Edge::with_weight(i, neighbor_idx, weight));
            }
        }
    }
    
    graph.add_edges(edges)
        .map_err(|e| PolarsError::ComputeError(format!("Failed to build graph: {:?}", e).into()))?;

    // 3. Run the Louvain Community Detection
    // Set weighted=true, using default resolution (1.0) and tolerances
    let communities = louvain_communities(&graph, true, None, None, Some(42))
        .map_err(|e| PolarsError::ComputeError(format!("Louvain failed: {:?}", e).into()))?;

    // 4. Map the communities back into a flat Vector
    let mut cluster_labels = vec![0u32; n_cells];
    
    for (cluster_id, community) in communities.iter().enumerate() {
        for &cell_idx in community {
            if cell_idx < n_cells {
                cluster_labels[cell_idx] = cluster_id as u32;
            } else {
                return Err(PolarsError::ComputeError(
                    format!("Louvain returned out-of-bounds cell index {} (n_cells={})", cell_idx, n_cells).into()
                ));
            }
        }
    }

    // Return the labels as a UInt32 series mapping back directly to the Polars index
    Ok(Series::new("louvain", cluster_labels))
}
