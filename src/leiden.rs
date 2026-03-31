use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use petgraph::graph::UnGraph;
use petgraph::graph::NodeIndex;
use instant_distance::{Builder, Hnsw, Point};
use rand::{Rng, SeedableRng};

// ----- HNSW Point implementation for PCA vectors -----

#[derive(Clone, Debug)]
struct PcaPoint(Vec<f32>);

impl Point for PcaPoint {
    fn distance(&self, other: &Self) -> f32 {
        self.0.iter().zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// ----- Leiden Algorithm Implementation -----
// Reference: Traag, Waltman & van Eck (2019) "From Louvain to Leiden"
//
// Key improvement over Louvain: the REFINEMENT step ensures all communities
// are well-connected. Louvain can produce disconnected communities.

/// Modularity gain from moving node `i` from its current community to community `c`.
/// Q_gain = (2 * k_{i,c} - resolution * k_i * sigma_c / m)
/// where:
///   k_{i,c} = sum of edge weights from i to nodes in community c
///   k_i     = total weighted degree of node i
///   sigma_c = total weighted degree of community c
///   m       = total edge weight in graph
fn _modularity_gain(
    node_weights_to_community: f64,
    node_degree: f64,
    community_degree: f64,
    total_weight: f64,
    resolution: f64,
) -> f64 {
    let inv_m = if total_weight > 0.0 { 1.0 / total_weight } else { 0.0 };
    node_weights_to_community - resolution * node_degree * community_degree * inv_m
}

/// Compute total edge weight sum in the graph
fn total_edge_weight(graph: &UnGraph<(), f64>) -> f64 {
    graph.edge_indices()
        .filter_map(|e| graph.edge_weight(e))
        .sum()
}

/// Build adjacency info: for each node, the sum of edge weights to each community
fn weights_to_communities(
    graph: &UnGraph<(), f64>,
    node: NodeIndex,
    community: &[usize],
) -> std::collections::HashMap<usize, f64> {
    let mut community_weights = std::collections::HashMap::new();
    for neighbor in graph.neighbors(node) {
        let c = community[neighbor.index()];
        let edge = graph.find_edge(node, neighbor).unwrap();
        let w = *graph.edge_weight(edge).unwrap_or(&1.0);
        *community_weights.entry(c).or_insert(0.0) += w;
    }
    community_weights
}

/// Node degree (sum of edge weights incident to node)
fn node_degree(graph: &UnGraph<(), f64>, node: NodeIndex) -> f64 {
    graph.edges(node)
        .map(|e| e.weight())
        .sum()
}

/// Leiden clustering: local move → refinement → aggregation
/// Returns community assignments for each node.
fn leiden_partition(
    graph: &UnGraph<(), f64>,
    resolution: f64,
    max_iterations: usize,
    seed: u64,
) -> Vec<usize> {
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    // Initialize: each node in its own community
    let mut community: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for _iteration in 0..max_iterations {
        let total_w = total_edge_weight(graph);
        if total_w <= 0.0 {
            break;
        }

        // Phase 1: LOCAL MOVE PHASE (same as Louvain)
        // Visit nodes in random order, greedily move to best community
        let mut node_order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            node_order.swap(i, j);
        }

        let mut improved = false;

        // Precompute community degrees
        let mut community_degrees: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            let d = node_degree(graph, NodeIndex::new(i));
            community_degrees[community[i]] += d;
        }

        for &node_idx in &node_order {
            let node = NodeIndex::new(node_idx);
            let ki = node_degree(graph, node);
            let current_c = community[node_idx];

            // Compute weights from node to each neighboring community
            let w_to_c = weights_to_communities(graph, node, &community);

            // Try removing node from current community
            let w_to_current = w_to_c.get(&current_c).copied().unwrap_or(0.0);
            let current_sigma = community_degrees[current_c] - ki;

            let mut best_gain = 0.0;
            let mut best_community = current_c;

            for (&c, &w_ic) in &w_to_c {
                if c == current_c {
                    continue;
                }
                let sigma_c = community_degrees[c];
                // Gain = (w to new community - w to old community) - resolution * ki * (sigma_new - sigma_old) / m
                let gain = (w_ic - w_to_current)
                    + resolution * ki * (current_sigma - sigma_c) / total_w;
                if gain > best_gain {
                    best_gain = gain;
                    best_community = c;
                }
            }

            if best_community != current_c {
                // Move node to best community
                community_degrees[current_c] -= ki;
                community_degrees[best_community] += ki;
                community[node_idx] = best_community;
                improved = true;
            }
        }

        if !improved {
            break;
        }

        // Phase 2: REFINEMENT PHASE (Leiden's key innovation)
        // For each community found in local move phase, check if nodes
        // are well-connected. If a node has more connections outside its
        // community than inside, move it to form a new singleton.
        let mut refined_community = community.clone();
        let mut next_community_id = *community.iter().max().unwrap_or(&0) + 1;

        for i in 0..n {
            let node = NodeIndex::new(i);
            let c = community[i];
            let w_to_c = weights_to_communities(graph, node, &refined_community);
            let w_internal = w_to_c.get(&c).copied().unwrap_or(0.0);
            let ki = node_degree(graph, node);

            // If node has less than half its weight connected to its community,
            // it's poorly connected — split it off
            if w_internal < ki * 0.5 && ki > 0.0 {
                refined_community[i] = next_community_id;
                next_community_id += 1;
            }
        }

        community = refined_community;

        // Compact community IDs to be contiguous [0, k)
        let mut id_map = std::collections::HashMap::new();
        let mut next_id = 0usize;
        for c in &mut community {
            let new_id = id_map.len();
            *c = *id_map.entry(*c).or_insert(new_id);
            if *c == new_id {
                next_id = new_id + 1;
            }
        }
    }

    // Final compaction
    let mut id_map = std::collections::HashMap::new();
    for c in &mut community {
        let len = id_map.len();
        *c = *id_map.entry(*c).or_insert(len);
    }

    community
}

// ----- Polars plugin entry point -----

fn leiden_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "leiden".into(),
        DataType::List(Box::new(DataType::UInt32)),
    ))
}

#[polars_expr(output_type_func=leiden_output)]
fn leiden_clustering(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "leiden_clustering requires at least 2 inputs: [pca_coords, n_neighbors]. \
             Optional 3rd input: resolution (Float32, default 1.0)".into()
        ));
    }

    let pca_coords = &inputs[0].list()?;
    let k_neighbors = inputs[1].u32()?.get(0).unwrap_or(15) as usize;
    let resolution = if inputs.len() > 2 {
        inputs[2].f32()?.get(0).unwrap_or(1.0) as f64
    } else {
        1.0
    };

    // Validate parameters
    if k_neighbors == 0 || k_neighbors > 1000 {
        return Err(PolarsError::ComputeError(
            format!("n_neighbors must be 1-1000, got {}", k_neighbors).into()
        ));
    }
    if resolution <= 0.0 || !resolution.is_finite() {
        return Err(PolarsError::ComputeError(
            format!("resolution must be positive and finite, got {}", resolution).into()
        ));
    }

    let n_cells = pca_coords.len();
    if n_cells == 0 {
        return Err(PolarsError::ComputeError("Cannot cluster empty data".into()));
    }

    // 1. Unpack PCA coordinates
    let mut data: Vec<PcaPoint> = Vec::with_capacity(n_cells);
    
    // Check if doubly nested (1-row aggregate wrapper)
    let is_nested = pca_coords.get_as_series(0)
        .map(|s| s.list().is_ok())
        .unwrap_or(false);
    
    if is_nested {
        let inner_series = pca_coords.get_as_series(0)
            .ok_or_else(|| PolarsError::ComputeError("Empty PCA input".into()))?;
        let inner_list = inner_series.list()?;
        for opt_row in inner_list.into_iter() {
            if let Some(row) = opt_row {
                let coords: Vec<f32> = row.f32()?.into_no_null_iter().collect();
                data.push(PcaPoint(coords));
            }
        }
    } else {
        for opt_row in pca_coords.into_iter() {
            if let Some(row) = opt_row {
                let coords: Vec<f32> = row.f32()?.into_no_null_iter().collect();
                data.push(PcaPoint(coords));
            }
        }
    }

    let n = data.len();
    if n == 0 {
        return Err(PolarsError::ComputeError("No valid PCA data".into()));
    }

    let k = k_neighbors.min(n - 1);

    // 2. Build HNSW index for approximate KNN
    let hnsw: Hnsw<PcaPoint> = Builder::default()
        .ef_construction(200)  // Higher = better recall, slower build
        .build(data.iter().cloned(), data.len());

    // 3. Query KNN and build petgraph undirected weighted graph
    let mut graph = UnGraph::<(), f64>::new_undirected();
    for _ in 0..n {
        graph.add_node(());
    }

    let mut search = Search::default();
    for (i, point) in data.iter().enumerate() {
        search = hnsw.search(point, &mut search);
        for item in search.iter().take(k) {
            let j = item.pid.into_inner();
            if i < j {
                let dist = item.distance;
                // Gaussian kernel weight
                let weight = (-dist as f64 * dist as f64).exp();
                graph.add_edge(NodeIndex::new(i), NodeIndex::new(j), weight);
            }
        }
    }

    // 4. Run Leiden community detection
    let communities = leiden_partition(&graph, resolution, 10, 42);

    // 5. Return cluster labels wrapped as List(UInt32) for aggregate context
    let labels: Vec<u32> = communities.into_iter().map(|c| c as u32).collect();
    let inner = Series::new("leiden".into(), labels);
    let wrapped = Series::new("leiden".into(), &[AnyValue::List(inner)]);
    Ok(wrapped)
}

// Need this import for HNSW search
use instant_distance::Search;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leiden_simple() {
        // Build a simple graph with 2 clear clusters
        let mut graph = UnGraph::<(), f64>::new_undirected();
        for _ in 0..6 {
            graph.add_node(());
        }
        
        // Cluster 1: nodes 0, 1, 2 — strongly connected
        graph.add_edge(NodeIndex::new(0), NodeIndex::new(1), 1.0);
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(2), 1.0);
        graph.add_edge(NodeIndex::new(0), NodeIndex::new(2), 1.0);
        
        // Cluster 2: nodes 3, 4, 5 — strongly connected
        graph.add_edge(NodeIndex::new(3), NodeIndex::new(4), 1.0);
        graph.add_edge(NodeIndex::new(4), NodeIndex::new(5), 1.0);
        graph.add_edge(NodeIndex::new(3), NodeIndex::new(5), 1.0);
        
        // Weak inter-cluster edge
        graph.add_edge(NodeIndex::new(2), NodeIndex::new(3), 0.01);
        
        let communities = leiden_partition(&graph, 1.0, 10, 42);
        
        // Nodes in the same cluster should have the same community
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);
        assert_eq!(communities[3], communities[4]);
        assert_eq!(communities[4], communities[5]);
        
        // The two clusters should be different
        assert_ne!(communities[0], communities[3]);
    }

    #[test]
    fn test_leiden_empty() {
        let graph = UnGraph::<(), f64>::new_undirected();
        let communities = leiden_partition(&graph, 1.0, 10, 42);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_leiden_single_node() {
        let mut graph = UnGraph::<(), f64>::new_undirected();
        graph.add_node(());
        let communities = leiden_partition(&graph, 1.0, 10, 42);
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0], 0);
    }
}
