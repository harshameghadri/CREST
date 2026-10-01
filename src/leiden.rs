use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;

// ----- Leiden Algorithm Implementation -----
// Reference: Traag, Waltman & van Eck (2019) "From Louvain to Leiden:
// guaranteeing well-connected communities", Sci. Rep. 9:5233.
//
// Quality function: modularity with a resolution parameter (RB configuration model),
//   Q = 1/(2m) * sum_ij [A_ij - gamma * k_i k_j / (2m)] delta(c_i, c_j)
// Moving node v into community C changes Q proportionally to
//   w(v, C) - gamma * k_v * K_C / (2m)
// where w(v, C) is the edge weight from v to C and K_C the summed degree of C.
//
// Each level: (1) fast local moving, (2) refinement of every community by
// randomised merging of well-connected singletons, (3) aggregation of the graph
// by the refined partition, with the unrefined partition as the aggregate's start.

/// Undirected weighted graph in CSR form (both directions stored, no self-loops).
/// `node_w` carries each node's weighted degree, including edges that became
/// internal (self-loops) during aggregation, so it is kept separately.
#[derive(Clone)]
pub(crate) struct Graph {
    offsets: Vec<usize>,
    nbrs: Vec<usize>,
    wts: Vec<f64>,
    node_w: Vec<f64>,
    two_m: f64,
}

impl Graph {
    fn n(&self) -> usize {
        self.node_w.len()
    }

    /// Build from undirected edges (i, j, w) with i != j, each pair listed once.
    pub(crate) fn from_edges<W: Copy + Into<f64>>(n: usize, edges: &[(usize, usize, W)]) -> Self {
        let mut node_w = vec![0.0f64; n];
        let mut deg = vec![0usize; n];
        for &(i, j, w) in edges {
            let w: f64 = w.into();
            node_w[i] += w;
            node_w[j] += w;
            deg[i] += 1;
            deg[j] += 1;
        }
        let mut offsets = vec![0usize; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + deg[i];
        }
        let mut pos = offsets[..n].to_vec();
        let mut nbrs = vec![0usize; offsets[n]];
        let mut wts = vec![0.0f64; offsets[n]];
        for &(i, j, w) in edges {
            let w: f64 = w.into();
            nbrs[pos[i]] = j;
            wts[pos[i]] = w;
            pos[i] += 1;
            nbrs[pos[j]] = i;
            wts[pos[j]] = w;
            pos[j] += 1;
        }
        let two_m = node_w.iter().sum();
        Graph { offsets, nbrs, wts, node_w, two_m }
    }

    #[inline]
    fn neighbors(&self, v: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.offsets[v]..self.offsets[v + 1];
        self.nbrs[r.clone()].iter().copied().zip(self.wts[r].iter().copied())
    }
}

/// Dense scratch accumulator of edge weight from one node to each community.
struct NeighborWeights {
    w: Vec<f64>,
    touched: Vec<usize>,
}

impl NeighborWeights {
    fn new(n: usize) -> Self {
        NeighborWeights { w: vec![0.0; n], touched: Vec::new() }
    }
    #[inline]
    fn add(&mut self, c: usize, w: f64) {
        if self.w[c] == 0.0 {
            self.touched.push(c);
        }
        self.w[c] += w;
    }
    fn clear(&mut self) {
        for &c in &self.touched {
            self.w[c] = 0.0;
        }
        self.touched.clear();
    }
}

/// Relabel communities to 0..k in order of first appearance; returns k.
fn renumber(part: &mut [usize]) -> usize {
    let mut map = vec![usize::MAX; part.len().max(1)];
    let mut next = 0;
    for c in part.iter_mut() {
        if map[*c] == usize::MAX {
            map[*c] = next;
            next += 1;
        }
        *c = map[*c];
    }
    next
}

/// Phase 1: queue-based fast local moving (Traag et al. 2019, Alg. A.2).
fn fast_move_nodes(g: &Graph, part: &mut [usize], gamma: f64, rng: &mut StdRng) {
    let n = g.n();
    let mut comm_w = vec![0.0f64; n];
    let mut comm_size = vec![0usize; n];
    for v in 0..n {
        comm_w[part[v]] += g.node_w[v];
        comm_size[part[v]] += 1;
    }
    let mut empty: Vec<usize> = (0..n).filter(|&c| comm_size[c] == 0).collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);
    let mut queue: VecDeque<usize> = order.into_iter().collect();
    let mut in_queue = vec![true; n];
    let mut nw = NeighborWeights::new(n);
    let scale = gamma / g.two_m;

    while let Some(v) = queue.pop_front() {
        in_queue[v] = false;
        let cur = part[v];
        let kv = g.node_w[v];

        for (u, w) in g.neighbors(v) {
            nw.add(part[u], w);
        }
        comm_w[cur] -= kv;
        comm_size[cur] -= 1;

        let mut best = cur;
        let mut best_gain = nw.w[cur] - kv * comm_w[cur] * scale;
        for &c in &nw.touched {
            let gain = nw.w[c] - kv * comm_w[c] * scale;
            if gain > best_gain + 1e-12 {
                best_gain = gain;
                best = c;
            }
        }
        // Moving to an empty community has gain 0.
        if best_gain < -1e-12 && comm_size[cur] > 0 {
            while let Some(c) = empty.pop() {
                if comm_size[c] == 0 {
                    best = c;
                    break;
                }
            }
        }
        nw.clear();

        part[v] = best;
        comm_w[best] += kv;
        comm_size[best] += 1;
        if best != cur {
            if comm_size[cur] == 0 {
                empty.push(cur);
            }
            for (u, _) in g.neighbors(v) {
                if !in_queue[u] && part[u] != best {
                    in_queue[u] = true;
                    queue.push_back(u);
                }
            }
        }
    }
}

/// SplitMix64: mix two integers into a well-distributed 64-bit seed, so each
/// community gets an independent-looking RNG stream from one base seed.
#[inline]
fn mix_seed(a: u64, b: u64) -> u64 {
    let mut z = a.wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Phase 2: refine each community of `part` by merging singletons into
/// well-connected sub-communities (Traag et al. 2019, Alg. A.2 MergeNodesSubset).
///
/// A merge only ever pulls a node into a sub-community of its *own* unrefined
/// community S (every candidate is filtered by `part[u] == s`), so communities
/// refine independently of one another. This runs each one on its own thread,
/// with its own RNG seeded from `(seed, community id)`: the result is the same
/// for any thread count, since which thread handles a community never affects
/// its RNG stream.
fn refine_partition(g: &Graph, part: &[usize], gamma: f64, theta: f64, seed: u64) -> Vec<usize> {
    let n = g.n();
    let scale = gamma / g.two_m;
    let k = part.iter().max().map(|m| m + 1).unwrap_or(0);

    // Group nodes by their unrefined community, and give each node a local
    // index within that group (its position in `members[part[v]]`).
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut local_idx = vec![0usize; n];
    for v in 0..n {
        local_idx[v] = members[part[v]].len();
        members[part[v]].push(v);
    }
    let mut s_w = vec![0.0f64; k];
    for v in 0..n {
        s_w[part[v]] += g.node_w[v];
    }

    let local: Vec<Vec<usize>> = members
        .par_iter()
        .enumerate()
        .map(|(s, nodes)| {
            refine_one_community(g, nodes, &local_idx, part, s_w[s], scale, theta, mix_seed(seed, s as u64))
        })
        .collect();

    let mut refined = vec![0usize; n];
    for (nodes, local_refined) in members.iter().zip(local.iter()) {
        for (i, &v) in nodes.iter().enumerate() {
            refined[v] = nodes[local_refined[i]];
        }
    }
    refined
}

/// Refine a single unrefined community `nodes` (Traag et al. 2019, Alg. A.2
/// MergeNodesSubset). `local_idx[u]` gives `u`'s position in `nodes` for any
/// `u` with `part[u]` equal to this community. Returns, for each position `i`
/// in `nodes`, the local index of the refined sub-community `nodes[i]` joined.
#[allow(clippy::too_many_arguments)]
fn refine_one_community(
    g: &Graph,
    nodes: &[usize],
    local_idx: &[usize],
    part: &[usize],
    s_w: f64,
    scale: f64,
    theta: f64,
    seed: u64,
) -> Vec<usize> {
    let m = nodes.len();
    if m == 0 {
        return Vec::new();
    }
    let s = part[nodes[0]];
    let mut rng = StdRng::seed_from_u64(seed);

    let mut refined: Vec<usize> = (0..m).collect();
    let mut ref_w: Vec<f64> = nodes.iter().map(|&v| g.node_w[v]).collect();
    let mut ref_size = vec![1usize; m];
    // ext[i]: weight from refined sub-community i to the rest of S.
    let mut ext = vec![0.0f64; m];
    for (i, &v) in nodes.iter().enumerate() {
        for (u, w) in g.neighbors(v) {
            if part[u] == s {
                ext[i] += w;
            }
        }
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.shuffle(&mut rng);
    let mut nw = NeighborWeights::new(m);
    let mut cands: Vec<(usize, f64)> = Vec::new();

    for i in order {
        if ref_size[refined[i]] != 1 {
            continue; // only singletons are moved
        }
        let v = nodes[i];
        let kv = g.node_w[v];
        // v must itself be well connected to S.
        if ext[i] < kv * (s_w - kv) * scale {
            continue;
        }

        for (u, w) in g.neighbors(v) {
            if part[u] == s {
                nw.add(refined[local_idx[u]], w);
            }
        }

        let own = refined[i];
        cands.clear();
        cands.push((own, 0.0));
        for &c in &nw.touched {
            if c == own {
                continue;
            }
            // Only merge into well-connected refined sub-communities.
            if ext[c] < ref_w[c] * (s_w - ref_w[c]) * scale {
                continue;
            }
            let gain = nw.w[c] - kv * ref_w[c] * scale;
            if gain >= 0.0 {
                cands.push((c, gain));
            }
        }

        let chosen = if cands.len() == 1 {
            own
        } else {
            // Randomised choice, Pr(C) ∝ exp(gain / theta).
            let gmax = cands.iter().map(|x| x.1).fold(f64::MIN, f64::max);
            let total: f64 = cands.iter().map(|x| ((x.1 - gmax) / theta).exp()).sum();
            let mut r = rng.gen::<f64>() * total;
            let mut pick = cands[cands.len() - 1].0;
            for &(c, gain) in &cands {
                r -= ((gain - gmax) / theta).exp();
                if r <= 0.0 {
                    pick = c;
                    break;
                }
            }
            pick
        };

        if chosen != own {
            let w_vc = nw.w[chosen];
            ext[chosen] = ext[chosen] + ext[own] - 2.0 * w_vc;
            ref_w[chosen] += kv;
            ref_size[chosen] += 1;
            ref_w[own] = 0.0;
            ref_size[own] = 0;
            refined[i] = chosen;
        }
        nw.clear();
    }
    refined
}

/// Phase 3: collapse each community of `by` (contiguous ids 0..k) into one node.
fn aggregate(g: &Graph, by: &[usize], k: usize) -> Graph {
    let n = g.n();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for v in 0..n {
        members[by[v]].push(v);
    }
    let mut node_w = vec![0.0f64; k];
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut nw = NeighborWeights::new(k);
    for c in 0..k {
        for &v in &members[c] {
            node_w[c] += g.node_w[v];
            for (u, w) in g.neighbors(v) {
                let d = by[u];
                if d != c {
                    nw.add(d, w);
                }
            }
        }
        for &d in &nw.touched {
            if d > c {
                edges.push((c, d, nw.w[d]));
            }
        }
        nw.clear();
    }
    let mut agg = Graph::from_edges(k, &edges);
    agg.node_w = node_w; // keep internal weight (self-loops) in degrees
    agg.two_m = g.two_m;
    agg
}

/// Leiden community detection. Returns a community id per node (0..k).
pub(crate) fn leiden_partition(
    graph: &Graph,
    resolution: f64,
    n_iterations: usize,
    seed: u64,
) -> Vec<usize> {
    let n = graph.n();
    if n == 0 {
        return vec![];
    }
    if graph.two_m <= 0.0 {
        return (0..n).collect();
    }
    let theta = 0.01;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result: Vec<usize> = (0..n).collect();

    for _ in 0..n_iterations.max(1) {
        // level 0 works on the caller's graph directly (no copy: a resolution sweep runs
        // several of these in parallel on one shared graph); later levels on aggregates
        let mut agg: Option<Graph> = None;
        let mut membership: Vec<usize> = (0..n).collect(); // original node -> aggregate node
        let mut part = result.clone(); // partition of aggregate nodes
        renumber(&mut part);

        for _level in 0..1000 {
            let g: &Graph = agg.as_ref().unwrap_or(graph);
            fast_move_nodes(g, &mut part, resolution, &mut rng);
            let n_comms = renumber(&mut part);
            if n_comms == g.n() {
                break;
            }
            let refine_seed = rng.gen::<u64>();
            let mut refined = refine_partition(g, &part, resolution, theta, refine_seed);
            let mut n_ref = renumber(&mut refined);
            if n_ref == g.n() {
                // Refinement merged nothing: aggregate by the unrefined partition.
                refined = part.clone();
                n_ref = n_comms;
            }
            let mut next_part = vec![0usize; n_ref];
            for v in 0..g.n() {
                next_part[refined[v]] = part[v];
            }
            for m in membership.iter_mut() {
                *m = refined[*m];
            }
            let next_g = aggregate(g, &refined, n_ref);
            agg = Some(next_g);
            part = next_part;
        }

        let mut next: Vec<usize> = membership.iter().map(|&m| part[m]).collect();
        renumber(&mut next);
        let changed = next != result;
        result = next;
        if !changed {
            break;
        }
    }
    result
}

/// Modularity of a partition (for tests and diagnostics).
#[cfg(test)]
fn modularity(g: &Graph, part: &[usize], gamma: f64) -> f64 {
    let k = part.iter().max().map(|m| m + 1).unwrap_or(0);
    let mut inside = vec![0.0f64; k];
    let mut tot = vec![0.0f64; k];
    for v in 0..g.n() {
        tot[part[v]] += g.node_w[v];
        for (u, w) in g.neighbors(v) {
            if part[u] == part[v] {
                inside[part[v]] += w;
            }
        }
    }
    (0..k).map(|c| inside[c] / g.two_m - gamma * (tot[c] / g.two_m).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_triangles() -> Graph {
        Graph::from_edges(6, &[
            (0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0),
            (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0),
            (2, 3, 0.01),
        ])
    }

    #[test]
    fn test_leiden_simple() {
        let communities = leiden_partition(&two_triangles(), 1.0, 2, 42);
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);
        assert_eq!(communities[3], communities[4]);
        assert_eq!(communities[4], communities[5]);
        assert_ne!(communities[0], communities[3]);
    }

    #[test]
    fn test_leiden_empty() {
        let g = Graph::from_edges::<f64>(0, &[]);
        assert!(leiden_partition(&g, 1.0, 2, 42).is_empty());
    }

    #[test]
    fn test_leiden_single_node() {
        let g = Graph::from_edges::<f64>(1, &[]);
        assert_eq!(leiden_partition(&g, 1.0, 2, 42), vec![0]);
    }

    /// Planted partition: 8 dense blocks of 50 nodes with sparse noise between them.
    /// Leiden must recover the blocks exactly and every community must be connected.
    #[test]
    fn test_leiden_planted_partition() {
        let mut rng = StdRng::seed_from_u64(1);
        let (blocks, size) = (8usize, 50usize);
        let n = blocks * size;
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let same = i / size == j / size;
                let p = if same { 0.3 } else { 0.005 };
                if rng.gen::<f64>() < p {
                    edges.push((i, j, 1.0));
                }
            }
        }
        let g = Graph::from_edges(n, &edges);
        let part = leiden_partition(&g, 1.0, 2, 0);
        let k = part.iter().max().unwrap() + 1;
        assert_eq!(k, blocks, "expected {} communities, got {}", blocks, k);
        for b in 0..blocks {
            let c = part[b * size];
            assert!((b * size..(b + 1) * size).all(|v| part[v] == c));
        }
        let truth: Vec<usize> = (0..n).map(|v| v / size).collect();
        assert!((modularity(&g, &part, 1.0) - modularity(&g, &truth, 1.0)).abs() < 1e-9);
    }

    /// Leiden's guarantee: every returned community is internally connected.
    #[test]
    fn test_leiden_communities_connected() {
        let mut rng = StdRng::seed_from_u64(3);
        let n = 300;
        let mut edges = Vec::new();
        for i in 0..n {
            for _ in 0..5 {
                let j = rng.gen_range(0..n);
                if j != i {
                    edges.push((i.min(j), i.max(j), rng.gen_range(0.1..1.0)));
                }
            }
        }
        edges.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        let g = Graph::from_edges(n, &edges);
        let part = leiden_partition(&g, 1.0, 2, 7);
        let k = part.iter().max().unwrap() + 1;
        for c in 0..k {
            let nodes: Vec<usize> = (0..n).filter(|&v| part[v] == c).collect();
            let mut seen = vec![false; n];
            let mut stack = vec![nodes[0]];
            seen[nodes[0]] = true;
            let mut count = 1;
            while let Some(v) = stack.pop() {
                for (u, _) in g.neighbors(v) {
                    if part[u] == c && !seen[u] {
                        seen[u] = true;
                        count += 1;
                        stack.push(u);
                    }
                }
            }
            assert_eq!(count, nodes.len(), "community {} is disconnected", c);
        }
        assert!(modularity(&g, &part, 1.0) > 0.3);
    }
}
