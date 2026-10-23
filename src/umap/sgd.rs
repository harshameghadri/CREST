//! UMAP layout optimisation (umap-learn `optimize_layout_euclidean` semantics),
//! with no per-sample allocation. umap-learn runs single-threaded whenever a
//! random_state is set (scanpy always sets one). CREST parallelises by giving
//! each persistent thread ownership of a contiguous range of source rows
//! (domain decomposition, as in parallel molecular dynamics): a thread only
//! writes the rows it owns and reads all other rows from a snapshot taken at
//! the start of each epoch. Work-stealing schedulers were ~8x slower here
//! because they migrate each chunk's working set between cores every epoch.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

#[inline(always)]
fn ld(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}
#[inline(always)]
fn st(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed)
}
/// Flushes denormals to zero on the current thread for as long as it is alive,
/// then restores the previous floating-point mode. UMAP's exp/pow terms produce
/// many subnormal floats, which are ~100x slower on x86 without FTZ/DAZ.
///
/// The control register (MXCSR) is per thread and is not saved across calls, so
/// it must be restored: the serial path runs on the Python caller's thread, and
/// leaving FTZ/DAZ set there would make every later numpy/scipy computation in
/// the process flush subnormals. RAII so a panic or early return restores it too.
struct FlushDenormals {
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}

impl FlushDenormals {
    #[inline]
    fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        #[allow(deprecated)]
        unsafe {
            use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
            let saved = _mm_getcsr();
            _mm_setcsr(saved | 0x8040); // FTZ | DAZ
            FlushDenormals { saved }
        }
        #[cfg(not(target_arch = "x86_64"))]
        FlushDenormals {}
    }
}

impl Drop for FlushDenormals {
    #[inline]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        #[allow(deprecated)]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[inline(always)]
fn clip(x: f32) -> f32 {
    x.clamp(-4.0, 4.0)
}

/// SplitMix64: cheap, well-mixed per-(edge, epoch) random stream.
#[inline(always)]
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Params<'a> {
    edges: &'a [(u32, u32, f32)],
    eps: &'a [f32],
    eps_neg: &'a [f32],
    pos: &'a [AtomicU32],
    snap: &'a [AtomicU32],
    dim: usize,
    n: usize,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    epoch: usize,
    seed: u64,
}

/// One chunk of edges for one epoch. `PAR = false`: umap-learn exact (move both
/// ends, live reads). `PAR = true`: move only the source row (the reverse edge
/// moves the target), positive samples twice per period so each node receives
/// the same expected force, and other rows are read from the epoch snapshot, so
/// threads never write rows they do not own nor read rows being written.
#[inline(never)]
fn run_chunk<const PAR: bool>(p: &Params, base: usize, nx: &mut [f32], nn: &mut [f32]) {
    let (dim, a, b, alpha) = (p.dim, p.a, p.b, p.alpha);
    let ep = p.epoch as f32;
    let other = |x: usize| if PAR { ld(&p.snap[x]) } else { ld(&p.pos[x]) };
    for t in 0..nx.len() {
        if nx[t] > ep {
            continue;
        }
        let e = base + t;
        let step = if PAR { 0.5 * p.eps[e] } else { p.eps[e] };
        let (i, j) = (p.edges[e].0 as usize * dim, p.edges[e].1 as usize * dim);
        let mut rs = p.seed ^ ((e as u64) << 21) ^ (p.epoch as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);

        while nx[t] <= ep {
            let mut d2 = 0.0f32;
            for d in 0..dim {
                let t = ld(&p.pos[i + d]) - other(j + d);
                d2 += t * t;
            }
            if d2 > 0.0 {
                let pb = (b * d2.ln()).exp();
                let gc = -2.0 * a * b * (pb / d2) / (a * pb + 1.0);
                for d in 0..dim {
                    let (ci, cj) = (ld(&p.pos[i + d]), other(j + d));
                    let g = clip(gc * (ci - cj)) * alpha;
                    st(&p.pos[i + d], ci + g);
                    if !PAR {
                        st(&p.pos[j + d], cj - g);
                    }
                }
            }
            nx[t] += step;
        }

        let n_neg = (((ep - nn[t]) / p.eps_neg[e]) as i64).max(0);
        for _ in 0..n_neg {
            let kk = (splitmix(&mut rs) % p.n as u64) as usize * dim;
            if kk == i {
                continue;
            }
            let mut d2 = 0.0f32;
            for d in 0..dim {
                let t = ld(&p.pos[i + d]) - other(kk + d);
                d2 += t * t;
            }
            if d2 > 0.0 {
                let gc = 2.0 * p.gamma * b / ((0.001 + d2) * (a * (b * d2.ln()).exp() + 1.0));
                for d in 0..dim {
                    let ci = ld(&p.pos[i + d]);
                    let g = clip(gc * (ci - other(kk + d))) * alpha;
                    st(&p.pos[i + d], ci + g);
                }
            }
        }
        nn[t] += n_neg as f32 * p.eps_neg[e];
    }
}

/// Optimise `embedding` (n × dim, row-major) in place.
pub fn optimize_layout(
    graph: &super::graph::UmapGraph,
    embedding: &mut [f32],
    dim: usize,
    n_epochs: usize,
    a: f32,
    b: f32,
    seed: u64,
) {
    let n = graph.n;
    if n == 0 || graph.edges.is_empty() || n_epochs == 0 {
        return;
    }
    let negative_sample_rate = 5.0f32;

    let wmax = graph.edges.iter().map(|e| e.weight).fold(0.0f32, f32::max);
    // umap-learn drops edges too weak to be sampled even once
    let mut edges: Vec<(u32, u32, f32)> = graph
        .edges
        .iter()
        .filter(|e| e.weight >= wmax / n_epochs as f32)
        .map(|e| (e.source as u32, e.target as u32, e.weight))
        .collect();
    edges.par_sort_unstable_by_key(|e| (e.0, e.1));

    let eps: Vec<f32> = edges.iter().map(|e| wmax / e.2).collect(); // n_epochs / (n_epochs * w / wmax)
    let eps_neg: Vec<f32> = eps.iter().map(|e| e / negative_sample_rate).collect();
    let mut next = eps.clone();
    let mut next_neg = eps_neg.clone();

    let pos: Vec<AtomicU32> = embedding.iter().map(|v| AtomicU32::new(v.to_bits())).collect();
    let chunk = 4096;
    // Parallel mode is deterministic for a fixed thread count: every thread writes
    // only its own rows and reads others from a per-epoch snapshot.
    let parallel = rayon::current_num_threads() > 1 && n >= 1000;

    let snap: Vec<AtomicU32> = embedding.iter().map(|v| AtomicU32::new(v.to_bits())).collect();
    if parallel {
        // Persistent threads, each owning a contiguous range of edges (hence of
        // source rows) for every epoch: data stays in that core's cache.
        let t = rayon::current_num_threads().max(1);
        let mut bounds = vec![0usize];
        for k in 1..t {
            let mut b = edges.len() * k / t;
            while b > 0 && b < edges.len() && edges[b].0 == edges[b - 1].0 {
                b += 1; // do not split a source row across threads
            }
            bounds.push(b.min(edges.len()));
        }
        bounds.push(edges.len());
        let barrier = std::sync::Barrier::new(t);
        // 16 snapshot refreshes per epoch: measured kNN-preservation 0.650 vs
        // umap-learn 0.655 (1 refresh: 0.607) at negligible cost.
        let substeps: usize = 16;
        let mut nx_parts: Vec<&mut [f32]> = Vec::new();
        let mut nn_parts: Vec<&mut [f32]> = Vec::new();
        {
            let (mut rx, mut rn): (&mut [f32], &mut [f32]) = (&mut next, &mut next_neg);
            for w in bounds.windows(2) {
                let (a1, b1) = rx.split_at_mut(w[1] - w[0]);
                let (a2, b2) = rn.split_at_mut(w[1] - w[0]);
                nx_parts.push(a1);
                nn_parts.push(a2);
                rx = b1;
                rn = b2;
            }
        }
        // rows owned by each thread: [first source, last source]
        let snap_plain: &[AtomicU32] = &snap;
        std::thread::scope(|sc| {
            for (tid, (nx, nn)) in nx_parts.into_iter().zip(nn_parts).enumerate() {
                let (lo, hi) = (bounds[tid], bounds[tid + 1]);
                let (edges, eps, eps_neg, pos, barrier) = (&edges, &eps, &eps_neg, &pos, &barrier);
                sc.spawn(move || {
                    // scoped threads exit at the end of the scope, but restore anyway
                    let _ftz = FlushDenormals::new();
                    let rows = if lo < hi {
                        (edges[lo].0 as usize * dim)..((edges[hi - 1].0 as usize + 1) * dim)
                    } else {
                        0..0
                    };
                    for epoch in 0..n_epochs {
                        let alpha = 1.0 - epoch as f32 / n_epochs as f32;
                        // several snapshot refreshes per epoch keep reads fresh
                        for sub in 0..substeps {
                            // publish own rows, then everyone reads the shared snapshot
                            // (no per-thread copy of all n rows); the second barrier stops
                            // the next refresh from overwriting rows still being read
                            for r in rows.clone() {
                                st(&snap_plain[r], ld(&pos[r]));
                            }
                            barrier.wait();
                            let len = hi - lo;
                            let (s0, s1) = (len * sub / substeps, len * (sub + 1) / substeps);
                            let p = Params {
                                edges, eps, eps_neg, pos, snap: snap_plain, dim, n, a, b, gamma: 1.0,
                                alpha, epoch, seed,
                            };
                            run_chunk::<true>(&p, lo + s0, &mut nx[s0..s1], &mut nn[s0..s1]);
                            barrier.wait();
                        }
                    }
                });
            }
        });
    } else {
        let empty: Vec<AtomicU32> = Vec::new();
        // the caller's thread (the Python thread): restored when the loop ends
        let _ftz = FlushDenormals::new();
        for epoch in 0..n_epochs {
            let p = Params {
                edges: &edges, eps: &eps, eps_neg: &eps_neg, pos: &pos, snap: &empty, dim, n, a, b, gamma: 1.0,
                alpha: 1.0 - epoch as f32 / n_epochs as f32, epoch, seed,
            };
            for (ci, (nx, nn)) in next.chunks_mut(chunk).zip(next_neg.chunks_mut(chunk)).enumerate() {
                run_chunk::<false>(&p, ci * chunk, nx, nn);
            }
        }
    }
    for (v, a) in embedding.iter_mut().zip(&pos) {
        *v = ld(a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::{Edge, UmapGraph};

    #[test]
    fn test_optimize_layout_attracts_neighbours() {
        // two 10-node cliques far apart in the init must stay separated and
        // each clique must contract
        let mut edges = vec![];
        for base in [0usize, 10] {
            for a in 0..10 {
                for b in 0..10 {
                    if a != b {
                        edges.push(Edge { source: base + a, target: base + b, weight: 1.0 });
                    }
                }
            }
        }
        let graph = UmapGraph { n: 20, edges };
        let mut emb: Vec<f32> = (0..20).flat_map(|i| {
            let off = if i < 10 { 0.0 } else { 20.0 };
            vec![off + (i % 10) as f32, (i * 3 % 7) as f32]
        }).collect();
        let spread = |e: &[f32], lo: usize| {
            let pts: Vec<(f32, f32)> = (lo..lo + 10).map(|i| (e[i * 2], e[i * 2 + 1])).collect();
            let cx = pts.iter().map(|p| p.0).sum::<f32>() / 10.0;
            let cy = pts.iter().map(|p| p.1).sum::<f32>() / 10.0;
            pts.iter().map(|p| (p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sum::<f32>()
        };
        let before = spread(&emb, 0);
        let (a, b) = super::super::utils::find_ab_params(1.0, 0.1);
        optimize_layout(&graph, &mut emb, 2, 200, a, b, 0);
        assert!(emb.iter().all(|v| v.is_finite()));
        assert!(spread(&emb, 0) < before);
    }

    /// The FTZ/DAZ mode must not leak to the caller's thread (the Python thread on
    /// the serial path, n < 1000) nor to rayon workers.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(deprecated)]
    fn test_optimize_layout_restores_mxcsr() {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let mut edges = vec![];
        for a in 0..20usize {
            for b in 0..20usize {
                if a != b && (a < 10) == (b < 10) {
                    edges.push(Edge { source: a, target: b, weight: 1.0 });
                }
            }
        }
        let graph = UmapGraph { n: 20, edges };
        let (a, b) = super::super::utils::find_ab_params(1.0, 0.1);
        let original = unsafe { _mm_getcsr() };
        // a sticky flag that the guard must also give back
        let before = (original & !0x8040) | 0x20;
        unsafe { _mm_setcsr(before) };
        // workers: compare the control bits only (FTZ, DAZ, rounding, masks); their
        // sticky exception flags change with any float work other tests run there
        let control = |_: rayon::BroadcastContext| unsafe { _mm_getcsr() } & !0x3f;
        let workers_before: Vec<u32> = rayon::broadcast(control);
        let mut emb: Vec<f32> = (0..40).map(|i| (i % 7) as f32).collect();
        optimize_layout(&graph, &mut emb, 2, 50, a, b, 0); // serial path (n < 1000)
        let after = unsafe { _mm_getcsr() };
        let workers_after: Vec<u32> = rayon::broadcast(control);
        unsafe { _mm_setcsr(original) };
        assert_eq!(after, before, "MXCSR changed on the caller's thread: {:#x} -> {:#x}", before, after);
        assert_eq!(workers_after, workers_before, "MXCSR changed on rayon workers");
        // and the guard does set FTZ|DAZ while alive
        let g = FlushDenormals::new();
        assert_eq!(unsafe { _mm_getcsr() } & 0x8040, 0x8040);
        drop(g);
        assert_eq!(unsafe { _mm_getcsr() }, original);
    }
}
