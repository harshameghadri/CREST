# CREST: Updated Production Action Plan

**Date**: 2026-04-04
**Status**: Supersedes PRODUCTION_ACTION_PLAN.md (2026-03-22)

---

## What's Changed Since Last Plan

### Done (on `main` branch)
- Leiden algorithm (Rust, petgraph) — replaces graphrs Louvain
- Halko randomized SVD (Rust, faer 0.24) — replaces single-svdlib
- HNSW approximate KNN (Rust, instant-distance 0.6.1) — replaces kiddo KD-tree
- Leiden output wrapping (List(UInt32) for aggregate context)
- PCA unpacking in all consumers (leiden, neighbors, umap_binding)
- Streaming incremental PCA (Python, sklearn + DuckDB)
- Landmark UMAP (Python, umap-learn)
- Hierarchical Leiden (Python, label transfer)
- Benchmarking scaffolding (download, metrics, suite)

### Done This Session
- **Fixed 29 compile errors**: faer 0.24 API (`compute_thin_Q`, `thin_svd` Result, `U()`/`S()`), instant-distance 0.6.1 API (`build_hnsw`, iterator-based search), nalgebra-sparse `triplet_iter` return types
- **CI fix**: Removed s390x/ppc64le targets (psm assembly errors on exotic arches), release trigger locked to tags only
- **CLAUDE.md**: Added to .gitignore, removed from tracking
- **20/20 tests pass** on `cargo test --release`

### Blocking Issues (from go_no_go_strategy.md)
- CI still needs the compile fixes pushed (this session's work)
- `Array(Float32, K)` migration deferred — Polars Array dtype needs compile-time width, `n_comps` is runtime

---

## New: Dual Louvain + Leiden (Like Seurat)

**Design principle**: Don't hard-replace Louvain. Both Seurat and Scanpy offer both algorithms. The bioinformatician chooses.

### API
```python
# Leiden (default, recommended)
df.select(pl.col("pca").bio.leiden(n_neighbors=15, resolution=1.0))

# Louvain (kept for backward compatibility and comparison)
df.select(pl.col("pca").bio.louvain(n_neighbors=15, resolution=1.0))
```

### Implementation
- `src/leiden.rs` — Leiden with refinement step (current, done)
- `src/louvain.rs` — Louvain WITHOUT refinement (extract from leiden.rs, skip Phase 2)
- Both share: HNSW KNN build, graph construction, modularity gain calculation
- Both return `List(UInt32)` wrapped output
- Python `__init__.py` already has `bio.louvain()` (currently marked deprecated → change to "alternative")

### Effort: ~2 hours (extract shared KNN→graph code, duplicate leiden_partition minus refinement)

---

## New: Harmony Batch Integration

### Why
The 320K FLEX dataset has 8 tissue types from different FFPE blocks. Without batch correction, tissue-of-origin dominates the UMAP/clustering. Harmony is the standard for fast linear batch correction.

### Harmony Algorithm (from source analysis)

**Total C++: 512 lines. Core math: 3 operations repeated until convergence.**

```
Input:  Z (N cells × d PCA dims), Phi (N × B batch indicator, sparse)
Output: Z_corrected (N × d)

1. INIT: K-means on cosine-normalized Z → soft cluster assignment R (N × K)
2. CLUSTER: Update R via batch-penalized soft-max:
   R[k,i] ∝ exp(-dist[k,i]/sigma[k]) × (expected[k,b] / observed[k,b])^theta
   where theta penalizes batch-enriched clusters
3. CORRECT: For each cluster k, ridge regression removes batch effect:
   W_k = (Phi_Rk @ Phi_Rk^T + lambda*I)^-1 @ Phi_Rk @ Z^T
   Z_corrected -= W_k^T @ Phi_Rk
4. Repeat steps 2-3 until convergence (typically 5-10 rounds)
```

### Rust Port Plan

**File**: `src/harmony.rs` (~800-1200 lines)

| Component | Rust Crate | Notes |
|---|---|---|
| Dense matrix ops | `faer` 0.24 (already in Cargo.toml) | MatMul, transpose, solve |
| Ridge regression | `faer` Cholesky solve | (A^T A + lambda I)^-1 A^T b |
| K-means init | `rand` + manual implementation | ~50 lines, uses cosine distance |
| Sparse batch indicator | `nalgebra-sparse` (already in Cargo.toml) | CSR for Phi matrix |
| Softmax / entropy | Manual | ~20 lines, with NaN guards |

**Key insight**: Harmony operates on the PCA embedding (N × d dense, d=20-50), NOT on the raw expression matrix. At 100K cells × 50 dims = 40 MB — fits entirely in memory. No streaming needed.

### API Design
```python
# Harmony integration on PCA coordinates
# Input: DataFrame with "pca" (List(Float32)) and "batch" (UInt32 or Utf8) columns
corrected = pca_df.with_columns(
    pl.col("pca").bio.harmony(
        pl.col("batch"),
        n_clusters=20,       # K for soft clustering (default: auto from data)
        theta=2.0,           # Diversity penalty (default: 2.0)
        max_iter=10,         # Outer iterations (default: 10)
        sigma=0.1,           # Cluster width (default: 0.1)
    ).alias("pca_corrected")
)

# Then proceed with corrected PCA for downstream
corrected.select(pl.col("pca_corrected").bio.leiden(n_neighbors=15))
```

### Numerical Stability Concerns
1. **Ridge solve**: Use Cholesky (faer has robust implementation), not matrix inverse
2. **Softmax overflow**: Subtract max before exp
3. **Log(0) in entropy**: Guard with `max(x, 1e-10)` before log
4. **Convergence**: Track objective, stop when relative change < 1e-5

### Estimated Effort: 1-2 weeks for Rust port + validation

---

## New: 10x GEM-X FLEX 320K Benchmark

### Dataset Overview
8 tissues from FFPE blocks, 16-plex GEM-X FLEX, ~320K total cells:

| Tissue | Cells | Use |
|---|---|---|
| Breast Cancer | 15,136 | |
| Glioblastoma | 33,323 | Single-tissue benchmark |
| Kidney | 33,690 | Single-tissue benchmark (default) |
| Lymph Node | 31,603 | Multi-tissue integration |
| Skin Melanoma | 45,153 | |
| Endo | 44,556 | |
| Colorectal | 52,541 | |
| Lung Cancer | 69,343 | Scale test |

### Phase A: Single-tissue validation (no integration needed)
- **Tissue**: Kidney (33,690 cells)
- **Download**: filtered_feature_bc_matrix.h5 + count_analysis.tar.gz
- **Three-way comparison**: CREST vs Scanpy vs Cell Ranger
- **Script**: `bench/validate_flex_320k.py --tissue kidney`

### Phase B: Multi-tissue integration (requires Harmony)
- **Tissues**: Glioblastoma + Kidney + Lymph Node (≈98K cells)
- **Pipeline**: Load → Normalize → HVG → PCA → Harmony → Leiden → UMAP
- **North star**: Cell Ranger's per-sample analysis as baseline

### Metrics
| Metric | Target | Notes |
|---|---|---|
| PCA cosine similarity (top 10 PCs) | ≥ 0.95 | CREST vs Scanpy, sign-invariant |
| ARI (CREST vs Scanpy Leiden) | ≥ 0.5 | Same resolution, stochastic → lower bound |
| ARI (CREST vs Cell Ranger) | ≥ 0.3 | Different algorithms, different preprocessing |
| Total pipeline speedup | > 1x | CREST should be faster on per-step basis |
| Memory (peak RSS) | ≤ 2× Scanpy | Guard against OOM |

---

## Execution Priority

### Week 1 (immediate)
1. Push compile fixes + CI fix to main → green CI
2. Run `bench/validate_flex_320k.py --tissue kidney` in Docker
3. Extract Louvain from Leiden (dual algorithm support)
4. Commit + push

### Week 2
5. Implement `src/harmony.rs` — core algorithm (init, cluster, correct)
6. Add `bio.harmony()` to Python API
7. Validate on 2-tissue integration (Glioblastoma + Kidney)

### Week 3
8. Run full 3-tissue integration benchmark (98K cells)
9. Compare with scanpy + harmonypy reference
10. Publication-quality figures

### Deferred
- Array(Float32, K) migration (blocked by runtime n_comps)
- Streaming Rust SVD (Python path works for now)
- Landmark UMAP at 50M scale
- Hierarchical Leiden at 50M scale
- HNSW sharding

---

## Files Modified/Created This Session

| File | Change |
|---|---|
| `src/svd.rs` | Fixed faer 0.24 API (QR, SVD), nalgebra-sparse triplet_iter |
| `src/leiden.rs` | Fixed instant-distance 0.6.1 API (build_hnsw, iterator search) |
| `src/neighbors.rs` | Fixed instant-distance 0.6.1 API |
| `src/umap/graph.rs` | Fixed instant-distance 0.6.1 API |
| `.github/workflows/CI.yml` | Removed s390x/ppc64le, locked release to tags |
| `.gitignore` | Added CLAUDE.md |
| `bench/validate_flex_320k.py` | New: 10x FLEX 320K validation benchmark |
| `crest/UPDATED_ACTION_PLAN.md` | This file |
