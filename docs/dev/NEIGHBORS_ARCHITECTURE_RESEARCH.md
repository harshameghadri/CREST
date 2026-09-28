# CREST Neighbors: Data Layout Architecture Research

**Date**: 2026-03-22
**Scope**: How to most efficiently pass PCA coordinates from Python to the Rust KNN plugin in CREST
**Trigger**: Scheduled research task — compare `List(List(Float32))` against alternatives

---

## 1. Current State: How CREST Uses `List(List(Float32))`

The current pipeline produces `List(List(Float32))` for all 2D matrix outputs (PCA, UMAP). The wrapping pattern is consistent across `svd.rs`, `umap_binding.rs`, and `neighbors.rs`:

```rust
// Build n_cells × k matrix
let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
    "pca", n_cells, k, DataType::Float32
);
for row in cells {
    builder.append_slice(&row_coords);
}
let s_orig = builder.finish().into_series();
// Wrap in length-1 outer list (required by aggregate plugin API)
let s_wrapped = Series::new("pca".into(), &[AnyValue::List(s_orig)]);
```

The outer wrapping in a length-1 Series is a **plugin API constraint**, not a data layout choice — aggregate plugins must return length-1 output, and the `AnyValue::List` wrapper is how pyo3-polars encodes "one row whose value is a list of all cells' vectors."

So the actual data structure to evaluate is the **inner** representation: for 20,000 cells × 50 PCA components, what is the best dtype for each cell's vector?

---

## 2. Alternatives Evaluated

### 2.1 `List(Float32)` — Current Approach

**How it works in Arrow:**
Variable-length list type. Stores:
1. An **offsets buffer** of `(N+1)` int32 values (4 bytes each) — maps row index to value offset
2. A **values buffer** of `N × K` float32 values

For 20,000 cells × 50 components:
- Offsets: 20,001 × 4 = ~80 KB
- Values: 20,000 × 50 × 4 = ~3.8 MB
- **Total overhead from offsets: ~2% of value data**

Random access to cell `j`'s vector requires: read `offsets[j]` → slice `values[offsets[j]..offsets[j+1]]`. This is one extra memory fetch (offset lookup) before reaching the data, which likely causes a cache miss on a cold access pattern.

**Variable-length penalty:** The type system does NOT guarantee all inner lists have the same length. This means:
- Polars cannot statically verify uniform width
- Any pass through `list().get_as_series()` must be defensive against varying lengths
- The KD-tree builder in `neighbors.rs` and `leiden.rs` already handles this with a `for (j, val) in float_ca.into_no_null_iter().enumerate() { if j >= MAX_DIMS { break; } }` pattern

**Cache locality for KNN:** KNN iterates over all N cells sequentially to build the KD-tree. The inner list values ARE stored contiguously in the values buffer (all 1,000,000 floats in one flat buffer), so sequential cell iteration is cache-friendly IF the entire values buffer fits in L3 cache. The offsets buffer is small (~80 KB) and will stay warm.

---

### 2.2 `Array(Float32, K)` — Arrow FixedSizeList (Recommended)

**How it works in Arrow:**
Fixed-size list type. Stores:
1. **No offsets buffer** — row `j` is always at position `j × K` in the values buffer
2. A **values buffer** of `N × K` float32 values (identical to List)

For 20,000 cells × 50 components:
- No offsets buffer
- Values: 20,000 × 50 × 4 = ~3.8 MB
- **Zero structural overhead** beyond the values themselves

**pyo3-polars support:** Yes, fully supported in Polars ≥ 0.16 via the `dtype-array` feature (which biopolars already enables via polars dependency). The `output_type_func` can return:

```rust
pub fn pca_array_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "pca",
        DataType::Array(Box::new(DataType::Float32), 50),
    ))
}
```

**Accessing rows in Rust:**

```rust
let arr_ca = inputs[0].array()?;  // ArrayChunked
for chunk in arr_ca.downcast_iter() {
    // chunk is FixedSizeListArray
    for i in 0..chunk.len() {
        let row: &[f32] = chunk.value(i)  // zero-copy slice into contiguous buffer
            .as_any()
            .downcast_ref::<PrimitiveArray<f32>>()
            .unwrap()
            .values()
            .as_ref();
        // row is exactly K elements, contiguous
    }
}
```

**Reference implementation:** `polars-distance` crate uses `Array(Float32/Float64, K)` for all per-row vector distance computations. Their `infer_distance_arr_output` pattern is the canonical pyo3-polars approach for fixed-width per-row vectors.

**Cache locality for KNN:** Slightly better than `List(Float32)` — no offset lookup before reaching row data, and the type system guarantees uniform width. In practice for L3-resident data the difference is negligible, but for cache-cold datasets the missing offset fetch improves prefetch predictability.

**Python API change:** DataFrame construction changes:
```python
# Current (List)
pl.Series("pca", [[1.0, 2.0, ...], [3.0, 4.0, ...]])  # inferred as List(Float32)

# With Array
pl.Series("pca", [[1.0, 2.0, ...], [3.0, 4.0, ...]],
          dtype=pl.Array(pl.Float32, 50))
```

---

### 2.3 `Struct` with Named Float Fields

**How it works:** Each dimension becomes a named field: `{pc0: f32, pc1: f32, ..., pc49: f32}`.

**Memory layout:** Polars stores Struct columns as separate Series per field (column-major within the struct). For KNN this is pathological — iterating over all cells' vectors requires N × K non-sequential memory accesses jumping between 50 separate column buffers.

**Serialization cost:** Building 50 named columns in the output type function is awkward:
```rust
let fields: Vec<Field> = (0..50).map(|i|
    Field::new(format!("pc{i}").into(), DataType::Float32)
).collect();
DataType::Struct(fields)
```

**Verdict: Not recommended.** Column-major layout is anti-cache for row-sequential KNN. Overhead is structural — unavoidable with this approach. No production bioinformatics tool uses struct columns for PCA coordinates.

---

### 2.4 Flat `Float32` Column with External Stride

**How it works:** Store all `N × K` floats in a single primitive `Float32` column (no nesting), with K passed as a separate parameter to the plugin.

For 20,000 cells × 50 components, this is 1,000,000 float32 values in a flat array. Cell `j` is at `data[j*K .. (j+1)*K]`.

**Memory layout:** Maximally compact — a single contiguous buffer. Identical to numpy's C-contiguous 2D array (which is exactly how scanpy stores `adata.obsm['X_pca']`).

**pyo3-polars access:**
```rust
let flat = inputs[0].f32()?.rechunk();
let values: &[f32] = flat.cont_slice()?;  // requires single contiguous chunk
let k = inputs[1].u32()?.get(0).unwrap() as usize;  // stride from 2nd input

for i in 0..n_cells {
    let row = &values[i * k .. (i + 1) * k];
    // build KD-tree
}
```

**Pros:**
- Zero overhead — single primitive array
- Identical memory layout to numpy's `X_pca` — zero-copy conceptually
- Fastest possible sequential scan for KD-tree construction
- `cont_slice()` gives a direct `&[f32]` into Arrow buffer

**Cons:**
- Polars type system doesn't know about the K stride — accidental misuse is easy
- Can't use polars `.len()` to get n_cells (would return `n_cells × K`)
- Python API requires explicitly managing stride
- Intermediate results (PCA output) must also be flat — harder to inspect/debug

**Verdict:** Optimal for raw performance, fragile for API ergonomics. Appropriate if performance is the primary concern and API is internal/low-level.

---

## 3. How Scanpy/AnnData Does It

### Data storage
```python
# After sc.tl.pca(adata, n_comps=50):
adata.obsm['X_pca']  # shape: (n_cells, 50), dtype float32, C-contiguous numpy array
```

### Passing to PyNNDescent
```python
# sc/neighbors.py: _neighbors_from_scratch()
X = _choose_representation(adata, use_rep='X_pca', n_pcs=n_pcs)
# X is np.ndarray, shape (n_cells, n_pcs), float32, C-contiguous
nn_model = PyNNDescentTransformer(n_neighbors=n_neighbors, ...)
nn_model.fit_transform(X)  # passes dense numpy array directly
```

The key insight: scanpy passes a **flat 2D C-contiguous numpy array** to PyNNDescent. This is structurally identical to the flat `Float32` column with stride approach. There is no per-row metadata overhead.

### PyNNDescent memory
PyNNDescent (HNSW + random projection forests) internally constructs an approximate graph. For exact KNN (brute-force), scikit-learn uses scipy's `cKDTree` on the raw numpy matrix. In both cases, the input format is a flat 2D numpy array — the most cache-friendly possible layout for row-sequential algorithms like KNN.

---

## 4. Benchmark Estimates

For 20,000 cells × 50 components:

| Operation | List(Float32) | Array(Float32, 50) | Flat Float32 |
|---|---|---|---|
| Memory (inner data) | 3.8 MB | 3.8 MB | 3.8 MB |
| Offset overhead | 80 KB | 0 | 0 |
| Row access (Rust) | `list.get_as_series(i)` + cast | `chunk.value(i)` downcast | `&values[i*k..(i+1)*k]` |
| Cache misses per row | 1 extra (offset lookup) | 0 | 0 |
| Type safety (fixed width) | No — variable-length | Yes — compiler-enforced | No — stride is runtime |
| Polars API ergonomics | Good (current) | Good (polars-distance proven) | Poor (stride implicit) |
| Plugin input parsing | `list()?.get_as_series()` | `array()?.downcast_iter()` | `f32()?.cont_slice()` |

For KD-tree construction (sequential scan of all cells), the dominant cost is building the tree, not data access overhead. The offset lookup overhead (~80 KB of offsets, likely L2-cached after first pass) is unlikely to be measurable at 20K cells. At 500K–1M cells, the offsets buffer (~2 MB) may cause some L2 pressure but is still a minor cost compared to KD-tree insert operations.

---

## 5. Concrete Recommendations

### Recommendation 1: Migrate to `Array(Float32, 50)` for PCA output (medium-term)

**Why:**
- Zero offset overhead (though small at current scale)
- Polars type system enforces fixed width — eliminates the `MAX_DIMS = 50` truncation footgun
- Proven in production by `polars-distance` crate
- Row access in Rust is cleaner (`chunk.value(i)` vs `list.get_as_series(i)`)
- The `width()` method on `ArrayChunked` gives K without a magic constant

**Implementation change in Rust:**
```rust
// output type function
pub fn pca_array_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "pca",
        DataType::Array(Box::new(DataType::Float32), 50),
    ))
}

// Building output — use FixedSizeListChunkedBuilder
use polars::prelude::*;
let mut builder = AnonymousListBuilder::new("pca", n_cells, Some(DataType::Float32));
// ... OR construct FixedSizeListArray directly from flat buffer
```

**Note:** The outer `AnyValue::List` wrapping for the length-1 aggregate plugin output stays the same — this is unrelated to the inner dtype choice.

**Migration cost:** Medium. Requires updating `svd.rs` output, `neighbors.rs` input parsing, `umap_binding.rs` input parsing, Python-side schema declarations.

### Recommendation 2: Keep `List(List(Float32))` for now, switch inner to `Array` when migrating (near-term)

**Why:** The current `List(List(Float32))` works correctly. The offset overhead at 20K–100K cells is negligible (~80 KB–400 KB) compared to the ~3.8–19 MB value buffers. The primary correctness issue is the `MAX_DIMS = 50` silent truncation — this should be fixed by reading the actual dimension from the first cell's `.len()` rather than hardcoding.

**Quick fix for current code (neighbors.rs + leiden.rs):**
```rust
// Instead of: const MAX_DIMS: usize = 50;
// Do: infer dims from first row
let first_row = pca_coords.get_as_series(0)
    .ok_or_else(|| PolarsError::ComputeError("Empty PCA input".into()))?;
let n_dims = first_row.len().min(100);  // cap at 100 but don't silently truncate
```

### Recommendation 3: Do NOT use Struct dtype

Struct's column-major storage is incompatible with row-sequential KNN access patterns. No production tool does this.

### Recommendation 4: Consider flat Float32 + stride for a future ultra-high-performance path

If CREST targets 1M+ cells, a flat `Float32` column with stride passed as a parameter would eliminate all Polars overhead and match numpy's layout exactly. But this should be a secondary API (internal or advanced), not the primary user-facing interface.

---

## 6. Priority Action Items

| Priority | Action | File | Rationale |
|---|---|---|---|
| P0 | Add `mod neighbors;` to `lib.rs` | `src/lib.rs` line 19 | Neighbors functions are unregistered; they don't compile |
| P0 | Fix `MAX_DIMS` silent truncation | `src/neighbors.rs`, `src/leiden.rs` | Infer dims from data, return error if > supported max |
| P1 | Add integration tests for neighbors | `bench/test_all.py` | No validation of List(List(Float32)) serialization |
| P1 | Migrate inner representation to `Array(Float32, K)` | `src/svd.rs`, `src/neighbors.rs` | Cleaner API, type-safe fixed width |
| P2 | Evaluate flat Float32 encoding for 1M+ cell scale | `src/neighbors.rs` | If benchmarks show List overhead at scale |

---

## 7. Summary

The current `List(List(Float32))` encoding works correctly for PCA coordinates at CREST's target scale (20K–500K cells). The memory overhead from offsets is ~2% — not a bottleneck. The real issues are:

1. **Correctness**: `mod neighbors;` is missing from `lib.rs` — the plugin doesn't compile
2. **Correctness**: `MAX_DIMS = 50` silently truncates PCA > 50 dims
3. **Type safety**: `List(Float32)` doesn't enforce fixed width — `Array(Float32, K)` does

The recommended migration path is **List(Float32) → Array(Float32, K)**, following the `polars-distance` crate's proven pattern. This is the Arrow-native representation for fixed-width per-row vectors, has zero offset overhead, and is fully supported by pyo3-polars. The outer `AnyValue::List` wrapping for aggregate plugin output is a separate concern and stays unchanged.

For ultra-high-performance scenarios (1M+ cells), a flat `Float32` + stride approach mirrors scanpy's numpy data layout exactly, but introduces API complexity. This should be evaluated against real benchmarks before committing to it.
