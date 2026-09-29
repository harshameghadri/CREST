# Developer guide

## Build and test

```bash
maturin develop --release -E test     # (re)build the Rust extension into crest/
cargo test --release                  # Rust unit tests
pytest -q                             # Python tests: scanpy, R-DESeq2, harmonypy, skmisc, skimage parity
```

A Python change needs no rebuild. A Rust change needs `maturin develop --release` before
`pytest` sees it. Always build with `--release`: debug builds are about 20× slower and make
the tests time out.

Build these docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html     # open docs/_build/html/index.html
```

## Repository layout

| path | what |
|---|---|
| `crest/core.py` | `BioFrame`, the three stores, `iter_ctx()` (the chunk protocol), AnnData / Parquet conversion |
| `crest/io.py` | `read_10x_h5` (+ `backed=` Parquet streaming), `read_10x_mtx`, `read_h5ad` |
| `crest/pp.py` | QC, filters, lazy transforms, HVG (`seurat`, `cell_ranger`, `seurat_v3`), neighbours |
| `crest/tl.py` | PCA, Leiden, `leiden_sweep`, UMAP, `rank_genes_groups`, `score_genes`; re-exports the modules below |
| `crest/deseq2.py` | `pseudobulk`, design formulas (`model_matrix`), `DESeq2` (Wald / LRT, `results()`, independent filtering), `pseudobulk_de` |
| `crest/harmony.py` | Python wrapper around the Rust Harmony |
| `crest/doublets.py` | Scrublet, and a port of `skimage.filters.threshold_minimum` |
| `crest/ingest.py` | projection onto a reference PCA, label and UMAP transfer |
| `crest/_loess.py` | port of netlib `dloess` (the loess behind R and skmisc), for `seurat_v3` |
| `src/kernels.rs` | fused chunk kernels over a `ChunkView` (QC, gene stats, Gram / projection, group sums, Wilcoxon) |
| `src/knn.rs` | kNN: exact tiled GEMM; IVF + NN-descent above 20k cells; `knn_query` (reference → query) |
| `src/leiden.rs` | Leiden (Traag 2019) + parallel resolution sweep |
| `src/umap/` | fuzzy simplicial set, spectral init, parallel SGD |
| `src/harmony.rs` | Harmony (harmony2 algorithm) |
| `src/deseq/` | DESeq2 (`mod.rs`), small dense LU (`linalg.rs`), R's `lowess` (`lowess.rs`) |
| `src/py.rs` | PyO3 bindings: NumPy in / out, GIL released, argument validation |
| `tests/test_crest.py`, `tests/data/` | the test suite; stored R DESeq2 outputs + the R script that makes them |
| `bench/paper/` | the paper benchmark harness (driven by `scripts/crest_paper_bench.sh`) |
| `bench/<module>/` | module comparisons with their `results/` |
| `scripts/` | `crest_paper_bench.sh`, `crest_git_housekeeping.sh` |
| `docs/` | this documentation (Sphinx + MyST); `.readthedocs.yaml` at the root |

## The chunk protocol (how Python talks to Rust)

`BioFrame.iter_ctx(transform=True)` yields one tuple per chunk:

```text
(genes u32[nnz], values f32[nnz], cell_map i64[n_store_cells], gene_map i32[n_store_genes],
 target_sum f64, log1p bool, cells u32[nnz] | None, indptr i64[k+1] | None, first_cell int)
```

* `genes` / `values` are raw entries of the store. `cells` (COO) or `indptr` + `first_cell`
  (CSR) say which cell each entry belongs to.
* `cell_map[store_cell]` is the current row, or −1 if the cell is filtered out. `gene_map`
  does the same for columns. This is how filtering works without copying.
* `target_sum` > 0 means normalise to it; `log1p` means apply log1p after that.

Each streaming native function takes `*ctx` followed by its own inputs and **preallocated
NumPy outputs that it accumulates into**, for example:

```python
out = np.zeros((bf.n_vars, 5))
for ctx in bf.iter_ctx():
    _native.gene_stats(*ctx, out)
```

In Rust, `src/py.rs` turns the tuple into a `kernels::ChunkView`. The kernel walks the chunk
cell by cell in parallel (rayon), applies the maps and transform into a per-thread buffer,
and merges per-thread partial results.

### Native functions (`crest.crest`)

| function | used by | does |
|---|---|---|
| `qc` | `calculate_qc_metrics`, filters | per-cell / per-gene totals, counts, mito fraction |
| `gene_stats` | HVG, `score_genes`, PCA scaling | per-gene Σx, Σx², Σexpm1, Σexpm1², nnz |
| `group_gene_sums(clip)` | `rank_genes_groups`, `seurat_v3` | per group × gene Σx, Σx², nnz (optionally clipped) |
| `group_gene_totals` | `pseudobulk` | per group × gene raw sums |
| `collect_gene_block`, `wilcoxon_rank_sums` | Wilcoxon | gathers a memory-bounded gene block; sparse rank sums with a closed-form zero-tie block |
| `gram_accumulate`, `pca_from_gram`, `project` | PCA, Scrublet, ingest | Gram matrix with implicit scale / clip; eigendecomposition; projection |
| `weighted_row_sums` | `score_genes` | per-cell weighted sum over genes |
| `materialize` | `to_scipy`, `write_parquet` | the filtered / transformed chunk as COO |
| `knn_graph`, `connectivities` | `neighbors`, Scrublet | kNN; umap-learn fuzzy connectivities |
| `knn_query` | `ingest` | reference → query kNN |
| `leiden`, `leiden_sweep` | clustering | Leiden, one or many (resolution, seed) runs in parallel |
| `umap` | `umap` | layout optimisation |
| `harmony` | `harmony` | full Harmony loop |
| `deseq2_fit`, `lowess` | `DESeq2` | the whole DESeq2 fit; R's lowess for independent filtering |
| `t_pvalues`, `normal_pvalues` | DE | vectorised p-values |

## Rules for changes

1. **Match the reference tool.** A new method needs a test that pins its agreement with the
   reference implementation (scanpy, R, harmonypy, …) *before* it is optimised. If a change
   moves a parity test, find out why. Never loosen the tolerance to make it pass.
2. **Bounded memory.** Work one chunk at a time through `iter_ctx()`. Never build a dense
   cells × genes array, and don't call `to_scipy()` inside library code.
3. **Heavy loops go in Rust.** Release the GIL (`py.allow_threads`), parallelise with rayon,
   and validate array lengths and indices in `py.rs` so bad input raises a Python error
   instead of panicking.
4. **Stay scanpy-compatible.** Keep scanpy's names and defaults unless there is a documented
   reason not to.
5. **Commits:** `feat:`, `fix:`, `bench:`, `docs:`, `chore:` prefixes. Before pushing, run
   `cargo test --release && pytest -q`.

### Recipe: adding a streamed statistic

1. Write the kernel in `src/kernels.rs` over `ChunkView`. Use a per-thread accumulator, then
   merge.
2. Bind it in `src/py.rs`: accept `*ctx` plus output arrays, check lengths, and wrap the
   compute in `py.allow_threads`.
3. Register it in the `#[pymodule]` list at the bottom of `src/py.rs`.
4. Call it from Python in a `for ctx in bf.iter_ctx(): ...` loop.
5. Add a parity test in `tests/test_crest.py` against the reference tool, and test that the
   result does not depend on `chunk_nnz`.

## Branches and releases

* Feature work goes into `dev` via pull requests. Never push to `main`.
* A release merges `dev` → `main` and tags `vX.Y.Z`. The version lives in `Cargo.toml`;
  `pyproject.toml` and these docs read it from there. `crest/__init__.py` has a copy, so
  update both.
* CI (`.github/workflows/CI.yml`) runs `cargo test` + `pytest` on pushes to main and on
  pull requests, and builds wheels for all platforms; a tag publishes them to PyPI.
