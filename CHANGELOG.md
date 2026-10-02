# Changelog

## Unreleased

### Changed
- Leiden aggregation (phase 3) now computes each community's outgoing edges on its own
  thread (rayon `map_init`, one reused `NeighborWeights` buffer per worker thread rather
  than one fresh O(k)-sized buffer per community): a community's edges to higher-numbered
  communities depend only on its own members, so this is independent per community and the
  result is bit-identical for any thread count (no RNG involved). Measured on a profiled
  split of Leiden's own wall time on `parse_pbmc_1m` (1M cells, 64 threads): phase 1
  (`fast_move_nodes`) ~73%, phase 3 (`aggregate`) ~21%, phase 2 (`refine_partition`) ~3%.
  This change alone takes `aggregate`'s share from ~10-13 s to ~5-5.6 s under load; combined
  with the refinement parallelisation below, Leiden at 1M cells is **17.9 s** (median of 5,
  64 threads, FRASER-quiet), down from 20.1 s with refinement alone and 23.9 s originally
  (~25% faster overall). Modularity vs leidenalg unchanged (20-seed sweep at resolution 1 on
  PBMC 68k: mean gap +0.00152, 2/20 seeds below — identical to the refinement-only baseline).
- **`fast_move_nodes` (phase 1) parallelisation: attempted, measured, rejected.** This is
  73% of Leiden's time and the obvious next target, but it is a fundamentally different
  algorithm from refinement/aggregation: moves are not independent per node, so a node's
  decision needs to see other nodes' just-applied moves to stay correct. Two designs were
  tried, both as synchronous rounds (parallel scan of neighbour weights, sequential commit
  one node at a time so `comm_w`/`comm_size`/the empty-community stack never race):
  - Deciding from the round-start snapshot (both which communities to consider *and* their
    weights) caused a real infinite loop on the existing test suite: two nodes could each see
    the other's not-yet-updated community as attractive and swap into each other's old spots
    every round forever. Not a hypothetical failure mode — it hung for hours before being
    killed.
  - Deciding from *live* `comm_w`/`comm_size` at commit time (only the candidate-community
    list and each candidate's raw edge weight came from the round-start scan) does terminate
    — every applied move is a certified modularity improvement under current weights, same as
    the original algorithm, so it inherits the same finite-state termination argument. But it
    measurably hurts solution quality: the same 20-seed modularity sweep that read +0.00152
    vs leidenalg with refinement+aggregation alone reads **-0.00083** with this change added,
    and 15/20 seeds fall below leidenalg instead of 2/20. A control run (reverting only this
    change, keeping the aggregation parallelisation) reproduced the +0.00152/2-of-20 baseline
    exactly, isolating this specific change as the cause.
  Per the project's accuracy rules, a parity-losing change does not ship even when it is
  faster: `fast_move_nodes` stays single-threaded. Design notes for a future attempt are in
  `HANDOVER.md`.
- Leiden refinement (phase 2) now runs each unrefined community on its own thread (rayon),
  with an RNG seeded from `(seed, community id)` via SplitMix64: a merge only ever pulls a
  node into a sub-community of its own unrefined community, so communities refine
  independently and the result is the same for any thread count. ~16% faster at 1M cells
  (20.1 s vs 23.9 s median of 5, 64 threads, FRASER-quiet rinamochana) with modularity
  statistically unchanged vs both the prior sequential code and leidenalg (mean gap vs
  leidenalg +0.0015 either way over 20 seeds at resolution 1; the exact per-seed partition
  changes along with the RNG draw sequence, so this is not bit-identical to the old code).
  `fast_move_nodes` (phase 1) is still single-threaded and is now the larger share of
  Leiden's wall time; parallelising it is the next step (HANDOVER.md).
- UMAP (parallel): threads read one shared snapshot of the embedding instead of each copying
  all n rows 16 times per epoch; removes O(threads x cells) copying and ~threads x cells x 8 B
  of memory. Output bit-identical.
- Leiden: the input graph is no longer cloned at the start of each iteration, and the graph is
  built without an intermediate f64 edge list; a resolution sweep now shares one graph across
  its parallel runs (was one copy per run). Output bit-identical.
- Benchmark accuracy: Leiden seed-to-seed baseline (3 seeds per tool on the same graph, untimed)
  reported next to CREST-vs-scanpy ARI.

### Changed
- Benchmark protocol (`bench/paper/`, `crest_paper_bench.sh`): the headline is the core workflow,
  run at 8 and all threads so every speed-up is reported at matched threads **and** best vs
  best (the smaller is the headline); optional modules run separately so their time and memory
  never mix into the headline; Scrublet is skipped above `--scrublet-max-cells` (150,000);
  the report adds a "where CREST is not faster" table, a modules table and skipped steps.
- New benchmark dataset `parse_pbmc`: the Parse ~1M PBMC cytokine atlas (figshare 28589774),
  used from a local copy; raw counts and donor / cytokine / cell-type columns detected
  automatically; `datasets.py --inspect FILE.h5ad` shows the layout.

### Fixed
- Documentation checked claim by claim against the code: what each function returns, what
  the stores hold, out-of-core memory (per-cell results still grow with the number of cells),
  thread scaling, reproducibility; the quickstart explains the Polars import and what "lazy"
  means; honest 0.3.0 benchmark results with the steps where CREST is slower.

## 0.3.0 (2026-09-29)

### Changed
- The acronym now reads **Chunked** Rust Engine for Single-cell Transcriptomics (was
  "Columnar"): the engine streams chunks of cells through fused Rust kernels; Polars is only
  the table and Parquet layer. Package and import names are unchanged (`crest-sc`, `crest`).

### New
- **Pseudobulk DESeq2 in Rust.** `crest.tl.DESeq2` ports DESeq2 1.42 `DESeq()` +
  `results()`: median-of-ratios size factors, Cox-Reid gene-wise dispersions,
  parametric trend, MAP shrinkage, NB-GLM Wald tests, Cook's filtering and
  outlier replacement, and independent filtering. It matches R to ~1e-9 on
  simulated designs, is ~18× faster than R and ~28× faster than pydeseq2 on
  Kang 2018 pseudobulk. See `docs/deseq2.md`.
- `crest.tl.pseudobulk` (streamed raw-count sums per sample × group) and
  `crest.tl.pseudobulk_de` (per-cell-type DESeq2 in one call).
- `bench/deseq2/`: R comparison on simulated designs and on Kang et al. 2018.
- **Harmony** batch integration in Rust (`crest.tl.harmony`), the harmony2 algorithm of
  R harmony >= 1.2 / harmonypy 2.x; 3.5x faster than harmonypy at equal integration quality.
- **Scrublet** doublet detection (`crest.pp.scrublet`), sparse and streamed; 12x faster
  than `scanpy.pp.scrublet` with the same AUROC on demuxlet-labelled doublets.
- `highly_variable_genes(flavor="seurat_v3" | "seurat_v3_paper", batch_key=...)`, with a
  port of netlib loess (`crest._loess`); identical to scanpy.
- `crest.tl.leiden_sweep`: many resolutions/seeds on one graph in parallel, with
  ARI-based stability; `crest.tl.adjusted_rand_index`.
- `crest.tl.ingest`: project a query onto a reference PCA, transfer labels and UMAP.
  `uns['pca']['projection']` now stores what the projection needs.
- DESeq2 likelihood-ratio test: `DESeq2(test="LRT", reduced="~ ...")`, matching R.
- Native `knn_query` (reference -> query kNN, exact or IVF).
- `scripts/crest_paper_bench.sh` + `bench/paper/`: one-command publication benchmark
  (datasets, per-core CPU/clock/memory monitoring, statistics, figures).
- `scripts/crest_git_housekeeping.sh`: repository housekeeping (dev branch, stale branches).
- **Documentation site** (Sphinx + MyST, Read the Docs): installation, quickstart, concepts,
  full API reference, benchmarks, developer guide; `.readthedocs.yaml`; CI `docs` job.
- `work.md`: the development history; `CLAUDE.md` / `HANDOVER.md` rewritten for new sessions.

### Removed
- The `.bio` Polars expression namespace and its Rust plugin code (duplicated the
  validated API, was not validated itself, and tied the wheel to Polars' plugin ABI).
  The `polars`/`pyo3-polars` Rust dependencies go with it.
- SLAF support (`BioFrame.from_slaf`, `crest/slaf_io.py`, the `slaf` extra).
- Legacy scripts: `bench/*.py` from the biopolars era, `notebooks/`, the Docker setup,
  `docs/dev/`, and `bench/whitepaper/`'s scripts (superseded by `bench/paper/`; its
  0.2.0 results stay).

### Fixed
- `BioFrame.to_anndata()` / `from_anndata()` no longer need pyarrow.
- `crest_paper_bench.sh` stops with a clear message when the work directory is not writable,
  and records the work directory's device and filesystem.

## 0.2.0

A correctness and performance release. Every step of the standard scanpy
workflow has been re-implemented and validated against scanpy.

### Fixed (results from 0.1.0 should not be used)
- **kNN / Leiden / UMAP**: the HNSW index shuffled points and its id map was
  ignored, so neighbour indices pointed at random cells (recall@15 = 0.005).
- **Leiden** was a single-level local-move heuristic (thousands of singleton
  clusters) with a modularity gain off by a factor of 2; now Traag et al. 2019
  Leiden (local moving, refinement, aggregation). Modularity equals or exceeds
  `leidenalg`.
- **normalize_cpm / qc / filter_cells / scale / score_genes** Polars plugins
  were declared elementwise, so Polars computed per-cell sums on partial
  batches; `normalize_cpm` ignored `target_sum`.
- **rank_genes_groups** divided by non-zero counts instead of group sizes.
- **Randomized PCA** lost trailing components (no re-orthonormalisation).
- **HVG** "seurat" flavour was raw variance/mean (883/2000 overlap with scanpy).
- **Wilcoxon** p-values underflowed to 0.

### New
- `crest.pp` / `crest.tl` scanpy-style API on `BioFrame`, with raw counts held
  in compact CSR, a Polars triplet frame, or a Parquet dataset streamed from disk.
  Filters and `normalize_total` / `log1p` / `scale` are applied lazily inside
  fused native kernels: no normalised, scaled or dense copy is ever made.
- Exact PCA from a streamed Gram matrix with `sc.pp.scale(max_value)` applied
  implicitly (0.000 deg from scanpy's ARPACK result).
- kNN: tiled GEMM brute force (small n) and IVF + NN-descent (large n).
- UMAP: umap-learn semantics, parallelised by domain decomposition.
- Sparse Wilcoxon ranking only non-zeros; all-groups Welch t-test.
- `score_genes` reproducing scanpy's control-gene sampling.
- Readers: `read_10x_h5` (optionally streamed to Parquet), `read_h5ad`,
  `read_10x_mtx`; `write_parquet` / `read_parquet`; AnnData conversion.
- `bench/whitepaper/` benchmark harness.

### Changed
- `bio.deseq2` renamed `bio.nb_glm` (it fits an NB GLM with given dispersions;
  not the full DESeq2 procedure) and returns null when IRLS does not converge.
- In `bio.leiden` / `bio.umap`, `n_neighbors` now counts the cell itself (scanpy convention).
- `crest.tl.sparse_masked_pca` / `incremental_pca` removed (superseded by `crest.tl.pca`).
- Distribution renamed `crest-sc` (the import name stays `crest`); Python >= 3.10.
