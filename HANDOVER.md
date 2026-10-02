# CREST handover

This file says where the project stands and what to do next. Read `CLAUDE.md` first (how
the code works, and the rules). Keep **Status** and **Next steps** current at the end of
every session.

_Last updated: 2026-10-01 (Leiden refinement parallelised; Parse re-run with seed baseline)._

## Status

| Area | State |
|---|---|
| Core workflow (QC → HVG → scale / PCA → kNN → Leiden → UMAP → t-test / Wilcoxon → `score_genes`) | Done; validated against scanpy 1.11. 0.3.0 on the 64-thread workstation: PBMC 68k core workflow **8.9× faster best-vs-best** (19.8× at matched 64 threads), 3.4× less memory (`docs/benchmarks.md`) |
| Out-of-core (Parquet) | Done; count matrix never in RAM; ~0.7–1.1 GB peak up to 200k cells (per-cell results still grow with cells); up to ~2× slower |
| Pseudobulk DESeq2 (Rust) | Done: Wald + LRT; matches R DESeq2 1.42; ~50× faster than R on rinamochana |
| Modules after clustering | Harmony, Scrublet, `seurat_v3` HVG, `leiden_sweep`, `ingest`, `knn_query`; each validated (`docs/downstream.md`) |
| Leiden | Phase 2 (`refine_partition`) now parallel, one thread per unrefined community, RNG seeded from `(seed, community id)`; ~16% faster at 1M cells, modularity unchanged vs leidenalg (2026-10-01, `docs/benchmarks.md`). Phase 1 (`fast_move_nodes`) still single-threaded and now the larger share of time |
| Paper benchmark | Quick + standard tiers done; Parse 1M + 9.7M re-run (2026-10-01, `3dbd760`, 1 repeat, core profile) with the Leiden seed-to-seed baseline and post-fix UMAP/Leiden timings in `docs/benchmarks.md`. **Next: final 5-repeat run with the parallel refinement + `performance` governor** |
| Input formats | 10x H5/MTX, `.h5ad` (native, no `anndata` needed, now keeps every obs/var column incl. categoricals) plus `BioFrame.from_anndata()`/`to_anndata()` — the documented path for anything else scanpy/anndata reads (Loom, Visium, Zarr-AnnData, CSV, ...), now with a parity test and a `quickstart.md` section (2026-10-02, Phase 0+1 of the plan below). Native streaming readers for Loom/Zarr (Phase 2+3) not started |
| Documentation | Sphinx site in `docs/`, checked claim by claim against the code (2026-09-29). Read the Docs connection: owner's step |
| Packaging | **0.3.0 published on PyPI** (tag `v0.3.0`, release job green, 2026-09-29) |
| CI | `test` (cargo + pytest), `docs` (sphinx `-W`), wheels for all platforms; publishes on tag |

### Git / GitHub state

* `main` = 0.3.0: `dev` merged into `main` on 2026-09-29 (0.2.0 was
  [PR #1](https://github.com/harshameghadri/CREST/pull/1); the 0.3.0 work came through
  [PR #2](https://github.com/harshameghadri/CREST/pull/2), [#3](https://github.com/harshameghadri/CREST/pull/3),
  [#4](https://github.com/harshameghadri/CREST/pull/4) and the rename/release PR into `dev`).
* Work branch `claude/awesome-lamport-dsvfv2` is restarted from `dev` for each new piece of work.
* Stale branches are archived as tags (`archive/feat-leiden-hnsw-faer`,
  `archive/fix-production-readiness`) and deleted.

## Next steps (in priority order)

1. **Parse benchmark re-run: done** (2026-10-01, `3dbd760`, 1 repeat, core profile;
   `docs/benchmarks.md`). 1M cells: CREST 85.1 s vs scanpy 994 s (11.7×), 4.7 vs 54.8 GB. 9.7M:
   CREST 21.4 min, 40.3 GB. Status of the 2026-09-30 follow-ups:
   - **Leiden agreement (ARI 0.66 vs scanpy at 1M): measured.** Seed-to-seed baseline: CREST 0.73,
     scanpy 0.64, cross-seed mean 0.67 — mostly ordinary Leiden seed variation on a million-node
     graph, not a correctness gap, though CREST's higher self-consistency (0.73 vs the 0.66–0.67
     cross numbers) is a modest residual nobody has explained yet.
   - **UMAP / Leiden speed at scale: measured.** UMAP 83 s → 42 s at 1M, 934 s → 584 s at 9.7M
     (the shared-snapshot fix). Leiden held at ~24 s / 400 s at this commit (fix didn't touch it).
   - **Leiden refinement: parallelised** (2026-10-01). `refine_partition` now runs each unrefined
     community on its own rayon thread (merges never cross community boundaries, so this is
     embarrassingly parallel); each gets an RNG seeded from `(seed, community id)` via SplitMix64,
     so results are the same for any thread count. ~16% faster at 1M cells (20.1 s vs 23.9 s,
     median of 5, FRASER-quiet), ~5.4% faster end-to-end at 9.7M (1,218.6 s vs 1,287.6 s, matches
     Leiden's ~31% time share). `cargo test --release` (26/26) and `pytest -q` (27/27) pass, incl.
     `test_leiden_sweep_equals_leiden_and_ari` (bit-identical to a standalone `leiden()` call with
     the same seed). Modularity vs leidenalg checked over 20 seeds at resolution 1 on PBMC 68k:
     mean gap +0.0015 both before and after (statistically unchanged; `docs/benchmarks.md` softens
     the "equal or higher" claim to match — it was never a strict per-seed guarantee, the
     pre-change code already dips ~1/20 seeds too). Not yet committed/pushed — do that first in a
     continuing session (`git diff src/leiden.rs` has the change; branch off `dev`, PR, don't merge
     without the owner).
     - **Still open: `fast_move_nodes` (phase 1) is single-threaded** and is now the larger share
       of Leiden's time (it was already queue-based local moving, which does not parallelise as
       cleanly as phase 2 — the standard approach processes a frontier of nodes in batches rather
       than community-by-community). Design needed for a local session.
   - Final paper numbers: 5 repeats, CPU governor `performance` (owner's step).
   - Resolution-sweep memory/time at 9.7M (484 s, 68 GB) predates the graph-copy fix; remeasure.
2. **Input format support beyond 10x** (owner's request, 2026-10-01; Phase 0+1 done 2026-10-02).
   `BioFrame.from_anndata()` already existed and covers Loom/Visium/Zarr-AnnData/CSV etc. via
   scanpy's readers — it was in the Sphinx API already (bare `:members:` on `BioFrame` picks up
   every public method), the actual gaps were narrower: no test, and not mentioned in
   `quickstart.md`. An earlier version of this file claimed otherwise ("undocumented... not in
   the Sphinx API") and that the `.obs.parquet` sidecar `bench/paper/datasets.py` writes for the
   Parse dataset was a `read_h5ad` limitation — both wrong, corrected here: the sidecar exists
   because that script converts Parse's `.h5ad` to a **10x-style `.h5`** for the benchmark harness,
   a format with no room for arbitrary obs columns at all regardless of `read_h5ad`; the two also
   use entirely different h5ad-reading code (`bench/paper/datasets.py`'s own `read_h5ad_obs` vs
   `crest.io.read_h5ad`). Done:
   - **Phase 0:** `from_anndata`/`to_anndata` docstrings now say what's preserved and point to this
     as the "other formats" path; `test_from_anndata_roundtrip` (categorical/numeric/string obs
     columns + obsm, round-tripped) in `tests/test_crest.py`; a "Reading other formats" section in
     `quickstart.md`.
   - **Phase 1:** `read_h5ad`'s native path (`crest/io.py`, no `anndata` needed) now keeps every
     obs/var column instead of just the index + `gene_ids`, including anndata's categorical
     encoding (a sub-group of `categories` + integer `codes`) — expanded to strings, matching what
     `from_anndata` already does for a pandas categorical column. `test_read_h5ad_keeps_obs_columns`
     writes a real `.h5ad` via installed `anndata` and reads it back with `crest.read_h5ad` to
     exercise the actual on-disk encoding, not a hand-rolled approximation of it.
   Not started: (2) native streaming reader for Loom (same chunked-`h5py` pattern as
   `read_10x_h5`); (3) native streaming reader for Zarr-backed AnnData (large cloud atlases, e.g.
   CELLxGENE Census use this; the one format where going through `anndata` first would defeat
   CREST's memory-bounded design at real scale); (4) Seurat RDS — docs-only (recommend SeuratDisk
   → h5ad), not a native Rust reader, unless there's real demand later.
3. **Connect Read the Docs** (`docs/readthedocs.md`): project `crest-sc`, default branch `main`.
4. **White paper**: headline from the core workflow best-vs-best; "not faster" table;
   modules table; accuracy table (all in `report.md`). Use 5 repeats and the new protocol.
   Set the CPU governor to `performance` for the final run.
5. **Performance work, measured first:**
   - `seurat_v3` HVG is 2× slower than scanpy at ≥ 10k cells: the loess fit runs in Python
     (`crest/_loess.py`); port it to Rust or vectorise it.
   - PCA has a ~1 s fixed cost: the full eigendecomposition of the 2,000 × 2,000 Gram matrix
     (measured 1.16 of 1.31 s on PBMC 3k). A partial eigensolver for the top 50 would remove it.
   - UMAP fixed cost on small data (500 epochs below 10k cells, as umap-learn).
   - Serial fraction ~24% (reading HDF5, Leiden, glue).
   - Scrublet at ≥ 100k cells without a batch key is ~quadratic (k ≈ 1.5·√n); low priority.
6. **Possible extensions:** DESeq2 `lfcShrink` / `lfcThreshold` / interactions; UMAP transform
   optimisation in `ingest`; `pearson_residuals` HVG.

## Decisions (and why)

Longer discussions are in `work.md`; search for the keywords.

| decision | reason |
|---|---|
| Rebuild around `BioFrame` + fused kernels instead of Polars expression plugins | Polars plugins compute per batch, which gave wrong per-cell sums; the plugin ABI tied wheels to Polars versions; nothing was validated |
| Delete the `.bio` namespace, SLAF support, notebooks, Docker, legacy benches | unvalidated duplicates of the real API; maintenance weight |
| PyPI name `crest-sc`, import name `crest` | the `crest` name was unavailable |
| Our own DESeq2 port instead of an existing crate | existing crates were immature or needed a newer Rust; a port on the streamed group sums could be validated line by line against R |
| DESeq2: Newton refinement instead of L-BFGS-B; exact integration for ≤ 3 residual df | deterministic and at least as accurate; documented in `docs/deseq2.md` |
| Harmony `nclust = min(round(N/30), 100)`, harmonypy defaults | parity with harmonypy 2.x / R harmony ≥ 1.2 |
| `ingest` centres the query on its own mean | what `scanpy.tl.ingest` does (accuracy 90.7% vs 86% otherwise) |
| Scrublet uses exact kNN up to 150k points | NN-descent is slow at Scrublet's large k |
| `scanpy.external.pp.harmony_integrate` not used in benchmarks | broken with harmonypy 2.x (transposed output); harmonypy is called directly |
| Benchmarks write only under `./crest-bench`, nothing in `$HOME` | the owner's `/home` is a small OS disk |
| Honest reporting rules (CLAUDE.md §5) | owner's explicit requirement for the paper |
| Acronym now "**Chunked** Rust Engine for Single-cell Transcriptomics" (0.3.0; was "Columnar") | the engine streams chunks of cells through fused Rust kernels; Polars is only the table / Parquet layer. Keeping the acronym and the `crest-sc` / `crest` names avoids breaking anyone |
| Leiden refinement parallelised by unrefined community, each with its own RNG from `(seed, community id)` (SplitMix64), instead of one shared RNG stream for all nodes | a merge only ever pulls a node into a sub-community of its own unrefined community (candidates are filtered by `part[u] == s`), so communities are already independent; giving each its own stream makes the result the same for any thread count instead of only being deterministic because it happened to be single-threaded. Changes the exact RNG draw sequence (no longer bit-identical to the old code for the same seed), which is fine — parity is against leidenalg/scanpy via modularity and ARI, not against CREST's own prior output |
| "Leiden modularity >= leidenalg" softened from a per-seed claim to a statistical one | measured: both the old sequential code and the new parallel one occasionally fall ~0.002 below leidenalg's modularity on a specific (resolution, seed) pair (1–2 times per 20 seeds); the mean gap is positive (+0.0015) and unchanged by the parallelisation. The original claim was from a small sample and was never a proven per-seed guarantee for a randomised heuristic |

## Known limitations

* HVG flavours: no `pearson_residuals`.
* Batch integration: Harmony only.
* DESeq2: additive formulas only (pass `design_matrix=` for interactions); no `lfcShrink`.
* UMAP is deterministic per thread count, not across thread counts.
* Leiden agrees with scanpy's clusterings at ARI 0.87–0.89, with the same number of clusters
  and equal or higher modularity.
* PCA needs ≤ 20k genes (the Gram matrix is genes²), so select HVGs first.
* CREST has no plotting: use `bf.to_anndata()` and scanpy's `sc.pl`.

## Setting up locally

```bash
git clone https://github.com/harshameghadri/CREST && cd CREST && git checkout dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y     # if no Rust
conda deactivate 2>/dev/null || true                                         # maturin + conda clash
uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install maturin
maturin develop --release -E test,bench
cargo test --release && pytest -q
```

Optional, only to regenerate the R comparisons:

* Debian / Ubuntu: `sudo apt install r-bioc-deseq2 r-cran-jsonlite`
* anywhere else: `R -e 'BiocManager::install("DESeq2")'`

## Reproducing the results

```bash
cd /mnt/scratch                                    # a roomy disk; never $HOME on rinamochana
curl -LO https://raw.githubusercontent.com/harshameghadri/CREST/dev/scripts/crest_paper_bench.sh
bash crest_paper_bench.sh --tier quick             # ~35 min
bash crest_paper_bench.sh --tier standard          # ~1 day on 64 cores
python bench/harmony/compare_harmonypy.py          # module comparisons: see docs/benchmarks.md
```
