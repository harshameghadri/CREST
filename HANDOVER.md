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
| Leiden | Phases 2 (`refine_partition`) and 3 (`aggregate`) now parallel, same per-community-independent pattern; 17.9 s at 1M cells (was 23.9 s), modularity unchanged vs leidenalg (2026-10-01, `docs/benchmarks.md`). Phase 1 (`fast_move_nodes`, ~73% of Leiden's time, confirmed by profiling) is still single-threaded: a parallel version was attempted and **measured-and-rejected** — it terminates but measurably loses modularity (see Next steps and `CHANGELOG.md`) |
| Paper benchmark | Quick + standard tiers done; Parse 1M + 9.7M re-run (2026-10-01, `3dbd760`, 1 repeat, core profile) with the Leiden seed-to-seed baseline and post-fix UMAP/Leiden timings in `docs/benchmarks.md`. **Next: final 5-repeat run with the parallel refinement + `performance` governor** |
| Input formats | 10x H5/MTX, `.h5ad` (native, no `anndata` needed) plus a generic `BioFrame.from_anndata()`/`to_anndata()` bridge (covers Loom/Visium/Zarr-AnnData/CSV via scanpy's readers) — **the bridge is undocumented and untested; `read_h5ad`'s native path drops all obs/var columns except the index and `gene_ids`**. Plan sketched 2026-10-01, not yet implemented (see Next steps) |
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
   - **Leiden refinement: parallelised, merged** (2026-10-01, PR #12). `refine_partition` runs
     each unrefined community on its own rayon thread (merges never cross community boundaries,
     so this is embarrassingly parallel); each gets an RNG seeded from `(seed, community id)` via
     SplitMix64, so results are the same for any thread count. ~16% faster at 1M cells (20.1 s vs
     23.9 s, median of 5, FRASER-quiet), ~5.4% faster end-to-end at 9.7M. `cargo test --release`
     (26/26) and `pytest -q` (27/27) pass, incl. `test_leiden_sweep_equals_leiden_and_ari`
     (bit-identical to a standalone `leiden()` call with the same seed). Modularity vs leidenalg
     checked over 20 seeds at resolution 1 on PBMC 68k: mean gap +0.0015 both before and after
     (statistically unchanged; `docs/benchmarks.md` softens the "equal or higher" claim to match —
     it was never a strict per-seed guarantee, the pre-change code already dips ~1/20 seeds too).
   - **Leiden aggregation: parallelised, same session, not yet committed.** Profiling
     `fast_move_nodes` / `refine_partition` / `aggregate` directly (temporary `Instant` timers,
     stripped before commit) on `parse_pbmc_1m` confirmed the shares: phase 1 ~73%, phase 3
     (`aggregate`) ~21%, phase 2 ~3%. `aggregate`'s per-community outgoing-edge computation is
     exploitable exactly like refinement (a community's edges to higher-numbered communities
     depend only on its own members), with no RNG involved at all — parallelised with rayon
     `map_init` (one reused `NeighborWeights` buffer per worker thread, not one fresh O(k) buffer
     per community, since that would cost O(k²) total). Output is **bit-identical** for any thread
     count (verified by diffing labels before/after on PBMC 68k across 5 seeds). Leiden at 1M cells
     is now **17.9 s** (median of 5, FRASER-quiet) — down from 20.1 s with refinement alone, 23.9 s
     originally. `git diff src/leiden.rs` in the working tree has this change; needs committing,
     testing once more, and a PR.
   - **`fast_move_nodes` (phase 1) parallelisation: attempted, measured, rejected** (same session).
     This is the 73% that actually matters, but unlike refinement/aggregation it is not
     embarrassingly parallel: a node's move decision genuinely depends on other nodes' concurrent
     moves. Two synchronous-round designs were tried (parallel scan of neighbour edge weights,
     sequential commit one node at a time so `comm_w`/`comm_size`/the empty-community stack can't
     race):
     1. Deciding from the round-start snapshot (both candidate communities *and* their weights):
        caused a genuine infinite loop — two nodes repeatedly swapping into each other's
        not-yet-updated communities, forever. Hung for hours before being killed (a real resource
        incident on rinamochana, not just a test failure — see note below).
     2. Deciding from *live* `comm_w`/`comm_size` at commit time (only the neighbour-weight scan
        itself comes from the round-start snapshot): terminates — every applied move is a
        certified modularity improvement under current weights, same finite-state termination
        argument as the original algorithm. But it measurably hurts quality: the 20-seed
        modularity-vs-leidenalg sweep (same test as refinement's gate) went from +0.00152 mean
        gap / 2-of-20 below, to **-0.00083 mean gap / 15-of-20 below**. A control run (this design
        reverted, aggregation parallelisation kept) reproduced +0.00152/2-of-20 exactly, isolating
        this specific change as the cause.
     Per CLAUDE.md's accuracy rules, this does not ship. `fast_move_nodes` stays single-threaded
     and sequential. **For a future attempt:** the failure mode in design 2 is neighbour-community
     staleness (not weight staleness, which design 2 already fixed) — a node's candidate list can
     miss a community that only became relevant because a neighbour moved earlier in the same
     round, or over-weight a candidate because a neighbour hasn't moved yet when it's about to.
     Shrinking the round size (process a bounded batch per round instead of the whole active set)
     would reduce this staleness at the cost of more synchronization overhead; whether that
     trade-off is worth it has not been measured. A different line of attack worth trying first:
     the queue-based algorithm is likely memory-latency-bound (every `part[u]`/`comm_w[c]` access
     is an effectively-random lookup into a multi-MB array for a 1M+-node graph) rather than
     compute-bound, given ~20-30 edges/node isn't enough raw arithmetic to explain 40 s — if so, a
     cache-friendlier node/edge layout (e.g. renumbering nodes so neighbours are close in memory)
     might speed up the existing *sequential* algorithm substantially with zero quality risk,
     which could make parallelising it less necessary.
     - **Resource note:** an early, buggier version of design 1 was left running by mistake and
       hung undetected for ~4.5 hours at near-full machine utilization (confirmed via `ps`:
       5679% CPU, 54 GB RSS) before being found and killed. If a `leiden`-tagged
       `target/release/deps/crest-*` process is ever found running for an implausibly long time,
       it's almost certainly a repeat of this — kill it, don't assume it's doing useful work.
   - Final paper numbers: 5 repeats, CPU governor `performance` (owner's step).
   - Resolution-sweep memory/time at 9.7M (484 s, 68 GB) predates the graph-copy fix; remeasure.
2. **Input format support beyond 10x: planned, not implemented** (2026-10-01, owner's request).
   CREST already reads `.h5ad` natively and has `BioFrame.from_anndata()` / `to_anndata()`, which
   covers Loom/Visium/Zarr-AnnData/CSV etc. via scanpy's readers — but that bridge has no test and
   isn't documented or in the Sphinx API, so nobody would find it. Planned phases (not started):
   (0) add a parity test + document `from_anndata` as the official "other formats" path;
   (1) fix `read_h5ad`'s native fast path to keep all obs/var columns, not just the index and
   `gene_ids` (removes the `.obs.parquet` sidecar-file workaround the Parse dataset needs today);
   (2) native streaming reader for Loom (same chunked-`h5py` pattern as `read_10x_h5`);
   (3) native streaming reader for Zarr-backed AnnData (large cloud atlases, e.g. CELLxGENE
   Census use this; the one format where going through `anndata` first would defeat CREST's
   memory-bounded design at real scale); (4) Seurat RDS — docs-only (recommend SeuratDisk →
   h5ad), not a native Rust reader, unless there's real demand later.
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
| `fast_move_nodes` (Leiden phase 1) stays single-threaded despite being ~73% of Leiden's time | two parallel designs were measured: one hangs forever (stale-weight decisions let two nodes swap into each other's communities every round), the other terminates but drops mean modularity vs leidenalg from +0.0015 to -0.0008 over 20 seeds (15/20 below instead of 2/20), isolated by a control run. CLAUDE.md's accuracy rules don't allow shipping a parity-losing change for speed; see `CHANGELOG.md` and HANDOVER's Next steps for what was tried and what to try next |

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
