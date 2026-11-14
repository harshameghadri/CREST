# Benchmarks and validation

CREST is held to two claims, and both are measured:

1. **Same results as the reference tool.** Parity for HVG, marker genes, `score_genes`, PCA,
   DESeq2, loess and `threshold_minimum` is checked by the test suite (CI, on pull requests
   and pushes to `main`); the other steps are checked on real data in the module benchmarks
   and the paper benchmark.
2. **Faster and leaner.** This is measured by the paper benchmark, on your own machine, with
   a protocol designed to be reported honestly.

## Accuracy (tests and real-data checks)

| step | reference | agreement |
|---|---|---|
| QC, filters | scanpy 1.11 (tested with 1.11.5) | identical |
| HVG `seurat`, `cell_ranger` | scanpy | identical gene set; `dispersions_norm` within 2e-6 |
| HVG `seurat_v3` | scanpy + skmisc loess | identical sets and ranks; `variances_norm` within 1e-14 |
| scale + PCA | scanpy (dense) | identical subspace (0.000° principal angles) |
| neighbours (connectivities) | scanpy / umap-learn | identical given the same kNN (1e-5) |
| Leiden | scanpy 1.11 `sc.tl.leiden(flavor="igraph", n_iterations=2)` (ARI, in the paper benchmark); leidenalg (modularity sweep) | similar number of clusters; modularity typically equal or higher than leidenalg's (mean +0.0015 over 20 seeds at resolution 1 on PBMC 68k), occasionally ~0.002 lower on a given seed — both tools' own seed-to-seed variation is of the same order, so this is not a per-seed guarantee. The modularity-sweep script is not yet in the repository |
| t-test, Wilcoxon | scanpy | t-test scores within 1e-4; Wilcoxon scores within 3e-4 on PBMC 3k (ties differ at the float32 ulp level of the normalisation; 1e-6 on identical input values); p-values within 4e-4 |
| `score_genes` | scanpy | identical incl. control-gene sampling (3e-7) |
| DESeq2 Wald / LRT | R DESeq2 (1.42 for the stored unit-test fixtures; the version of every benchmark run is in that run's environment record) | ~1e-9 (Wald, simulated), 1e-6 (LRT) |
| Harmony | harmonypy 2.1.0 (current figures measured with 2.0.2) | same iLISI / cLISI / ARI; neighbourhood overlap at seed-noise level |
| Scrublet | scanpy 1.11 | same AUROC on demuxlet doublets; Spearman 0.98 (measured with `include_self_neighbor=False`, see {doc}`downstream`) |
| ingest | scanpy 1.11 | 90.7% vs 89.8% accuracy; same label for 96.4% of cells |

Details: {doc}`downstream`, {doc}`deseq2`.

## The paper benchmark (one command)

```bash
cd /a/disk/with/space          # everything is written under ./crest-bench (nothing in $HOME)
curl -LO https://raw.githubusercontent.com/harshameghadri/CREST/v0.4.0/scripts/crest_paper_bench.sh
bash crest_paper_bench.sh --tier quick      # ~35 min: 3 datasets, 2 repeats
bash crest_paper_bench.sh --tier standard   # ~1 day on 64 cores: 6 datasets, 5 repeats, thread scan
bash crest_paper_bench.sh --tier full       # adds the 647k COVID atlas and 1.3M neurons
```

The script:

1. checks the tools (git, curl, uv, Rust) and the free disk;
2. clones CREST at `--ref` (default `dev`; pass `--ref v0.4.0` to reproduce the release) into a throwaway uv venv and builds it in release mode;
3. runs `cargo test` and `pytest`, and stops if they fail;
4. records the machine: CPU model, caches, governor, cores, RAM, disk, OS, compilers, BLAS,
   the SMT sibling map, the numba threading layer, the R, DESeq2 and BiocParallel versions
   (`R_versions.txt`, `R_sessionInfo.txt`) and every package version;
5. downloads and converts the datasets (resumable, size-checked, SHA-256 recorded; the MTX sets are
   converted with scipy, not with CREST's reader) and verifies every converted file count for count
   against independent readers (`datasets.py --verify`);
6. runs every (dataset × tool × repeat) in its own process, after an optional untimed tuning
   pilot (`--pilot`);
7. checks agreement with scanpy (ARI/NMI, kNN Jaccard, principal angles, HVG and DE overlap);
8. runs the module benchmarks (DESeq2, Harmony, Scrublet, ingest);
9. computes statistics and writes tables (CSV + LaTeX), figures (PDF + PNG), `report.md`
   and a `.tar.gz`.

Datasets: PBMC 3k, PBMC 10k, Kang 2018 (29k, 8 donors × 2 conditions), PBMC 68k, the
Stephenson 2021 COVID atlas (647k), 10x 1.3M neurons, the Parse Biosciences PBMC cytokine
atlas (12 donors × 90 cytokines + PBS: `parse_pbmc`, all 9.7M cells × 2,000 genes, and
`parse_pbmc_1m`, a seeded random 1M-cell subset of it). The scaling series are **random
subsamples of real datasets** (`<parent>_sub<N>`, for example `neurons_1m_sub100k`; seeded and
nested, so the sizes of one series differ only in the number of cells;
`datasets.py --subsample PARENT --sizes ...`). The synthetic 25k–1M sets, interpolated from 68k
real PBMCs with CREST's own normalisation, PCA and kNN, are a **labelled supplement**: their
neighbourhood structure is defined by CREST, so no scaling statement rests on them. At 9.7M cells scanpy cannot run in 128 GB (its dense scaled matrix alone would
be 78 GB), so the tool comparison uses `parse_pbmc_1m` and the full atlas shows CREST alone
at scale. The Parse matrix holds only 2,000 genes, so its QC threshold is scaled to the same
fraction of genes (`min_genes` 20 of 2,000 instead of 200 of ~20,000), for both tools.

The Parse file (13 GB, [figshare 28589774](https://figshare.com/articles/dataset/pbmc_parse/28589774))
cannot always be downloaded by a script. Download it in a browser, put it in
`crest-bench/data/`, and run with `--datasets "parse_pbmc_1m parse_pbmc"` (or `--tier full`). The raw counts
are taken from `layers/counts`, `raw/X` or `X` (the first that holds integer counts), and the
donor, cytokine and cell-type columns are detected automatically. The pseudobulk DESeq2 step
compares IFN-beta with PBS (or, if absent, the most frequent condition with the control).
`datasets.py --manifest` writes `dataset_manifest.json` (source URLs, cell and gene counts, SHA-256 of the source and converted
files, conversion path, verification result); the licence field of each dataset stays empty until it has been
checked at the data provider.

The 10x 1.3M neuron download is a Cell Ranger v1 file (one group per genome, 64-bit column indices, 2.6 billion
non-zeros, more than a 32-bit index can address); `datasets.py` streams it into the v3 layout the harness reads, with
32-bit column indices and 64-bit row pointers, and verifies the result against the download non-zero by non-zero.
The Stephenson atlas is converted from `raw.X` of the CELLxGENE file with h5py only (gene symbols as names, Ensembl
identifiers as ids, no CREST reader), and its pseudobulk contrast is `Status` Covid against Healthy: the `disease`
column also calls the six LPS-stimulated volunteers "normal". `--verify` adds to `verification.json` (a run on one
dataset keeps the records of the others). The two read-back checks (scanpy's and CREST's reader
against h5py) hold the whole matrix in memory (about 25 and 50 bytes per non-zero), so a file with more than
`--reader-max-nnz` non-zeros (default 250 million; 0 reads the whole file at once) is read back in contiguous blocks of
whole cells, each block a temporary file (set `TMPDIR`) read by both readers; every cell is still read by both, and
the check's detail says how many blocks were used. What the block mode does not test is a reader's behaviour on one
file with that many non-zeros. To check
what will be used before a long run:

```bash
python bench/paper/datasets.py --inspect crest-bench/data/Parse_1M_adata_for_cellflow_datasets_with_embeddings.h5ad
```

### Protocol (why the numbers can be trusted)

The values of a run (datasets with licence and checksums, grid, tuning, repeats, pinning, statistics,
headline rule) are fixed before it in `bench/paper/PREREGISTRATION.md`.

* **Kinds of runs.** The *core workflow* of every dataset × tool runs at each thread count of the
  grid (`--thread-grid`, default 1, 2, 4, 8, 16, 32, 64, clamped to the logical CPUs), so every
  speed-up can be compared at matched thread counts and at each tool's best. The *optional modules*
  run separately, so their time and memory never leak into the headline. Scrublet is skipped there
  above `--scrublet-max-cells` (150,000), because its neighbour search grows roughly with the
  square of the number of cells. A *thread scan* runs the core workflow at 1, 2, 4, … threads on one
  dataset. *Controls* (below) are kept out of every statistic.
* **Fresh process per run.** No cache, JIT or allocator state leaks between runs. The numba JIT of
  scanpy (and the file cache for both tools) is warmed up in the same process before timing; the
  warm cache is the primary condition.
* **Interleaved, shuffled repeats.** Repeat *r* of every configuration runs before repeat
  *r+1* of any, in a seeded random order. Thermal drift and background load then spread over
  all tools instead of biasing one.
* **Endpoints.** The primary endpoint is the **end-to-end process wall time**: from process creation to the end of
  the last pipeline step. It contains interpreter start-up, the imports the workflow needs (every module
  the tool loads for it is imported inside the timed start-up block) and the timed pipeline: every step plus the
  garbage collections and result extraction between steps. The in-process warm-up and the untimed accuracy
  extras are not part of it. The secondary endpoint is the sum of the step timers (`core_seconds`), which
  excludes start-up, imports and the untracked time between steps. `tables/endpoints.csv` and `tables/main.csv`
  report both, with the import and untracked time of each tool (they differ between the tools), and
  `tables/runs.csv` has every repeat.
* **Per-step comparisons use fused stages.** CREST applies `normalize_total` and `log1p` inside the next
  kernel, so its `normalize_log1p` step takes almost no time and the cost appears in `hvg`; dividing the
  raw step times gives ratios in the thousands that compare different amounts of work. The report therefore gives
  normalize + log1p + HVG as one stage, forms no ratio below the 0.01 s timer floor and never prints one above
  1000. The raw steps stay in `tables/steps.csv`.
* **Thread control and pinning.** `RAYON`, `OMP`, `OPENBLAS`, `MKL`, `VECLIB` and `NUMBA` thread variables
  are set for every run, and scanpy's `n_jobs` with them. Every run, at every thread count, is pinned to the first
  *p* logical CPUs in physical-core-first order (one thread context of every core before any second SMT sibling);
  with *p* equal to the number of logical CPUs this is the full set of CPUs. The CPUs, the SMT sibling map and
  the numba threading layer and thread count the process reports are recorded for every run and listed in the
  report. At 64 threads on a 32-core SMT machine the run therefore uses both contexts of every core, while 8 to 32 threads use
  one context per core: thread count and SMT use change together, and the *unpinned control* (the same run
  without pinning, at selected thread counts) measures what pinning itself changes. `--no-pin` switches
  pinning off for every run.
* **Tuning pilot.** scanpy's Wilcoxon ranking runs in a numba-parallel regime in which more threads can make it slower,
  so the settings of a tool at a given thread count are not left to the defaults. An untimed pilot
  (`--pilot`) tries the same number of settings variants for both tools (the numba threading layer, the numba, BLAS,
  OpenMP and rayon thread counts, and `n_jobs`), keeps the default unless another variant is at least 5% faster, and
  writes the choice to `tuning.json` and to the run plan. The timed runs use the choice; a *default-settings control*
  runs the untuned settings where the pilot chose otherwise, and is reported separately and labelled. A step that is
  at least 1.5× AND at least 10 s slower at the highest thread count than at a lower one is listed in the
  report (`thread_regressions.csv`).
* **Quiet host.** Every run records the busy cores of the machine that are not its own. A run whose median over its samples
  reaches one core violates the rule and is re-run once, at the end of the pass; a cell that violates twice is
  kept, flagged, with its load in the report. The first attempt's files stay under `raw/_quiet/`.
* **Required results.** The run plan lists the result files each run must leave (its JSON; for a successful
  run its timeline, and the per-cell outputs of the run that saves them). `summarize.py` stops with a non-zero
  exit status if a file is missing (`--allow-incomplete` writes a draft marked as such).
* **Memory.** Peak RSS is sampled every 50 ms (`sample_interval_s` is recorded), per step and per run. The
  baseline is the RSS after the imports and the release of the warm-up's memory (the RSS right after the
  imports is recorded as well), and memory is reported as the absolute peak and as peak minus baseline. A
  peak shorter than the interval can be missed; the lifetime `ru_maxrss` includes the warm-up and is not a run
  metric. Runs that read from disk report page-cache effects only as I/O counters. `--tools ... scanpy-lean` adds an optional
  memory-lean scanpy profile (the highly-variable-gene subset is taken in place; no computed value changes),
  reported next to the idiomatic one in its own table.
* **Monitoring.** RSS every 50 ms, per-core utilisation and clock speed, RAPL energy where
  available, and per-step wall and CPU time.
* **Failures are data.** Out-of-memory kills and timeouts are recorded as failed runs with
  the reason, never dropped.
* **Statistics.** Medians with the min–max range and the number of repeats, the min–max envelope of every
  speed-up ratio, Mann–Whitney U with Hodges–Lehmann estimates on the samples behind the headline, log–log
  scaling exponents per series (real subsamples as the primary series, the synthetic supplement labelled), and
  Amdahl serial fractions from the thread scan. A Mann–Whitney p of 0.0079 is the smallest value possible with
  5 against 5 samples; it says that the samples do not overlap and nothing more.

### Honest reporting rules

These rules apply to every number CREST publishes:

* The **headline** compares the `core` pipeline, which is what CREST is designed to speed up.
  Optional modules (Scrublet, Harmony, the resolution sweep, pseudobulk DESeq2) are reported
  in a **separate** table.
* Speed-ups are quoted **both** at each tool's *best* thread count and at matched thread
  counts. `tables/main.csv` lists every candidate; its `speedup` column is the smallest of them, and the
  pre-registration states which one is quoted as the headline. The matched columns say which thread settings
  they used: *default thread settings* (only the thread count was set) or the tuned settings of the pilot.
  Comparing both tools at the highest thread count would not be a comparison of scanpy at its best, because the
  default settings can leave scanpy in a regime in which it is slower than at fewer threads.
* Steps where CREST is **not** faster, steps that get slower with more threads, CREST's limited thread
  scaling, and every failed run are reported, not hidden.
* Every speed-up is reported with the number of repeats and the min–max range of the ratio.

## Results: 0.4.0 (commit `d75475e`), standard tier, performance governor

Same Threadripper PRO 3975WX (32 cores / 64 threads, 123 GB), CPU governor set to
`performance`, 5 repeats, current protocol (core runs at 8 and 64 threads, modules in separate
3-repeat runs, Scrublet skipped above 150,000 cells). Commit `d75475e` is `dev` with both
Leiden parallelisations merged (phases 2 and 3; phase 1 stays single-threaded, see below) and
the `read_h5ad`/`from_anndata` I/O work. 284/284 runs succeeded, zero failures. This
supersedes the 0.3.0 numbers below for every dataset (those used `powersave` and only had a
thread scan on PBMC 68k); the full `report.md` and figures are archived at
`crest-bench-rinamochana-20261002-0323.tar.gz` on rinamochana.

**Core workflow**, median seconds (read → Wilcoxon markers), matched at 64 threads (default thread
settings for both tools) and best vs best per tool; headline is the smaller of the two:

| dataset | cells | CREST | CREST best (threads) | scanpy | scanpy best (threads) | matched | best vs best | headline | memory ratio |
|---|---|---|---|---|---|---|---|---|---|
| PBMC 3k | 2,700 | 4.35 | 2.79 (8) | 5.21 | 4.68 (8) | 1.20× | 1.68× | **1.20×** | 1.58× |
| PBMC 10k | 11,537 | 5.09 | 5.09 (64) | 31.0 | 17.5 (8) | 6.09× | 3.44× | **3.44×** | 1.59× |
| Kang 2018 | 28,871 | 5.28 | 5.28 (64) | 82.4 | 34.1 (8) | 15.61× | 6.46× | **6.46×** | 2.38× |
| PBMC 68k | 68,551 | 9.13 | 9.13 (64) | 210 | 93.8 (8) | 22.97× | 10.28× | **10.28×** | 3.17× |
| synthetic 100k | 99,779 | 11.5 | 11.5 (64) | 306 | 142 (8) | 26.50× | 12.31× | **12.31×** | 3.93× |
| synthetic 200k | 199,531 | 20.0 | 20.0 (64) | 555 | 333 (8) | 27.83× | 16.70× | **16.70×** | 4.70× |

Mann-Whitney p = 0.0079 on every dataset is the smallest value possible with 5 vs 5 samples: the two
samples do not overlap, and nothing more. CREST's own best thread
count is 64 everywhere except PBMC 3k (8); scanpy's best is consistently 8 — it gets *slower*
above that (PBMC 68k: 93.8 s at 8 threads vs 210 s at 64, worse than its own 1-thread time).
The 64-thread scanpy times are those of the default thread settings: scanpy's Wilcoxon ranking runs in a
numba-parallel regime that is slower at high thread counts, which is why the protocol above adds a tuning
pilot and a default-settings control instead of treating the 64-thread column as scanpy's best.
Out-of-core CREST trails in-memory CREST by 5–18%, as before.

**Where CREST is not faster (< 1.2×):**

| step | CREST vs scanpy |
|---|---|
| `seurat_v3` HVG (10k–200k cells) | 0.48–0.56× (loess fit runs in Python; at 2,700 cells, PBMC 3k, it's 1.14×) |
| scale + PCA, PBMC 3k | 0.72× (fixed eigendecomposition cost) |
| UMAP, PBMC 3k | 0.83× (fixed cost below 10k cells) |
| QC filter, PBMC 3k | 1.02× (both near the measurement floor) |
| reading the file (all datasets) | 1.13–1.19× (near parity; both HDF5-decompression-bound) |

**Thread scaling** (PBMC 68k, core workflow, 1→64 threads): CREST 46.3 s → 9.13 s (5.08×,
Amdahl serial fraction 0.19, still improving at 64). scanpy 195 s → 93.8 s at its best (8
threads, 2.08×) then *regresses* to 210 s at 64 (Amdahl serial fraction 0.59).

**Scaling with cells.** No scaling exponent is quoted. The exponent that was published
earlier came from three dataset sizes (one real dataset and two synthetic ones derived from
it), too few points for a meaningful interval. A fit needs at least five sizes from one
generator (the `full` tier) and is reported with its 95% interval and number of points
(`tables/scaling.csv`) once such a run exists.

**Agreement with scanpy:** PCA subspace cosine 1.000 and 15-NN neighbour-set Jaccard in the
two PCA spaces 1.000 (both computed with CREST's kNN; each tool builds and clusters its own
graph) on every dataset; HVG Jaccard 0.994–1.000 (`seurat`) and 0.998–1.000 (`seurat_v3`); Leiden ARI
0.70–0.94 / NMI 0.86–0.94 (same range as 0.3.0); UMAP trustworthiness 0.86–0.96, matching
scanpy's own to within 0.01; top-50 marker overlap 0.90–0.96.

**Optional modules** (64 threads, 3 repeats): `leiden_sweep` 14.6–22.7× faster; Scrublet
1.3–48.8× (shrinks with cell count, as before — both tools' neighbour search grows
roughly quadratically); Harmony 2.3× on Kang (ARI vs cell type 0.82 CREST / 0.818 harmonypy,
15-NN neighbourhood overlap 0.98). Three module runs stand out as new comparisons this round:
Scrublet-equivalent doublet detection vs `scanpy.pp.scrublet` on Kang is **49× faster** at
identical AUROC (0.863 both, Spearman 0.983 on doublet scores); `ingest` label transfer is
**17.6× faster** and *more accurate* than scanpy 1.11 (90.7% vs 89.8%, label agreement 96.4%); pseudobulk
DESeq2 on Kang (8 cell types) matches R's significant-gene calls almost exactly (identical
significant sets in 6 of 8 cell types; 1 and 21 discordant genes in the other two). Its speed
relative to R DESeq2 and pydeseq2 is not quoted here: the earlier figures were not
thread-matched, and the thread-matched comparison (`bench/deseq2/kang_pseudobulk.py`) is being
re-run. pydeseq2 itself diverges from R noticeably more than CREST does on some
cell types (e.g. 914 vs R's 794 significant genes on B cells).

## Results: 0.3.0, standard tier

Threadripper PRO 3975WX (32 cores / 64 threads, 128 GB), Ubuntu 26.04, CPU governor
`powersave`, 5 repeats. The code measured (commit `3db58a8`) is identical to the 0.3.0
release except for docstrings and version strings. This run used the earlier protocol: the
main runs were full-profile runs at 64 threads only, so the core-workflow times below are the
sum of the core steps inside those runs, and scanpy's best thread count was measured only on
pbmc68k.

**Core workflow**, median seconds at 64 threads (read → Wilcoxon markers):

| dataset | cells | CREST | scanpy | speed-up, matched threads | speed-up, best vs best |
|---|---|---|---|---|---|
| PBMC 3k | 2,700 | 4.44 | 5.38 | 1.2× | not measured |
| PBMC 10k | 11,537 | 5.10 | 30.5 | 6.0× | not measured |
| Kang 2018 | 28,871 | 5.79 | 80.9 | 14.0× | not measured |
| PBMC 68k | 68,551 | 10.6 | 211 (94.3 at 8 threads) | 19.8× | **8.9×** (8.7–9.0) |
| synthetic 100k | 99,779 | 13.6 | 305 | 22.4× | not measured |
| synthetic 200k | 199,531 | 26.2 | 555 | 21.2× | not measured |

On PBMC 68k, the only dataset with a thread scan, scanpy was 2.2× faster on 8 threads than on
64. The matched-thread speed-ups on the other datasets are therefore upper bounds, and the
defensible headline is the best-vs-best **8.9×** on PBMC 68k. Core-workflow peak memory
(clean only in the PBMC 68k thread-scan runs) was 3.4× lower with CREST. The current protocol
(core runs at 8 and 64 threads) measures best-vs-best on every dataset.

**Where CREST is not faster:**

| step | CREST vs scanpy |
|---|---|
| `seurat_v3` HVG (10k–200k cells) | 0.49–0.55×: **2× slower** (the loess fit runs in Python) |
| UMAP, PBMC 3k | 0.81× (fixed cost: 500 epochs below 10k cells) |
| scale + PCA, PBMC 3k | 0.90× (fixed cost: ~1 s eigendecomposition of the 2,000 × 2,000 matrix) |
| reading the file | 1.06–1.17× (both are limited by HDF5 decompression) |

**Thread scaling** (PBMC 68k, core workflow): CREST 45.6 s on 1 thread → 10.6 s on 64 (4.3×;
Amdahl serial fraction 0.24). scanpy 193 s → 94.3 s at 8 threads (its best), 211 s at 64.

**Optional modules** (64 threads): resolution sweep 11–15× faster, Harmony 2.5× (Kang),
Scrublet 21–50× on 10k–29k cells but only 1.2–2.7× at 68k–200k (both tools' neighbour
search grows roughly with cells²; 2,590 s vs 3,020 s at 200k).

**Agreement with scanpy:** PCA subspaces identical (cosine 1.000000), 15-NN neighbour-set
Jaccard in the two PCA spaces ≥ 0.99999 (both computed with CREST's kNN; each tool builds and
clusters its own graph), HVG sets 99.4–100% identical (Jaccard), Leiden ARI 0.80–0.94 with the
same number of clusters on 4 of 6 datasets (57 vs 59 and 64 vs 62 on the synthetic sets), UMAP
trustworthiness equal or slightly higher (0.862–0.959 vs 0.863–0.957), top-50 marker overlap
0.92–0.99.

## Results: Parse PBMC atlas, 1M and 9.7M cells

Same machine, commit `3dbd760` (`dev`; includes the copy-free Leiden graph and shared UMAP
snapshot below, but not yet the parallel Leiden refinement further down), 1 repeat, core
workflow at 8 and 64 threads, modules in separate runs, Scrublet skipped (above 150,000
cells). The Parse matrix has 2,000 genes; QC keeps cells with ≥ 20 of them (996,779 of
1,000,000 in the subset; 9,666,140 of 9,697,974 in the full atlas).

**Core workflow, 1M-cell subset** (scanpy was fastest at 64 threads here):

| | CREST | CREST out-of-core | scanpy |
|---|---|---|---|
| time, 64 threads | **85.1 s** | 89.5 s | 994 s |
| time, 8 threads | 154.0 s | 162.3 s | 1,379 s |
| peak memory | 4.7 GB | 3.2 GB | 54.8 GB |

Speed-up **11.7×** at 64 threads (best vs best and matched are the same here; 9.0× at matched
8 threads), memory **11.7× lower**. 1 repeat only (this was the Leiden-seed-baseline /
Leiden-UMAP-speed re-run, not the final paper run); use 5 repeats with the CPU governor set
to `performance` for the paper.

Per step at 64 threads (CREST vs scanpy): scale + PCA 2.7 vs 84.7 s (31×; scanpy's dense
scaled matrix is the 54.8 GB peak), neighbours 8.8 vs 106 s, Leiden 23.7 vs 64.5 s, UMAP 42.0
vs 647 s, Wilcoxon 3.5 vs 79.4 s. Against the previous (`ee87e3f`) measurement, **UMAP
dropped from 83 s to 42 s** (the shared-snapshot fix) while **Leiden held at ~24 s**
(unaffected by that fix, still single-threaded at this commit) — UMAP is now 49% of CREST's
time and Leiden 28%. `seurat_v3` HVG is again slower than scanpy (2.6 vs 1.1 s, unchanged).

**Full atlas, 9.7M cells (CREST only):** the core workflow now takes **21.4 min** in memory
(1,282 s at 64 threads, 40.3 GB peak, vs 27 min / 37 GB previously) and **24.3 min**
out-of-core (1,460 s, 24.9 GB peak, vs 30 min / 30 GB previously). Per step at 64 threads:
UMAP dropped from 934 s to **584 s**, Leiden held at **400 s** (single-threaded, unaffected by
the UMAP/graph-copy fixes) — together still 77% of the time. At 8 threads: 44.3 min in memory,
49.7 min out-of-core. Out-of-core's memory saving shrank further (15.4 GB vs the previous
7 GB) as the in-memory peak itself grew slightly (37 → 40.3 GB) while per-cell structures stayed
the dominant cost either way. The resolution-sweep number below (484 s, 68 GB peak) predates
this commit's graph-copy fix and still needs remeasuring.

**Agreement with scanpy (1M):** identical PCA subspace and HVG sets, 15-NN neighbour-set
Jaccard in the two PCA spaces 1.0 (both computed with CREST's kNN); UMAP trustworthiness 0.944 vs 0.946; top-50 marker overlap 0.94. **Leiden ARI is 0.66**
(27 vs 27 clusters), lower than on the smaller datasets (0.80–0.94), although the two PCA
spaces give the same 15-NN sets (each tool still builds and clusters its own graph).
**Seed-to-seed baseline** (2 extra untimed Leiden runs per tool, each on that tool's own graph): CREST seed-to-seed ARI 0.73, scanpy seed-to-seed ARI 0.64, CREST-vs-scanpy
cross-seed mean ARI 0.67. So CREST-vs-scanpy disagreement (0.66) is close to scanpy's own
seed noise (0.64) and a bit below CREST's own seed noise (0.73): **most of the gap is ordinary
Leiden seed variation on a million-node graph, not a correctness difference**, though CREST's
higher self-consistency leaves a modest residual not yet explained. The resolution sweep shows
the same pattern (ARI 0.96 at resolution 0.2, 0.65–0.69 at 1.0–2.0). Harmony neighbourhoods
agree at 0.82 (0.97–0.98 on Kang).

### Leiden, parallelised phase by phase

The above numbers predate the parallel Leiden work in `CHANGELOG.md` (0.4.0). Profiling
Leiden's own three phases directly (temporary timers, `parse_pbmc_1m`, 64 threads) gives their
share of its wall time: phase 1 (`fast_move_nodes`) ~73%, phase 3 (`aggregate`) ~21%, phase 2
(`refine_partition`) ~3%.

**Phase 2 (refinement):** each unrefined community now runs on its own thread instead of one
sequential pass over all nodes. Measured separately (not through the harness above, same
machine, "FRASER-quiet", i.e. with the shared machine's other heavy jobs not running; the same
applies below; median of 5 `crest.tl.leiden` calls on the 1M-cell graph at 64
threads): **20.1 s vs 23.9 s before, ~16% faster**, modularity unchanged (within the seed
noise quantified above against leidenalg; the 20-seed modularity-sweep script is not yet in
the repository).

**Phase 3 (aggregation):** each community's outgoing edges now compute on its own thread too
(no RNG involved, so this is bit-identical for any thread count — verified by diffing labels
before/after on PBMC 68k). Combined with phase 2: **17.9 s at 1M cells** (median of 5,
FRASER-quiet), down from 20.1 s with phase 2 alone and 23.9 s originally — a 25% reduction
overall. At 9.7M cells, a full core-workflow run (`bench/paper/run_one.py`, FRASER-quiet)
went 1,287.6 s (original) → 1,218.6 s (phase 2) → **1,165.9 s (phase 2+3, -9.5% overall)**,
peak memory ~42.7 GB (up slightly from ~40.3 GB; within the run-to-run variation already seen
at this scale, not attributed to the change). Modularity vs leidenalg unchanged (20-seed sweep
at resolution 1 on PBMC 68k: mean gap +0.00152, 2/20 below — identical to phase-2-alone).

**Phase 1 (`fast_move_nodes`, ~73% of the time): attempted, measured, rejected.** Unlike
phases 2 and 3, a node's move here is not independent of other nodes' concurrent moves, so
this isn't the same safe per-item parallelisation. Two synchronous-round designs were tried
(parallel scan of neighbour weights, sequential commit so the community bookkeeping can't
race): deciding from a round-start snapshot caused a genuine infinite loop (two nodes
repeatedly swapping into each other's not-yet-updated communities); deciding from *live*
community weights at commit time does terminate, but the 20-seed modularity sweep against
leidenalg went from +0.00152 mean / 2-of-20 below to **-0.00083 mean / 15-of-20 below** — a
real, isolated regression (a control run reverting only this change reproduced the
+0.00152/2-of-20 baseline exactly). Per the accuracy rules above, this does not ship.
`fast_move_nodes` stays single-threaded. A future attempt could process a bounded batch of
nodes per round, which reduces the staleness of neighbouring communities at the cost of more
synchronisation, or first make the sequential algorithm cache-friendlier by renumbering nodes
so that neighbours are close in memory, which carries no risk to quality.

## Module benchmarks

Each writes its results to `bench/<module>/results/`:

```bash
python bench/deseq2/compare_r.py --genes 5000      # vs R DESeq2 on simulated designs (needs R)
python bench/deseq2/kang_pseudobulk.py             # vs R and pydeseq2 on Kang 2018
python bench/harmony/compare_harmonypy.py          # vs harmonypy
python bench/doublets/compare_scrublet.py          # vs scanpy, with demuxlet ground truth
python bench/ingest/compare_scanpy_ingest.py       # vs scanpy.tl.ingest
```
