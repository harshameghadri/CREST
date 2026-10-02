# Benchmarks and validation

CREST is held to two claims, and both are measured:

1. **Same results as the reference tool.** This is checked by the test suite on every
   change, and on real data in the module benchmarks.
2. **Faster and leaner.** This is measured by the paper benchmark, on your own machine, with
   a protocol designed to be reported honestly.

## Accuracy (what the tests pin)

| step | reference | agreement |
|---|---|---|
| QC, filters | scanpy 1.11 | identical |
| HVG `seurat`, `cell_ranger` | scanpy | identical gene set; `dispersions_norm` within 2e-6 |
| HVG `seurat_v3` | scanpy + skmisc loess | identical sets and ranks; `variances_norm` within 1e-14 |
| scale + PCA | scanpy (dense) | identical subspace (0.000° principal angles) |
| neighbours (connectivities) | scanpy / umap-learn | identical given the same kNN (1e-5) |
| Leiden | leidenalg | similar number of clusters; modularity typically equal or higher (mean +0.0015 over 20 seeds at resolution 1 on PBMC 68k), occasionally ~0.002 lower on a given seed — both tools' own seed-to-seed variation is of the same order, so this is not a per-seed guarantee |
| t-test, Wilcoxon | scanpy | scores within 1e-4, same p-values |
| `score_genes` | scanpy | identical incl. control-gene sampling (3e-7) |
| DESeq2 Wald / LRT | R DESeq2 1.42 | ~1e-9 (Wald, simulated), 1e-6 (LRT) |
| Harmony | harmonypy 2.0 | same iLISI / cLISI / ARI; neighbourhood overlap at seed-noise level |
| Scrublet | scanpy | same AUROC on demuxlet doublets; Spearman 0.98 |
| ingest | scanpy | 90.7% vs 89.8% accuracy; same label for 96.4% of cells |

Details: {doc}`downstream`, {doc}`deseq2`.

## The paper benchmark (one command)

```bash
cd /a/disk/with/space          # everything is written under ./crest-bench (nothing in $HOME)
curl -LO https://raw.githubusercontent.com/harshameghadri/CREST/dev/scripts/crest_paper_bench.sh
bash crest_paper_bench.sh --tier quick      # ~35 min: 3 datasets, 2 repeats
bash crest_paper_bench.sh --tier standard   # ~1 day on 64 cores: 6 datasets, 5 repeats, thread scan
bash crest_paper_bench.sh --tier full       # adds the 647k COVID atlas and 1.3M neurons
```

The script:

1. checks the tools (git, curl, uv, Rust) and the free disk;
2. clones CREST at `--ref` (default `dev`) into a throwaway uv venv and builds it in release mode;
3. runs `cargo test` and `pytest`, and stops if they fail;
4. records the machine: CPU model, caches, governor, cores, RAM, disk, OS, compilers, BLAS
   and every package version;
5. downloads and converts the datasets (resumable, size-checked, SHA-256 recorded);
6. runs every (dataset × tool × repeat) in its own process;
7. checks agreement with scanpy (ARI/NMI, kNN Jaccard, principal angles, HVG and DE overlap);
8. runs the module benchmarks (DESeq2, Harmony, Scrublet, ingest);
9. computes statistics and writes tables (CSV + LaTeX), figures (PDF + PNG), `report.md`
   and a `.tar.gz`.

Datasets: PBMC 3k, PBMC 10k, Kang 2018 (29k, 8 donors × 2 conditions), PBMC 68k, the
Stephenson 2021 COVID atlas (647k), 10x 1.3M neurons, the Parse Biosciences PBMC cytokine
atlas (12 donors × 90 cytokines + PBS: `parse_pbmc`, all 9.7M cells × 2,000 genes, and
`parse_pbmc_1m`, a seeded random 1M-cell subset of it), and synthetic 100k–1M sets derived
from PBMC 68k. At 9.7M cells scanpy cannot run in 128 GB (its dense scaled matrix alone would
be 78 GB), so the tool comparison uses `parse_pbmc_1m` and the full atlas shows CREST alone
at scale. The Parse matrix holds only 2,000 genes, so its QC threshold is scaled to the same
fraction of genes (`min_genes` 20 of 2,000 instead of 200 of ~20,000), for both tools.

The Parse file (13 GB, [figshare 28589774](https://figshare.com/articles/dataset/pbmc_parse/28589774))
cannot always be downloaded by a script. Download it in a browser, put it in
`crest-bench/data/`, and run with `--datasets "parse_pbmc_1m parse_pbmc"` (or `--tier full`). The raw counts
are taken from `layers/counts`, `raw/X` or `X` (the first that holds integer counts), and the
donor, cytokine and cell-type columns are detected automatically. The pseudobulk DESeq2 step
compares IFN-beta with PBS (or, if absent, the most frequent condition with the control). To check what will be used before a long run:

```bash
python bench/paper/datasets.py --inspect crest-bench/data/Parse_1M_adata_for_cellflow_datasets_with_embeddings.h5ad
```

### Protocol (why the numbers can be trusted)

* **Three kinds of runs.** The *core workflow* of every dataset × tool runs at each of
  `--main-threads` (default: 8 and all cores), so every speed-up can be compared at matched
  thread counts and at each tool's best. The *optional modules* run separately, so their time
  and memory never leak into the headline. Scrublet is skipped there above
  `--scrublet-max-cells` (150,000), because its neighbour search grows roughly with the square
  of the number of cells. A *thread scan* runs the core workflow at 1, 2, 4, … threads on one
  dataset.
* **Fresh process per run.** No cache, JIT or allocator state leaks between runs. scanpy's
  numba JIT is warmed up before timing.
* **Interleaved, shuffled repeats.** Repeat *r* of every configuration runs before repeat
  *r+1* of any, in a seeded random order. Thermal drift and background load then spread over
  all tools instead of biasing one.
* **Thread control.** `RAYON/OMP/OPENBLAS/MKL/NUMBA` thread variables are set and, for the
  thread scan, the process is pinned to the first *p* cores.
* **Monitoring.** RSS every 50 ms, per-core utilisation and clock speed, RAPL energy where
  available, and per-step wall and CPU time.
* **Failures are data.** Out-of-memory kills and timeouts are recorded as failed runs with
  the reason, never dropped.
* **Statistics.** Medians with bootstrap 95% CIs of the speed-up ratio, Mann–Whitney U with
  Hodges–Lehmann estimates, log–log scaling exponents, and Amdahl serial fractions from the
  thread scan.

### Honest reporting rules

These rules apply to every number CREST publishes:

* The **headline** compares the `core` pipeline, which is what CREST is designed to speed up.
  Optional modules (Scrublet, Harmony, the resolution sweep, pseudobulk DESeq2) are reported
  in a **separate** table.
* Speed-ups are quoted **both** at each tool's *best* thread count and at matched thread
  counts. The headline uses the more conservative of the two. For example, scanpy is
  **slower** at 64 threads than at 8 (thread-pool contention), so comparing both tools at 64
  threads would flatter CREST.
* Steps where CREST is **not** faster, CREST's limited thread scaling, and every failed run
  are reported, not hidden.
* Every speed-up has a confidence interval and the number of repeats.

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

**Agreement with scanpy:** PCA subspaces identical (cosine 1.000000), 15-NN graphs identical
(Jaccard ≥ 0.99999), HVG sets 99.4–100% identical (Jaccard), Leiden ARI 0.80–0.94 with the
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

**Agreement with scanpy (1M):** identical PCA subspace, 15-NN graph (Jaccard 1.0) and HVG
sets; UMAP trustworthiness 0.944 vs 0.946; top-50 marker overlap 0.94. **Leiden ARI is 0.66**
(27 vs 27 clusters), lower than on the smaller datasets (0.80–0.94), although both tools
cluster the identical graph. **Seed-to-seed baseline** (2 extra untimed Leiden runs per tool,
same graph): CREST seed-to-seed ARI 0.73, scanpy seed-to-seed ARI 0.64, CREST-vs-scanpy
cross-seed mean ARI 0.67. So CREST-vs-scanpy disagreement (0.66) is close to scanpy's own
seed noise (0.64) and a bit below CREST's own seed noise (0.73): **most of the gap is ordinary
Leiden seed variation on a million-node graph, not a correctness difference**, though CREST's
higher self-consistency leaves a modest residual not yet explained. The resolution sweep shows
the same pattern (ARI 0.96 at resolution 0.2, 0.65–0.69 at 1.0–2.0). Harmony neighbourhoods
agree at 0.82 (0.97–0.98 on Kang).

### Leiden, parallelised phase by phase

The above numbers predate the parallel Leiden work in `CHANGELOG.md` (Unreleased). Profiling
Leiden's own three phases directly (temporary timers, `parse_pbmc_1m`, 64 threads) gives their
share of its wall time: phase 1 (`fast_move_nodes`) ~73%, phase 3 (`aggregate`) ~21%, phase 2
(`refine_partition`) ~3%.

**Phase 2 (refinement):** each unrefined community now runs on its own thread instead of one
sequential pass over all nodes. Measured separately (not through the harness above, same
machine, FRASER-quiet, median of 5 `crest.tl.leiden` calls on the 1M-cell graph at 64
threads): **20.1 s vs 23.9 s before, ~16% faster**, modularity unchanged (within the seed
noise quantified above against leidenalg).

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
`fast_move_nodes` stays single-threaded; see `HANDOVER.md` for what a future attempt should
try differently.

## Module benchmarks

Each writes its results to `bench/<module>/results/`:

```bash
python bench/deseq2/compare_r.py --genes 5000      # vs R DESeq2 on simulated designs (needs R)
python bench/deseq2/kang_pseudobulk.py             # vs R and pydeseq2 on Kang 2018
python bench/harmony/compare_harmonypy.py          # vs harmonypy
python bench/doublets/compare_scrublet.py          # vs scanpy, with demuxlet ground truth
python bench/ingest/compare_scanpy_ingest.py       # vs scanpy.tl.ingest
```
