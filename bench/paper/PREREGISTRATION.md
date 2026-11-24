# Pre-registration of the timing benchmark (DRAFT template)

**Status: DRAFT.** This file is a template. It is filled in and committed **before** the timed run starts;
after that point only the *Deviations* section at the end may be appended to. Every `FILL:` marks a value the
final run supplies, every `PENDING:` marks a decision that has not been taken yet (the default written next
to it applies if none is taken), and nothing below has been measured. The code that implements the protocol
is `bench/paper/` (`protocol.py` holds the rules, `run_all.py` the plan, `run_one.py` one run,
`summarize.py` the report); the user-facing description is `docs/benchmarks.md`.

Machine-readable inputs of the run are copied next to this file when it is committed: `plan.json`,
`tuning.json`, `env/`, `dataset_manifest.json` and `verification.json`.

## 1. Code and wheel

| item | value |
|---|---|
| repository and commit of the release candidate | FILL: URL, 40-character SHA |
| release candidate tag | FILL |
| wheel file name | FILL |
| wheel sha256 | FILL (`sha256sum` of the file that was installed into the benchmark environment) |
| `crest.__version__` and build date | FILL |
| built with | FILL: `rustc -V`, `cargo -V`, maturin version (recorded in `env/toolchain.txt`) |
| harness commit (if different from the above) | FILL |
| date of the run | FILL |

The run uses the installed wheel, not a source tree. The test suites (`cargo test`, `pytest`) pass on the
same commit before the run starts (`logs/cargo_test.log`, `logs/pytest.log`).

## 2. Environments and lockfiles

| item | value |
|---|---|
| Python environment of the timed runs | FILL: lockfile name; currently `bench/paper/requirements-bench.txt` (scanpy 1.11.5) |
| scanpy version(s) of the headline comparison | PENDING: default is the newest release at the time of the run in its fastest supported configuration, with the 1.11.5 environment as a continuity table on two datasets. Every sentence that names scanpy states the version |
| lockfile of the second scanpy environment | FILL |
| anndata, numba, llvmlite, pynndescent, umap-learn, igraph, leidenalg, harmonypy | FILL: pinned versions of each environment (all appear in `env/pip_freeze.txt` and `env/python_env.json`) |
| R environment of the DESeq2 reference | FILL: lockfile; the R, DESeq2 and BiocParallel versions are taken from the recorded environment (`env/R_versions.txt`, `env/R_versions.json`, `env/R_sessionInfo.txt`), never from this text |
| numba threading layer and thread count | recorded per run (`env.numba_threading_layer`, `env.numba_num_threads` in each run's JSON) and once in `env/python_env.json` |
| machine | FILL: CPU model, cores, SMT, RAM, OS, governor, turbo setting (`env/summary.md`, `env/lscpu.txt`, `env/smt_siblings.txt`) |

## 3. Datasets

Every converted file is checked count for count against independent readers (`datasets.py --verify`: scipy for the
MTX sources, scanpy's readers, and CREST's readers against the independent read; the h5ad sources and the
subsamples against their parents). A mismatch stops the run. The MTX sets are converted with scipy, not with CREST's
reader. The licence of a dataset is not filled in by any script: it is read at the data provider and entered here.

| dataset | role | cells | source (from the registry) | licence | retrieved | source sha256 | converted sha256 | check |
|---|---|---|---|---|---|---|---|---|
| `pbmc3k` | continuity | 2,700 | 10x Genomics, `cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/` | FILL | FILL | FILL | FILL | FILL |
| `pbmc10k` | continuity | 11,769 | 10x Genomics, `pbmc_10k_v3` (v3 chemistry) | FILL | FILL | FILL | FILL | FILL |
| `pbmc68k` | continuity | 68,579 | 10x Genomics, `fresh_68k_pbmc_donor_a` | FILL | FILL | FILL | FILL | FILL |
| `kang` | continuity; module inputs | 29,065 | GEO GSE96583 (batch 2), Kang et al. 2018 | FILL | FILL | FILL | FILL | FILL |
| `neurons_1m` | non-blood, scaling parent | 1,306,127 | 10x Genomics, `1M_neurons` (mouse brain) | FILL | FILL | FILL | FILL | FILL |
| `covid_stephenson` or another multi-donor case/control set | pseudobulk (donor-level, design `~ condition`) | FILL | PENDING: CELLxGENE collection of Stephenson et al. 2021 by default | FILL | FILL | FILL | FILL | FILL |
| `parse_pbmc` | split-pool technology, CREST only at 9.7M cells (a 2,000-gene panel) | 9,697,974 | figshare 28589774 (Parse Biosciences) | FILL | FILL | FILL | FILL | FILL |
| `parse_pbmc_1m`, `parse_pbmc_sub250k`, `parse_pbmc_sub500k` | real-subsample scaling series | 1M / 250k / 500k | seeded random subsets of `parse_pbmc` | as parent | as parent | as parent | FILL | FILL |
| `neurons_1m_sub25k` ... `neurons_1m_sub500k` | real-subsample scaling series | 25k, 50k, 100k, 200k, 500k | seeded random subsets of `neurons_1m` | as parent | as parent | as parent | FILL | FILL |
| `synth_25k` ... `synth_200k` | **labelled supplement only** | 25k to 200k | interpolated from 68k real PBMCs with CREST's own normalisation, PCA and kNN | n/a (derived) | n/a | n/a | FILL | reader check only |

PENDING: the dataset panel and the subsample sizes are drafts until the data owner confirms availability and
licences. The subsets of one parent and seed are nested (prefixes of one random permutation), the seed is
`SUBSAMPLE_SEED = 0` unless changed here: FILL. The cell-count scaling statement rests on the real subsamples
only; the synthetic sets are shown separately and labelled.

Scope sentence for the paper: the comparison is validated on UMI-count droplet data and split-pool data; it
makes no claim for Smart-seq or spatial data. The Parse matrix holds 2,000 genes and is labelled as such
wherever its 9.7M cells are mentioned.

## 4. Endpoints

* **Primary: end-to-end process wall time** = process creation to the end of the last pipeline step:
  interpreter start-up, the imports the workflow needs (all modules each tool loads for it are imported in the timed
  start-up block), and the timed pipeline including the garbage collections and result extraction between steps.
  Field: `end_to_end_seconds` = `startup_seconds` + `total_seconds`.
* **Secondary: sum of the step timers** (`core_seconds`: read to Wilcoxon markers).
* Also recorded for every run and tool: `import_seconds`, `untracked_seconds` (timed block minus the step
  timers) and `process_wall_seconds` (parent's wall clock around the whole process, which includes the warm-up and the
  untimed accuracy extras and is **not** an endpoint).
* The JIT / file-cache warm-up inside the process is not part of any endpoint. The **warm numba cache is the primary
  condition.** Cold-cache cost note: FILL (three datasets, run by hand with `run_one.py --no-warmup` and an empty
  `NUMBA_CACHE_DIR`; this is not part of the automated plan).
* Both endpoints are reported with the full per-repeat distribution (`tables/runs.csv`).
* Per-step comparisons are made on fused stages (normalize + log1p + HVG as one stage, scale + PCA as one); no ratio
  is formed for a stage below 0.01 s and none above 1000 is printed.

## 5. Thread grid

* Both tools, every dataset: 1, 2, 4, 8, 16, 32, 64 threads (`--thread-grid`), clamped to the logical CPUs.
  FILL: the grid actually used.
* If the runtime does not allow the full grid on every dataset: the untimed pilot (section 6) fixes each tool's best
  grid point per dataset (`--grid-mode best`), and the points run are 1, that best point and all CPUs. FILL: which mode.
* Out-of-core CREST runs the same grid unless stated: FILL.
* The thread scan on one dataset (`--thread-dataset`, `--threads-scan`) is part of the grid on that dataset: FILL: dataset.

## 6. Tuning pilot (untimed)

* Purpose: scanpy's Wilcoxon ranking runs in a numba-parallel regime that can get slower with more threads, so
  settings are tuned instead of left at their defaults. **CREST gets the same tuning budget**: both tools run the same
  number of variants at every thread count (`protocol.VARIANTS`; a test checks that the numbers are equal).
* Variants (at *p* threads; `default` sets every thread variable to *p* and `n_jobs` to *p*):

  | tool | variants |
  |---|---|
  | scanpy | `default`; `blas1` (BLAS and OpenMP threads 1); `numba_cap8` (numba threads at most 8); `workqueue` (`NUMBA_THREADING_LAYER=workqueue`) |
  | CREST | `default`; `blas1`; `rayon_cap8` (rayon threads at most 8); `rayon_half` (rayon threads p/2) |

  PENDING: further scanpy configurations that the chosen scanpy version supports (for example another Wilcoxon
  backend) are added to `protocol.VARIANTS` before the pilot, with the same number of variants for CREST.
* Pilot datasets: FILL (default: the thread-scan dataset and one smaller dataset); one repeat per
  (dataset, tool, thread count, variant); results in `pilot/`, never in the report.
* Choice: per tool and thread count the variant with the lowest mean of time over fastest-variant time across the
  pilot datasets; the default keeps its place unless the winner is more than 5% faster. The choice is written to
  `tuning.json` and into `plan.json` (`tuning`, and the resolved `settings` of every run).
* A **default-settings control** (the untuned settings, at 8 threads and all CPUs, on the thread-scan dataset) is
  reported separately and labelled; steps with `t_many >= 1.5 * t_few` and `t_many - t_few >= 10 s` are listed.

## 7. Repeats, interleaving, pinning

* 5 repeats per configuration (module runs: FILL, default 3), each in a fresh process; repeat *r* of every configuration
  runs before repeat *r*+1 of any, in a seeded random order (seed `20260928` unless changed: FILL).
* **Pinning policy, identical at every thread count:** every run is pinned (`sched_setaffinity`) to the first *p* logical CPUs
  in physical-core-first order (one context of every core before any second SMT sibling); with *p* = all logical CPUs
  the mask is all CPUs. The CPUs of every run, the affinity and the SMT sibling map are recorded. On a 32-core machine with
  SMT, 64 threads therefore use both contexts of each core while 1 to 32 use one: thread count and SMT use change
  together, and this is stated wherever the 64-thread numbers appear.
* **Controls (never in any statistic):** the same run unpinned at 8 threads, and the default-settings control above,
  on the thread-scan dataset. FILL: datasets and thread counts if different.
* Resume: a run whose JSON exists with status `ok` is skipped; every attempt is appended to `attempts.jsonl`.

## 8. Memory

* Peak RSS of the process (children included) sampled every 50 ms (`--sample-interval`; recorded as
  `sample_interval_s`), per step and per run.
* The baseline is taken after the imports and after the warm-up's memory is released to the operating system
  (`malloc_trim` on glibc); the RSS right after the imports is recorded too. Memory is reported as the absolute peak and as
  peak minus baseline, from core-profile runs only (the module runs contain the optional modules).
* A peak shorter than the sampling interval can be missed; the lifetime `ru_maxrss` includes the warm-up and is not used.
* Out-of-core runs report page-cache effects only through the I/O counters; they do not drop the cache between repeats.
* A **memory-lean scanpy profile** (`scanpy-lean`, in-place HVG subsetting, no computed value changes) runs alongside the
  idiomatic scanpy on FILL: datasets, and is reported next to it in `tables/memory_lean.csv`.

## 9. Statistics and headline rule

* Statistic: median and min-max range over the repeats (n = 5), and the min-max envelope of every ratio. No bootstrap
  intervals. The Mann-Whitney p and the Hodges-Lehmann shift are written to the tables for completeness; no significance
  statement is made from them (with 5 against 5 the smallest possible p is 0.0079).
* **Headline rule. PENDING owner decision.** Default: the headline is **best-vs-best on the identical thread grid**,
  with scanpy tuned (section 6) at its fastest supported configuration, quoted **together with** the single-thread
  ratio and the matched-thread curve. The earlier wording "the lower of the matched and best-vs-best ratios" is dropped
  because it is a data-dependent minimum. Alternative if the owner keeps the earlier rule: the abstract quotes both the
  best-vs-best and the single-thread ratios and the smaller one is the headline. The report's `speedup` column is the
  minimum over all candidates and is not the quoted number unless the alternative is chosen; the candidate columns
  (`speedup_best`, `speedup_t<p>`) are all in `tables/main.csv`.
* Not-faster rules: a step is listed as not clearly faster below 1.2x; a step is listed as slower with more threads when
  `t_many >= 1.5 * t_few` and `t_many - t_few >= 10 s`.
* Scaling: a log-log fit with its 95% interval and the number of points, per series; real subsamples are the evidence,
  the synthetic series is shown separately and labelled. No exponent is quoted from fewer than 5 sizes of one series.

## 10. Quiet host and re-run cap

* A run violates the quiet-host rule when the **median** over its samples of the busy cores that are not the run's own
  (machine-wide utilisation minus the process's CPU use, per 50 ms sample) is **1 core or more**.
* Only violating runs are re-run, **once**, at the end of the pass (`quiet_pass`); the first attempt's files are kept in
  `raw/_quiet/`. A cell that violates twice is kept with its flag and its load in the report; it is not dropped and not
  re-run again. FILL: the machine's arrangement for an exclusive window (agreed no-job period or cgroup).

## 11. Failures and required files

* A CREST or R failure on a dataset is a reported result, unless the problem is critical by the release rules (silently
  wrong results on plausible input, a crash on plausible input, or a false published claim); then the code is fixed before
  the run, not during it.
* Out-of-memory kills and timeouts (`--timeout`, FILL) are recorded as failed runs with their reason. Every run starts in its own
  memory scope (`systemd-run --user --scope`, `MemoryMax` = the cap, swap off), so that one run's kill ends that run only: it is
  recorded with status `oom`, the cap and the scope's peak memory, and the pass goes on. Per-run memory cap: FILL (`--memory-cap-gb`;
  default 3/4 of the RAM, 92 GB on the 123 GB workstation of the shake-down). A resumed pass keeps an `oom` result obtained at the
  same cap and runs the other failed runs again.
* **Required result files** per planned (tool, dataset, threads, repeat): its JSON always; for a successful run also its
  timeline CSV; for the run that saves outputs also the per-cell `.outputs.npz`. `summarize.py` exits non-zero when
  one is missing; the report of a draft made with `--allow-incomplete` carries an INCOMPLETE banner.
* The module scripts end with a non-zero status if any module failed (`module_failures.txt`).

## 12. Duration estimate

* The published run (standard tier, 6 datasets, 3 tools, main thread counts 8 and 64, thread scan on one dataset): 284 runs,
  **7.7 h measured wall time**. The planner (`run_all.py --dry-run`) reproduces the 284 runs and estimates 8.3 h for it.
* Planner estimates for the draft protocol on the same six datasets (a 64-CPU machine; the cost model is calibrated on the
  published run and is good to about +-50%; it does not know the added datasets):

  | configuration | runs | estimated wall time |
  |---|---|---|
  | full grid 1-64 (CREST, out-of-core CREST, scanpy), 5 repeats, modules with 3 repeats | 684 | 17.3 h |
  | the same with controls and the memory-lean profile | 909 | 29.1 h |
  | best-grid mode (1, pilot best, all CPUs) with the thread scan | 264 | 9.6 h |
  | tuning pilot, one dataset (2 tools x 7 thread counts x 4 variants) | 56 | 1.8 h |

  FILL: the estimate printed by the planner for the final configuration, including the datasets added to the panel
  and the second scanpy environment. An exclusive window of that length plus the re-runs is requested.
* Resume: after an interruption the same command continues; finished runs are skipped, failed ones are retried, and
  every attempt stays in `attempts.jsonl`.

## 13. Modules in the same run

* DESeq2 thread-matched against R (`bench/deseq2/kang_pseudobulk.py`, thread counts 1 to 64; R's version from the
  recorded environment).
* ingest like-for-like against `scanpy.tl.ingest`.
* Scrublet with matched windows (per-dataset ratios, no pooled number).
* Harmony with the donor as the batch.
* kNN recall at scale on each dataset's real embedding, scanpy's kNN scored with the corrected graph reader; CREST's kNN is never
  used on both sides of a comparison.
* Leiden modularity sweep against leidenalg (`bench/paper/modularity_sweep.py`, 20 seeds).

The untimed agreement run (ARI / NMI against annotation, common-input decomposition, DE concordance, kNN recall,
UMAP trustworthiness, DESeq2 concordance) is registered separately; its thresholds are fixed before the freeze.

## 14. Deviations

*(append-only, after the run starts: date, what differed from the above, why)*

## 15. Checklist before the run starts

- [ ] sections 1 to 13 have no `FILL:` left and every `PENDING:` is decided or its default accepted in writing
- [ ] the wheel's sha256 equals the installed one; `crest.__version__` is the release candidate's
- [ ] `datasets.py --verify` passed for every dataset of the panel; `dataset_manifest.json` has every licence filled in
- [ ] the tuning pilot has run and `tuning.json` is committed with this file
- [ ] `run_all.py --dry-run` shows the planned runs and the estimate stated in section 12
- [ ] the machine is in its exclusive window (no foreign jobs), governor and turbo recorded
- [ ] this file is committed, and its commit is named in the run's `meta.json`
