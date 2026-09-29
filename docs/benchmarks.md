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
| Leiden | leidenalg | same number of clusters, equal or higher modularity |
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
Stephenson 2021 COVID atlas (647k), 10x 1.3M neurons, and synthetic 100k–1M sets derived
from PBMC 68k.

### Protocol (why the numbers can be trusted)

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

## Module benchmarks

Each writes its results to `bench/<module>/results/`:

```bash
python bench/deseq2/compare_r.py --genes 5000      # vs R DESeq2 on simulated designs (needs R)
python bench/deseq2/kang_pseudobulk.py             # vs R and pydeseq2 on Kang 2018
python bench/harmony/compare_harmonypy.py          # vs harmonypy
python bench/doublets/compare_scrublet.py          # vs scanpy, with demuxlet ground truth
python bench/ingest/compare_scanpy_ingest.py       # vs scanpy.tl.ingest
```
