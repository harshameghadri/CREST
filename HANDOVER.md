# CREST handover

Where the project stands and how to continue on a local machine. Keep this
file current: update **Status** and **Next steps** at the end of every work
session, cloud or local.

_Last updated: 2026-09-28 (session 2)._

## Status

| Area | State |
|---|---|
| Core pipeline (QC → HVG → scale/PCA → kNN → Leiden → UMAP → t-test/Wilcoxon) | Done, validated against scanpy 1.11 (`tests/`), 6–7× faster, 4–10× less memory |
| Out-of-core (Parquet) mode | Done, ~1 GB peak regardless of dataset size |
| Pseudobulk DESeq2 in Rust | Done: Wald + **LRT (new)**, matches R DESeq2 1.42 (`docs/deseq2.md`) |
| **Downstream tools (new)** | Harmony (Rust), Scrublet, seurat_v3 HVG (loess port), Leiden resolution sweep, ingest/label transfer, `knn_query`; all validated, see `docs/downstream.md` |
| **Paper benchmark (new)** | `scripts/crest_paper_bench.sh` + `bench/paper/`: 6 real datasets + synthetic series, per-core CPU/clock/memory monitoring, stats, figures. Tested end to end on pbmc3k/Kang in the cloud; **the real run is for your machine** |
| Pruning (new) | Removed `.bio` Polars plugin layer (+ Rust polars deps), SLAF, legacy `bench/*.py`, notebooks, Docker, `docs/dev/`, old whitepaper scripts |
| Packaging | PyPI name `crest-sc`, import name `crest`, version 0.2.0 (DESeq2 + downstream tools not yet released) |
| CI | `.github/workflows/CI.yml`: `cargo test` + `pytest`, then maturin wheels for all platforms (release on tag) |

### Git / GitHub state

* [PR #1](https://github.com/harshameghadri/CREST/pull/1) (0.2.0 + DESeq2) is **merged into `main`**.
* `dev` exists (created from `main` after the merge). Feature work goes into `dev` via PRs.
* This session's work is on `claude/awesome-lamport-dsvfv2`, restarted from `main` after the merge,
  with a PR into `dev`.
* Still to do by hand (Claude Code's safety check blocks deleting shared branches):
  archive-tag and delete `feat/leiden-hnsw-faer` and `fix/production-readiness`. Run
  `bash scripts/crest_git_housekeeping.sh` (dry run) and then with `--apply`.

## Setting up locally

```bash
git clone https://github.com/harshameghadri/CREST && cd CREST
git checkout claude/awesome-lamport-dsvfv2     # or main once PR #1 is merged

# toolchain: Rust (stable >= 1.80), Python >= 3.10
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
uv venv .venv --python 3.11 && source .venv/bin/activate     # or python3 -m venv .venv
uv pip install maturin
maturin develop --release -E test,bench          # builds the Rust extension in place

cargo test --release                              # Rust unit tests (26)
pytest -q                                         # Python tests incl. scanpy + R-DESeq2 parity (27)
```

Optional, needed only to re-run the R comparison (the tests don't need R):

```bash
# Debian/Ubuntu:  sudo apt install r-bioc-deseq2 r-cran-jsonlite
# macOS / other:  R -e 'install.packages(c("BiocManager","jsonlite")); BiocManager::install("DESeq2")'
pip install pydeseq2                              # only for the speed comparison
```

On macOS use `--scanpy-max-cells` in the benchmark runner (macOS swaps
instead of OOM-killing).

## Reproducing the results

```bash
# everything (build, tests, data, runs, stats, figures, tarball), on the local machine.
# Run it from a roomy disk: all files and caches go under ./crest-bench (nothing in $HOME).
cd /mnt/scratch
bash scripts/crest_paper_bench.sh --tier quick        # ~20 min sanity run
bash scripts/crest_paper_bench.sh --tier standard     # the paper tables; --tier full adds 647k + 1.3M cells

# individual module comparisons (each writes bench/<module>/results/)
python bench/deseq2/compare_r.py --genes 5000         # DESeq2 vs R on simulated designs (needs R + DESeq2)
python bench/deseq2/kang_pseudobulk.py                # DESeq2 vs R / pydeseq2 on Kang 2018
python bench/harmony/compare_harmonypy.py             # Harmony vs harmonypy
python bench/doublets/compare_scrublet.py             # Scrublet vs scanpy, demuxlet ground truth
python bench/ingest/compare_scanpy_ingest.py          # label transfer vs scanpy.tl.ingest
Rscript tests/data/make_lrt_fixtures.R                # regenerate the R LRT reference outputs
```

## Code map

| Path | What |
|---|---|
| `src/kernels.rs` | Fused chunk kernels (QC, gene stats, Gram/PCA, projection, group sums, Wilcoxon) over a `ChunkView` |
| `src/knn.rs`, `src/leiden.rs`, `src/umap/` | kNN (GEMM / IVF + NN-descent), Leiden, UMAP |
| `src/deseq/mod.rs` | DESeq2 pipeline: size factors, dispersions, trend, MAP, IRLS/Newton, Cook's, outlier refit |
| `src/deseq/linalg.rs`, `src/deseq/lowess.rs` | Small dense LU; port of R's `clowess` |
| `src/harmony.rs` | Harmony (harmony2) |
| `src/knn.rs` | also `knn_query` (reference → query) |
| `src/py.rs` | PyO3 bindings (numpy in/out, GIL released) |
| `crest/core.py` | `BioFrame` and the stores (CSR, Polars frame, Parquet) |
| `crest/pp.py`, `crest/tl.py` | scanpy-style API |
| `crest/deseq2.py` | `pseudobulk`, `DESeq2` (design formulas, Wald/LRT, `results()`, independent filtering), `pseudobulk_de` |
| `crest/harmony.py`, `crest/doublets.py`, `crest/ingest.py`, `crest/_loess.py` | Harmony, Scrublet (+ `threshold_minimum`), ingest, loess |
| `tests/test_crest.py`, `tests/data/` | Test suite; stored R DESeq2 reference outputs |
| `bench/paper/` | Paper benchmark harness (driven by `scripts/crest_paper_bench.sh`) |
| `bench/deseq2/`, `bench/harmony/`, `bench/doublets/`, `bench/ingest/` | Module comparisons + results |
| `bench/whitepaper/results/` | 0.2.0 benchmark results (README figures); scripts replaced by `bench/paper/` |
| `scripts/` | `crest_paper_bench.sh` (benchmark), `crest_git_housekeeping.sh` (branches) |
| `docs/memory_model.md`, `docs/deseq2.md`, `docs/downstream.md` | Design notes and validation |

## Next steps (in priority order)

1. **Merge the PR into `dev`** once CI is green; run `scripts/crest_git_housekeeping.sh --apply`.
2. **Run the paper benchmark locally**: `bash scripts/crest_paper_bench.sh --tier standard`
   (then `--tier full` on a machine with >= 64 GB RAM for scanpy at 647k cells). Send back the tarball.
3. **Release 0.3.0**: bump `Cargo.toml` + `crest/__init__.py` + `CHANGELOG.md`, merge `dev → main`, tag `v0.3.0`.
   Needs a `PYPI_API_TOKEN` secret or trusted publishing on PyPI for `crest-sc`.
4. **White paper**: results from step 2, plus `docs/deseq2.md` and `docs/downstream.md` validation tables.
5. Possible extensions: `lfcShrink` (apeglm), `lfcThreshold`, interaction formulas; UMAP transform
   optimisation in `ingest`; `pearson_residuals` HVG; per-call allocation cuts in DESeq2.

## Conventions

* Every numerical method is validated against its reference implementation
  (scanpy, R DESeq2). Add a test that pins the agreement before optimising.
* Keep memory bounded: kernels take one chunk at a time. Never materialise
  cells × genes dense arrays.
* Commit messages: `feat:`, `fix:`, `bench:`, `docs:`, `chore:` prefixes.
* Before pushing, run `cargo test --release && pytest -q`.
