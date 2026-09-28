# CREST handover

Where the project stands and how to continue on a local machine. Keep this
file current: update **Status** and **Next steps** at the end of every work
session, cloud or local.

_Last updated: 2026-09-28._

## Status

| Area | State |
|---|---|
| Core pipeline (QC → HVG → scale/PCA → kNN → Leiden → UMAP → t-test/Wilcoxon) | Done, validated against scanpy 1.11 (`tests/`), 6–7× faster, 4–10× less memory |
| Out-of-core (Parquet) mode | Done, ~1 GB peak regardless of dataset size |
| White-paper benchmark (68k / 100k / 200k) | Done, `bench/whitepaper/results/` |
| **Pseudobulk DESeq2 in Rust** | **New this session**: `crest.tl.DESeq2` / `pseudobulk` / `pseudobulk_de`. Matches R DESeq2 1.42 to ~1e-9 on simulated data and ≤1e-7 on 5 of 8 Kang cell types; ~19× faster than R and ~30× faster than pydeseq2. See `docs/deseq2.md` |
| Packaging | PyPI name `crest-sc`, import name `crest`, version 0.2.0 (DESeq2 not yet in a release) |
| CI | `.github/workflows/CI.yml`: `cargo test` + `pytest`, then maturin wheels for all platforms (release on tag) |

### Git / GitHub state

* Working branch: `claude/awesome-lamport-dsvfv2` (all work is here and pushed).
* [PR #1](https://github.com/harshameghadri/CREST/pull/1) `claude/awesome-lamport-dsvfv2 → main` is open. It fast-forwards
  `main` to 0.2.0 and now also carries the DESeq2 work.
* Remote branches to clean up once PR #1 is merged:
  `feat/leiden-hnsw-faer` and `fix/production-readiness`. Both are superseded; tag them
  first if you want an archive (`git tag archive/<name> origin/<name>`).
* Planned workflow: create a `dev` branch from `main` after PR #1 merges, open
  feature PRs into `dev`, and merge `dev → main` for releases.
* In the cloud session, pushing straight to `main`, creating shared branches and tags,
  and deleting branches were blocked by Claude Code's safety check, which needs
  explicit confirmation for such actions. They are not GitHub permission
  problems. Locally, run them yourself or confirm them when Claude asks.

## Setting up locally

```bash
git clone https://github.com/harshameghadri/CREST && cd CREST
git checkout claude/awesome-lamport-dsvfv2     # or main once PR #1 is merged

# toolchain: Rust (stable >= 1.80), Python >= 3.10
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release -E test,bench          # builds the Rust extension in place

cargo test --release                              # Rust unit tests (~33)
pytest -q                                         # Python tests incl. scanpy + R-DESeq2 parity (~18)
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
# white-paper pipeline benchmark (downloads data into ./bench_data)
bash bench/whitepaper/run_local_benchmark.sh --sizes "100000 200000" --repeats 3

# DESeq2 vs R on simulated designs            -> bench/deseq2/results/comparison.{md,csv}
python bench/deseq2/compare_r.py --genes 5000
# DESeq2 on real pseudobulk (Kang 2018, GEO)  -> bench/deseq2/results/kang_pseudobulk.{md,csv}
python bench/deseq2/kang_pseudobulk.py        # add --skip-r / --skip-pydeseq2 if not installed
```

## Code map

| Path | What |
|---|---|
| `src/kernels.rs` | Fused chunk kernels (QC, gene stats, Gram/PCA, projection, group sums, Wilcoxon) over a `ChunkView` |
| `src/knn.rs`, `src/leiden.rs`, `src/umap/` | kNN (GEMM / IVF + NN-descent), Leiden, UMAP |
| `src/deseq/mod.rs` | DESeq2 pipeline: size factors, dispersions, trend, MAP, IRLS/Newton, Cook's, outlier refit |
| `src/deseq/linalg.rs`, `src/deseq/lowess.rs` | Small dense LU; port of R's `clowess` |
| `src/py.rs` | PyO3 bindings (numpy in/out, GIL released) |
| `src/nb_glm.rs`, `src/lib.rs` | Legacy Polars expression plugins (`.bio` namespace) |
| `crest/core.py` | `BioFrame` and the stores (CSR, Polars frame, Parquet) |
| `crest/pp.py`, `crest/tl.py` | scanpy-style API |
| `crest/deseq2.py` | `pseudobulk`, `DESeq2` (design formulas, `results()`, independent filtering), `pseudobulk_de` |
| `tests/test_crest.py`, `tests/data/` | Test suite; stored R DESeq2 reference outputs |
| `bench/whitepaper/`, `bench/deseq2/` | Benchmarks used in the paper |
| `docs/memory_model.md`, `docs/deseq2.md` | Design notes |
| `docs/dev/` | Old planning notes from March–April 2026; historical only, superseded by this file |
| `bench/*.py` (top level) | Old exploratory scripts, many using the removed `biopolars` API; candidates for deletion |

## Next steps (in priority order)

1. **Merge PR #1** once CI is green, create `dev`, and delete or archive the stale branches (see above).
2. **Release 0.3.0** with DESeq2: bump `Cargo.toml` and `CHANGELOG.md`, then tag `v0.3.0`; CI builds and publishes the wheels.
   Needs a `PYPI_API_TOKEN` secret or trusted publishing set up on PyPI for `crest-sc`.
3. **White paper**: add a DESeq2 section from `bench/deseq2/results/` and `docs/deseq2.md`
   (accuracy table + speed vs R/pydeseq2 + the known-differences paragraph).
4. DESeq2 extensions, if reviewers ask: LRT (`test="LRT"`), `lfcShrink` (apeglm is
   the usual request), `lfcThreshold`, interaction terms in the formula parser.
5. DESeq2 speed: ~0.3 ms/gene at p = 9. Per-call allocations in
   `log_posterior` / `fit_beta` (`Vec`s, the LU copy) are the obvious next win, if needed.
6. HVG `seurat_v3`, and Harmony batch integration (listed under README "Scope and limitations").
7. Housekeeping: delete the legacy `bench/*.py` scripts and `docs/dev/`.

## Conventions

* Every numerical method is validated against its reference implementation
  (scanpy, R DESeq2). Add a test that pins the agreement before optimising.
* Keep memory bounded: kernels take one chunk at a time. Never materialise
  cells × genes dense arrays.
* Commit messages: `feat:`, `fix:`, `bench:`, `docs:`, `chore:` prefixes.
* Before pushing, run `cargo test --release && pytest -q`.
