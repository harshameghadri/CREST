# CREST handover

This file says where the project stands and what to do next. Read `CLAUDE.md` first (how
the code works, and the rules). Keep **Status** and **Next steps** current at the end of
every session.

_Last updated: 2026-09-29 (0.3.0 merged to main)._

## Status

| Area | State |
|---|---|
| Core workflow (QC → HVG → scale / PCA → kNN → Leiden → UMAP → t-test / Wilcoxon → `score_genes`) | Done; validated against scanpy 1.11 in `tests/`. 0.2.0 run (4-core cloud VM): 6–7× faster, 4–10× less memory |
| Out-of-core (Parquet) | Done; ~1 GB peak regardless of size, ~2× slower than in-memory |
| Pseudobulk DESeq2 (Rust) | Done: Wald + LRT; matches R DESeq2 1.42 (`docs/deseq2.md`) |
| Modules after clustering | Harmony, Scrublet, `seurat_v3` HVG, `leiden_sweep`, `ingest`, `knn_query`; each validated (`docs/downstream.md`) |
| Paper benchmark | `scripts/crest_paper_bench.sh` + `bench/paper/`. Quick tier done on rinamochana (30/30 ok). **Standard tier running** (see Next steps 1) |
| Documentation | **New:** Sphinx + MyST site in `docs/`, `.readthedocs.yaml`, beginner setup guide `docs/readthedocs.md`, CI `docs` job. Not yet connected to Read the Docs (owner's step) |
| Packaging | Version **0.3.0** (`Cargo.toml`), merged `dev` → `main` on 2026-09-29. Tag `v0.3.0` + PyPI publish still to do (owner) |
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

1. **Let the standard-tier benchmark finish** on rinamochana.
   - It runs in `/mnt/scratch/crest-bench`, started 2026-09-28 17:56 (160 runs, ~20–25 h;
     39/160 done at 2026-09-29 morning).
   - Don't change benchmark code while it runs. It can be resumed: re-running the same
     command skips finished runs.
   - The owner sends back `crest-bench/results/crest-bench-rinamochana-<date>.tar.gz`.
2. **Write the results tables with the honest-reporting rules** (`CLAUDE.md` §5, and
   `docs/benchmarks.md`). The changes go in `bench/paper/summarize.py`:
   - headline = `core` pipeline, at each tool's best thread count **and** at matched
     threads. In the thread scan, scanpy's pbmc68k `core` time is 152 s at 8 threads but
     301 s at 64, so a 64-vs-64 comparison flatters CREST 2×;
   - a separate table for the optional modules (Scrublet, Harmony, sweep, pseudobulk);
   - report CREST's weak spots: thread scaling on pbmc68k `core` is 63.5 s at 1 thread and
     24 s at 64 (~1/3 serial; find which steps from `steps.csv`); steps where CREST is not
     faster; failed runs.
3. **Connect Read the Docs.** The owner follows `docs/readthedocs.md`: project name
   `crest-sc`, default branch `main`.
4. **Publish 0.3.0.** The version bump, CHANGELOG and `dev` → `main` merge are done. Left:
   - tag `v0.3.0` on `main` (`git tag v0.3.0 origin/main && git push origin v0.3.0`); the
     tag triggers the CI wheel builds and the PyPI upload;
   - the upload needs a `PYPI_API_TOKEN` repository secret or PyPI trusted publishing for
     `crest-sc`.
5. **White paper**: results from step 2, plus the validation tables in `docs/deseq2.md` and
   `docs/downstream.md`.
6. **Performance work, after the paper numbers exist** (measure first, from `steps.csv`):
   - the serial fraction of the core pipeline (candidates: 10x HDF5 gzip reading, Leiden
     local moving, Python glue);
   - Scrublet at ≥ 100k cells with no batch key is ~quadratic, because the approximate kNN
     is run with k ≈ 1.5·√n (541 s at 100k, 2,633 s at 200k). Low priority: Scrublet is an
     optional module, and users may prefer scDblFinder. A cheap fix is exact tiled search at
     any size.
7. **Possible extensions:**
   - DESeq2: `lfcShrink` (apeglm), `lfcThreshold`, interaction formulas;
   - UMAP transform optimisation in `ingest`;
   - `pearson_residuals` HVG.

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
