# CREST handover

This file says where the project stands and what to do next. Read `CLAUDE.md` first (how
the code works, and the rules). Keep **Status** and **Next steps** current at the end of
every session.

_Last updated: 2026-09-29 (0.3.0 on PyPI; standard benchmark analysed)._

## Status

| Area | State |
|---|---|
| Core workflow (QC → HVG → scale / PCA → kNN → Leiden → UMAP → t-test / Wilcoxon → `score_genes`) | Done; validated against scanpy 1.11. 0.3.0 on the 64-thread workstation: PBMC 68k core workflow **8.9× faster best-vs-best** (19.8× at matched 64 threads), 3.4× less memory (`docs/benchmarks.md`) |
| Out-of-core (Parquet) | Done; count matrix never in RAM; ~0.7–1.1 GB peak up to 200k cells (per-cell results still grow with cells); up to ~2× slower |
| Pseudobulk DESeq2 (Rust) | Done: Wald + LRT; matches R DESeq2 1.42; ~50× faster than R on rinamochana |
| Modules after clustering | Harmony, Scrublet, `seurat_v3` HVG, `leiden_sweep`, `ingest`, `knn_query`; each validated (`docs/downstream.md`) |
| Paper benchmark | Quick + standard tiers done on rinamochana (190 runs, 0 failures). Harness now runs core at 8 + 64 threads, modules separately, Scrublet capped; supports the Parse ~1M PBMC dataset (`parse_pbmc`). **Next: the Parse run** |
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

1. **Parse benchmark on rinamochana.** The file
   (`/mnt/scratch/crest-bench/data/Parse_1M_adata_for_cellflow_datasets_with_embeddings.h5ad`)
   is the **full 9.7M-cell** atlas with raw counts of **2,000 genes** in `X` (checked with
   `--inspect`). Two datasets are built from it: `parse_pbmc` (all 9.7M cells; scanpy cannot fit
   in 128 GB, so CREST only) and `parse_pbmc_1m` (seeded random 1M cells; the scanpy
   comparison). QC uses `min_genes` 20 (200 scaled to 2,000 genes); DESeq2 contrast IFN-beta
   vs PBS. Run from `/mnt/scratch`:
   `bash crest_paper_bench.sh --ref <branch with this change> --datasets "parse_pbmc_1m parse_pbmc" --repeats 3 --timeout 21600 --skip-thread-scan`
2. **Connect Read the Docs** (`docs/readthedocs.md`): project `crest-sc`, default branch `main`.
3. **White paper**: headline from the core workflow best-vs-best; "not faster" table;
   modules table; accuracy table (all in `report.md`). Use 5 repeats and the new protocol.
   Set the CPU governor to `performance` for the final run.
4. **Performance work, measured first:**
   - `seurat_v3` HVG is 2× slower than scanpy at ≥ 10k cells: the loess fit runs in Python
     (`crest/_loess.py`); port it to Rust or vectorise it.
   - PCA has a ~1 s fixed cost: the full eigendecomposition of the 2,000 × 2,000 Gram matrix
     (measured 1.16 of 1.31 s on PBMC 3k). A partial eigensolver for the top 50 would remove it.
   - UMAP fixed cost on small data (500 epochs below 10k cells, as umap-learn).
   - Serial fraction ~24% (reading HDF5, Leiden, glue).
   - Scrublet at ≥ 100k cells without a batch key is ~quadratic (k ≈ 1.5·√n); low priority.
5. **Possible extensions:** DESeq2 `lfcShrink` / `lfcThreshold` / interactions; UMAP transform
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
