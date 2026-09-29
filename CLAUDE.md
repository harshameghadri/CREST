# CREST: operating manual for Claude Code

Read this file completely at the start of every session. Then read `HANDOVER.md` (current
status, open work, decisions). If you wonder *why* something is the way it is, search
`work.md`: it is the full history of the sessions that built the project, with every request
and every reply. Use `grep -n "<keyword>" work.md` rather than reading it top to bottom. The
"Context summary" sections are dense recaps. `scripts/export_chat.py` regenerates it from a
Claude Code transcript (`~/.claude/projects/<project>/<session>.jsonl`); append new sessions
rather than replacing the history.

## 1. What this project is

CREST is a Python package (`pip install crest-sc`, `import crest`) with a Rust core
(PyO3 + maturin). It runs the scanpy single-cell RNA-seq workflow and gives **the same
results as scanpy** (and as R DESeq2 / harmonypy for those modules), while being several
times faster and using far less memory. The owner is a computational biologist who works
mostly in R (Seurat); the goal is a publishable package with an arXiv/bioRxiv-grade
performance paper.

The steps covered:

* **Workflow:** QC → filter → `normalize_total` → `log1p` → HVG → `scale` → PCA →
  neighbours → Leiden → UMAP → marker genes (t-test / Wilcoxon) → `score_genes`.
* **After clustering:** pseudobulk DESeq2 (Wald + LRT), Harmony, Scrublet, `seurat_v3`
  HVGs, Leiden resolution sweep, ingest (label transfer).

History in one paragraph: the project began (Feb 2026) as "biopolars", Polars expression
plugins for single-cell data. That design was wrong (per-cell sums over partial batches,
broken kNN id mapping, a fake Leiden). In Sept 2026 it was rebuilt around a `BioFrame` with
lazy transforms and fused Rust kernels, validated step by step against scanpy, and released
as 0.2.0. Downstream modules followed (0.3.0, unreleased on `dev`). The Polars plugin layer
was deleted. Polars remains only as the metadata / table / Parquet layer. Whether the name
"Columnar Rust Engine" still fits is an open question (see HANDOVER, Decisions).

## 2. The mental model (read before touching code)

**Raw counts are stored once and never modified.** Everything else is recorded, and applied
on the fly inside Rust kernels as the data is streamed chunk by chunk:

```text
BioFrame
  store  : CSRStore (in-memory CSR) | FrameStore (Polars triplets) | ParquetStore (on disk, out-of-core)
  obs    : polars.DataFrame, kept cells; obs["cell_id"] indexes the store
  var    : polars.DataFrame, kept genes; var["gene_id"] indexes the store
  ops    : [("normalize_total", 1e4), ("log1p",)]   <- recorded transforms
  uns    : {"scale": {...}, "hvg": {...}, "pca": {...}, "neighbors": {...}, ...}
  obsm / varm : embeddings (X_pca, X_umap, X_pca_harmony) / PCs

for ctx in bf.iter_ctx():          # one raw chunk + cell_map/gene_map (-1 = filtered) + transform
    _native.<kernel>(*ctx, outputs...)   # Rust: filter + normalise + log1p (+ scale) per cell, accumulate
```

The consequences are rules, not suggestions:

* **Filters return a new BioFrame that shares the store:** `bf = crest.pp.filter_cells(bf, ...)`.
  All other functions mutate `bf` in place and also return it.
* **Nothing is ever cells × genes and dense.** Results are per-gene, per-cell, per-group, or
  genes × genes (the PCA Gram matrix, with HVGs only; capped at 20k genes).
* **PCA with `scale` is exact without densifying.** Zeros map to a per-gene constant
  `b_j`, so `Z = 1·bᵀ + S` with `S` sparse. Centring removes `1·bᵀ`, so the Gram matrix
  is accumulated from sparse rows (`docs/memory_model.md`).
* **What each step reads:**
  - raw counts: QC, filters, `seurat_v3`, Scrublet, pseudobulk / DESeq2;
  - log-normalised values: `seurat` / `cell_ranger` HVG, DE, `score_genes`;
  - scaled values: PCA;
  - `obsm` embeddings: neighbours, Leiden, UMAP, Harmony, ingest.
* `ParquetStore` = out-of-core: one Parquet part in RAM at a time, ~1 GB peak at any size,
  ~2× slower.

Full user-level explanation: `docs/concepts.md`. The chunk protocol tuple and the table of
all native functions: `docs/development.md`.

## 3. Build, test, docs

```bash
# one-time: Rust stable >= 1.80, Python >= 3.10; the owner uses uv
uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install maturin
maturin develop --release -E test      # rebuild after ANY change under src/ (always --release)
cargo test --release                   # 26 Rust unit tests
pytest -q                              # 27 Python tests: scanpy / R-DESeq2 / harmonypy / skmisc / skimage parity
pip install -r docs/requirements.txt && sphinx-build -W -b html docs docs/_build/html   # docs
```

* A Python-only change needs no rebuild.
* If `maturin` says "Both VIRTUAL_ENV and CONDA_PREFIX are set", run `conda deactivate`
  (the owner's shell starts in conda `base`).
* The R-DESeq2 reference outputs are stored in `tests/data/`, so the tests don't need R.
  They are regenerated with `Rscript tests/data/make_lrt_fixtures.R` and the scripts in
  `bench/deseq2/`.

## 4. Where things are

| need | file |
|---|---|
| BioFrame, stores, `iter_ctx`, AnnData / Parquet I/O | `crest/core.py` |
| readers | `crest/io.py` |
| QC, filters, lazy transforms, HVG (3 flavours), neighbours | `crest/pp.py` |
| PCA, Leiden, `leiden_sweep`, UMAP, DE, `score_genes` | `crest/tl.py` |
| pseudobulk + DESeq2 (formulas, Wald / LRT, results, independent filtering) | `crest/deseq2.py` + `src/deseq/` |
| Harmony / Scrublet / ingest / loess | `crest/harmony.py` + `src/harmony.rs`, `crest/doublets.py`, `crest/ingest.py`, `crest/_loess.py` |
| streaming kernels | `src/kernels.rs` (`ChunkView`) |
| kNN (exact GEMM ≤ 20k cells; IVF + NN-descent above; `knn_query`) | `src/knn.rs` |
| Leiden, UMAP | `src/leiden.rs`, `src/umap/` |
| Python ↔ Rust bindings (validation, GIL release) | `src/py.rs` |
| tests | `tests/test_crest.py`, `tests/data/` |
| paper benchmark | `scripts/crest_paper_bench.sh` → `bench/paper/{datasets,run_one,run_all,monitor,accuracy,summarize}.py` |
| module benchmarks | `bench/{deseq2,harmony,doublets,ingest}/` (results in `results/`) |
| user docs (Read the Docs) | `docs/*.md`, `docs/conf.py`, `.readthedocs.yaml`, `docs/readthedocs.md` |
| design / validation notes | `docs/concepts.md`, `docs/memory_model.md`, `docs/deseq2.md`, `docs/downstream.md`, `docs/benchmarks.md` |
| history | `work.md`, `CHANGELOG.md` |

## 5. Rules

1. **Parity first.** Results must match the reference tool: scanpy 1.11 for the workflow,
   R DESeq2 1.42 for `DESeq2`, harmonypy 2.x for Harmony, `skmisc` for loess and `skimage`
   for `threshold_minimum`.
   - A new method gets a parity test before it is optimised.
   - If a change moves a parity test, find out why. **Never loosen a tolerance** to make
     it pass.
   - Where exact parity is impossible (random streams: Scrublet pairs, Harmony init), compare
     outcomes statistically and document it.
2. **Bounded memory.** Stream through `iter_ctx()`. Never build dense cells × genes, and never
   call `to_scipy()` inside library code.
3. **Speed lives in Rust.** Loops over non-zeros or cells go in `src/`. Release the GIL
   (`py.allow_threads`), parallelise with rayon, and validate inputs in `py.rs` so errors
   raise instead of panicking.
4. **scanpy-compatible names and defaults.** Deviations are documented in the docstring and
   in `docs/`.
5. **Honest benchmarks.** These rules come from the owner:
   - The headline comparison is the `core` pipeline; optional modules go in a separate table.
   - Speed-ups are quoted at each tool's *best* thread count and at matched threads; the
     conservative one is the headline.
   - Report the steps where CREST is not faster, its thread-scaling limits, and every failed
     run.
   - No cherry-picking.
6. **Docs move with code.**
   - A new public function gets a docstring and an entry in `docs/api.md`.
   - A behaviour change updates `docs/` and `CHANGELOG.md` (Unreleased).
   - `sphinx-build -W` must stay clean (CI enforces it).

## 6. Git and GitHub

* Branches: `main` (releases only; **never push to it**), `dev` (integration; PRs land
  here), and feature branches. A release merges `dev` → `main` and tags `vX.Y.Z`.
* The version lives in `Cargo.toml`. `pyproject.toml` and `docs/conf.py` read it from there.
  `crest/__init__.py` has a copy: bump both.
* Commit prefixes: `feat:`, `fix:`, `bench:`, `docs:`, `chore:`. Before pushing, run
  `cargo test --release && pytest -q`.
* CI (`.github/workflows/CI.yml`) runs on PRs and pushes to main:
  - the `test` job (cargo + pytest);
  - the `docs` job (Sphinx, warnings are errors);
  - wheel builds for all platforms, published to PyPI on a tag.
* Deleting or tagging shared branches is done by the owner with
  `scripts/crest_git_housekeeping.sh`; don't try to work around permission refusals.
* Old branches are archived as tags `archive/feat-leiden-hnsw-faer` and
  `archive/fix-production-readiness`.

## 7. The owner's machine (where benchmarks run)

* **Machine:** `rinamochana`, AMD Threadripper PRO 3975WX (32 cores / 64 threads),
  128 GB DDR4-2667 ECC, Ubuntu 26.04.
* **Disks:**
  - `/home` is small (OS only): **never write data or caches there**;
  - `/mnt/scratch` is a 1.8 TB NVMe (ext4, `/dev/nvme0n1`); run benchmarks here;
  - `/storage` is a large ZFS pool on HDDs.
* **Benchmark:**
  - Run from `/mnt/scratch` with
    `bash crest_paper_bench.sh --tier quick|standard|full`.
  - Everything, including uv / cargo / numba caches, goes under `./crest-bench`.
  - Results go to `crest-bench/results/<host>-<date>/`, and a `.tar.gz` is written to send
    back.
* The owner uses `uv` for Python environments.

## 8. Working style the owner expects

* **Think thoroughly and weigh trade-offs.** Quality comes first, and every claim should be
  verified: run the code, rebuild, re-test.
* Scripts they run locally must work from the current directory, log everything, and fail
  early with a clear message.
* When you finish a session that changed anything, update `HANDOVER.md` (Status and Next
  steps), and add a `CHANGELOG.md` entry for user-visible changes.
* Never put model identifiers in commits, code or docs.
