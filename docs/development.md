# Developer guide

## Support and contributing

Report bugs or ask questions at <https://github.com/harshameghadri/CREST/issues>. Pull
requests go into `dev`; see
[CONTRIBUTING.md](https://github.com/harshameghadri/CREST/blob/main/CONTRIBUTING.md) for the
checklist. CREST is maintained by one person on a best-effort basis.

## Build and test

```bash
maturin develop --release -E test     # (re)build the Rust extension into crest/
cargo test --release                  # Rust unit tests
pytest -q                             # Python tests: scanpy, R-DESeq2, skmisc, skimage parity + behaviour tests
```

A Python change needs no rebuild. A Rust change needs `maturin develop --release` before
`pytest` sees it. Always build with `--release`: debug builds are about 20× slower and make
the tests time out.

Build these docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html     # open docs/_build/html/index.html
sphinx-build -n -b html docs docs/_build/html     # optional: also report broken cross-references
```

## Repository layout

| path | what |
|---|---|
| `crest/core.py` | `BioFrame`, the three stores, `iter_ctx()` (the chunk protocol), AnnData / Parquet conversion |
| `crest/io.py` | `read_10x_h5` (+ `backed=` Parquet streaming), `read_10x_mtx`, `read_h5ad` |
| `crest/pp.py` | QC, filters, lazy transforms, HVG (`seurat`, `cell_ranger`, `seurat_v3`), neighbours |
| `crest/tl.py` | PCA, Leiden, `leiden_sweep`, UMAP, `rank_genes_groups`, `score_genes`; re-exports the modules below |
| `crest/deseq2.py` | `pseudobulk`, design formulas (`model_matrix`), `DESeq2` (Wald / LRT, `results()`, independent filtering), `pseudobulk_de` |
| `crest/harmony.py` | Python wrapper around the Rust Harmony |
| `crest/doublets.py` | Scrublet, and a port of `skimage.filters.threshold_minimum` |
| `crest/ingest.py` | projection onto a reference PCA, label and UMAP transfer |
| `crest/_loess.py` | port of netlib `dloess` (the loess behind R and skmisc), for `seurat_v3` |
| `src/kernels.rs` | fused chunk kernels over a `ChunkView` (QC, gene stats, Gram / projection, group sums, Wilcoxon) |
| `src/knn.rs` | kNN: exact tiled GEMM; IVF + NN-descent above 20k cells; `knn_query` (reference → query) |
| `src/leiden.rs` | Leiden (Traag 2019) + parallel resolution sweep |
| `src/umap/` | fuzzy simplicial set, spectral init, parallel SGD |
| `src/harmony.rs` | Harmony (harmony2 algorithm) |
| `src/deseq/` | DESeq2 (`mod.rs`), small dense LU (`linalg.rs`), R's `lowess` (`lowess.rs`) |
| `src/py.rs` | PyO3 bindings: NumPy in / out, GIL released, argument validation |
| `tests/test_crest.py`, `tests/data/` | the test suite; stored R DESeq2 outputs + the R script that makes them |
| `bench/paper/` | the paper benchmark harness (driven by `scripts/crest_paper_bench.sh`) |
| `bench/<module>/` | module comparisons with their `results/` |
| `scripts/` | `crest_paper_bench.sh`; `bump_version.py`, `assemble_changelog.py`, `fetch_test_data.py` (release and test infrastructure, described below) |
| `tests/conftest.py` | the `baseline_fails` marker and the skip allowlist (both described below) |
| `changes/` | one change fragment per pull request; see `changes/README.md` |
| `KNOWN_ISSUES.md` | documented limitations of the current release |
| `docs/` | this documentation (Sphinx + MyST); `.readthedocs.yaml` at the root |
| `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff` | contribution guide, code of conduct, citation metadata |

## The chunk protocol (how Python talks to Rust)

`BioFrame.iter_ctx(transform=True)` yields one tuple per chunk:

```text
(genes u32[nnz], values f32[nnz], cell_map i64[n_store_cells], gene_map i32[n_store_genes],
 target_sum f64, log1p bool, cell_totals f64[n_store_cells] | None,
 cells u32[nnz] | None, indptr i64[k+1] | None, first_cell int)
```

* `genes` / `values` are raw entries of the store. `cells` (COO) or `indptr` + `first_cell`
  (CSR) say which cell each entry belongs to.
* `cell_map[store_cell]` is the current row, or −1 if the cell is filtered out. `gene_map`
  does the same for columns. This is how filtering works without copying.
* `target_sum` > 0 means normalise to it; `log1p` means apply log1p after that.
* `cell_totals` (passed when `target_sum` > 0) is what each cell is divided by, indexed by
  store cell id: the totals `normalize_total` computed over the genes kept at that time
  (`BioFrame.cell_totals`), so filtering genes later does not change the values, as in
  scanpy. `None` falls back to each cell's total over the genes kept by `gene_map` (only
  for a legacy two-element `("normalize_total", target)` op; a fingerprinted op with no
  stored totals raises). `py.rs` checks only its length (O(1) per chunk);
  `normalize_total` checks once that the totals are finite and ≥ 0.

Each streaming native function takes `*ctx` followed by its own inputs and **preallocated
NumPy outputs that it accumulates into**, for example:

```python
out = np.zeros((bf.n_vars, 5))
for ctx in bf.iter_ctx():
    _native.gene_stats(*ctx, out)
```

In Rust, `src/py.rs` turns the tuple into a `kernels::ChunkView`. The kernel walks the chunk
cell by cell in parallel (rayon), applies the maps and transform into a per-thread buffer,
and merges per-thread partial results.

### Native functions (`crest.crest`)

| function | used by | does |
|---|---|---|
| `qc` | `calculate_qc_metrics`, filters | per-cell / per-gene totals, counts, mito fraction |
| `gene_stats` | HVG, `score_genes`, PCA scaling | per-gene Σx, Σx², Σexpm1, Σexpm1², nnz |
| `group_gene_sums(clip)` | `rank_genes_groups`, `seurat_v3` | per group × gene Σx, Σx², nnz (optionally clipped) |
| `group_gene_totals` | `pseudobulk` | per group × gene raw sums |
| `collect_gene_block`, `wilcoxon_rank_sums` | Wilcoxon | gathers a memory-bounded gene block; sparse rank sums with a closed-form zero-tie block |
| `gram_accumulate`, `pca_from_gram`, `project` | PCA, Scrublet, ingest | Gram matrix with implicit scale / clip; eigendecomposition; projection |
| `weighted_row_sums` | `score_genes` | per-cell weighted sum over genes |
| `materialize` | `to_scipy`, `write_parquet` | the filtered / transformed chunk as COO |
| `knn_graph`, `connectivities` | `neighbors`, Scrublet | kNN; umap-learn fuzzy connectivities |
| `knn_query` | `ingest` | reference → query kNN |
| `leiden`, `leiden_sweep` | clustering | Leiden, one or many (resolution, seed) runs in parallel |
| `umap` | `umap` | layout optimisation |
| `harmony` | `harmony` | full Harmony loop |
| `deseq2_fit`, `lowess` | `DESeq2` | the whole DESeq2 fit; R's lowess for independent filtering |
| `t_pvalues`, `normal_pvalues` | DE | vectorised p-values |

## Rules for changes

1. **Match the reference tool.** A new method needs a test that pins its agreement with the
   reference implementation (scanpy, R, harmonypy, …) *before* it is optimised. If a change
   moves a parity test, find out why. Never loosen the tolerance to make it pass.
2. **Bounded memory.** Work one chunk at a time through `iter_ctx()`. Never build a dense
   cells × genes array, and don't call `to_scipy()` inside library code.
3. **Heavy loops go in Rust.** Release the GIL (`py.allow_threads`), parallelise with rayon,
   and validate array lengths and indices in `py.rs` so bad input raises a Python error
   instead of panicking.
4. **Stay scanpy-compatible.** Keep scanpy's names and defaults unless there is a documented
   reason not to.
5. **Commits:** `feat:`, `fix:`, `bench:`, `docs:`, `chore:`, `test:` prefixes. Before pushing, run
   `cargo test --release && pytest -q`.

### Recipe: adding a streamed statistic

1. Write the kernel in `src/kernels.rs` over `ChunkView`. Use a per-thread accumulator, then
   merge.
2. Bind it in `src/py.rs`: accept `*ctx` plus output arrays, check lengths, and wrap the
   compute in `py.allow_threads`.
3. Register it in the `#[pymodule]` list at the bottom of `src/py.rs`.
4. Call it from Python in a `for ctx in bf.iter_ctx(): ...` loop.
5. Add a parity test in `tests/test_crest.py` against the reference tool, and test that the
   result does not depend on `chunk_nnz`.

## Branches and releases

* Feature work goes into `dev` via pull requests. Never push to `main`.
* A release merges `dev` → `main` and tags `vX.Y.Z`. The version lives in `Cargo.toml`;
  `pyproject.toml` and these docs read it from there. `crest/__init__.py` has a copy, so
  update both.
* CI (`.github/workflows/CI.yml`) is described in the next section; a tag publishes the
  wheels as described under "Releasing".

## Continuous integration

`.github/workflows/CI.yml` runs on every pull request and on pushes to `main`, `dev` and
`release/**`. A new run of a pull request cancels the previous one.

| job | what it does |
|---|---|
| `tests` | builds a wheel, installs it with the `test` extra and runs `cargo test --release` and the whole pytest suite on Linux x86-64 with Python 3.10 and NumPy 1.24 (the oldest supported), Linux x86-64 with Python 3.13 and the latest NumPy, and macOS arm64. The tests run from a copy of `tests/`, so they import the installed wheel and not the source tree. |
| `test` | one check that summarises `tests` (use this name for branch protection) |
| `baseline` | runs the tests marked `baseline_fails` against `crest-sc==0.4.0` from PyPI and against a wheel built from the `rc0` commit; see below |
| `docs` | `sphinx-build -W`, the same build as Read the Docs |
| `sdist` | builds the source distribution and checks that the repository and the sdist contain only package files |
| `linux`, `musllinux`, `windows`, `macos` | build the wheels (Linux x86-64 and aarch64, macOS arm64 and x86-64, Windows x64) |
| `smoke` | installs the wheel of each family in a fresh environment, imports it and runs three quick core checks |
| `version` | checks that the tag, `Cargo.toml` and `crest.__version__` name the same version |
| `publish`, `github-release` | run only for a tag or a manual TestPyPI dry run; see "Releasing" |

**Reference stack.** The `test` extra in `pyproject.toml` pins scanpy 1.11.5 and the
pynndescent, umap-learn, harmonypy and leidenalg releases it was validated with, because the
parity tests compare against them. Do not widen these pins without running the whole suite
(`tests/test_reference_stack.py` fails if the scanpy calls the parity tests rely on break).
The runtime dependencies stay unpinned except `numpy>=1.24,<3`.

**Skipped tests.** CI runs pytest with `--skip-allowlist=tests/skip_allowlist.txt` and fails
if any test is skipped that is not listed there. Every reference package is installed, so a
skip means that something is broken; install the missing package rather than listing the
test. An entry is `<node id>  # <reason>`, and a module that is skipped as a whole is listed
by its path.

**Test data.** `python scripts/fetch_test_data.py` downloads PBMC 3k and PBMC 10k v3 from
10x Genomics into `$CREST_TEST_DATA` (default `~/.cache/crest-test-data`), verifies their
sha256 and skips files that are already present. Some tests read real data and skip when it is
not available; they are controlled by environment variables:

| variable | content | how to set it |
|---|---|---|
| `CREST_TEST_PBMC3K` | the extracted PBMC 3k 10x directory (`matrix.mtx`, `genes.tsv`, `barcodes.tsv`) | `python scripts/fetch_test_data.py --print-env` |
| `CREST_TEST_PBMC_CONNECTIVITIES` | `.npz` of the scanpy connectivities matrix of PBMC 3k | `python scripts/fetch_test_data.py --derive --print-env` (needs the `test` extra) |

`--print-env` prints `export` lines for your shell; `--github-env FILE` appends them to a
GitHub Actions env file. CI does this in the composite action `.github/actions/test-data`
(cached, keyed on the script and `pyproject.toml`) in both the `tests` and the `baseline` jobs,
so these tests run there. A test that needs data must not be listed in
`tests/skip_allowlist.txt`.

### Tests that document a defect: `baseline_fails`

A test that describes a defect in an earlier build carries the marker

```python
@pytest.mark.baseline_fails(ref="v0.4.0", raises=IndexError, match="out of bounds")
def test_chunk_boundary_is_checked():
    ...
```

* `ref` names the earlier build: `v0.4.0` is the wheel on PyPI, `rc0` is the integrated tree
  (commit `d1aaa42`) that the fix branches started from.
* `raises` is the exception class (or a tuple of classes) that the test must raise on that
  build, and `match` an optional regular expression searched in `str(exception)`. For a test
  that fails an assertion, use `raises=AssertionError`; for a `pytest.raises(...)` block that
  finds nothing to catch, use `raises=pytest.fail.Exception, match="DID NOT RAISE"`.
* In a normal run the marker does nothing: the test is an ordinary test and must pass.
* With `pytest --baseline-ref=v0.4.0` (or `CREST_BASELINE_REF=v0.4.0`), only the tests marked
  for that reference run, and each **passes only if it raises the declared exception**. It
  fails if it passes, raises something else, or is skipped. Collection errors fail the whole
  run, so import the new API inside the test, not at module level. A run with no test marked
  for the reference is a pass. The run ends with a line such as
  `baseline v0.4.0: 54 marked test(s) selected, 54 failed as declared`; CI repeats it as an
  annotation of the job.

To reproduce the `baseline` job locally, install the old build and the test extra in a
fresh environment and run pytest from a copy of the tests:

```bash
pip install "crest-sc==0.4.0" -r <(python .github/scripts/print_extra.py test)
mkdir /tmp/run && cp -r tests pyproject.toml /tmp/run && cd /tmp/run
CREST_SOURCE_DIR=<path to the repository> pytest --baseline-ref=v0.4.0
```

A test that passes the baseline job proves that it tests the defect and not something else.
Tests for behaviour that was already correct (pins) carry no marker and must pass on both
builds.

## Changelog fragments

Do not edit `CHANGELOG.md`. Every pull request adds one file `changes/<branch-name-with-dashes>.md`
with one to four bullets (`changes/README.md` has the format). At release time
`python scripts/assemble_changelog.py <version>` turns the fragments into a new
`CHANGELOG.md` section and deletes them.

## Releasing

1. Merge the release branch into `main`. Fold the change fragments into the changelog with
   `scripts/assemble_changelog.py`, and set the version with `scripts/bump_version.py 0.5.0rc1`
   (it edits `Cargo.toml`, `Cargo.lock`, `crest/__init__.py`, `CITATION.cff` and the pinned
   tags in the README and docs; `tests/test_version_consistency.py` checks that they agree).
2. Tag `v<version>` in PEP 440 spelling (`v0.5.0rc1`, `v0.5.0`). CI fails at the `version` job
   unless the tag, `Cargo.toml` and `crest.__version__` are the same version.
3. A tag with a pre-release marker (`rc`, `a`, `b`) is published to **TestPyPI** only. A final
   tag is published to **PyPI** after the owner approves the `pypi` environment, and a GitHub
   Release is then created from the tag (Zenodo archives releases). Neither happens until
   the tests, docs, sdist, all wheel builds and the smoke test of every wheel family passed.
4. Dry run: *Actions → CI → Run workflow* with `publish_testpypi` ticked builds everything
   and publishes to TestPyPI. TestPyPI refuses to upload a file twice, so repeat a dry run only
   after changing the version.

Publishing uses PyPI trusted publishing (OIDC); the `PYPI_API_TOKEN` repository secret is a
fallback that is used only if trusted publishing fails.

### One-time settings (repository owner)

1. **pypi.org**, project `crest-sc` → Publishing → add a GitHub trusted publisher: owner
   `harshameghadri`, repository `CREST`, workflow `CI.yml`, environment `pypi`.
2. **test.pypi.org**: the same entry with environment `testpypi` (create the project `crest-sc`
   on TestPyPI first, or add it as a pending publisher).
3. **GitHub** → Settings → Environments: create `pypi` with *Required reviewers* (the owner)
   and, under *Deployment branches and tags*, a tag rule `v*`; create `testpypi` with no
   reviewer.
4. Optionally add a `TEST_PYPI_API_TOKEN` repository secret as a fallback for TestPyPI.
   Once a release has gone through trusted publishing, the `PYPI_API_TOKEN` secret can be
   deleted.
5. Branch protection for `dev` and `main`: require the checks `test`, `docs`, `sdist`,
   `baseline (v0.4.0)` and `baseline (rc0)`.
6. Enable the repository in the Zenodo GitHub integration before the first final tag.

## Read the Docs

The site <https://crest-sc.readthedocs.io> is built by Read the Docs from
`.readthedocs.yaml` (Ubuntu 24.04, Python 3.12, `docs/requirements.txt`, warnings are
errors as in CI). The package is not installed there: `docs/conf.py` imports `crest` from the
source tree and mocks the compiled Rust module (`autodoc_mock_imports`), so no Rust build is
needed and a build takes about a minute.

* **Versions.** The default branch is `main`, so `latest` shows the released code and
  `stable` the newest tag. Activate each release tag (e.g. `v0.4.0`) in the **Versions** tab
  so it stays readable, and `dev` for a development version.
* **Pull request previews.** Settings → Pull request builds → *Build pull requests for this
  project*: each pull request gets a docs preview and a GitHub check.
* **Editing.** Function documentation comes from the docstrings in `crest/*.py`. A new public
  function also needs a line in `docs/api.md` (`.. autofunction:: crest.<module>.<name>`); a
  new page needs its file name in a `toctree` in `docs/index.md`.

| message in the build log | fix |
|---|---|
| `Config file not found` / `.readthedocs.yaml` missing | the branch being built doesn't have the file; check **Settings → Default branch** |
| `No module named ...` during `autodoc` | a module imports a package at top level that isn't in `docs/requirements.txt`: add it there, or to `autodoc_mock_imports` in `docs/conf.py` if it is heavy or compiled |
| `Extension error` / `Could not import extension` | a package in `extensions` of `docs/conf.py` is missing from `docs/requirements.txt` |
