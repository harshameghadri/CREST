# Contributing to CREST

Thank you for your interest in CREST. Bug reports, questions, documentation fixes and code
contributions are all welcome.

## Reporting bugs and asking for help

Open an issue at <https://github.com/harshameghadri/CREST/issues>. This is the place for bug
reports, feature requests and usage questions.

For a bug, please include:

* the CREST version (`python -c "import crest; print(crest.__version__)"`), your Python
  version and operating system;
* a minimal piece of code that shows the problem, and the full error message or the output
  you expected instead;
* if the problem is a result that differs from scanpy (or R DESeq2, harmonypy), the versions
  of both tools and the parameters you used.

CREST is maintained by a single person on a best-effort basis. Issues are usually read within
a few days, but there is no guaranteed response time.

## Development setup

You need Rust (stable, >= 1.80) and Python >= 3.10. The steps below use
[uv](https://docs.astral.sh/uv/); `python -m venv` works too.

```bash
git clone https://github.com/harshameghadri/CREST
cd CREST
git checkout dev
uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install maturin
maturin develop --release -E test      # rebuild after ANY change under src/ (always --release)
cargo test --release                   # Rust unit tests
pytest -q                              # Python tests, including the parity tests
pip install -r docs/requirements.txt && sphinx-build -W -b html docs docs/_build/html   # docs
```

* A change to the Python files under `crest/` needs no rebuild; a change under `src/` needs
  `maturin develop --release` again. Debug builds are about 20× slower and make the tests
  time out.
* If maturin says "Both VIRTUAL_ENV and CONDA_PREFIX are set", run `conda deactivate` first.
* The R DESeq2 reference outputs are stored in `tests/data/`, so the tests do not need R.

The [developer guide](https://crest-sc.readthedocs.io/en/latest/development.html) explains
the repository layout, the chunk protocol between Python and Rust, and how to add a streamed
statistic.

## Pull requests

* Branch from `dev` and open the pull request **into `dev`**. `main` only receives releases.
* Use a commit prefix: `feat:`, `fix:`, `bench:`, `docs:` or `chore:`.
* Before pushing, run `cargo test --release && pytest -q`, and `sphinx-build -W` if you changed
  the documentation or a docstring. CI runs the same checks on every pull request.
* A new public function needs a docstring and an entry in `docs/api.md`. A behaviour change
  updates `docs/` and the `Unreleased` section of `CHANGELOG.md`.

### The parity rule

CREST's results must match the reference tool: scanpy 1.11 for the workflow, R DESeq2 1.42
for `DESeq2`, harmonypy 2.x for Harmony, `skmisc` for loess and `skimage` for
`threshold_minimum`.

* A new method gets a parity test against its reference before it is optimised.
* If a change moves a parity test, find out why. Never loosen a tolerance to make a test pass.
* Where exact parity is impossible (random number streams), compare outcomes statistically and
  document it.
* Keep scanpy's names and defaults; document any deviation in the docstring and in `docs/`.

Two further rules keep CREST fast and lean: work one chunk at a time through
`BioFrame.iter_ctx()` (never build a dense cells × genes array in library code), and put loops
over non-zeros or cells in Rust.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By taking part you agree
to abide by it.
