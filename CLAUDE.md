# CREST — notes for Claude Code

Read `HANDOVER.md` first: it has the current status, open PRs, the next steps and the code map.
Update its **Status** / **Next steps** before ending a session that changed anything.

## Build and test

```bash
maturin develop --release -E test      # rebuild the Rust extension after any src/ change
cargo test --release                   # Rust unit tests
pytest -q                              # Python tests (scanpy parity, R-DESeq2 parity via tests/data)
```

A Python change needs no rebuild. A Rust change needs `maturin develop --release`
before `pytest` sees it.

## Rules

* Results must match the reference tools: scanpy for the pipeline, R DESeq2 1.42
  for `crest.tl.DESeq2`. If a change moves a parity test, find out why; don't loosen the tolerance.
* Memory: work one chunk at a time (`BioFrame.iter_ctx()`); never build a dense cells × genes array.
* Native code releases the GIL (`py.allow_threads`) and parallelises with rayon.
* The distribution is `crest-sc`; the import name is `crest`. The version lives in `Cargo.toml` (pyproject reads it dynamically).
* Branches: feature work goes into `dev` via PRs, and releases go `dev → main` plus a `vX.Y.Z` tag.
  Don't push to `main` directly.
* Benchmarks write into `bench/*/results/`. Downloaded data goes in `bench_data/` (gitignored).
