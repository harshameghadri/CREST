# Installation

## From PyPI

```bash
pip install crest-sc                 # the import name is `crest`
pip install "crest-sc[anndata]"      # + AnnData / scipy interoperability
```

The distribution is called `crest-sc`, because `crest` was taken on PyPI. You still write
`import crest`. Wheels are built for Linux (x86_64, aarch64, …), macOS and Windows on
Python ≥ 3.10.

Runtime dependencies are small: `numpy`, `polars` and `h5py`. scanpy, anndata and scipy are
only needed to hand results over to the scverse ecosystem (`BioFrame.to_anndata()`).

:::{note}
The analysis tools added in 0.3.0 are on the `dev` branch until the 0.3.0 release: Harmony,
Scrublet, `seurat_v3` HVGs, `leiden_sweep`, `ingest` and the DESeq2 likelihood-ratio test.
To use them now, install from source (below).
:::

## From source

You need a Rust toolchain and [maturin](https://www.maturin.rs), which compiles the Rust
extension and installs the package in place.

```bash
# 1. Rust (skip if `cargo --version` already works)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 2. the code
git clone https://github.com/harshameghadri/CREST
cd CREST
git checkout dev

# 3. a virtual environment (uv shown; `python -m venv .venv` works too)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install maturin

# 4. build + install (release mode matters: debug builds are ~20x slower)
maturin develop --release -E test

# 5. check
python -c "import crest; print(crest.__version__)"
pytest -q
```

:::{tip}
**conda users:** if a conda environment is active, maturin refuses to build because
`VIRTUAL_ENV` and `CONDA_PREFIX` are both set. Run `conda deactivate` first (or
`unset CONDA_PREFIX`).
:::

After changing any Rust file under `src/`, run `maturin develop --release` again. A
change to the Python files under `crest/` needs no rebuild.
