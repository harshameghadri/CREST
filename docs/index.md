# CREST

CREST runs the standard single-cell RNA-seq workflow with native Rust kernels behind a
scanpy-style Python API. The steps are QC, normalisation, highly variable genes, scaling and
PCA, neighbours, Leiden, UMAP and differential expression. After clustering it adds Harmony
integration, doublet detection, reference mapping and pseudobulk DESeq2.

The standard is to **give the same answers as the reference tools** (scanpy 1.11,
R DESeq2 1.42, harmonypy 2.x) while being faster and using less memory. How much faster
depends on dataset size and thread count; {doc}`benchmarks` reports the measured numbers,
including where CREST is not faster.

CREST never makes a normalised, scaled or dense copy of the count matrix. Filters and
transforms are recorded and applied on the fly inside each kernel. Data can also be streamed
from disk, with memory bounded by one chunk.

```python
import crest

bf = crest.read_10x_h5("filtered_feature_bc_matrix.h5")
bf = crest.pp.filter_cells(bf, min_genes=200)
bf = crest.pp.filter_genes(bf, min_cells=3)
crest.pp.normalize_total(bf, target_sum=1e4)   # recorded, applied lazily
crest.pp.log1p(bf)                             # recorded, applied lazily
crest.pp.highly_variable_genes(bf, n_top_genes=2000)
crest.pp.scale(bf, max_value=10)               # applied exactly inside PCA
crest.tl.pca(bf, n_comps=50)
crest.pp.neighbors(bf, n_neighbors=15)
crest.tl.leiden(bf)
crest.tl.umap(bf)
markers = crest.tl.rank_genes_groups(bf, "leiden", method="wilcoxon")
```

New here? Start with {doc}`installation` and {doc}`quickstart`, then read {doc}`concepts`.
{doc}`concepts` explains the one idea everything else rests on: lazy transforms applied
chunk by chunk.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

concepts
memory_model
downstream
deseq2
benchmarks
```

```{toctree}
:maxdepth: 2
:caption: API reference

api
```

```{toctree}
:maxdepth: 1
:caption: Project

development
readthedocs
changelog
```
