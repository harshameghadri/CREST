# API reference

```python
import crest
```

The API mirrors scanpy: `crest.pp` (preprocessing) and `crest.tl` (tools) take a
{class}`~crest.core.BioFrame` as their first argument. Top-level functions read data. Tables
are returned as Polars DataFrames and arrays as NumPy.

## Data: `BioFrame` and stores

```{eval-rst}
.. autoclass:: crest.BioFrame
   :members:
   :exclude-members: store, obs, var, obsm, varm, uns, ops, chunk_nnz

.. autoclass:: crest.CSRStore
.. autoclass:: crest.FrameStore
.. autoclass:: crest.ParquetStore
```

## Reading and writing

```{eval-rst}
.. autofunction:: crest.read_10x_h5
.. autofunction:: crest.read_10x_mtx
.. autofunction:: crest.read_h5ad
.. autofunction:: crest.read_parquet
```

Writing: {meth}`BioFrame.write_parquet <crest.core.BioFrame.write_parquet>`,
{meth}`BioFrame.to_anndata <crest.core.BioFrame.to_anndata>` and
{meth}`BioFrame.to_scipy <crest.core.BioFrame.to_scipy>`.

## Preprocessing: `crest.pp`

### Quality control and filtering

```{eval-rst}
.. autofunction:: crest.pp.calculate_qc_metrics
.. autofunction:: crest.pp.filter_cells
.. autofunction:: crest.pp.filter_genes
```

### Normalisation (recorded, applied lazily)

```{eval-rst}
.. autofunction:: crest.pp.normalize_total
.. autofunction:: crest.pp.log1p
.. autofunction:: crest.pp.scale
```

### Feature selection

```{eval-rst}
.. autofunction:: crest.pp.highly_variable_genes
.. autofunction:: crest.pp.gene_stats
```

### Neighbour graph

```{eval-rst}
.. autofunction:: crest.pp.neighbors
.. autofunction:: crest.pp.connectivities_matrix
```

### Doublets

```{eval-rst}
.. autofunction:: crest.doublets.scrublet
.. autofunction:: crest.doublets.threshold_minimum
```

`crest.pp.scrublet` is the same function.

## Tools: `crest.tl`

### Dimensionality reduction and embedding

```{eval-rst}
.. autofunction:: crest.tl.pca
.. autofunction:: crest.tl.umap
```

### Clustering

```{eval-rst}
.. autofunction:: crest.tl.leiden
.. autofunction:: crest.tl.leiden_sweep
.. autofunction:: crest.tl.adjusted_rand_index
```

### Marker genes and scores

```{eval-rst}
.. autofunction:: crest.tl.rank_genes_groups
.. autofunction:: crest.tl.score_genes
```

### Batch integration (Harmony)

```{eval-rst}
.. autofunction:: crest.harmony.harmony
.. autofunction:: crest.harmony.run_harmony
```

`crest.tl.harmony` is the same function.

### Reference mapping

```{eval-rst}
.. autofunction:: crest.ingest.ingest
.. autofunction:: crest.ingest.project_pca
```

`crest.tl.ingest` is the same function.

### Pseudobulk differential expression (DESeq2)

```{eval-rst}
.. autofunction:: crest.deseq2.pseudobulk
.. autofunction:: crest.deseq2.pseudobulk_de
.. autoclass:: crest.deseq2.DESeq2
   :members: results, results_names, dispersions, size_factors
.. autofunction:: crest.deseq2.deseq2
.. autoclass:: crest.deseq2.Pseudobulk
   :members: subset, to_anndata
.. autofunction:: crest.deseq2.model_matrix
```

All of these are also available as `crest.tl.<name>`.

## Native module

`crest.crest` is the compiled Rust extension. Its functions are internal: they take the
argument tuples yielded by {meth}`BioFrame.iter_ctx <crest.core.BioFrame.iter_ctx>` plus
preallocated NumPy output arrays. Use the Python API above. The native functions are listed
in {doc}`development` for contributors.
