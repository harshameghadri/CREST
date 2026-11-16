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
.. autofunction:: crest.io.convert_h5_to_parquet_stream
```

Writing: {meth}`BioFrame.write_parquet <crest.BioFrame.write_parquet>`,
{meth}`BioFrame.to_anndata <crest.BioFrame.to_anndata>` and
{meth}`BioFrame.to_scipy <crest.BioFrame.to_scipy>`.

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

### Recall of the approximate neighbour search

Above 20,000 cells (or with `exact=False`) the neighbour search is approximate: cells are grouped
into inverted lists of about 512 cells and every cell is searched against the `nprobe` lists whose
centroids are nearest to the cell itself (`nprobe = max(16, ceil(0.5 * sqrt(n_lists)))` unless
given), then refined by NN-descent. Recall is the share of a cell's 14 exact neighbours that the
graph contains; it was measured on the PCA embeddings (50 components) of the agreement run
against exact float64 brute force on 10,000 cells sampled with a fixed seed, with 16 threads and
`random_state=0`, on a build of `eaeca867` (`release/0.5` at `ef5e974` plus the change that
introduced this table). `synth_200k` is synthetic (interpolated from pbmc68k). A search that
probed the lists around the centre of each cell's own list instead (the previous behaviour)
reached a mean recall of 0.9587 to 0.9976 on the same embeddings, with 0.4% to 7.9% of cells
below 0.8.

| data set | cells | inverted lists | `nprobe` | mean recall | cells < 0.8 | cells < 0.5 |
|---|---|---|---|---|---|---|
| kang | 28,871 | 56 | 16 | 1.0000 | 0.00% | 0.00% |
| pbmc68k | 68,551 | 133 | 16 | 0.9998 | 0.00% | 0.00% |
| synth_200k (synthetic) | 199,531 | 389 | 16 | 0.9992 | 0.03% | 0.00% |
| covid_stephenson | 647,365 | 1264 | 18 | 0.9988 | 0.08% | 0.00% |
| neurons_1m_sub500k | 494,803 | 966 | 16 | 0.9993 | 0.01% | 0.00% |
| parse_pbmc_1m | 996,653 | 1946 | 23 | 0.9989 | 0.03% | 0.00% |
| neurons_1m | 1,292,537 | 2524 | 26 | 0.9992 | 0.02% | 0.00% |
| parse_pbmc | 9,666,140 | 8192 | 46 | 0.9980 | 0.04% | 0.00% |

The recall of a data set not in this table can be checked with the same comparison: the
environment variable `CREST_TEST_KNN_EMBEDDING` points the test in `tests/test_knn_recall.py` at an
embedding of more than 20,000 cells and applies the thresholds mean recall >= 0.99, at most 1.0% of
cells below 0.8 and at most 0.5% below 0.5. A larger `nprobe` raises the recall at a cost that
grows about linearly with it.

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
.. autoclass:: crest.deseq2.NotEstimableError
.. autoclass:: crest.deseq2.LevelOrderWarning
```

All of these are also available as `crest.tl.<name>`.

## Native module

`crest.crest` is the compiled Rust extension. Its functions are internal: they take the
argument tuples yielded by {meth}`BioFrame.iter_ctx <crest.BioFrame.iter_ctx>` plus
preallocated NumPy output arrays. Use the Python API above. The native functions are listed
in {doc}`development` for contributors.

The bindings validate what they are given and raise `ValueError` (never a silent wrong
result) for arrays that are not C-contiguous (including Fortran-ordered matrices), for
indices out of range, for NaN, infinite or negative edge weights, for `k < 1`, for NaN or
infinite values in the points given to `knn_graph` / `knn_query` (the message names the
first bad row), and for more Wilcoxon entries than cells. `knn_graph` / `knn_query` warn when `k` exceeds the
number of available points and clamp it. Leiden warns when the largest edge weight is below
1e-6: its tolerances are absolute, and the partition collapses to singletons near 1e-12.
A panic inside a kernel is reported as `RuntimeError`.
