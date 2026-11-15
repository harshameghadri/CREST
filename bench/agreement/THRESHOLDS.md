# Agreement run: registered metrics, thresholds and actions

Status: registered on 2026-11-14, before any agreement result was computed. The registration
point is the commit that introduces this file. After that commit nothing in the sections
"Metrics and thresholds" and "Actions on failure" changes; a needed change is written into
"Deviations" at the end of this file, with date and reason, and reported with the results.

Scope: untimed. Nothing measured by this run is a performance result. The run compares the
outputs of CREST and scanpy on the same data, with the same parameters, and records what
agrees, what does not, and by how much.

## 1. Code, environment and data

* CREST: the `release/0.5` head at the time of the run, built from a fresh clone with
  `maturin build --release`; the SHA and the wheel are recorded in every result file.
* Reference tool: scanpy 1.11.5 with the versions pinned in `bench/paper/requirements-bench.txt`
  (anndata 0.12.19, numpy 2.4.6, scipy 1.17.1, pynndescent 0.6.0, umap-learn 0.5.12, numba 0.68.0,
  igraph 1.0.0, leidenalg 0.12.0, scikit-learn 1.9.1, Python 3.11). The recorded versions of the
  run are the authoritative ones.
* Threads: 24 for both tools. Every dataset, seed count, SHA and version is stated next to each number.
* Datasets (files in the shared benchmark data directory, read by both tools):

  | dataset | cells (raw) | metrics | curated annotation |
  |---|---|---|---|
  | kang | 29,065 | 1 to 6 | `cell` (cell types; cells without a label are left out of metric 1) |
  | pbmc68k | 68,579 | 2 to 6 | none |
  | parse_pbmc_1m | 1,000,000 (random subset of the 9.7M-cell Parse set, 2,000 genes, seed 0) | 1 to 6 | `cell_type` (as supplied with the data) |
  | neurons_1m | 1,306,127 | 2 to 6 | none |
  | covid_stephenson | 647,366 | 1 to 6 | `cell_type` |
  | synth_200k | 200,000, synthetic (interpolated from pbmc68k with CREST's own PCA and kNN) | 5 only | none |
  | parse_pbmc | 9,697,974, 2,000 genes | 5 only, CREST only (scanpy is not run at this size) | none |

  neurons_1m and covid_stephenson are run only when the dataset manifest in the shared data
  directory lists them as verified when their run starts; otherwise they are reported as pending.
  The synthetic set is labelled synthetic wherever it appears.

## 2. Workflow and seeds

Both tools run the benchmark workflow with identical explicit parameters (`bench/paper/run_one.py`):
cell filter `min_genes` from the dataset metadata (200 unless stated there) and gene filter
`min_cells=3`; `normalize_total(target_sum=1e4)`, `log1p`; 2,000 highly variable genes (flavor
`seurat`); `scale(max_value=10)`; 50 principal components (scanpy: `svd_solver="arpack"`);
`n_neighbors=15` (14 neighbours per cell, the cell itself excluded); Leiden at resolution 1 with two
iterations (scanpy: `flavor="igraph"`, `n_iterations=2`); UMAP with default parameters; Wilcoxon
rank-sum differential expression of every Leiden cluster against the rest, over all genes.
Cells are matched across tools by barcode; every comparison uses the cells present in both outputs.

Seeds are `s = 0..9` (ten per tool). Seed `s` sets both the neighbour search (`random_state` of
`sc.pp.neighbors`; `random_state` of `crest.pp.neighbors`) and the Leiden run, on a fixed PCA. The
PCA is computed once per tool.

Arms:

* A (scanpy): scanpy PCA, scanpy kNN graph (seed `s`), scanpy Leiden (seed `s`). `A_leiden`: the
  seed-0 graph of A with ten Leiden seeds, i.e. only the clustering step varies.
* D (CREST): CREST PCA, CREST kNN graph (seed `s`), CREST Leiden (seed `s`).
* B (clustering step alone): scanpy's seed-0 connectivities, clustered by CREST's Leiden with seeds `0..9`.
* C (kNN step and clustering): scanpy's PCA, CREST's kNN graph and connectivities (seed `s`),
  CREST's Leiden (seed `s`).

"Within-scanpy spread" of a reference arm is the set of adjusted Rand indices (ARI) over the 45
pairs of its ten partitions. A cross-tool value is the mean ARI over the 100 (seed, seed) pairs
of the two arms compared.

## 3. Metrics and thresholds

A metric passes or fails on each dataset separately. A metric is reported as passing only when
it passes on every dataset it was computed on; the table of results lists each dataset.

### Metric 1. Agreement with curated annotation

ARI and normalised mutual information (NMI, arithmetic normalisation) of each seed's partition
against the curated annotation, on the labelled cells. For each tool, the mean over the ten seeds
and the range (maximum minus minimum) over the ten seeds are reported (arms D and A).

* Pass: for ARI and for NMI separately, `|mean(D) - mean(A)| <= max(range(A), 0.01)`.
  The 0.01 floor is the smallest difference in either index that is treated as material when
  the scanpy seeds barely differ.

### Metric 2. Cross-tool agreement against the within-scanpy seed spread, with a common-input decomposition

Let `W` be the within-scanpy spread of the reference arm and `L = min(min(W), mean(W) - 0.02)`.
A comparison passes when its mean cross-tool ARI is at least `L`.

* 2a (whole workflow): D against A. Reference spread `W`: A.
* 2b (clustering step on the same graph): B against `A_leiden`. Reference spread: `A_leiden`.
* 2c (kNN step on the same PCA): C against A. Reference spread: A.

Reported without a threshold: the within-tool spreads of D, B and C, cluster counts, the ARI
between the seed-0 partitions, the Jaccard index of the neighbour sets of the two tools' kNN graphs
(own PCA and common PCA), the cosines of the principal angles between the two PCA subspaces, and the
Jaccard index of the highly variable gene sets.

### Metric 3. Modularity

Modularity `Q` is computed with igraph's `Graph.modularity` (weighted, resolution 1), an
implementation independent of both tools, on the graph given to the clustering.

* Primary: the common graph, scanpy's seed-0 connectivities. Partitions: CREST's Leiden
  (two iterations) and the reference, `leidenalg.find_partition` with
  `RBConfigurationVertexPartition`, `resolution_parameter=1`, `n_iterations=2`, leidenalg 0.12.0.
  Seeds `0..19` on kang and pbmc68k, `0..9` on datasets of 500,000 cells or more.
  Pass: `mean Q(CREST) >= mean Q(leidenalg) - 0.003`.
* Secondary (same threshold, reported separately): the same comparison on CREST's own seed-0
  connectivities.
* Reported without a threshold: the gap `mean Q(CREST) - mean Q(scanpy, flavor="igraph", n_iterations=2)`
  on the common graph, and leidenalg with `n_iterations=-1`.

### Metric 4. Differential expression

Wilcoxon rank-sum scores and log2 fold changes, all genes passing both tools' gene filter,
group against rest.

* 4a (own clusters): the seed-0 partitions of D and A. Clusters are paired one to one by maximising
  the total Jaccard index of their cell sets (Hungarian assignment); a pair counts when its Jaccard
  index is at least 0.6. Fewer than three counted pairs makes the metric inconclusive, which is
  not a pass. Per counted pair: Spearman correlation of the scores over all genes, Spearman correlation
  of the log2 fold changes over all genes, Jaccard index of the top-50 genes by score, and Jaccard
  index of the significant sets (adjusted p-value below 0.05, log2 fold change above 0.5, score above 0).
  Pass: the median over counted pairs is at least 0.90 (score correlation), 0.90 (fold-change
  correlation), 0.50 (top-50 Jaccard) and 0.70 (significant-set Jaccard). The minimum over pairs
  and the number and cell fraction of unpaired clusters are reported.
* 4b (same labels): scanpy's seed-0 partition is given to CREST's differential expression, so both
  tools test the same groups. Pass: the median over groups is at least 0.99 (score correlation),
  0.99 (fold-change correlation), 0.80 (top-50 Jaccard) and 0.90 (significant-set Jaccard).
* Context, not thresholded, datasets up to 100,000 cells: the same four values between scanpy's
  seed-0 partition and scanpy's seed-1 and seed-2 partitions (paired as in 4a).

### Metric 5. kNN recall against exact search

For the real embedding of each dataset, the recall of a kNN graph is measured on 10,000
uniformly sampled query cells (seed 0; all cells when a dataset has fewer, never fewer than 5,000;
the count is stated with each result), against exact brute-force neighbours in float64 on the embedding
the graph was built in, `k = 14`, the cell itself excluded. A neighbour counts as correct when its
distance does not exceed the exact k-th distance (so ties at the boundary are not misses). Recall of a
query is correct neighbours divided by `k`. Reported: mean recall, the fraction of queries below 0.8 and
the fraction below 0.5.

* 5.1: CREST's graph built by `crest.pp.neighbors` (default search) on CREST's PCA, scored on CREST's PCA.
* 5.2: scanpy's graph (`sc.pp.neighbors`, pynndescent) on scanpy's PCA, scored on scanpy's PCA, by the same
  function. Reported without a threshold.
* 5.3: CREST's graph built on scanpy's PCA, scored on scanpy's PCA, so that both tools are scored on one
  embedding.
* 5.4 (queries from other batches): reference points are the cells of the first half of the levels of
  the dataset's batch column (kang: `stim`; parse and covid: donor), queries are 10,000 sampled cells
  of the remaining levels, searched with `crest.crest.knn_query` on CREST's PCA, scored the same way.
  Computed where a batch column exists.

* Pass (5.1, 5.3, 5.4): mean recall at least 0.99, fraction of queries below 0.5 at most 0.5 percent,
  fraction of queries below 0.8 at most 1.0 percent.
* The synthetic set synth_200k and the 9.7M-cell Parse set are scored by 5.1 (and 5.2 and 5.3 for synth_200k)
  only; the QC and normalisation of the workflow are applied as above, and for the 9.7M-cell set CREST runs
  out of core within the memory limit of the lane.

### Metric 6. UMAP

UMAP is computed from each tool's own seed-0 graph (default parameters), seeds `0, 1, 2` up to 100,000
cells and seed 0 above. Both are scored in one common space, scanpy's PCA.

* Local: trustworthiness at 15 neighbours, `T = 1 - 2 / (n k (2n - 3k - 1)) * sum of rank penalties`, with the
  neighbourhoods taken among all `n` cells and the penalty sum estimated from 5,000 sampled query cells
  (seed 0; the estimate equals scikit-learn's `trustworthiness` when every cell is a query).
  Pass: `T(CREST) >= T(scanpy) - 0.01` (means over the UMAP seeds).
* Global (the one global metric): Spearman correlation between the pairwise Euclidean distances of
  cluster centroids in the common PCA space and in the UMAP, clusters of scanpy's seed-0 partition with at least
  100 cells, all cluster pairs. Pass: `rho(CREST) >= rho(scanpy) - 0.10` (means over the UMAP seeds).
  UMAP claims are restricted to local structure regardless of this result.

## 4. Actions on failure

Each failed metric is assigned one action in advance.

| metric | failure | action |
|---|---|---|
| 2b | Leiden on identical input disagrees more than scanpy with itself | defect in the Leiden step: reported with the dataset and numbers for a fix before code freeze |
| 2c, 5.1, 5.3, 5.4 | kNN step disagrees or recall below threshold | defect in the kNN index settings: reported for a fix before code freeze (candidate: the index probe settings); if the fix costs unacceptable runtime, the measured recall is written into the documentation as a limitation |
| 3 | mean Q below the reference by more than 0.003 | defect in the Leiden optimisation: fix before code freeze |
| 4b | same-label differential expression disagrees | defect in the differential expression step: fix before code freeze |
| 1, 2a with 2b, 2c passing | the whole workflow disagrees more than the scanpy seed spread, while the steps that follow the PCA agree | attributed to the steps before the kNN graph (gene selection, scaling, PCA solver); the measured values replace any statement that CREST reproduces scanpy's clusters; the PCA stage is reported for a decision |
| 4a with 4b passing | cluster definitions differ, expression statistics agree | limitation sentence with the measured values and the within-scanpy context |
| 6 local | trustworthiness below the threshold | reported for a decision (layout optimiser) before code freeze |
| 6 global | correlation below the threshold | limitation: UMAP claims stay restricted to local structure; no code change |
| inconclusive | fewer than three counted pairs in 4a, or a missing input | reported as not evaluated; never counted as a pass |

A reported defect is not fixed in this run. Any numbers produced by arms that failed are reported as
measured, together with the pre-registered action.

## 5. Provenance and conduct

* Results are not committed to the repository. Raw per-run outputs (per seed labels, per-cluster tables,
  recall vectors, configuration and environment records) are archived with the results, with the commit
  and command that produced each file.
* Datasets are never substituted or subsampled without being named as such in the result.
* The recall thresholds (mean 0.99, at most 0.5 percent of queries below 0.5) and the local UMAP threshold (0.01)
  are the acceptance values set for the corresponding code changes before this run. The other thresholds were
  chosen without any measurement of these metrics on these datasets.

## Deviations

No threshold, metric definition or action has been changed. One diagnostic was added after results were seen.

* 2026-11-15, post-hoc diagnostic (no threshold). After the kang results were seen (metric 2c and the ARI of metric 1
  failed while 2b passed, and scanpy's own kNN recall on kang was 0.94), an arm E was added: scanpy's PCA, CREST's exact kNN
  search (`exact=True`), CREST's connectivities and Leiden with seeds 0..9, i.e. the clustering of the true kNN graph.
  The mean ARI of E against A and against C, and the Jaccard index of each approximate kNN graph with the exact graph, are
  reported under `diagnostics.post_hoc_exact_graph_arm` and are labelled post-hoc wherever they appear. They are run on every
  dataset the registered metrics were run on, and do not enter any pass/fail decision.
* 2026-11-15, memory measure (no change to any value). scanpy's workflow on parse_pbmc_1m (1,000,000 cells x 2,000 genes) was
  killed by the 40 GB memory limit of the run: the scaled matrix is a dense float64 array of 16 GB and scikit-learn's PCA
  copies it before centering. For datasets of this size the scanpy stage runs with `--pca-in-place`, which centres the scaled
  matrix in place (scikit-learn's `copy=False`). The result is the same floating-point numbers (the matrix is not used after
  the PCA; a test compares both paths bit for bit on a small matrix), and the setting is recorded in `scanpy_meta.json`.
* 2026-11-15, reported-only item not computed at scale (no threshold affected). The modularity of leidenalg run to convergence
  (`n_iterations=-1`, reported without a threshold in metric 3) was started on parse_pbmc_1m with two seeds and had not finished
  after 60 minutes; it is computed up to 100,000 cells (five seeds) and not computed on larger datasets. The registered comparison
  (`n_iterations=2`) is computed on every dataset.
* 2026-11-15, datasets. (a) neurons_1m: the scanpy stage was attempted on the full file under the 40 GB memory limit of this run and was killed by the limit
  while reading it (exit 137, 1.7 minutes after the start; log in the results archive); the Datasets lane's untested read-back
  estimate for scanpy's reader is about 65 GB at 1,306,127 cells, which is consistent with that. Whether scanpy runs on the full file
  with a higher limit is a decision for the benchmark protocol. so metrics 1 to 6 were computed on its 500,000-cell real subsample `neurons_1m_sub500k` (seed 0, nested subset of
  neurons_1m), and metric 5.1 was also computed on the full neurons_1m for CREST alone (scanpy not run). The subsample is named
  as such wherever it appears. neurons_1m has no curated annotation, so metric 1 does not apply to it. (b) parse_pbmc_1m is the
  rebuilt, nested seed-0 subset of the 9.7M-cell Parse set (file written 2026-11-15 00:08 +01:00, sha256 equal to the dataset
  manifest); every result of this run read that file. (c) covid_stephenson uses the annotation column `cell_type` named in its
  metadata.

