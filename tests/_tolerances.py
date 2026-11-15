"""Tolerances of the scanpy parity matrix (tests/test_scanpy_parity_matrix.py): one named constant per comparison.

Each value is the largest deviation measured over PBMC 3k and PBMC 10k, x86-64 (Linux) and arm64 (macOS), and scanpy 1.11.5,
1.12.6 and 1.13.0, times MARGIN, rounded up to two significant digits (`tests/_parity_cases.py` defines every comparison and
how its deviation is measured; a deviation is non-negative, a shortfall is max(0, reference - CREST)). Two rules bound the margin:

* a constant is never looser than a tolerance that an existing test already applies to the same comparison (EXISTING_TESTS);
  where the margin would exceed it, the existing value is kept for the data set that test uses (PBMC 3k) and a separate
  constant with the suffix _PBMC10K carries 10k;
* a deviation that measured exactly 0 keeps tolerance 0 for counts of differing genes, cells or entries, and gets the floor
  NOISE_FLOAT (two float32 ulp) or NOISE_FRACTION (one in a thousand) otherwise.

Measured with CREST release/0.5 at d6a1f16. Changing a value needs the measurement that justifies it (see the parity matrix
published with the release); never loosen a constant to make a test pass.
"""

MARGIN = 2.0
NOISE_FLOAT = 2.4e-07
NOISE_FRACTION = 0.001

QC_CELLS_AFTER_MIN_GENES_N_DIFFERENT = 0.0  # measured max: pbmc3k 0, pbmc10k 0
QC_GENES_AFTER_MIN_CELLS_N_DIFFERENT = 0.0  # measured max: pbmc3k 0, pbmc10k 0
QC_N_GENES_BY_COUNTS_MAX_ABS = 0.0  # measured max: pbmc3k 0, pbmc10k 0
QC_TOTAL_COUNTS_MAX_REL = 2.4e-07  # measured max: pbmc3k 0, pbmc10k 0
QC_PCT_COUNTS_MT_MAX_ABS = 1.1e-05  # measured max: pbmc3k 1.34e-06, pbmc10k 5.39e-06
QC_CELLS_AFTER_MAX_FILTERS_N_DIFFERENT = 0.0  # measured max: pbmc3k 0, pbmc10k 0
NORMALIZE_TOTAL_TARGET_1E4_PATTERN_DIFFERENCES = 0.0  # measured max: pbmc3k 0, pbmc10k 0
NORMALIZE_TOTAL_TARGET_1E4_MAX_ABS = 0.00049  # measured max: pbmc3k 0.000122, pbmc10k 0.000244
NORMALIZE_TOTAL_TARGET_1E4_MAX_REL = 2.4e-07  # measured max: pbmc3k 1.19e-07, pbmc10k 1.19e-07
NORMALIZE_TOTAL_DEFAULT_PATTERN_DIFFERENCES = 0.0  # measured max: pbmc3k 0, pbmc10k 0
NORMALIZE_TOTAL_DEFAULT_MAX_ABS = 0.00025  # measured max: pbmc3k 1.53e-05, pbmc10k 0.000122
NORMALIZE_TOTAL_DEFAULT_MAX_REL = 2.4e-07  # measured max: pbmc3k 1.19e-07, pbmc10k 1.19e-07
LOG1P_PATTERN_DIFFERENCES = 0.0  # measured max: pbmc3k 0, pbmc10k 0
LOG1P_MAX_ABS = 9.6e-07  # measured max: pbmc3k 4.77e-07, pbmc10k 4.77e-07
LOG1P_MAX_REL = 2.9e-07  # measured max: pbmc3k 1.19e-07, pbmc10k 1.44e-07
HVG_SEURAT_NTOP_N_GENES_DIFFERENT = 0.0  # measured max: pbmc3k 0, pbmc10k 0
HVG_SEURAT_NTOP_MEANS_MAX_REL = 7.4e-07  # measured max: pbmc3k 3.66e-07, pbmc10k 2.34e-07
HVG_SEURAT_NTOP_DISPERSIONS_MAX_REL = 0.00042  # measured max: pbmc3k 2.69e-07, pbmc10k 0.000206
HVG_SEURAT_NTOP_DISPERSIONS_NORM_MAX_ABS = 3.6e-06  # measured max: pbmc3k 1.78e-06, pbmc10k 1.1e-06
HVG_CELL_RANGER_NTOP_N_GENES_DIFFERENT = 0.0  # measured max: pbmc3k 0, pbmc10k 0
HVG_CELL_RANGER_NTOP_MEANS_MAX_REL = 2.4e-07  # measured max: pbmc3k 8.2e-08, pbmc10k 8.49e-08
HVG_CELL_RANGER_NTOP_DISPERSIONS_MAX_REL = 3.1e-07  # measured max: pbmc3k 1.04e-07, pbmc10k 1.53e-07
HVG_CELL_RANGER_NTOP_DISPERSIONS_NORM_MAX_ABS = 3.4e-05  # measured max: pbmc3k 1.65e-05, pbmc10k 6.6e-06
HVG_SEURAT_TUTORIAL_N_GENES_DIFFERENT = 0.0  # measured max: pbmc3k 0, pbmc10k 0
HVG_SEURAT_TUTORIAL_MEANS_MAX_REL = 7.4e-07  # measured max: pbmc3k 3.66e-07, pbmc10k 2.34e-07
HVG_SEURAT_TUTORIAL_DISPERSIONS_MAX_REL = 0.00042  # measured max: pbmc3k 2.69e-07, pbmc10k 0.000206
HVG_SEURAT_TUTORIAL_DISPERSIONS_NORM_MAX_ABS = 3.6e-06  # measured max: pbmc3k 1.78e-06, pbmc10k 1.1e-06
HVG_SEURAT_V3_N_GENES_DIFFERENT = 0.0  # measured max: pbmc3k 0, pbmc10k 0
HVG_SEURAT_V3_VARIANCES_NORM_MAX_REL = 1e-10  # pbmc3k; measured max: pbmc3k 3.22e-13, pbmc10k 2.68e-10
HVG_SEURAT_V3_VARIANCES_NORM_MAX_REL_PBMC10K = 2.4e-07  # pbmc10k; measured max: pbmc3k 3.22e-13, pbmc10k 2.68e-10
HVG_SEURAT_V3_RANK_N_DIFFERENT = 0.0  # measured max: pbmc3k 0, pbmc10k 0
PCA_SCALE_MAX10_ANGLE_TOP10_DEG = 1.9e-05  # measured max: pbmc3k 9.06e-06, pbmc10k 7.63e-06
PCA_SCALE_MAX10_ANGLE_ALL50_DEG = 4.9e-05  # measured max: pbmc3k 2.41e-05, pbmc10k 1.89e-05
PCA_SCALE_MAX10_PC_SCORES_TOP10_MAX_REL_STD = 3e-06  # measured max: pbmc3k 9.67e-07, pbmc10k 1.46e-06
PCA_SCALE_MAX10_VARIANCE_RATIO_MAX_ABS = 2.4e-07  # measured max: pbmc3k 5.46e-11, pbmc10k 2.25e-11
PCA_SCALE_NOCLIP_ANGLE_TOP10_DEG = 2.3e-05  # measured max: pbmc3k 1.13e-05, pbmc10k 6.32e-06
KNN_CREST_EXACT_RECALL_SHORTFALL = 0.001  # measured max: pbmc3k 0, pbmc10k 0
CONNECTIVITIES_SAME_KNN_PATTERN_DIFFERENCES = 0.0  # measured max: pbmc3k 0, pbmc10k 0
CONNECTIVITIES_SAME_KNN_MAX_ABS = 2.4e-07  # measured max: pbmc3k 1.19e-07, pbmc10k 1.19e-07
LEIDEN_MODULARITY_SHORTFALL_VS_LEIDENALG = 0.001  # measured max: pbmc3k 0, pbmc10k 0
LEIDEN_MODULARITY_SHORTFALL_VS_SCANPY_IGRAPH = 0.0046  # measured max: pbmc3k 0.00228, pbmc10k 0
LEIDEN_N_CLUSTERS_MEAN_DIFF_VS_LEIDENALG = 0.4  # pbmc3k; measured max: pbmc3k 0.2, pbmc10k 1.2
LEIDEN_N_CLUSTERS_MEAN_DIFF_VS_LEIDENALG_PBMC10K = 2.4  # pbmc10k; measured max: pbmc3k 0.2, pbmc10k 1.2
LEIDEN_N_CLUSTERS_MEAN_DIFF_VS_SCANPY_IGRAPH = 0.8  # measured max: pbmc3k 0.4, pbmc10k 0.4
UMAP_TRUSTWORTHINESS_GAP = 0.0035  # measured max: pbmc3k 0.0005, pbmc10k 0.00171
UMAP_TRUSTWORTHINESS_SHORTFALL = 0.0013  # measured max: pbmc3k 0, pbmc10k 0.000612
RANK_GENES_GROUPS_TTEST_IDENTICAL_SCORES_MAX_ABS = 6.9e-05  # pbmc3k; measured max: pbmc3k 3.44e-05, pbmc10k 6.77e-05
RANK_GENES_GROUPS_TTEST_IDENTICAL_SCORES_MAX_ABS_PBMC10K = 0.0001  # pbmc10k; measured max: pbmc3k 3.44e-05, pbmc10k 6.77e-05
RANK_GENES_GROUPS_TTEST_IDENTICAL_LOGFOLDCHANGES_MAX_ABS = 2e-06  # measured max: pbmc3k 9.54e-07, pbmc10k 9.54e-07
RANK_GENES_GROUPS_TTEST_IDENTICAL_PVALS_MAX_ABS = 1.3e-06  # measured max: pbmc3k 6.41e-07, pbmc10k 4.71e-07
RANK_GENES_GROUPS_TTEST_IDENTICAL_PVALS_ADJ_MAX_ABS = 1.9e-06  # measured max: pbmc3k 9.49e-07, pbmc10k 6.49e-07
RANK_GENES_GROUPS_TTEST_IDENTICAL_TOP50_OVERLAP_SHORTFALL = 0.001  # measured max: pbmc3k 0, pbmc10k 0
RANK_GENES_GROUPS_WILCOXON_IDENTICAL_SCORES_MAX_ABS = 1e-06  # pbmc3k; measured max: pbmc3k 9.47e-07, pbmc10k 3.77e-06
RANK_GENES_GROUPS_WILCOXON_IDENTICAL_SCORES_MAX_ABS_PBMC10K = 7.6e-06  # pbmc10k; measured max: pbmc3k 9.47e-07, pbmc10k 3.77e-06
RANK_GENES_GROUPS_WILCOXON_IDENTICAL_LOGFOLDCHANGES_MAX_ABS = 2e-06  # measured max: pbmc3k 9.54e-07, pbmc10k 9.54e-07
RANK_GENES_GROUPS_WILCOXON_IDENTICAL_PVALS_MAX_ABS = 2.4e-07  # measured max: pbmc3k 4.94e-11, pbmc10k 4.94e-11
RANK_GENES_GROUPS_WILCOXON_IDENTICAL_PVALS_ADJ_MAX_ABS = 2.4e-07  # measured max: pbmc3k 4.57e-11, pbmc10k 1.02e-10
RANK_GENES_GROUPS_WILCOXON_IDENTICAL_TOP50_OVERLAP_SHORTFALL = 0.001  # measured max: pbmc3k 0, pbmc10k 0
RANK_GENES_GROUPS_TTEST_LAZY_SCORES_MAX_ABS = 6.2e-05  # pbmc3k; measured max: pbmc3k 3.06e-05, pbmc10k 5.15e-05
RANK_GENES_GROUPS_TTEST_LAZY_SCORES_MAX_ABS_PBMC10K = 0.0001  # pbmc10k; measured max: pbmc3k 3.06e-05, pbmc10k 5.15e-05
RANK_GENES_GROUPS_TTEST_LAZY_LOGFOLDCHANGES_MAX_ABS = 2.2e-06  # measured max: pbmc3k 1.03e-06, pbmc10k 1.07e-06
RANK_GENES_GROUPS_TTEST_LAZY_PVALS_MAX_ABS = 1.4e-06  # measured max: pbmc3k 6.59e-07, pbmc10k 5.93e-07
RANK_GENES_GROUPS_TTEST_LAZY_PVALS_ADJ_MAX_ABS = 1.6e-06  # measured max: pbmc3k 7.76e-07, pbmc10k 7.4e-07
RANK_GENES_GROUPS_TTEST_LAZY_TOP50_OVERLAP_SHORTFALL = 0.001  # measured max: pbmc3k 0, pbmc10k 0
RANK_GENES_GROUPS_WILCOXON_LAZY_SCORES_MAX_ABS = 0.0005  # pbmc3k; measured max: pbmc3k 0.000269, pbmc10k 0.00014
RANK_GENES_GROUPS_WILCOXON_LAZY_SCORES_MAX_ABS_PBMC10K = 0.00029  # pbmc10k; measured max: pbmc3k 0.000269, pbmc10k 0.00014
RANK_GENES_GROUPS_WILCOXON_LAZY_LOGFOLDCHANGES_MAX_ABS = 2.2e-06  # measured max: pbmc3k 1.03e-06, pbmc10k 1.07e-06
RANK_GENES_GROUPS_WILCOXON_LAZY_PVALS_MAX_ABS = 0.0003  # measured max: pbmc3k 0.000147, pbmc10k 9.7e-05
RANK_GENES_GROUPS_WILCOXON_LAZY_PVALS_ADJ_MAX_ABS = 0.00066  # measured max: pbmc3k 0.000326, pbmc10k 0.000174
RANK_GENES_GROUPS_WILCOXON_LAZY_TOP50_OVERLAP_SHORTFALL = 0.001  # measured max: pbmc3k 0, pbmc10k 0
SCORE_GENES_B_MARKERS_MAX_ABS = 9.1e-07  # measured max: pbmc3k 3.79e-07, pbmc10k 4.52e-07
SCORE_GENES_RANDOM_25_MAX_ABS = 2.4e-07  # measured max: pbmc3k 4.38e-08, pbmc10k 4.94e-08
SCRUBLET_MEAN_SCORE_RATIO_DEV = 0.012  # measured max: pbmc3k 0.00428, pbmc10k 0.00594
SCRUBLET_SPEARMAN_SHORTFALL_WORST_SEED = 0.06  # pbmc3k; measured max: pbmc3k 0.0493, pbmc10k 0.0193
SCRUBLET_SPEARMAN_SHORTFALL_WORST_SEED_PBMC10K = 0.039  # pbmc10k; measured max: pbmc3k 0.0493, pbmc10k 0.0193
SCRUBLET_THRESHOLD_ABS_DIFF_MAX = 0.086  # measured max: pbmc3k 0.0429, pbmc10k 0.00317
SCRUBLET_PREDICTED_RATE_ABS_DIFF_MAX = 0.016  # measured max: pbmc3k 0.00772, pbmc10k 0.00154
SCRUBLET_AUROC_SPIKED_ABS_DIFF_MAX = 0.028  # measured max: pbmc3k 0.0138, pbmc10k 0.00333
INGEST_MATCHING_CENTRE_PCA_TOP10_MAX_REL_STD = 4.9e-06  # measured max: pbmc3k 2.45e-06, pbmc10k 1.63e-06
INGEST_MATCHING_CENTRE_LABEL_DISAGREEMENT = 0.043  # measured max: pbmc3k 0.0121, pbmc10k 0.0214
INGEST_MATCHING_CENTRE_LABEL_ACCURACY_ABS_DIFF = 0.031  # measured max: pbmc3k 0.000758, pbmc10k 0.0152

# comparison id -> name of the constant above, per data set
TOLERANCES = {
    "qc.cells_after_min_genes.n_different": {"pbmc3k": "QC_CELLS_AFTER_MIN_GENES_N_DIFFERENT", "pbmc10k": "QC_CELLS_AFTER_MIN_GENES_N_DIFFERENT"},
    "qc.genes_after_min_cells.n_different": {"pbmc3k": "QC_GENES_AFTER_MIN_CELLS_N_DIFFERENT", "pbmc10k": "QC_GENES_AFTER_MIN_CELLS_N_DIFFERENT"},
    "qc.n_genes_by_counts.max_abs": {"pbmc3k": "QC_N_GENES_BY_COUNTS_MAX_ABS", "pbmc10k": "QC_N_GENES_BY_COUNTS_MAX_ABS"},
    "qc.total_counts.max_rel": {"pbmc3k": "QC_TOTAL_COUNTS_MAX_REL", "pbmc10k": "QC_TOTAL_COUNTS_MAX_REL"},
    "qc.pct_counts_mt.max_abs": {"pbmc3k": "QC_PCT_COUNTS_MT_MAX_ABS", "pbmc10k": "QC_PCT_COUNTS_MT_MAX_ABS"},
    "qc.cells_after_max_filters.n_different": {"pbmc3k": "QC_CELLS_AFTER_MAX_FILTERS_N_DIFFERENT", "pbmc10k": "QC_CELLS_AFTER_MAX_FILTERS_N_DIFFERENT"},
    "normalize_total.target_1e4.pattern_differences": {"pbmc3k": "NORMALIZE_TOTAL_TARGET_1E4_PATTERN_DIFFERENCES", "pbmc10k": "NORMALIZE_TOTAL_TARGET_1E4_PATTERN_DIFFERENCES"},
    "normalize_total.target_1e4.max_abs": {"pbmc3k": "NORMALIZE_TOTAL_TARGET_1E4_MAX_ABS", "pbmc10k": "NORMALIZE_TOTAL_TARGET_1E4_MAX_ABS"},
    "normalize_total.target_1e4.max_rel": {"pbmc3k": "NORMALIZE_TOTAL_TARGET_1E4_MAX_REL", "pbmc10k": "NORMALIZE_TOTAL_TARGET_1E4_MAX_REL"},
    "normalize_total.default.pattern_differences": {"pbmc3k": "NORMALIZE_TOTAL_DEFAULT_PATTERN_DIFFERENCES", "pbmc10k": "NORMALIZE_TOTAL_DEFAULT_PATTERN_DIFFERENCES"},
    "normalize_total.default.max_abs": {"pbmc3k": "NORMALIZE_TOTAL_DEFAULT_MAX_ABS", "pbmc10k": "NORMALIZE_TOTAL_DEFAULT_MAX_ABS"},
    "normalize_total.default.max_rel": {"pbmc3k": "NORMALIZE_TOTAL_DEFAULT_MAX_REL", "pbmc10k": "NORMALIZE_TOTAL_DEFAULT_MAX_REL"},
    "log1p.pattern_differences": {"pbmc3k": "LOG1P_PATTERN_DIFFERENCES", "pbmc10k": "LOG1P_PATTERN_DIFFERENCES"},
    "log1p.max_abs": {"pbmc3k": "LOG1P_MAX_ABS", "pbmc10k": "LOG1P_MAX_ABS"},
    "log1p.max_rel": {"pbmc3k": "LOG1P_MAX_REL", "pbmc10k": "LOG1P_MAX_REL"},
    "hvg.seurat_ntop.n_genes_different": {"pbmc3k": "HVG_SEURAT_NTOP_N_GENES_DIFFERENT", "pbmc10k": "HVG_SEURAT_NTOP_N_GENES_DIFFERENT"},
    "hvg.seurat_ntop.means.max_rel": {"pbmc3k": "HVG_SEURAT_NTOP_MEANS_MAX_REL", "pbmc10k": "HVG_SEURAT_NTOP_MEANS_MAX_REL"},
    "hvg.seurat_ntop.dispersions.max_rel": {"pbmc3k": "HVG_SEURAT_NTOP_DISPERSIONS_MAX_REL", "pbmc10k": "HVG_SEURAT_NTOP_DISPERSIONS_MAX_REL"},
    "hvg.seurat_ntop.dispersions_norm.max_abs": {"pbmc3k": "HVG_SEURAT_NTOP_DISPERSIONS_NORM_MAX_ABS", "pbmc10k": "HVG_SEURAT_NTOP_DISPERSIONS_NORM_MAX_ABS"},
    "hvg.cell_ranger_ntop.n_genes_different": {"pbmc3k": "HVG_CELL_RANGER_NTOP_N_GENES_DIFFERENT", "pbmc10k": "HVG_CELL_RANGER_NTOP_N_GENES_DIFFERENT"},
    "hvg.cell_ranger_ntop.means.max_rel": {"pbmc3k": "HVG_CELL_RANGER_NTOP_MEANS_MAX_REL", "pbmc10k": "HVG_CELL_RANGER_NTOP_MEANS_MAX_REL"},
    "hvg.cell_ranger_ntop.dispersions.max_rel": {"pbmc3k": "HVG_CELL_RANGER_NTOP_DISPERSIONS_MAX_REL", "pbmc10k": "HVG_CELL_RANGER_NTOP_DISPERSIONS_MAX_REL"},
    "hvg.cell_ranger_ntop.dispersions_norm.max_abs": {"pbmc3k": "HVG_CELL_RANGER_NTOP_DISPERSIONS_NORM_MAX_ABS", "pbmc10k": "HVG_CELL_RANGER_NTOP_DISPERSIONS_NORM_MAX_ABS"},
    "hvg.seurat_tutorial.n_genes_different": {"pbmc3k": "HVG_SEURAT_TUTORIAL_N_GENES_DIFFERENT", "pbmc10k": "HVG_SEURAT_TUTORIAL_N_GENES_DIFFERENT"},
    "hvg.seurat_tutorial.means.max_rel": {"pbmc3k": "HVG_SEURAT_TUTORIAL_MEANS_MAX_REL", "pbmc10k": "HVG_SEURAT_TUTORIAL_MEANS_MAX_REL"},
    "hvg.seurat_tutorial.dispersions.max_rel": {"pbmc3k": "HVG_SEURAT_TUTORIAL_DISPERSIONS_MAX_REL", "pbmc10k": "HVG_SEURAT_TUTORIAL_DISPERSIONS_MAX_REL"},
    "hvg.seurat_tutorial.dispersions_norm.max_abs": {"pbmc3k": "HVG_SEURAT_TUTORIAL_DISPERSIONS_NORM_MAX_ABS", "pbmc10k": "HVG_SEURAT_TUTORIAL_DISPERSIONS_NORM_MAX_ABS"},
    "hvg.seurat_v3.n_genes_different": {"pbmc3k": "HVG_SEURAT_V3_N_GENES_DIFFERENT", "pbmc10k": "HVG_SEURAT_V3_N_GENES_DIFFERENT"},
    "hvg.seurat_v3.variances_norm.max_rel": {"pbmc3k": "HVG_SEURAT_V3_VARIANCES_NORM_MAX_REL", "pbmc10k": "HVG_SEURAT_V3_VARIANCES_NORM_MAX_REL_PBMC10K"},
    "hvg.seurat_v3.rank.n_different": {"pbmc3k": "HVG_SEURAT_V3_RANK_N_DIFFERENT", "pbmc10k": "HVG_SEURAT_V3_RANK_N_DIFFERENT"},
    "pca.scale_max10.angle_top10_deg": {"pbmc3k": "PCA_SCALE_MAX10_ANGLE_TOP10_DEG", "pbmc10k": "PCA_SCALE_MAX10_ANGLE_TOP10_DEG"},
    "pca.scale_max10.angle_all50_deg": {"pbmc3k": "PCA_SCALE_MAX10_ANGLE_ALL50_DEG", "pbmc10k": "PCA_SCALE_MAX10_ANGLE_ALL50_DEG"},
    "pca.scale_max10.pc_scores_top10.max_rel_std": {"pbmc3k": "PCA_SCALE_MAX10_PC_SCORES_TOP10_MAX_REL_STD", "pbmc10k": "PCA_SCALE_MAX10_PC_SCORES_TOP10_MAX_REL_STD"},
    "pca.scale_max10.variance_ratio.max_abs": {"pbmc3k": "PCA_SCALE_MAX10_VARIANCE_RATIO_MAX_ABS", "pbmc10k": "PCA_SCALE_MAX10_VARIANCE_RATIO_MAX_ABS"},
    "pca.scale_noclip.angle_top10_deg": {"pbmc3k": "PCA_SCALE_NOCLIP_ANGLE_TOP10_DEG", "pbmc10k": "PCA_SCALE_NOCLIP_ANGLE_TOP10_DEG"},
    "knn.crest_exact.recall_shortfall": {"pbmc3k": "KNN_CREST_EXACT_RECALL_SHORTFALL", "pbmc10k": "KNN_CREST_EXACT_RECALL_SHORTFALL"},
    "connectivities.same_knn.pattern_differences": {"pbmc3k": "CONNECTIVITIES_SAME_KNN_PATTERN_DIFFERENCES", "pbmc10k": "CONNECTIVITIES_SAME_KNN_PATTERN_DIFFERENCES"},
    "connectivities.same_knn.max_abs": {"pbmc3k": "CONNECTIVITIES_SAME_KNN_MAX_ABS", "pbmc10k": "CONNECTIVITIES_SAME_KNN_MAX_ABS"},
    "leiden.modularity_shortfall_vs_leidenalg": {"pbmc3k": "LEIDEN_MODULARITY_SHORTFALL_VS_LEIDENALG", "pbmc10k": "LEIDEN_MODULARITY_SHORTFALL_VS_LEIDENALG"},
    "leiden.modularity_shortfall_vs_scanpy_igraph": {"pbmc3k": "LEIDEN_MODULARITY_SHORTFALL_VS_SCANPY_IGRAPH", "pbmc10k": "LEIDEN_MODULARITY_SHORTFALL_VS_SCANPY_IGRAPH"},
    "leiden.n_clusters_mean_diff_vs_leidenalg": {"pbmc3k": "LEIDEN_N_CLUSTERS_MEAN_DIFF_VS_LEIDENALG", "pbmc10k": "LEIDEN_N_CLUSTERS_MEAN_DIFF_VS_LEIDENALG_PBMC10K"},
    "leiden.n_clusters_mean_diff_vs_scanpy_igraph": {"pbmc3k": "LEIDEN_N_CLUSTERS_MEAN_DIFF_VS_SCANPY_IGRAPH", "pbmc10k": "LEIDEN_N_CLUSTERS_MEAN_DIFF_VS_SCANPY_IGRAPH"},
    "umap.trustworthiness_gap": {"pbmc3k": "UMAP_TRUSTWORTHINESS_GAP", "pbmc10k": "UMAP_TRUSTWORTHINESS_GAP"},
    "umap.trustworthiness_shortfall": {"pbmc3k": "UMAP_TRUSTWORTHINESS_SHORTFALL", "pbmc10k": "UMAP_TRUSTWORTHINESS_SHORTFALL"},
    "rank_genes_groups.ttest.identical.scores.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_SCORES_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_SCORES_MAX_ABS_PBMC10K"},
    "rank_genes_groups.ttest.identical.logfoldchanges.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_LOGFOLDCHANGES_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_LOGFOLDCHANGES_MAX_ABS"},
    "rank_genes_groups.ttest.identical.pvals.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_PVALS_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_PVALS_MAX_ABS"},
    "rank_genes_groups.ttest.identical.pvals_adj.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_PVALS_ADJ_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_PVALS_ADJ_MAX_ABS"},
    "rank_genes_groups.ttest.identical.top50_overlap_shortfall": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_TOP50_OVERLAP_SHORTFALL", "pbmc10k": "RANK_GENES_GROUPS_TTEST_IDENTICAL_TOP50_OVERLAP_SHORTFALL"},
    "rank_genes_groups.wilcoxon.identical.scores.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_SCORES_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_SCORES_MAX_ABS_PBMC10K"},
    "rank_genes_groups.wilcoxon.identical.logfoldchanges.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_LOGFOLDCHANGES_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_LOGFOLDCHANGES_MAX_ABS"},
    "rank_genes_groups.wilcoxon.identical.pvals.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_PVALS_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_PVALS_MAX_ABS"},
    "rank_genes_groups.wilcoxon.identical.pvals_adj.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_PVALS_ADJ_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_PVALS_ADJ_MAX_ABS"},
    "rank_genes_groups.wilcoxon.identical.top50_overlap_shortfall": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_TOP50_OVERLAP_SHORTFALL", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_IDENTICAL_TOP50_OVERLAP_SHORTFALL"},
    "rank_genes_groups.ttest.lazy.scores.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_LAZY_SCORES_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_TTEST_LAZY_SCORES_MAX_ABS_PBMC10K"},
    "rank_genes_groups.ttest.lazy.logfoldchanges.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_LAZY_LOGFOLDCHANGES_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_TTEST_LAZY_LOGFOLDCHANGES_MAX_ABS"},
    "rank_genes_groups.ttest.lazy.pvals.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_LAZY_PVALS_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_TTEST_LAZY_PVALS_MAX_ABS"},
    "rank_genes_groups.ttest.lazy.pvals_adj.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_LAZY_PVALS_ADJ_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_TTEST_LAZY_PVALS_ADJ_MAX_ABS"},
    "rank_genes_groups.ttest.lazy.top50_overlap_shortfall": {"pbmc3k": "RANK_GENES_GROUPS_TTEST_LAZY_TOP50_OVERLAP_SHORTFALL", "pbmc10k": "RANK_GENES_GROUPS_TTEST_LAZY_TOP50_OVERLAP_SHORTFALL"},
    "rank_genes_groups.wilcoxon.lazy.scores.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_LAZY_SCORES_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_LAZY_SCORES_MAX_ABS_PBMC10K"},
    "rank_genes_groups.wilcoxon.lazy.logfoldchanges.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_LAZY_LOGFOLDCHANGES_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_LAZY_LOGFOLDCHANGES_MAX_ABS"},
    "rank_genes_groups.wilcoxon.lazy.pvals.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_LAZY_PVALS_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_LAZY_PVALS_MAX_ABS"},
    "rank_genes_groups.wilcoxon.lazy.pvals_adj.max_abs": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_LAZY_PVALS_ADJ_MAX_ABS", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_LAZY_PVALS_ADJ_MAX_ABS"},
    "rank_genes_groups.wilcoxon.lazy.top50_overlap_shortfall": {"pbmc3k": "RANK_GENES_GROUPS_WILCOXON_LAZY_TOP50_OVERLAP_SHORTFALL", "pbmc10k": "RANK_GENES_GROUPS_WILCOXON_LAZY_TOP50_OVERLAP_SHORTFALL"},
    "score_genes.b_markers.max_abs": {"pbmc3k": "SCORE_GENES_B_MARKERS_MAX_ABS", "pbmc10k": "SCORE_GENES_B_MARKERS_MAX_ABS"},
    "score_genes.random_25.max_abs": {"pbmc3k": "SCORE_GENES_RANDOM_25_MAX_ABS", "pbmc10k": "SCORE_GENES_RANDOM_25_MAX_ABS"},
    "scrublet.mean_score_ratio_dev": {"pbmc3k": "SCRUBLET_MEAN_SCORE_RATIO_DEV", "pbmc10k": "SCRUBLET_MEAN_SCORE_RATIO_DEV"},
    "scrublet.spearman_shortfall_worst_seed": {"pbmc3k": "SCRUBLET_SPEARMAN_SHORTFALL_WORST_SEED", "pbmc10k": "SCRUBLET_SPEARMAN_SHORTFALL_WORST_SEED_PBMC10K"},
    "scrublet.threshold_abs_diff_max": {"pbmc3k": "SCRUBLET_THRESHOLD_ABS_DIFF_MAX", "pbmc10k": "SCRUBLET_THRESHOLD_ABS_DIFF_MAX"},
    "scrublet.predicted_rate_abs_diff_max": {"pbmc3k": "SCRUBLET_PREDICTED_RATE_ABS_DIFF_MAX", "pbmc10k": "SCRUBLET_PREDICTED_RATE_ABS_DIFF_MAX"},
    "scrublet.auroc_spiked_abs_diff_max": {"pbmc3k": "SCRUBLET_AUROC_SPIKED_ABS_DIFF_MAX", "pbmc10k": "SCRUBLET_AUROC_SPIKED_ABS_DIFF_MAX"},
    "ingest.matching_centre.pca_top10.max_rel_std": {"pbmc3k": "INGEST_MATCHING_CENTRE_PCA_TOP10_MAX_REL_STD", "pbmc10k": "INGEST_MATCHING_CENTRE_PCA_TOP10_MAX_REL_STD"},
    "ingest.matching_centre.label_disagreement": {"pbmc3k": "INGEST_MATCHING_CENTRE_LABEL_DISAGREEMENT", "pbmc10k": "INGEST_MATCHING_CENTRE_LABEL_DISAGREEMENT"},
    "ingest.matching_centre.label_accuracy_abs_diff": {"pbmc3k": "INGEST_MATCHING_CENTRE_LABEL_ACCURACY_ABS_DIFF", "pbmc10k": "INGEST_MATCHING_CENTRE_LABEL_ACCURACY_ABS_DIFF"},
}

# comparison id -> (value, where) of the tolerance an existing test applies to the same comparison
EXISTING_TESTS = {
    "normalize_total.target_1e4.max_rel": (2e-06, "tests/test_scanpy_defaults.py NORMALIZE_RTOL (explicit target_sum case)"),
    "normalize_total.default.max_rel": (2e-06, "tests/test_scanpy_defaults.py NORMALIZE_RTOL"),
    "hvg.seurat_ntop.dispersions_norm.max_abs": (0.0001, "tests/test_crest.py test_hvg_matches_scanpy atol"),
    "hvg.cell_ranger_ntop.dispersions_norm.max_abs": (0.0001, "tests/test_crest.py test_hvg_matches_scanpy atol"),
    "hvg.seurat_tutorial.dispersions_norm.max_abs": (0.0001, "tests/test_crest.py test_hvg_matches_scanpy atol"),
    "hvg.seurat_v3.variances_norm.max_rel": (1e-10, "tests/test_crest.py test_hvg_seurat_v3_matches_scanpy rtol"),
    "pca.scale_max10.angle_top10_deg": (0.01, "tests/test_scanpy_defaults.py PCA_ANGLE_DEG"),
    "pca.scale_noclip.angle_top10_deg": (0.01, "tests/test_scanpy_defaults.py PCA_ANGLE_DEG"),
    "knn.crest_exact.recall_shortfall": (0.001, "tests/test_knn_precision.py RECALL_VS_TRUTH (0.999)"),
    "connectivities.same_knn.max_abs": (1e-06, "tests/test_graph_pins.py CONNECTIVITY_MAX_ABS_DIFF"),
    "leiden.modularity_shortfall_vs_leidenalg": (0.003, "tests/test_graph_pins.py LEIDEN_Q_MARGIN"),
    "leiden.n_clusters_mean_diff_vs_leidenalg": (1.0, "tests/test_graph_pins.py LEIDEN_MAX_MEAN_CLUSTER_DIFF"),
    "umap.trustworthiness_gap": (0.005, "tests/test_graph_pins.py TRUSTWORTHINESS_MAX_GAP"),
    "rank_genes_groups.ttest.identical.scores.max_abs": (0.0001, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.ttest.identical.logfoldchanges.max_abs": (0.0001, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.ttest.identical.pvals.max_abs": (1e-05, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.ttest.identical.pvals_adj.max_abs": (1e-05, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.wilcoxon.identical.scores.max_abs": (1e-06, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.wilcoxon.identical.logfoldchanges.max_abs": (0.0001, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.wilcoxon.identical.pvals.max_abs": (1e-05, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.wilcoxon.identical.pvals_adj.max_abs": (1e-05, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.ttest.lazy.scores.max_abs": (0.0001, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.ttest.lazy.logfoldchanges.max_abs": (0.0001, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.ttest.lazy.pvals.max_abs": (1e-05, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.ttest.lazy.pvals_adj.max_abs": (1e-05, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.wilcoxon.lazy.scores.max_abs": (0.0005, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.wilcoxon.lazy.logfoldchanges.max_abs": (0.0001, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.wilcoxon.lazy.pvals.max_abs": (0.001, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "rank_genes_groups.wilcoxon.lazy.pvals_adj.max_abs": (0.001, "tests/test_rank_genes_rest.py TOL / tests/test_wilcoxon_signed.py (atol part)"),
    "score_genes.b_markers.max_abs": (1e-05, "tests/test_crest.py test_score_genes_matches_scanpy atol"),
    "score_genes.random_25.max_abs": (1e-05, "tests/test_crest.py test_score_genes_matches_scanpy atol"),
    "scrublet.mean_score_ratio_dev": (0.02, "tests/test_scrublet_ingest_leftovers.py MEAN_SCORE_RTOL"),
    "scrublet.spearman_shortfall_worst_seed": (0.06, "tests/test_scrublet_ingest_leftovers.py SPEARMAN_FLOOR (0.94)"),
}
