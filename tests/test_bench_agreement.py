"""Tests of the agreement-run script (bench/agreement/run_agreement.py): its metric functions on hand-built
inputs with known answers, the consistency of its constants with THRESHOLDS.md, and the three stages end to end
on a small synthetic dataset. No benchmark data is needed."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("igraph")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "bench" / "agreement"), str(REPO / "bench" / "paper")]


class _Script:
    """``run_agreement`` imported on first use: the module does not exist in the source tree of an older reference
    build, which these tests are collected against (their tests are deselected there)."""

    def __getattr__(self, name):
        import run_agreement
        return getattr(run_agreement, name)


ra = _Script()


def _thresholds_text() -> str:
    return (REPO / "bench" / "agreement" / "THRESHOLDS.md").read_text()

# --------------------------------------------------------------------------- tolerances
EXACT = 1e-12                # closed-form values
TRUST_AGREE = 1e-9           # trustworthiness against scikit-learn when every cell is a query
TRUST_ESTIMATE = 0.01        # a 40 % sample of the queries against the full value
N_POINTS = 120               # points in the random-embedding tests
N_SMALL_CELLS_PER_TYPE = 1200
N_SMALL_GENES = 2500
SMALL_SEEDS = 3


def _two_triangles():
    import igraph as ig
    g = ig.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
    g.es["weight"] = [1.0] * 7
    return g


# --------------------------------------------------------------------------- partitions and seed spreads
def test_pair_aris_counts_and_identity():
    runs = [np.array([0, 0, 1, 1, 2, 2])] * 4
    assert len(ra.pair_aris(runs)) == 6
    assert np.allclose(ra.pair_aris(runs), 1.0, atol=EXACT)
    other = [np.array([0, 1, 0, 1, 0, 1])] * 3
    cross = ra.pair_aris(runs, other)
    assert cross.shape == (12,)
    assert (cross < 1.0).all()


@pytest.mark.parametrize("within,expected", [
    ([0.80, 0.92, 0.95], 0.80),          # the lowest pair is below mean - 0.02: it is the bound
    ([0.9000, 0.9005, 0.9010], 0.8805),  # nearly identical seeds: the floor applies (mean - 0.02)
])
def test_seed_floor_takes_the_lower_of_minimum_and_mean_minus_floor(within, expected):
    assert abs(ra.seed_floor(np.array(within)) - expected) < EXACT


def test_cross_ari_passes_at_the_bound_and_fails_below_it():
    within = np.array([0.90, 0.92, 0.94])
    assert ra.eval_cross_ari(np.array([0.90]), within)["pass"]
    assert not ra.eval_cross_ari(np.array([0.899]), within)["pass"]


def test_annotation_rule_uses_the_scanpy_range_with_a_floor():
    wide = ra.eval_annotation(0.80, [0.78, 0.82, 0.80])      # range 0.04: |diff| = 0 is inside
    assert wide["pass"] and abs(wide["threshold"] - 0.04) < EXACT
    tight = ra.eval_annotation(0.805, [0.800, 0.8001, 0.7999])  # range 2e-4: the 0.01 floor applies
    assert tight["pass"] and abs(tight["threshold"] - ra.ARI_FLOOR_ANNOT) < EXACT
    assert not ra.eval_annotation(0.82, [0.800, 0.8001, 0.7999])["pass"]


# --------------------------------------------------------------------------- modularity
def test_modularity_of_two_triangles_is_the_closed_form_value():
    g = _two_triangles()
    q = ra.modularity(g, [0, 0, 0, 1, 1, 1])
    assert abs(q - 2 * (3 / 7 - 0.25)) < EXACT
    assert abs(ra.modularity(g, [0] * 6)) < EXACT


def test_modularity_graph_from_connectivities_keeps_each_edge_once():
    import scipy.sparse as sp
    C = sp.csr_matrix(np.array([[0, 2.0, 0], [2.0, 0, 1.0], [0, 1.0, 0]]))
    g = ra.igraph_from_connectivities(C)
    assert g.ecount() == 2 and sorted(g.es["weight"]) == [1.0, 2.0]


def test_modularity_rule_tolerance_is_a_bound_on_the_gap():
    assert ra.eval_modularity([0.6600], [0.6625])["pass"]       # gap -0.0025
    assert not ra.eval_modularity([0.6600], [0.6635])["pass"]   # gap -0.0035
    r = ra.eval_modularity([0.66, 0.67], [0.65, 0.65])
    assert r["pass"] and abs(r["gap"] - 0.015) < EXACT and r["n_seeds"] == 2


# --------------------------------------------------------------------------- cluster pairing and DE agreement
def test_match_clusters_pairs_permuted_labels_and_lists_unpaired_ones():
    a = np.repeat([0, 1, 2, 3], [40, 30, 20, 10])
    b = a.copy()
    b[a == 3] = 2                  # clusters 2 and 3 of `a` merge into label 2 of `b`
    b = np.where(b == 0, 7, b)     # a relabelled cluster
    pairs, jac, un_a, un_b = ra.match_clusters(a, b)
    # cluster 2 (20 cells) is paired with the merged cluster (30 cells, Jaccard 2/3); cluster 3 is left over
    assert sorted((int(x), int(y)) for x, y in pairs) == [(0, 7), (1, 1), (2, 2)]
    assert all(j >= ra.DE_MIN_JACCARD_PAIR for j in jac)
    assert [int(x) for x in un_a] == [3] and not un_b
    pairs, _, un_a, _ = ra.match_clusters(a, b, min_jaccard=0.7)
    assert [int(x) for x in un_a] == [2, 3] and len(pairs) == 2


def test_match_clusters_identical_partitions_pair_everything():
    a = np.repeat([0, 1, 2], 10)
    pairs, jac, un_a, un_b = ra.match_clusters(a, (a + 5))
    assert len(pairs) == 3 and np.allclose(jac, 1.0) and not un_a and not un_b


def test_de_group_metrics_identical_and_reversed_scores():
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    lfc = s / 3
    p = np.where(s > 1.5, 1e-6, 0.5)
    same = ra.de_group_metrics(s, lfc, p, s, lfc, p)
    assert abs(same["spearman_scores"] - 1) < EXACT and abs(same["top50_jaccard"] - 1) < EXACT
    assert abs(same["sig_jaccard"] - 1) < EXACT
    rev = ra.de_group_metrics(s, lfc, p, -s, -lfc, p)
    assert abs(rev["spearman_scores"] + 1) < EXACT and rev["top50_jaccard"] == 0.0


def test_de_top50_jaccard_counts_the_overlap():
    s1 = np.arange(200, dtype=float)
    s2 = s1.copy()
    s2[150:175] = -1.0             # 25 of the 50 top genes (150..199) drop out
    s2[:25] = 1000.0               # and 25 other genes enter
    r = ra.de_group_metrics(s1, s1, np.ones(200), s2, s2, np.ones(200))
    assert abs(r["top50_jaccard"] - 25 / 75) < EXACT


def test_de_significant_set_needs_padj_log_fold_change_and_positive_score():
    s = np.array([5.0, 5.0, 5.0, -5.0, 5.0])
    lfc = np.array([1.0, 1.0, 0.2, 1.0, 1.0])
    padj = np.array([0.001, 0.2, 0.001, 0.001, 0.001])
    r = ra.de_group_metrics(s, lfc, padj, s, lfc, padj)
    assert r["n_sig_a"] == 2        # genes 0 and 4 only


def test_eval_de_is_inconclusive_below_three_groups_and_uses_the_median():
    row = {"spearman_scores": 0.95, "spearman_lfc": 0.95, "top50_jaccard": 0.6, "sig_jaccard": 0.8}
    assert ra.eval_de([row, row], ra.DE_OWN_MIN)["pass"] is None
    bad = {**row, "top50_jaccard": 0.1}
    assert ra.eval_de([row, row, bad], ra.DE_OWN_MIN)["pass"] is True     # the median is 0.6
    assert ra.eval_de([row, bad, bad], ra.DE_OWN_MIN)["pass"] is False
    nan_row = {**row, "sig_jaccard": float("nan")}
    assert ra.eval_de([row, row, nan_row], ra.DE_OWN_MIN)["median"]["sig_jaccard"] == 0.8


# --------------------------------------------------------------------------- kNN recall
def _exact_knn(X, k):
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def test_recall_of_an_exact_graph_is_one_and_a_wrong_neighbour_costs_one_over_k():
    X = np.random.default_rng(1).normal(size=(N_POINTS, 5))
    k = 6
    nb = _exact_knn(X, k)
    q = np.arange(N_POINTS)
    assert np.allclose(ra.recall_per_query(X, X, nb, k, self_idx=q), 1.0)
    bad = nb.copy()
    far = np.argsort(((X - X[0]) ** 2).sum(1))[-1]
    bad[0, 0] = far
    rec = ra.recall_per_query(X, X, bad, k, self_idx=q)
    assert abs(rec[0] - (k - 1) / k) < EXACT and np.allclose(rec[1:], 1.0)


def test_recall_does_not_count_a_tied_boundary_neighbour_as_a_miss():
    X = np.array([[0.0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [10, 10]])
    # point 0 has four neighbours at distance 1; any 3 of them are an exact 3-NN set
    nb = np.array([[1, 2, 3]] + [[0, 1, 2]] * 5)
    assert ra.recall_per_query(X, X[:1], nb[:1], 3, self_idx=np.array([0]))[0] == 1.0
    nb2 = np.array([[1, 2, 4]])
    assert ra.recall_per_query(X, X[:1], nb2, 3, self_idx=np.array([0]))[0] == 1.0
    nb3 = np.array([[1, 2, 5]])
    assert abs(ra.recall_per_query(X, X[:1], nb3, 3, self_idx=np.array([0]))[0] - 2 / 3) < EXACT


def test_recall_raises_when_a_list_holds_its_own_query_and_ignores_padding():
    X = np.random.default_rng(2).normal(size=(20, 3))
    nb = _exact_knn(X, 4)
    with pytest.raises(ValueError, match="itself"):
        bad = nb.copy()
        bad[3, 0] = 3
        ra.recall_per_query(X, X, bad, 4, self_idx=np.arange(20))
    padded = nb.copy()
    padded[:, 3] = -1
    assert np.allclose(ra.recall_per_query(X, X, padded, 4, self_idx=np.arange(20)), 0.75)


def test_recall_of_queries_outside_the_reference_set():
    rng = np.random.default_rng(3)
    ref, q = rng.normal(size=(80, 4)), rng.normal(size=(15, 4)) + 3.0
    D = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(D, axis=1)[:, :5]
    assert np.allclose(ra.recall_per_query(ref, q, nb, 5), 1.0)


def test_recall_summary_fractions_and_acceptance():
    rec = np.array([1.0] * 197 + [0.7, 0.4, 0.4])
    s = ra.recall_summary(rec)
    assert abs(s["frac_below_0.8"] - 3 / 200) < EXACT and abs(s["frac_below_0.5"] - 2 / 200) < EXACT
    good = {"mean": 0.995, "frac_below_0.8": 0.005, "frac_below_0.5": 0.004, "n_query": 10000}
    assert ra.eval_recall(good)["pass"]
    assert not ra.eval_recall({**good, "mean": 0.985})["pass"]
    assert not ra.eval_recall({**good, "frac_below_0.5": 0.006})["pass"]
    assert not ra.eval_recall({**good, "frac_below_0.8": 0.011})["pass"]
    assert not ra.eval_recall({**good, "n_query": 4000})["pass"]


def test_neighbour_set_jaccard():
    A = np.array([[1, 2, 3], [0, 2, 3]])
    B = np.array([[1, 2, 4], [0, 2, 3]])
    assert abs(ra.jaccard_neighbor_rows(A, B) - (0.5 + 1.0) / 2) < EXACT


# --------------------------------------------------------------------------- UMAP
def test_trustworthiness_matches_scikit_learn_when_every_cell_is_a_query():
    from sklearn.manifold import trustworthiness
    rng = np.random.default_rng(4)
    Xh = rng.normal(size=(N_POINTS, 8))
    Xl = Xh[:, :2] + 0.3 * rng.normal(size=(N_POINTS, 2))
    mine = ra.trustworthiness_sampled(Xh, Xl, np.arange(N_POINTS), k=10)
    ref = trustworthiness(Xh, Xl, n_neighbors=10)
    assert abs(mine - ref) < TRUST_AGREE
    assert abs(ra.trustworthiness_sampled(Xh, Xh, np.arange(N_POINTS), k=10) - 1.0) < EXACT


def test_trustworthiness_of_a_sample_of_queries_estimates_the_full_value():
    rng = np.random.default_rng(5)
    Xh = rng.normal(size=(300, 8))
    Xl = Xh[:, :2] + 0.3 * rng.normal(size=(300, 2))
    full = ra.trustworthiness_sampled(Xh, Xl, np.arange(300), k=10)
    part = ra.trustworthiness_sampled(Xh, Xl, np.sort(rng.choice(300, 120, replace=False)), k=10)
    assert abs(full - part) < TRUST_ESTIMATE


def test_centroid_distance_spearman_is_one_for_a_similarity_transform_and_lower_when_scrambled():
    rng = np.random.default_rng(6)
    lab = np.repeat(np.arange(6), 150)
    centers = rng.normal(size=(6, 4)) * 5
    Xh = centers[lab] + 0.1 * rng.normal(size=(900, 4))
    assert abs(ra.centroid_distance_spearman(Xh, 3.0 * Xh, lab)["rho"] - 1.0) < 1e-9
    scrambled = Xh[:, :2][rng.permutation(len(Xh))]
    assert ra.centroid_distance_spearman(Xh, scrambled, lab)["rho"] < 0.9
    small = ra.centroid_distance_spearman(Xh, Xh, np.where(np.arange(900) < 450, 0, 1))
    assert np.isnan(small["rho"]) and small["n_clusters"] == 2


def test_threshold_difference_rule():
    assert ra.eval_threshold_diff(0.955, 0.96, 0.01)["pass"] is True
    assert ra.eval_threshold_diff(0.9, 0.96, 0.01)["pass"] is False
    assert ra.eval_threshold_diff(0.97, 0.96, 0.01)["pass"] is True


# --------------------------------------------------------------------------- consistency with THRESHOLDS.md
def test_thresholds_document_states_the_constants_of_the_script():
    t = " ".join(_thresholds_text().split())   # line breaks in the document do not matter
    assert f"{ra.N_SEEDS} " in t or "ten per tool" in t
    for text in (f"{ra.MOD_TOL}", f"{ra.RECALL_MEAN_MIN}", "0.5 percent", "1.0 percent", f"{ra.TRUST_TOL}", f"{ra.GLOBAL_TOL}",
                 f"{ra.ARI_FLOOR_ANNOT}", f"{ra.ARI_FLOOR_SEEDS}", str(f"{ra.DE_MIN_JACCARD_PAIR}")):
        assert text in t, text
    for key, label in (("spearman_scores", "score correlation"), ("spearman_lfc", "fold-change correlation"),
                       ("top50_jaccard", "top-50 Jaccard"), ("sig_jaccard", "significant-set Jaccard")):
        for table in (ra.DE_OWN_MIN, ra.DE_SAME_MIN):
            assert f"{table[key]:.2f} ({label})" in t, (key, table[key])
    assert f"{ra.RECALL_QUERIES:,}" in t and f"{ra.TRUST_QUERIES:,}" in t
    assert f"Seeds `0..{ra.MOD_SEEDS_SMALL - 1}`" in t and f"`0..{ra.MOD_SEEDS_LARGE - 1}`" in t
    assert ra.K == 14 and "14 neighbours" in t


FORBIDDEN_PUBLIC = re.compile(r"\b(?:F|PY|WP|G|D|B|SB|BL)-\d+[ab]?\b|\bB\d{1,2}[ab]?\b")   # internal item identifiers (shape only)


def test_public_files_of_the_agreement_run_carry_no_item_identifiers():
    files = [REPO / "bench" / "agreement" / "THRESHOLDS.md", REPO / "bench" / "agreement" / "run_agreement.py",
             Path(__file__)]
    hits = [f"{p.name}: {m.group(0)}" for p in files for m in FORBIDDEN_PUBLIC.finditer(p.read_text())]
    assert not hits, hits


# --------------------------------------------------------------------------- the stages end to end
def _write_small_dataset(dirpath: Path, name: str = "tiny"):
    """A 3-type, 2-batch synthetic count matrix as a 10x v3 .h5 with an observation sidecar and metadata."""
    import h5py
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    n = 3 * N_SMALL_CELLS_PER_TYPE
    ctype = np.repeat(np.arange(3), N_SMALL_CELLS_PER_TYPE)
    base = rng.gamma(0.7, 0.5, size=N_SMALL_GENES)
    prof = np.tile(base, (3, 1))
    for t in range(3):
        sel = rng.choice(N_SMALL_GENES, 150, replace=False)
        prof[t, sel] *= 8.0
    counts = rng.poisson(prof[ctype] * rng.uniform(0.8, 1.2, size=(n, 1))).astype(np.float32)
    X = sp.csc_matrix(counts.T)  # genes x cells
    barcodes = np.array([f"CELL{i:05d}-1" for i in range(n)])
    with h5py.File(dirpath / f"{name}.h5", "w") as f:
        g = f.create_group("matrix")
        g["barcodes"] = barcodes.astype("S")
        g["data"], g["indices"], g["indptr"] = X.data.astype(np.int32), X.indices.astype(np.int64), X.indptr.astype(np.int64)
        g["shape"] = np.array(X.shape, dtype=np.int32)
        ft = g.create_group("features")
        ids = np.array([f"G{i:05d}" for i in range(N_SMALL_GENES)])
        ft["id"], ft["name"] = ids.astype("S"), np.array([f"gene{i}" for i in range(N_SMALL_GENES)]).astype("S")
        ft["feature_type"] = np.array(["Gene Expression"] * N_SMALL_GENES).astype("S")
        ft["genome"] = np.array(["x"] * N_SMALL_GENES).astype("S")
    pl.DataFrame({"barcode": barcodes, "ctype": [f"type{t}" for t in ctype],
                  "batch": np.where(np.arange(n) % 2 == 0, "b1", "b2")}).write_parquet(dirpath / f"{name}.obs.parquet")
    (dirpath / f"{name}.meta.json").write_text(json.dumps({"celltype": "ctype", "batch": "batch"}))
    return n


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    pytest.importorskip("crest")
    import crest  # noqa: F401
    d = tmp_path_factory.mktemp("agreement")
    n = _write_small_dataset(d)
    import run_agreement as script
    mp = pytest.MonkeyPatch()
    mp.setattr(script, "N_SEEDS", SMALL_SEEDS)
    mp.setattr(script, "MOD_SEEDS_SMALL", SMALL_SEEDS)
    mp.setattr(script, "UMAP_SEEDS_SMALL", 1)
    mp.setattr(script, "DE_REF_SEEDS_SMALL", 1)
    mp.setattr(script, "_threads", lambda n: None)   # leave the thread variables of the test process alone
    out = d / "out"
    for stage in ("scanpy", "crest", "exact", "compare"):
        ra.main([stage, "--dataset", "tiny", "--data-dir", str(d), "--out", str(out), "--threads", "2"])
    yield n, out / "tiny"
    mp.undo()


def test_leidenalg_partitions_are_the_same_with_forked_workers():
    pytest.importorskip("leidenalg")
    import scipy.sparse as sp
    rng = np.random.default_rng(8)
    n = 300
    block = np.repeat(np.arange(3), n // 3)
    P = np.where(block[:, None] == block[None, :], 0.12, 0.01)
    A = np.triu((rng.random((n, n)) < P).astype(float), 1)
    g = ra.igraph_from_connectivities(sp.csr_matrix(A + A.T))
    one = ra.leidenalg_partitions(g, range(4), 2, workers=1)
    many = ra.leidenalg_partitions(g, range(4), 2, workers=2)
    assert one.shape == (4, n) and np.array_equal(one, many)
    assert ra.modularity(g, one[0]) > 0.3


def test_in_place_pca_gives_the_same_numbers_as_scanpy_default():
    sc = pytest.importorskip("scanpy")
    import anndata as ad
    import sklearn.decomposition as dec
    rng = np.random.default_rng(7)
    X = rng.gamma(2.0, 1.0, size=(400, 60)).astype(np.float32)
    runs = []
    for in_place in (False, True):
        a = ad.AnnData(X.copy())
        sc.pp.scale(a, max_value=10)
        with ra._pca_in_place(in_place):
            sc.tl.pca(a, n_comps=10, svd_solver="arpack")
        runs.append(np.asarray(a.obsm["X_pca"]))
    assert np.array_equal(runs[0], runs[1])
    assert dec.PCA.__name__ == "PCA"     # the patch is undone


def test_end_to_end_writes_every_registered_metric(small_run):
    n, d = small_run
    res = json.loads((d / "results.json").read_text())
    M = res["metrics"]
    for key in ("1_annotation_ari", "1_annotation_nmi", "2a_whole_workflow", "2b_clustering_step", "2c_knn_step",
                "3_modularity_common_graph", "3_modularity_crest_graph", "4_de", "5_knn_recall", "6_umap"):
        assert key in M, key
    assert {"5.1_crest_own_pca", "5.2_scanpy_own_pca", "5.3_crest_on_scanpy_pca", "5.4_crest_query_other_batches"} <= set(M["5_knn_recall"])
    assert "4a_own_clusters" in M["4_de"] and "4b_same_labels" in M["4_de"]
    assert res["n_cells"]["crest"] == res["n_cells"]["scanpy"] == res["n_common_cells"]
    ex = res["diagnostics"]["post_hoc_exact_graph_arm"]
    assert "post-hoc" in ex["note"] and 0.0 <= ex["mean_ari_E_vs_A"] <= 1.0 and ex["knn_jaccard_vs_exact_graph"]["scanpy"] > 0.99
    assert (d / "results.md").read_text().count("|") > 30


def test_end_to_end_agrees_on_well_separated_clusters(small_run):
    _, d = small_run
    M = json.loads((d / "results.json").read_text())["metrics"]
    assert M["1_annotation_ari"]["mean_crest"] > 0.9 and M["1_annotation_ari"]["mean_scanpy"] > 0.9
    assert M["2a_whole_workflow"]["pass"] and M["2b_clustering_step"]["pass"] and M["2c_knn_step"]["pass"]
    assert M["3_modularity_common_graph"]["pass"]
    for key in ("5.1_crest_own_pca", "5.2_scanpy_own_pca"):
        assert M["5_knn_recall"][key]["mean"] > 0.999      # exact search on both sides below 20,000 cells
    de = M["4_de"]["4b_same_labels"]
    assert de["pass"] is True and de["median"]["spearman_scores"] > 0.999


def test_end_to_end_stages_resume_without_recomputing(small_run):
    _, d = small_run
    before = {f.name: f.stat().st_mtime_ns for f in d.glob("*.np*")}
    out = d.parent
    ra.main(["crest", "--dataset", "tiny", "--data-dir", str(out.parent), "--out", str(out), "--threads", "2"])
    after = {f.name: f.stat().st_mtime_ns for f in d.glob("*.np*")}
    assert before == after


def test_crest_stage_run_before_the_scanpy_stage_is_completed_by_a_second_run(tmp_path, monkeypatch):
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    pytest.importorskip("crest")
    import run_agreement as script
    monkeypatch.setattr(script, "_threads", lambda n: None)
    _write_small_dataset(tmp_path)
    out = tmp_path / "out"
    common = ["--dataset", "tiny", "--data-dir", str(tmp_path), "--out", str(out), "--threads", "2", "--mode", "knn"]
    script.main(["crest", *common])
    assert "knn0_on_scanpy_pca" not in np.load(out / "tiny" / "crest_core.npz").files
    script.main(["scanpy", *common])
    script.main(["crest", *common])
    z = np.load(out / "tiny" / "crest_core.npz")
    assert z["knn0_on_scanpy_pca"].shape == z["knn0"].shape
