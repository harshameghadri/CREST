"""Tools (scanpy ``sc.tl`` equivalents) on :class:`~crest.core.BioFrame`."""

from __future__ import annotations

import re
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import polars as pl

from .core import BioFrame
from . import crest as _native
from .pp import gene_stats, _mean_var
from .deseq2 import DESeq2, Pseudobulk, deseq2, pseudobulk, pseudobulk_de  # noqa: F401
from .harmony import harmony  # noqa: F401
from .ingest import _unique_positions, ingest  # noqa: F401

__all__ = ["pca", "leiden", "umap", "rank_genes_groups", "score_genes",
           "pseudobulk", "pseudobulk_de", "deseq2", "DESeq2", "Pseudobulk", "harmony",
           "leiden_sweep", "adjusted_rand_index", "ingest"]


# --------------------------------------------------------------------------- PCA
def _scale_params(bf: BioFrame, mask: np.ndarray):
    """Per-gene log-scale mean/std (ddof=1) of the current cells and ops.

    Reuses the var ``mean_log``/``std_log`` cached by ``highly_variable_genes`` only
    while ``uns["hvg"]["ops"]`` equals ``bf.ops``; a cell filter drops that key (the
    statistics belong to the old cells), so the values are recomputed in one pass.
    """
    cached = bf.uns.get("hvg", {}).get("ops") == bf.ops and "mean_log" in bf.var.columns
    if cached:
        mean, std = bf.var["mean_log"].to_numpy(), bf.var["std_log"].to_numpy()
    else:
        st = gene_stats(bf)
        mean, var = _mean_var(st[:, 0], st[:, 1], bf.n_obs)
        std = np.sqrt(var)
        std[std == 0] = 1.0
    return np.ascontiguousarray(mean[mask], np.float64), np.ascontiguousarray(std[mask], np.float64)


def pca(bf: BioFrame, n_comps: int = 50, use_highly_variable: Optional[bool] = None,
        scale: Optional[bool] = None, max_value: Optional[float] = None) -> BioFrame:
    """Exact PCA of the (optionally scaled) matrix via its covariance, streamed.

    One pass accumulates the gene × gene Gram matrix from sparse rows, a
    symmetric eigendecomposition gives the loadings, and a second pass projects
    the cells. With scaling (``crest.pp.scale`` or ``scale=True``) the result
    equals ``sc.pp.scale(max_value) + sc.tl.pca`` on the dense matrix, but the
    dense matrix is never formed; ``scale=True`` without ``crest.pp.scale`` does not clip
    (``max_value=None``, as ``sc.pp.scale``). Memory: one chunk + a genes² matrix.
    If ``n_comps`` is at least the number of genes (or cells), ``min(n_comps, genes, cells - 1)``
    components are returned with a ``UserWarning`` (scanpy raises an error).

    Adds ``obsm['X_pca']``, ``varm['PCs']``, ``uns['pca']`` (variance, variance_ratio).
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in bf.var.columns
    mask = bf.var["highly_variable"].to_numpy().astype(bool) if use_highly_variable else np.ones(bf.n_vars, bool)
    d = int(mask.sum())
    if d < 2:
        raise ValueError("need at least 2 genes for PCA")
    if d > 20_000:
        raise ValueError(f"{d} genes: select highly variable genes first (Gram PCA is O(genes^2) memory)")
    if scale is None:
        scale = "scale" in bf.uns
    if max_value is None:
        max_value = bf.uns.get("scale", {}).get("max_value", 0.0) if scale else 0.0
    gmap = np.full(bf.n_vars, -1, np.int32)
    gmap[mask] = np.arange(d, dtype=np.int32)
    mean = std = None
    if scale:
        mean, std = _scale_params(bf, mask)

    n = bf.n_obs
    gram = np.zeros((d, d), np.float64)
    colsum = np.zeros(d, np.float64)
    for ctx in bf.iter_ctx():
        _native.gram_accumulate(*ctx, gmap, gram, colsum, scale, mean, std, float(max_value or 0.0))
    k = min(n_comps, d, n - 1)
    if k < n_comps:
        warnings.warn(f"n_comps={n_comps} exceeds the {d} genes / {n} cells available; returning {k} components "
                      "(sc.tl.pca raises an error here)", UserWarning, stacklevel=2)
    variance, loadings, total = _native.pca_from_gram(gram, colsum, n, k)
    del gram

    shift = (colsum / n) @ loadings
    X = np.empty((n, k), np.float32)
    X[:] = -shift.astype(np.float32)
    for ctx in bf.iter_ctx():
        _native.project(*ctx, gmap, loadings, X, scale, mean, std, float(max_value or 0.0))

    pcs = np.zeros((bf.n_vars, k), np.float32)
    pcs[mask] = loadings
    bf.obsm["X_pca"] = X
    bf.varm["PCs"] = pcs
    bf.uns["pca"] = {"variance": variance, "variance_ratio": variance / total,
                     "params": {"scale": bool(scale), "max_value": max_value, "use_highly_variable": bool(use_highly_variable)},
                     # what crest.tl.ingest needs to project new cells onto these PCs
                     "projection": {"genes": [g for g, m in zip(bf.var_names, mask) if m],
                                    "gene_ids": None if bf.gene_ids is None else [g for g, m in zip(bf.gene_ids, mask) if m],
                                    "loadings": loadings,
                                    "mean": mean, "std": std, "shift": shift, "ops": list(bf.ops)}}
    return bf


# --------------------------------------------------------------------------- clustering / embedding
def _graph(bf: BioFrame):
    if "neighbors" not in bf.uns:
        raise KeyError("run crest.pp.neighbors first")
    nb = bf.uns["neighbors"]
    return nb["rows"], nb["cols"], nb["weights"]


def leiden(bf: BioFrame, resolution: float = 1.0, n_iterations: int = 2, random_state: int = 0,
           key_added: str = "leiden") -> BioFrame:
    """Leiden clustering of the neighbour graph (labels ordered by cluster size, "0" largest).

    Differs from scanpy: the default is ``n_iterations=2``. scanpy 1.11's
    ``sc.tl.leiden`` defaults to ``flavor="leidenalg"`` with ``n_iterations=-1``
    (iterate until no improvement), and warns that its future default is
    ``flavor="igraph", n_iterations=2``. CREST has one implementation (Traag et
    al. 2019) and no ``flavor`` argument.

    Edge weights must be finite and non-negative (``ValueError`` otherwise). Leiden's move
    tolerances are absolute, so the graph should be on the scale of the ``crest.pp.neighbors``
    connectivities (weights up to about 1): a ``RuntimeWarning`` is emitted when the largest
    weight is below 1e-6, and the partition collapses to singletons near 1e-12. Rescale such
    weights, e.g. by dividing them by their maximum.

    ``n_iterations`` is the number of Leiden iterations, as in scanpy; iterating
    stops early once the partition no longer changes, and a negative value
    (scanpy's ``-1``) iterates until that happens (capped at 10,000 iterations,
    with a ``RuntimeWarning`` if the cap is reached).
    """
    rows, cols, w = _graph(bf)
    lab = _native.leiden(bf.n_obs, rows, cols, w, resolution, n_iterations, random_state)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes, kind="stable")
    relabel = np.empty_like(order)
    relabel[order] = np.arange(len(order))
    bf.obs = bf.obs.with_columns(pl.Series(key_added, relabel[lab].astype(str)))
    bf.uns[key_added] = {"params": {"resolution": resolution, "n_iterations": n_iterations, "random_state": random_state}}
    return bf


def _relabel_by_size(lab: np.ndarray) -> np.ndarray:
    sizes = np.bincount(lab)
    order = np.argsort(-sizes, kind="stable")
    relabel = np.empty_like(order)
    relabel[order] = np.arange(len(order))
    return relabel[lab]


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labelings (Hubert & Arabie 1985)."""
    a = np.unique(np.asarray(a), return_inverse=True)[1]
    b = np.unique(np.asarray(b), return_inverse=True)[1]
    n = len(a)
    if n < 2:
        return 1.0
    cont = np.bincount(a * (b.max() + 1) + b, minlength=(a.max() + 1) * (b.max() + 1)).astype(np.float64)
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    sum_ij = comb(cont).sum()
    sum_a = comb(np.bincount(a).astype(np.float64)).sum()
    sum_b = comb(np.bincount(b).astype(np.float64)).sum()
    expected = sum_a * sum_b / comb(float(n))
    max_idx = (sum_a + sum_b) / 2.0
    if max_idx == expected:
        return 1.0
    return float((sum_ij - expected) / (max_idx - expected))


def leiden_sweep(bf: BioFrame, resolutions: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0),
                 n_seeds: int = 1, n_iterations: int = 2, random_state: int = 0,
                 key_prefix: str = "leiden_") -> pl.DataFrame:
    """Leiden at several resolutions (and seeds) on one neighbour graph, all runs in
    parallel. Replaces a loop of :func:`leiden` calls, each of which rebuilds the
    graph and runs on one core.

    Adds obs columns ``{key_prefix}{resolution}`` (labels of the first seed,
    ordered by cluster size like :func:`leiden`) and returns a table with, per
    resolution: ``n_clusters``, ``ari_prev`` (agreement with the next lower
    resolution; clustree-style stability) and, with ``n_seeds > 1``,
    ``ari_seeds`` (mean pairwise ARI across seeds: how reproducible the
    partition is). Also stored in ``uns['leiden_sweep']``. ``n_iterations`` is as
    in :func:`leiden` (negative = until the partition no longer changes).
    """
    rows, cols, w = _graph(bf)
    res = [float(r) for r in resolutions]
    jobs_r = [r for r in res for _ in range(n_seeds)]
    jobs_s = [random_state + s for _ in res for s in range(n_seeds)]
    labels = _native.leiden_sweep(bf.n_obs, rows, cols, w, jobs_r, jobs_s, n_iterations)
    out, prev, new_cols = [], None, []
    for i, r in enumerate(res):
        runs = [labels[i * n_seeds + s].astype(np.int64) for s in range(n_seeds)]
        lab = _relabel_by_size(runs[0])
        new_cols.append(pl.Series(f"{key_prefix}{r:g}", lab.astype(str)))
        row = {"resolution": r, "n_clusters": int(lab.max() + 1),
               "ari_prev": adjusted_rand_index(prev, lab) if prev is not None else None}
        if n_seeds > 1:
            pairs = [adjusted_rand_index(runs[a], runs[b]) for a in range(n_seeds) for b in range(a + 1, n_seeds)]
            row["ari_seeds"] = float(np.mean(pairs))
        out.append(row)
        prev = lab
    bf.obs = bf.obs.with_columns(new_cols)
    table = pl.DataFrame(out)
    bf.uns["leiden_sweep"] = {"table": table, "params": {"n_seeds": n_seeds, "n_iterations": n_iterations,
                                                         "random_state": random_state}}
    return table


def umap(bf: BioFrame, min_dist: float = 0.5, spread: float = 1.0, n_components: int = 2,
         n_epochs: Optional[int] = None, random_state: int = 0,
         spectral_max_matvec: Optional[int] = None) -> BioFrame:
    """UMAP embedding of the neighbour graph (scanpy defaults: min_dist=0.5).

    umap-learn's objective and default parameters, with CREST's own spectral
    initialisation. The initial layout is the set of smallest non-trivial
    eigenvectors of the normalised graph Laplacian, as in umap-learn, computed by a
    restarted Lanczos iteration that runs until the eigenvector residuals are below
    1e-7 (it is deterministic given ``random_state`` and independent of the number
    of threads). A graph with several connected components is laid out component by
    component, as umap-learn does: for up to ``2 * n_components`` components the
    anchors are the points ``+-e_i``, for more components they sit on a regular grid
    (umap-learn places them from the data centroids, which are not available from a
    graph), and each component is scaled into a box around its anchor.

    ``min_dist > spread`` is accepted, as in ``scanpy.tl.umap`` (``umap.UMAP`` itself raises), with a
    ``UserWarning``: the layout curve is then fitted outside the range umap-learn supports. The graph
    weights must be finite and non-negative.

    ``spectral_max_matvec`` caps the matrix-vector products spent per component
    (default: five per cell, between 1,000 and 50,000). When the cap is reached
    before the tolerance is met a ``RuntimeWarning`` is issued and the layout starts
    from the approximate eigenvectors.

    With more than one thread (and at least 1,000 cells) the SGD runs in parallel
    with a different schedule than on one thread, so layouts are deterministic for
    a fixed thread count but differ between thread counts. Layouts differ from
    those of earlier releases, whose initial layout was not converged.
    """
    if min_dist > spread:
        warnings.warn(f"umap: min_dist={min_dist} exceeds spread={spread}; scanpy accepts this, but umap-learn "
                      "raises because the curve fit is outside its supported range", UserWarning, stacklevel=2)
    rows, cols, w = _graph(bf)
    bf.obsm["X_umap"] = _native.umap(bf.n_obs, rows, cols, w, n_components, min_dist, spread, n_epochs,
                                      spectral_max_matvec, random_state)
    return bf


# --------------------------------------------------------------------------- differential expression
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def rank_genes_groups(bf: BioFrame, groupby: str, method: str = "t-test", groups: Union[str, Sequence[str]] = "all",
                      n_genes: Optional[int] = 100, tie_correct: bool = False,
                      memory_budget_gb: float = 0.25) -> pl.DataFrame:
    """Each group vs the rest, as ``sc.tl.rank_genes_groups(reference="rest")``.

    ``method="t-test"`` (Welch) or ``"wilcoxon"`` (Mann-Whitney U, normal
    approximation). The Wilcoxon test ranks only non-zero values: all implicit
    zeros of a gene form one tie block whose rank is known in closed form.
    Returns a long DataFrame (group, names, scores, logfoldchanges, pvals,
    pvals_adj) with the top ``n_genes`` per group by score; also stored in
    ``uns['rank_genes_groups']``.

    The reference ("rest") of each group is every other cell of the frame, as in scanpy:
    the cells of groups that are not selected through ``groups`` and the cells whose label
    is null count as "rest" as well. ``groups`` is ``"all"``, one group name or a list of
    names; an unknown name raises ``ValueError`` (scanpy raises ``IndexError``), as does a
    group with fewer than two cells or one that holds every cell.

    Differs from scanpy: the default is ``n_genes=100``; scanpy's default
    (``None``) keeps every gene. Pass ``n_genes=None`` for all genes. The arguments
    ``reference``, ``corr_method``, ``pts``, ``layer`` and ``use_raw`` and
    ``method="t-test_overestim_var"`` / ``"logreg"`` are not supported; the adjusted
    p-values are always Benjamini-Hochberg.
    """
    if method not in ("t-test", "wilcoxon"):
        raise ValueError('method must be "t-test" or "wilcoxon"')
    labels = bf.obs[groupby].cast(pl.Utf8).to_numpy()
    all_cats = sorted({x for x in labels if x is not None}, key=_natural_key)
    if isinstance(groups, str):
        groups = "all" if groups == "all" else [groups]
    if isinstance(groups, str) and groups == "all":
        cats = all_cats
    else:
        wanted = [str(g) for g in groups]
        unknown = [g for g in dict.fromkeys(wanted) if g not in set(all_cats)]
        if unknown:
            raise ValueError(f"groups {unknown} are not categories of {groupby!r}; available: {all_cats}")
        cats = [c for c in all_cats if c in set(wanted)]
    if not cats:
        raise ValueError("no groups selected")
    code = {c: i for i, c in enumerate(cats)}
    G, V = len(cats), bf.n_vars
    # unselected groups and null labels form one extra pseudo-group G: together with the other
    # selected groups it is the "rest" of every group (all n_obs cells take part in the sums)
    grp = np.array([code.get(x, G) if x is not None else G for x in labels], dtype=np.uint32)
    sizes_full = np.bincount(grp, minlength=G + 1).astype(np.float64)
    sizes = sizes_full[:G]
    N = float(sizes_full.sum())
    small = [c for c, n in zip(cats, sizes) if n < 2]
    if small:
        raise ValueError(f"groups {small} have fewer than two cells; statistics cannot be computed")
    whole = [c for c, n in zip(cats, sizes) if N - n < 1]
    if whole:
        raise ValueError(f"group {whole} holds every cell, so there is no rest to compare with")

    s = np.zeros((G + 1, V)); q = np.zeros((G + 1, V)); z = np.zeros((G + 1, V))
    for ctx in bf.iter_ctx():
        _native.group_gene_sums(*ctx, grp, s, q, z, None)
    S_all, Q_all = s.sum(0), q.sum(0)
    nnz_gene = z.sum(0)  # exact non-zeros per gene (sizes Wilcoxon gene blocks)
    del z

    if method == "wilcoxon":
        wscores = np.zeros((G, V))
        budget = memory_budget_gb * 1e9 / 12.0
        lo = 0
        while lo < V:
            hi, acc = lo, 0.0
            while hi < V and (acc + nnz_gene[hi] <= budget or hi == lo):
                acc += nnz_gene[hi]
                hi += 1
            parts = [_native.collect_gene_block(*ctx, grp, lo, hi) for ctx in bf.iter_ctx()]
            cat = [np.concatenate([p[i] for p in parts]) if len(parts) > 1 else parts[0][i] for i in range(3)]
            del parts
            rs, ties = _native.wilcoxon_rank_sums(cat[0], cat[1], cat[2], hi - lo, sizes_full.astype(np.uint64))
            rs = rs[:G]
            del cat
            T = 1.0 - ties / (N ** 3 - N) if tie_correct else np.ones(hi - lo)
            std = np.sqrt(T[None, :] * sizes[:, None] * (N - sizes)[:, None] * (N + 1) / 12.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                wscores[:, lo:hi] = (rs - sizes[:, None] * (N + 1) / 2.0) / std
            lo = hi

    names = np.array(bf.var_names, dtype=object)
    frames, uns = [], {"params": {"groupby": groupby, "method": method, "reference": "rest"}, "groups": cats}
    k = V if n_genes is None else min(n_genes, V)
    for i, c in enumerate(cats):  # one group at a time: O(genes) temporaries
        n1, n2 = sizes[i], N - sizes[i]
        m1 = s[i] / n1
        m2 = (S_all - s[i]) / n2
        logfc = np.log2((np.expm1(m1) + 1e-9) / (np.expm1(m2) + 1e-9))
        if method == "t-test":
            v1 = np.maximum((q[i] - n1 * m1 ** 2) / (n1 - 1), 0)
            v2 = np.maximum(((Q_all - q[i]) - n2 * m2 ** 2) / (n2 - 1), 0)
            a, b = v1 / n1, v2 / n2
            with np.errstate(divide="ignore", invalid="ignore"):
                sc_ = (m1 - m2) / np.sqrt(a + b)
                df = (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
            df = np.where(np.isnan(df), 1.0, df)
            pv = _native.t_pvalues(np.ascontiguousarray(sc_), np.ascontiguousarray(df))
        else:
            sc_ = wscores[i]
            pv = _native.normal_pvalues(np.ascontiguousarray(sc_))
        sc_ = np.where(np.isnan(sc_), 0.0, sc_)
        pv = np.where(np.isnan(pv), 1.0, pv)
        padj = _bh(pv)
        top = np.argsort(-sc_, kind="stable")[:k]
        frames.append(pl.DataFrame({
            "group": [c] * k, "names": names[top].astype(str), "scores": sc_[top],
            "logfoldchanges": logfc[top], "pvals": pv[top], "pvals_adj": padj[top],
        }))
    out = pl.concat(frames) if frames else pl.DataFrame()
    uns["table"] = out
    bf.uns["rank_genes_groups"] = uns
    return out


# --------------------------------------------------------------------------- gene set scores
def score_genes(bf: BioFrame, gene_list: Sequence[str], ctrl_size: int = 50, n_bins: int = 25,
                random_state: int = 0, score_name: str = "score", ctrl_as_ref: bool = True) -> BioFrame:
    """Gene-set score (mean of set − mean of expression-matched control genes),
    reproducing ``sc.tl.score_genes`` including its control-gene sampling.

    Genes in ``gene_list`` are looked up by name (``bf.var_names``) and, if ``var`` has a ``gene_ids``
    column, also by gene id. Names that are not in ``var`` are skipped. A name that matches
    several genes of ``var`` (Cell Ranger feature tables repeat some symbols) raises ``ValueError``
    naming it: pass the gene's id instead, or filter the duplicates out first. (scanpy refuses every
    AnnData with non-unique ``var_names``; CREST refuses only a repeated name in ``gene_list``.)
    """
    names = bf.var_names
    name_pos: dict = {}
    for i, g in enumerate(names):
        name_pos.setdefault(g, []).append(i)
    ids = bf.gene_ids
    id_pos = {} if ids is None else _unique_positions(ids)
    pos, glist, ambiguous = {}, [], []
    for g in dict.fromkeys(gene_list):
        where = name_pos.get(g, [])
        if len(where) == 1:
            pos[g] = where[0]
        elif g in id_pos:
            pos[g] = id_pos[g]
        elif len(where) > 1:
            ambiguous.append(g)
            continue
        else:
            continue
        glist.append(g)
    if ambiguous:
        raise ValueError(f"score_genes: {len(ambiguous)} names in gene_list match more than one gene in var: "
                         f"{ambiguous[:10]}; give the gene ids instead, or remove the duplicates first")
    if not glist:
        raise ValueError("no genes of gene_list are in var")
    n = bf.n_obs
    st = gene_stats(bf)
    obs_avg = st[:, 0] / n
    finite = np.isfinite(obs_avg)
    pool = np.flatnonzero(finite)
    avg = obs_avg[pool]
    sorted_avg = np.sort(avg)
    rank_min = np.searchsorted(sorted_avg, avg, side="left") + 1.0  # pandas rank(method="min")
    n_items = int(np.round(len(avg) / (n_bins - 1)))
    cut = rank_min // n_items
    cut_of = dict(zip(pool.tolist(), cut.tolist()))
    in_list = np.isin(pool, [pos[g] for g in glist])
    rs = np.random.RandomState(random_state)  # same stream as scanpy's np.random.seed
    ctrl = set()
    for c in np.unique([cut_of[pos[g]] for g in glist if pos[g] in cut_of]):
        sel = (cut == c) & (np.ones_like(in_list) if ctrl_as_ref else ~in_list)
        r_genes = pool[sel]
        if ctrl_size < len(r_genes):
            r_genes = r_genes[rs.choice(len(r_genes), ctrl_size, replace=False)]
        if ctrl_as_ref:
            r_genes = r_genes[~np.isin(r_genes, [pos[g] for g in glist])]
        ctrl.update(r_genes.tolist())
    if not ctrl:
        raise RuntimeError("no control genes found")
    w = np.zeros(bf.n_vars, np.float64)
    w[[pos[g] for g in glist]] += 1.0 / len(glist)
    w[sorted(ctrl)] -= 1.0 / len(ctrl)
    score = np.zeros(n, np.float64)
    for ctx in bf.iter_ctx():
        _native.weighted_row_sums(*ctx, w, score)
    bf.obs = bf.obs.with_columns(pl.Series(score_name, score))
    return bf
