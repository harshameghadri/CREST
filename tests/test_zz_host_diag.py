"""Temporary: host and numerics diagnostics of the parity matrix on CI runners (removed before merge)."""
import hashlib
import json
import os
import platform
import warnings

import numpy as np
import polars as pl
import pytest

import _parity_cases as pc

PBMC3K = os.environ.get("CREST_TEST_PBMC3K")


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def _cpu():
    out = {"machine": platform.machine(), "libc": platform.libc_ver(), "python": platform.python_version()}
    try:
        txt = open("/proc/cpuinfo").read()
        out["model"] = next((l.split(":", 1)[1].strip() for l in txt.splitlines() if l.startswith("model name")), None)
        flags = next((l for l in txt.splitlines() if l.startswith("flags")), "")
        out["flags"] = [f for f in ("avx2", "fma", "avx512f", "avx512dq", "avx512vl", "avx512bw", "avx512_vnni") if f in flags.split()]
    except OSError:
        pass
    try:
        out["numpy_features_enabled"] = sorted(k for k, v in np._core._multiarray_umath.__cpu_features__.items() if v and ("AVX" in k or "FMA" in k))
    except Exception as e:  # noqa: BLE001
        out["numpy_features_enabled"] = repr(e)
    out["numpy"] = np.__version__
    return out


@pytest.mark.skipif(not PBMC3K, reason="CREST_TEST_PBMC3K not set")
def test_zz_host_diag():
    import scanpy as sc
    from scipy import stats
    rep = {"host": _cpu()}
    v = np.float32(np.linspace(0, 12, 200001))
    rep["np_log1p_f32_hash"], rep["np_expm1_f32_hash"], rep["np_log2_f32_hash"] = _h(np.log1p(v)), _h(np.expm1(v)), _h(np.log2(v + 1))
    X, names, ids = pc.load_raw(PBMC3K)
    m = pc.Measure("pbmc3k", X, names, ids, pc.DATASETS["pbmc3k"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.qc()
        a, bf = m.a, m.bf
        sc.pp.normalize_total(a, target_sum=1e4)
        import crest
        crest.pp.normalize_total(bf, 1e4)
        rep["scanpy_norm_hash"], rep["crest_norm_hash"] = _h(a.X.data), _h(bf.to_scipy().data)
        sc.pp.log1p(a)
        crest.pp.log1p(bf)
        rep["scanpy_log_hash"], rep["crest_log_hash"] = _h(a.X.data), _h(bf.to_scipy().data)
        rep["log_max_rel"] = float(np.max(np.abs(a.X.data - bf.to_scipy().data) / np.abs(a.X.data)))
        # Wilcoxon, identical input, fixed labels
        g = m.labels
        import anndata as ad
        ai = ad.AnnData(bf.to_scipy(), var=a.var[[]].copy())
        ai.obs["g"] = g
        ai.uns["log1p"] = {"base": None}
        sc.tl.rank_genes_groups(ai, "g", method="wilcoxon", n_genes=ai.n_vars, use_raw=False, reference="rest")
        bf.obs = bf.obs.with_columns(pl.Series("g", g))
        res = crest.tl.rank_genes_groups(bf, "g", method="wilcoxon", n_genes=None)
        rg = ai.uns["rank_genes_groups"]
        worst = (0, None)
        for grp in rg["names"].dtype.names:
            ref = {n: i for i, n in enumerate(rg["names"][grp])}
            sub = res.filter(pl.col("group") == grp)
            idx = np.array([ref[n] for n in sub["names"]])
            sref = np.asarray(rg["scores"][grp])[idx]
            d = np.abs(sub["scores"].to_numpy() - sref)
            j = int(np.nanargmax(d))
            if d[j] > worst[0]:
                worst = (float(d[j]), dict(group=grp, gene=sub["names"][j], crest=float(sub["scores"][j]), scanpy=float(sref[j]), dtype=str(np.asarray(rg["scores"][grp]).dtype), ulp=float(np.spacing(np.float32(abs(sref[j])))), n_ulp=float(d[j] / np.spacing(np.float32(abs(sref[j]))))))
        rep["wilcoxon_identical_worst"] = worst
        rep["wilcoxon_scores_hash_scanpy"] = _h(np.concatenate([np.asarray(rg["scores"][gr], np.float32) for gr in rg["names"].dtype.names]))
    warnings.warn("PARITY-DIAG " + json.dumps(rep, default=str))
