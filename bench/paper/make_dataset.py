"""Build a larger benchmark dataset from a real 10x matrix.

Each synthetic cell mixes a real cell with one of its 10 nearest neighbours
(in PCA space): both count vectors are binomially thinned (weights w, 1 - w with
w ~ U(0.3, 0.7), overall capture ~ U(0.6, 1.0)) and summed. Cells are unique
and interpolate locally, so cluster structure, sparsity and library sizes stay
realistic without the near-duplicate cliques that plain resampling creates
(which inflate Leiden to hundreds of tiny clusters). Written as a Cell Ranger
v3 .h5 so both scanpy and crest use their native 10x readers.

    python bench/paper/make_dataset.py --src pbmc68k.h5 --cells 500000 --out synth_500k.h5
"""

import argparse

import h5py
import numpy as np


def synthesize(src_path, n_cells: int, out_path, seed: int = 0) -> None:
    with h5py.File(src_path, "r") as f:
        m = f["matrix"] if "matrix" in f else f[next(iter(f.keys()))]
        indptr, indices, data = m["indptr"][:], m["indices"][:], m["data"][:]
        if "features" in m:
            ids, names = m["features"]["id"][:], m["features"]["name"][:]
            ftype = m["features"]["feature_type"][:] if "feature_type" in m["features"] else np.array([b"Gene Expression"] * len(ids))
        else:
            ids, names = m["genes"][:], m["gene_names"][:]
            ftype = np.array([b"Gene Expression"] * len(ids))
        n_genes = len(ids)

    import crest
    import scipy.sparse as sp

    src = sp.csr_matrix((data.astype(np.float32), indices, indptr), shape=(len(indptr) - 1, n_genes))
    bf = crest.BioFrame.from_scipy(src)
    crest.pp.normalize_total(bf, 1e4)
    crest.pp.log1p(bf)
    crest.pp.highly_variable_genes(bf, n_top_genes=2000)
    crest.tl.pca(bf, n_comps=30)
    nn, _ = crest.crest.knn_graph(np.ascontiguousarray(bf.obsm["X_pca"]), 10)

    rng = np.random.default_rng(seed)
    src_n = src.shape[0]
    a_idx = rng.integers(0, src_n, n_cells)
    b_idx = nn[a_idx, rng.integers(0, nn.shape[1], n_cells)]
    w = rng.uniform(0.3, 0.7, n_cells)
    cap = rng.uniform(0.6, 1.0, n_cells)

    out_ptr = np.zeros(n_cells + 1, np.int64)
    ind_parts, dat_parts = [], []
    block = 5_000  # bounded memory
    for lo in range(0, n_cells, block):
        hi = min(lo + block, n_cells)
        A, B = src[a_idx[lo:hi]].tocoo(), src[b_idx[lo:hi]].tocoo()
        pa = (cap[lo:hi] * w[lo:hi])[A.row]
        pb = (cap[lo:hi] * (1 - w[lo:hi]))[B.row]
        ca = rng.binomial(A.data.astype(np.int64), pa)
        cb = rng.binomial(B.data.astype(np.int64), pb)
        M = sp.coo_matrix((np.r_[ca, cb], (np.r_[A.row, B.row], np.r_[A.col, B.col])),
                          shape=(hi - lo, n_genes)).tocsr()
        M.sum_duplicates()
        M.eliminate_zeros()
        M.sort_indices()
        ind_parts.append(M.indices.astype(np.int32))
        dat_parts.append(M.data.astype(np.int32))
        out_ptr[lo + 1:hi + 1] = out_ptr[lo] + M.indptr[1:]
    ind = np.concatenate(ind_parts)
    dat = np.concatenate(dat_parts)

    tmp = str(out_path) + ".tmp"
    with h5py.File(tmp, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([f"cell{i:08d}-1".encode() for i in range(n_cells)]))
        m.create_dataset("data", data=dat, compression="gzip", compression_opts=1)
        m.create_dataset("indices", data=ind, compression="gzip", compression_opts=1)
        m.create_dataset("indptr", data=out_ptr)
        m.create_dataset("shape", data=np.array([n_genes, n_cells], np.int32))
        ft = m.create_group("features")
        ft.create_dataset("id", data=ids)
        ft.create_dataset("name", data=names)
        ft.create_dataset("feature_type", data=ftype)
        ft.create_dataset("genome", data=np.array([b"GRCh38"] * n_genes))
    import os
    os.replace(tmp, out_path)
    print(f"wrote {out_path}: {n_cells} cells, {n_genes} genes, {len(dat):,} non-zeros")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Cell Ranger .h5 (v2 or v3)")
    ap.add_argument("--cells", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    synthesize(args.src, args.cells, args.out, args.seed)


if __name__ == "__main__":
    main()
