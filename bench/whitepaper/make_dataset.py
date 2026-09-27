"""Build a larger benchmark dataset from a real 10x matrix by resampling cells.

Each synthetic cell is a real cell's count vector binomially thinned with a
random capture rate in [0.6, 1.0], so every cell is distinct while gene-gene
structure, sparsity and library-size distributions stay realistic. Written as a
Cell Ranger v3 .h5 so both scanpy and crest use their native 10x readers.

    python bench/whitepaper/make_dataset.py --src pbmc68k.h5 --cells 500000 --out pbmc_500k.h5
"""

import argparse

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Cell Ranger .h5 (v2 or v3)")
    ap.add_argument("--cells", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with h5py.File(args.src, "r") as f:
        m = f["matrix"] if "matrix" in f else f[next(iter(f.keys()))]
        indptr, indices, data = m["indptr"][:], m["indices"][:], m["data"][:]
        if "features" in m:
            ids, names = m["features"]["id"][:], m["features"]["name"][:]
            ftype = m["features"]["feature_type"][:] if "feature_type" in m["features"] else np.array([b"Gene Expression"] * len(ids))
        else:
            ids, names = m["genes"][:], m["gene_names"][:]
            ftype = np.array([b"Gene Expression"] * len(ids))
        n_genes = len(ids)

    rng = np.random.default_rng(args.seed)
    src_n = len(indptr) - 1
    pick = rng.integers(0, src_n, args.cells)
    rate = rng.uniform(0.6, 1.0, args.cells)

    out_ptr = np.zeros(args.cells + 1, np.int64)
    ind_parts, dat_parts = [], []
    block = 50_000
    for lo in range(0, args.cells, block):
        hi = min(lo + block, args.cells)
        cells = pick[lo:hi]
        lens = indptr[cells + 1] - indptr[cells]
        idx = np.concatenate([np.arange(indptr[c], indptr[c + 1]) for c in cells])
        counts = rng.binomial(data[idx].astype(np.int64), np.repeat(rate[lo:hi], lens)).astype(np.int32)
        keep = counts > 0
        row = np.repeat(np.arange(hi - lo), lens)[keep]
        ind_parts.append(indices[idx][keep].astype(np.int32))
        dat_parts.append(counts[keep])
        out_ptr[lo + 1:hi + 1] = out_ptr[lo] + np.cumsum(np.bincount(row, minlength=hi - lo))
    ind = np.concatenate(ind_parts)
    dat = np.concatenate(dat_parts)

    with h5py.File(args.out, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([f"cell{i:08d}-1".encode() for i in range(args.cells)]))
        m.create_dataset("data", data=dat, compression="gzip", compression_opts=1)
        m.create_dataset("indices", data=ind, compression="gzip", compression_opts=1)
        m.create_dataset("indptr", data=out_ptr)
        m.create_dataset("shape", data=np.array([n_genes, args.cells], np.int32))
        ft = m.create_group("features")
        ft.create_dataset("id", data=ids)
        ft.create_dataset("name", data=names)
        ft.create_dataset("feature_type", data=ftype)
        ft.create_dataset("genome", data=np.array([b"GRCh38"] * n_genes))
    print(f"wrote {args.out}: {args.cells} cells, {n_genes} genes, {len(dat):,} non-zeros")


if __name__ == "__main__":
    main()
