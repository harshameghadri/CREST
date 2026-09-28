"""Download the real 10x datasets used by the benchmark into a directory.

    python bench/whitepaper/fetch_data.py DATA_DIR

Produces:
  pbmc10k_v3.h5 - 10x PBMC 10k, v3 chemistry (source for the synthetic scale-up sets)
  pbmc68k.h5    - 10x fresh 68k PBMC (Zheng et al. 2017), converted from MTX to .h5
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np

URLS = {
    "pbmc10k_v3.h5": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5",
    "pbmc68k.tar.gz": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name}")
        return
    print(f"  [download] {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def mtx_to_h5(mtx_dir: Path, out: Path) -> None:
    import crest

    bf = crest.read_10x_mtx(mtx_dir)
    X = bf.to_scipy(transform=False).tocsr()
    X.sort_indices()
    enc = lambda xs: np.array([str(x).encode() for x in xs])  # noqa: E731
    with h5py.File(out, "w") as f:
        m = f.create_group("matrix")
        m["barcodes"] = enc(bf.obs["barcode"].to_list())
        m["data"] = X.data.astype(np.int32)
        m["indices"] = X.indices.astype(np.int32)
        m["indptr"] = X.indptr.astype(np.int64)
        m["shape"] = np.array([X.shape[1], X.shape[0]], np.int32)
        ft = m.create_group("features")
        ft["id"] = enc(bf.var["gene_ids"].to_list())
        ft["name"] = enc(bf.var["gene_name"].to_list())
        ft["feature_type"] = enc(["Gene Expression"] * X.shape[1])
        ft["genome"] = enc(["hg19"] * X.shape[1])
    print(f"  [convert] {out.name}: {X.shape[0]} cells, {X.nnz:,} non-zeros")


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    d.mkdir(parents=True, exist_ok=True)
    download(URLS["pbmc10k_v3.h5"], d / "pbmc10k_v3.h5")
    if not (d / "pbmc68k.h5").exists():
        tgz = d / "pbmc68k.tar.gz"
        download(URLS["pbmc68k.tar.gz"], tgz)
        with tarfile.open(tgz) as t:
            t.extractall(d / "pbmc68k_mtx")
        mtx = next((d / "pbmc68k_mtx").rglob("matrix.mtx*")).parent
        mtx_to_h5(mtx, d / "pbmc68k.h5")
    else:
        print("  [skip] pbmc68k.h5")


if __name__ == "__main__":
    main()
