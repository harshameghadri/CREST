#!/usr/bin/env python3
"""Download the public datasets used by the larger tests and verify their checksums.

Usage:
    python scripts/fetch_test_data.py                  # both datasets into $CREST_TEST_DATA
    python scripts/fetch_test_data.py pbmc3k           # one dataset
    python scripts/fetch_test_data.py --dest DIR       # explicit directory
    python scripts/fetch_test_data.py --path pbmc10k   # print the path of a dataset (fetch if missing)
    python scripts/fetch_test_data.py --derive --print-env     # also build the PBMC 3k graph; print export lines
    python scripts/fetch_test_data.py --derive --github-env "$GITHUB_ENV"    # CI: export them to later steps

Besides the archives, the script prepares what the environment-gated tests read:
    CREST_TEST_PBMC3K                the extracted PBMC 3k 10x directory (matrix.mtx, genes.tsv, barcodes.tsv)
    CREST_TEST_PBMC_CONNECTIVITIES   .npz of the scanpy connectivities matrix of PBMC 3k (needs --derive, scanpy and anndata)

The cache directory is $CREST_TEST_DATA, or ~/.cache/crest-test-data when it is not set.
A file that is already present with the right sha256 is not downloaded again; a file with the
wrong sha256 is deleted and fetched again, and the script exits with an error if the new copy
does not match either. The standard library is enough: no extra packages are needed.

The datasets are public 10x Genomics example data (CC BY 4.0):
    pbmc3k    PBMC 3k, filtered_gene_bc_matrices (hg19, Cell Ranger 1.1.0), 10x MTX tarball
    pbmc10k   PBMC 10k v3, filtered_feature_bc_matrix.h5 (Cell Ranger 3.0.0)
"""
import argparse
import hashlib
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASETS = {
    "pbmc3k": {
        "filename": "pbmc3k.tar.gz",
        "url": "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
        "sha256": "847d6ebd9a1ec9a768f2be7e40ca42cbfe75ebeb6d76a4c24167041699dc28b5",
        "size": 7621991,
    },
    "pbmc10k": {
        "filename": "pbmc10k.h5",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5",
        "sha256": "ebc5dedc938830e20f8e1aafb893b0a9e0bf88584f3ef2d00b232dd277a188af",
        "size": 37491093,
    },
}
CHUNK = 1 << 20
# the 10x download host answers the default urllib user agent with 403, so identify the script
USER_AGENT = "crest-fetch-test-data/1.0 (+https://github.com/harshameghadri/CREST)"


def cache_dir(dest=None):
    if dest:
        return Path(dest).expanduser()
    env = os.environ.get("CREST_TEST_DATA")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "crest-test-data"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target, retries=3, timeout=60):
    """Stream `url` into `target` (via a .part file); retries on network errors."""
    part = Path(str(target) + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as resp, open(part, "wb") as out:
                for block in iter(lambda: resp.read(CHUNK), b""):
                    out.write(block)
            os.replace(part, target)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}: {last}")


def fetch(name, dest=None, spec=None, log=print):
    """Make sure the dataset `name` is in the cache with the right checksum; return its path."""
    spec = spec or DATASETS[name]
    directory = cache_dir(dest)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec["filename"]
    if target.exists():
        if sha256_of(target) == spec["sha256"]:
            log(f"{name}: cached, checksum ok ({target})")
            return target
        log(f"{name}: cached file has the wrong checksum, fetching again")
        target.unlink()
    log(f"{name}: downloading {spec['url']}")
    download(spec["url"], target)
    got = sha256_of(target)
    if got != spec["sha256"]:
        target.unlink()
        raise RuntimeError(f"{name}: sha256 mismatch after download (expected {spec['sha256']}, got {got})")
    log(f"{name}: downloaded, checksum ok ({target})")
    return target


PBMC3K_SUBDIR = "pbmc3k"
PBMC3K_MTX = Path("filtered_gene_bc_matrices") / "hg19"
CONNECTIVITIES_FILE = "pbmc3k_connectivities.npz"


def extract_pbmc3k(dest=None, log=print):
    """Unpack the PBMC 3k archive into the cache (once); return the directory with matrix.mtx."""
    directory = cache_dir(dest)
    archive = directory / DATASETS["pbmc3k"]["filename"]
    target = directory / PBMC3K_SUBDIR
    mtx_dir = target / PBMC3K_MTX
    if (mtx_dir / "matrix.mtx").exists():
        return mtx_dir
    if not archive.exists():
        raise RuntimeError(f"{archive} is missing; fetch pbmc3k first")
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if not (target / member.name).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe path in {archive}: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unexpected entry in {archive}: {member.name}")
        # the 'data' filter (Python 3.12+; backported to 3.10.12 and 3.11.4) refuses links and special files too
        tf.extractall(target, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    if not (mtx_dir / "matrix.mtx").exists():
        raise RuntimeError(f"{archive} does not contain {PBMC3K_MTX}/matrix.mtx")
    log(f"pbmc3k: extracted to {mtx_dir}")
    return mtx_dir


def derive_connectivities(dest=None, log=print):
    """Scanpy connectivities matrix of PBMC 3k (standard workflow, 15 neighbours), saved as .npz.

    Needs scanpy and anndata, which the `test` extra installs; the result is cached.
    """
    directory = cache_dir(dest)
    out = directory / CONNECTIVITIES_FILE
    if out.exists():
        return out
    import scanpy as sc
    import scipy.sparse as sp

    adata = sc.read_10x_mtx(extract_pbmc3k(dest, log), var_names="gene_symbols")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, random_state=0)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=0)
    sp.save_npz(out, adata.obsp["connectivities"].tocsr())
    log(f"pbmc3k: wrote the scanpy connectivities matrix ({adata.n_obs} cells) to {out}")
    return out


def environment(dest=None, derive=False, log=print):
    """The variables the environment-gated tests read, after making sure their files exist."""
    directory = cache_dir(dest)
    env = {"CREST_TEST_DATA": str(directory)}
    if (directory / DATASETS["pbmc3k"]["filename"]).exists():
        env["CREST_TEST_PBMC3K"] = str(extract_pbmc3k(dest, log))
        if derive:
            env["CREST_TEST_PBMC_CONNECTIVITIES"] = str(derive_connectivities(dest, log))
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="datasets to fetch: " + ", ".join(DATASETS) + " (default: all)")
    ap.add_argument("--dest", help="cache directory (default: $CREST_TEST_DATA or ~/.cache/crest-test-data)")
    ap.add_argument("--path", action="store_true", help="print only the path(s) of the datasets")
    ap.add_argument("--derive", action="store_true", help="also build the PBMC 3k connectivities .npz (needs scanpy)")
    ap.add_argument("--github-env", metavar="FILE", help="append the CREST_TEST_* variables to FILE (a GitHub Actions env file)")
    ap.add_argument("--print-env", action="store_true", help="print the CREST_TEST_* variables as shell export lines")
    args = ap.parse_args(argv)
    names = args.names or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        ap.error(f"unknown dataset {unknown[0]!r}; choose from {', '.join(DATASETS)}")
    log = (lambda *_: None) if args.path else print
    try:
        for name in names:
            path = fetch(name, args.dest, log=log)
            if args.path:
                print(path)
        if args.github_env or args.print_env:
            env = environment(args.dest, args.derive, log=log)
            if args.github_env:
                with open(args.github_env, "a") as fh:
                    fh.writelines(f"{k}={v}\n" for k, v in env.items())
            if args.print_env:
                for k, v in env.items():
                    print(f"export {k}={v}")
        elif args.derive:
            derive_connectivities(args.dest, log=log)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
