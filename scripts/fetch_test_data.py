#!/usr/bin/env python3
"""Download the public datasets used by the larger tests and verify their checksums.

Usage:
    python scripts/fetch_test_data.py                  # both datasets into $CREST_TEST_DATA
    python scripts/fetch_test_data.py pbmc3k           # one dataset
    python scripts/fetch_test_data.py --dest DIR       # explicit directory
    python scripts/fetch_test_data.py --path pbmc10k   # print the path of a dataset (fetch if missing)

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
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="datasets to fetch: " + ", ".join(DATASETS) + " (default: all)")
    ap.add_argument("--dest", help="cache directory (default: $CREST_TEST_DATA or ~/.cache/crest-test-data)")
    ap.add_argument("--path", action="store_true", help="print only the path(s) of the datasets")
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
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
