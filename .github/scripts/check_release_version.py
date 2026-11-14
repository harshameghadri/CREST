"""Fail unless the release tag, Cargo.toml and crest/__init__.py name the same version.

Usage:
    python .github/scripts/check_release_version.py v0.5.0rc1     # tag build: all three must agree
    python .github/scripts/check_release_version.py               # manual run: Cargo.toml == __init__.py

Versions are compared in PEP 440 form, so the tag v0.5.0rc1, the Cargo version 0.5.0-rc.1 and
__version__ = "0.5.0rc1" are the same version. With --github-output FILE the script also
writes `version=<PEP 440 version>` and `prerelease=true|false` for later workflow steps.
"""
import argparse
import re
import sys
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version


def read_cargo(root):
    with open(root / "Cargo.toml", "rb") as fh:
        return tomllib.load(fh)["package"]["version"]


def read_init(root):
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', (root / "crest" / "__init__.py").read_text(), re.M)
    if not m:
        raise SystemExit("error: no __version__ in crest/__init__.py")
    return m.group(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tag", nargs="?", help="release tag, e.g. v0.5.0rc1 (omit for a manual run)")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--github-output", type=Path)
    args = ap.parse_args(argv)

    found = {"Cargo.toml": read_cargo(args.root), "crest/__init__.py": read_init(args.root)}
    if args.tag:
        if not re.match(r"^v\d", args.tag):
            raise SystemExit(f"error: tag {args.tag!r} does not look like v<version>")
        found["tag"] = args.tag[1:]
    try:
        parsed = {k: Version(v) for k, v in found.items()}
    except InvalidVersion as exc:
        raise SystemExit(f"error: {exc}")
    for k, v in found.items():
        print(f"{k:20s} {v!r:16} -> {parsed[k]}")
    if len(set(parsed.values())) != 1:
        raise SystemExit("error: the versions above are not the same version")
    version = next(iter(parsed.values()))
    print(f"ok: {version} ({'pre-release' if version.is_prerelease or version.is_devrelease else 'final'})")
    if args.github_output:
        with open(args.github_output, "a") as fh:
            fh.write(f"version={version}\nprerelease={'true' if (version.is_prerelease or version.is_devrelease) else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
