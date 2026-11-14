#!/usr/bin/env python3
"""Set the CREST version everywhere it is written down.

Usage:
    python scripts/bump_version.py 0.5.0            # final release
    python scripts/bump_version.py 0.5.0rc1         # release candidate (PEP 440 spelling)
    python scripts/bump_version.py 0.5.0 --date 2026-12-01 --dry-run

The version is accepted in PEP 440 form (0.5.0rc1) or in Cargo form (0.5.0-rc.1) and is
written in the form each file expects:

    Cargo.toml, Cargo.lock           0.5.0-rc.1   (semantic versioning; maturin converts it to 0.5.0rc1)
    crest/__init__.py                0.5.0rc1     (PEP 440)
    CITATION.cff                     version: and date-released:
    README.md, docs/*.md             pinned tags in clone and download commands (v0.5.0rc1)

tests/test_version_consistency.py checks that these agree.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

PRE_TO_PEP440 = {"a": "a", "alpha": "a", "b": "b", "beta": "b", "rc": "rc", "c": "rc", "pre": "rc", "preview": "rc"}
PRE_TO_CARGO = {"a": "alpha", "b": "beta", "rc": "rc"}

VERSION_RE = re.compile(
    r"""^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)
        (?:[-._]?(?P<pre>alpha|beta|preview|pre|rc|a|b|c)[-._]?(?P<pre_n>\d+))?
        (?:[-._]?dev[-._]?(?P<dev>\d+))?$""",
    re.VERBOSE | re.IGNORECASE,
)

# tags that are pinned in commands: .../CREST/v0.4.0/..., --ref v0.4.0, git checkout v0.4.0
PIN_RE = re.compile(
    r"(?P<prefix>(?:CREST/|--ref[ =]|git checkout |--branch[ =]))v"
    r"(?P<version>\d+\.\d+\.\d+(?:[-._]?(?:alpha|beta|preview|pre|rc|a|b|c)[-._]?\d+)?(?:[-._]?dev[-._]?\d+)?)"
)
PIN_FILES = ("README.md", "docs/benchmarks.md", "docs/installation.md")


def parse(version):
    """Split a version string into (release tuple, pre-release kind or None, pre number, dev number)."""
    m = VERSION_RE.match(version.strip())
    if not m:
        raise ValueError(f"not a version I understand: {version!r} (expected like 0.5.0, 0.5.0rc1 or 0.5.0-rc.1)")
    pre = PRE_TO_PEP440[m["pre"].lower()] if m["pre"] else None
    return (
        (int(m["major"]), int(m["minor"]), int(m["patch"])),
        pre,
        int(m["pre_n"]) if m["pre_n"] else None,
        int(m["dev"]) if m["dev"] else None,
    )


def to_pep440(version):
    release, pre, pre_n, dev = parse(version)
    out = ".".join(map(str, release))
    if pre:
        out += f"{pre}{pre_n}"
    if dev is not None:
        out += f".dev{dev}"
    return out


def to_cargo(version):
    release, pre, pre_n, dev = parse(version)
    out = ".".join(map(str, release))
    ident = []
    if pre:
        ident.append(f"{PRE_TO_CARGO[pre]}.{pre_n}")
    if dev is not None:
        ident.append(f"dev.{dev}")
    return out + ("-" + ".".join(ident) if ident else "")


def normalize(version):
    """The canonical (PEP 440) spelling; two spellings of one version compare equal."""
    return to_pep440(version)


# --------------------------------------------------------------------------- readers
def _cargo_package_version(text):
    in_package = False
    for line in text.splitlines():
        if line.strip().startswith("["):
            in_package = line.strip() == "[package]"
        elif in_package:
            m = re.match(r'version\s*=\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    raise ValueError("no version in the [package] table of Cargo.toml")


LOCK_BLOCK_RE = re.compile(r'(\[\[package\]\]\nname = "(?P<name>crest-sc|crest)"\nversion = ")(?P<version>[^"]+)(")')


def read_versions(root):
    """Every place the version is written: {label: version string as written}."""
    root = Path(root)
    found = {"Cargo.toml": _cargo_package_version((root / "Cargo.toml").read_text())}
    for m in LOCK_BLOCK_RE.finditer((root / "Cargo.lock").read_text()):
        found[f"Cargo.lock [{m['name']}]"] = m["version"]
    init = re.search(r'^__version__\s*=\s*"([^"]+)"', (root / "crest/__init__.py").read_text(), re.M)
    found["crest/__init__.py"] = init.group(1) if init else None
    cff = (root / "CITATION.cff").read_text()
    m = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', cff, re.M)
    found["CITATION.cff"] = m.group(1) if m else None
    return found


def read_pins(root):
    """Pinned release tags in the documentation: {(file, line number): version without the v}."""
    pins = {}
    for rel in PIN_FILES:
        path = Path(root) / rel
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for m in PIN_RE.finditer(line):
                pins[(rel, n)] = m["version"]
    return pins


# --------------------------------------------------------------------------- writers
def _sub_once(text, pattern, repl, what, flags=0):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise ValueError(f"could not find {what}")
    return new


def bump(root, version, date=None, dry_run=False):
    """Rewrite the version in every file under `root`. Returns the list of files that changed."""
    root = Path(root)
    pep, cargo = to_pep440(version), to_cargo(version)
    date = date or datetime.date.today().isoformat()
    datetime.date.fromisoformat(date)  # validates
    edits = {}

    text = (root / "Cargo.toml").read_text()
    old = _cargo_package_version(text)
    # replace only the [package] version line
    pkg = re.search(r"(\[package\][^\[]*?\nversion\s*=\s*\")" + re.escape(old) + '"', text)
    if not pkg:
        raise ValueError("could not find the [package] version line in Cargo.toml")
    edits["Cargo.toml"] = text[: pkg.start()] + pkg.group(1) + cargo + '"' + text[pkg.end():]

    lock = (root / "Cargo.lock").read_text()
    new_lock, n = LOCK_BLOCK_RE.subn(lambda m: m.group(1) + cargo + m.group(4), lock)
    if n < 1:
        raise ValueError("no crest package entry in Cargo.lock")
    edits["Cargo.lock"] = new_lock

    edits["crest/__init__.py"] = _sub_once(
        (root / "crest/__init__.py").read_text(), r'^(__version__\s*=\s*")[^"]+(")', rf"\g<1>{pep}\g<2>",
        "__version__ in crest/__init__.py", re.M,
    )

    cff = (root / "CITATION.cff").read_text()
    cff = _sub_once(cff, r'^(version:\s*)"?[^"\s]+"?\s*$', rf'\g<1>"{pep}"', "version in CITATION.cff", re.M)
    cff = _sub_once(cff, r'^(date-released:\s*)"?[^"\s]+"?\s*$', rf'\g<1>"{date}"', "date-released in CITATION.cff", re.M)
    edits["CITATION.cff"] = cff

    for rel in PIN_FILES:
        path = root / rel
        if path.exists():
            text = path.read_text()
            edits[rel] = PIN_RE.sub(lambda m: f"{m['prefix']}v{pep}", text)

    changed = []
    for rel, new in edits.items():
        path = root / rel
        if path.read_text() != new:
            changed.append(rel)
            if not dry_run:
                path.write_text(new)
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version", help="new version, e.g. 0.5.0 or 0.5.0rc1")
    ap.add_argument("--date", help="release date for CITATION.cff (default: today, YYYY-MM-DD)")
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path, help="repository root")
    ap.add_argument("--dry-run", action="store_true", help="report the files that would change and write nothing")
    args = ap.parse_args(argv)
    try:
        changed = bump(args.root, args.version, args.date, args.dry_run)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    verb = "would change" if args.dry_run else "changed"
    print(f"version {to_pep440(args.version)} (Cargo form {to_cargo(args.version)}); {verb}:")
    for rel in changed:
        print(f"  {rel}")
    if not changed:
        print("  nothing: every file already carries this version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
