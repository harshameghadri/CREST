#!/usr/bin/env python3
"""Turn the change fragments in changes/ into a CHANGELOG.md section and delete them.

Usage:
    python scripts/assemble_changelog.py 0.5.0 [--date 2026-12-01] [--dry-run]
    python scripts/assemble_changelog.py --check        # validate the fragments only

Every pull request adds one fragment, changes/<branch-name-with-dashes>.md, holding one to
four bullets (see changes/README.md). A bullet may start with a category tag (`Added:`,
`Changed:`, `Fixed:`, `Removed:`); untagged bullets go under Changed. The new section is
inserted above the newest existing entry of CHANGELOG.md and the fragments are removed.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

CATEGORIES = ("Added", "Changed", "Fixed", "Removed")
MAX_BULLETS = 4
TAG_RE = re.compile(r"^(?P<tag>" + "|".join(CATEGORIES) + r")\s*:\s*(?P<rest>.*)$", re.DOTALL)


class FragmentError(ValueError):
    pass


def fragment_files(changes_dir):
    return sorted(p for p in Path(changes_dir).glob("*.md") if p.name.lower() != "readme.md")


def parse_fragment(path):
    """Return the bullets of one fragment as a list of (category, text); text keeps its line breaks."""
    bullets, current = [], None
    for raw in Path(path).read_text().splitlines():
        if re.match(r"^[-*]\s+\S", raw):
            current = [raw[2:].strip()]
            bullets.append(current)
        elif raw.strip() == "":
            current = None
        elif raw.startswith((" ", "\t")) and current is not None:
            current.append(raw.strip())
        else:
            raise FragmentError(f"{path}: only bullet lines ('- ...') and their indented continuations are allowed, got {raw!r}")
    if not 1 <= len(bullets) <= MAX_BULLETS:
        raise FragmentError(f"{path}: a fragment needs 1 to {MAX_BULLETS} bullets, found {len(bullets)}")
    out = []
    for lines in bullets:
        text = "\n  ".join(lines)
        m = TAG_RE.match(text)
        if m:
            rest = m["rest"].strip()
            out.append((m["tag"], rest[:1].upper() + rest[1:]))
        else:
            out.append(("Changed", text))
        if not out[-1][1]:
            raise FragmentError(f"{path}: empty bullet")
    return out


def render_section(version, date, fragments):
    """Markdown for one release from {fragment path: bullets}."""
    by_cat = {c: [] for c in CATEGORIES}
    for path in sorted(fragments):
        for cat, text in fragments[path]:
            by_cat[cat].append(text)
    lines = [f"## {version} ({date})", ""]
    for cat in CATEGORIES:
        if by_cat[cat]:
            lines.append(f"### {cat}")
            lines.extend(f"- {t}" for t in by_cat[cat])
            lines.append("")
    return "\n".join(lines)


def insert_section(changelog_text, section, version):
    if re.search(rf"^##\s+\[?{re.escape(version)}\]?(\s|$)", changelog_text, re.M):
        raise FragmentError(f"CHANGELOG.md already has a section for {version}")
    m = re.search(r"^## ", changelog_text, re.M)
    if m is None:
        head = changelog_text.rstrip("\n") + "\n\n" if changelog_text.strip() else "# Changelog\n\n"
        return head + section.rstrip("\n") + "\n"
    return changelog_text[: m.start()] + section.rstrip("\n") + "\n\n" + changelog_text[m.start():]


def assemble(root, version, date=None, dry_run=False):
    """Fold the fragments under `root`/changes into `root`/CHANGELOG.md. Returns the new section text."""
    root = Path(root)
    files = fragment_files(root / "changes")
    if not files:
        raise FragmentError("no change fragments found in changes/")
    fragments = {p.name: parse_fragment(p) for p in files}
    date = date or datetime.date.today().isoformat()
    datetime.date.fromisoformat(date)
    section = render_section(version, date, fragments)
    changelog = root / "CHANGELOG.md"
    new_text = insert_section(changelog.read_text() if changelog.exists() else "", section, version)
    if not dry_run:
        changelog.write_text(new_text)
        for p in files:
            p.unlink()
    return section


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version", nargs="?", help="version of the new section, e.g. 0.5.0")
    ap.add_argument("--date", help="date of the section (default: today, YYYY-MM-DD)")
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path, help="repository root")
    ap.add_argument("--dry-run", action="store_true", help="print the section; change nothing")
    ap.add_argument("--check", action="store_true", help="validate the fragments in changes/ and exit")
    args = ap.parse_args(argv)
    try:
        if args.check:
            files = fragment_files(Path(args.root) / "changes")
            for p in files:
                parse_fragment(p)
            print(f"{len(files)} fragment(s) valid")
            return 0
        if not args.version:
            ap.error("a version is required unless --check is given")
        print(assemble(args.root, args.version, args.date, args.dry_run))
    except FragmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
