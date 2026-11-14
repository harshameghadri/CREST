"""Copy a source tree to a scratch directory for running the tests against an installed wheel.

Usage:
    python .github/scripts/make_test_dir.py DEST [--tree TREE] [--overlay-from CURRENT PATH ...]

Copies the files that git tracks in TREE (default: the current directory; every file except
build and cache directories when TREE is not a git checkout) except the Python
package `crest/` and the Rust sources `src/`, so that `import crest` finds the installed wheel
and not the source tree, while tests that read repository files (docs, bench/, scripts/) still
find them next to `tests/`.

With --overlay-from CURRENT PATH ..., the given paths (for example `tests pyproject.toml`) come
from the tracked files of CURRENT instead of TREE. The baseline jobs use this to run the
current tests against an older source tree.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXCLUDED = ("crest/", "src/")


SKIP_DIRS = {".git", ".venv", "target", "__pycache__", "dist", "node_modules"}


def tracked(tree):
    """Files git tracks in `tree`; outside a git checkout, every file except build and cache directories."""
    done = subprocess.run(["git", "-C", str(tree), "ls-files", "-z"], capture_output=True)
    if done.returncode == 0 and done.stdout:
        return [p for p in done.stdout.decode().split("\0") if p]
    root = Path(tree)
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not SKIP_DIRS & set(p.relative_to(root).parts)
    )


def under(path, prefixes):
    return any(path == p.rstrip("/") or path.startswith(p.rstrip("/") + "/") for p in prefixes)


def copy(tree, dest, files):
    for rel in files:
        src = Path(tree) / rel
        if src.is_symlink() or not src.is_file():
            continue
        target = Path(dest) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dest", type=Path)
    ap.add_argument("--tree", type=Path, default=Path("."))
    ap.add_argument("--overlay-from", nargs="+", metavar=("CURRENT", "PATH"),
                    help="a tree followed by the paths to take from it instead of --tree")
    args = ap.parse_args(argv)
    if args.dest.exists() and any(args.dest.iterdir()):
        sys.exit(f"error: {args.dest} exists and is not empty")
    overlay_tree, overlay = (Path(args.overlay_from[0]), args.overlay_from[1:]) if args.overlay_from else (None, [])
    files = [p for p in tracked(args.tree) if not under(p, EXCLUDED) and not under(p, overlay)]
    args.dest.mkdir(parents=True, exist_ok=True)
    copy(args.tree, args.dest, files)
    n_over = 0
    if overlay_tree:
        extra = [p for p in tracked(overlay_tree) if under(p, overlay)]
        copy(overlay_tree, args.dest, extra)
        n_over = len(extra)
    print(f"{args.dest}: {len(files)} files from {args.tree}" + (f" + {n_over} from {overlay_tree}" if n_over else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
