"""Check that the repository and the source distribution contain only package files.

Usage: python3 .github/scripts/check_package_contents.py [SDIST.tar.gz ...]

1. Every top-level entry of `git ls-files` must be in REPO_ALLOWED.
2. Every top-level entry inside each sdist must be in SDIST_ALLOWED.
3. No path in either may match DENIED (local development notes and review material).

Exit status 1 lists every offending path. CI runs this in the `sdist` job, so a release
cannot be built from a tree that carries internal files.
"""
import re
import subprocess
import sys
import tarfile

REPO_ALLOWED = {
    ".github", ".gitignore", ".readthedocs.yaml",
    "CHANGELOG.md", "CITATION.cff", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "KNOWN_ISSUES.md",
    "Cargo.lock", "Cargo.toml", "LICENSE", "README.md", "pyproject.toml",
    "bench", "changes", "crest", "docs", "scripts", "src", "tests",
}
SDIST_ALLOWED = {
    "PKG-INFO", "pyproject.toml", "Cargo.toml", "Cargo.lock", "README.md", "LICENSE",
    "CHANGELOG.md", "CITATION.cff", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "KNOWN_ISSUES.md",
    ".gitignore", ".readthedocs.yaml", "crest", "src", "tests", "docs",
}
DENIED = re.compile(
    r"(^|/)(CLAUDE(\.local)?\.md|AGENTS\.md|HANDOVER\.md|work\.md)$"
    r"|(^|/)(\.claude|review)/"
    r"|^scripts/(export_chat\.py|crest_git_housekeeping\.sh)$"
)


def check(paths, allowed, label):
    problems = []
    for p in paths:
        top = p.split("/", 1)[0]
        if top not in allowed:
            problems.append(f"{label}: not an allowed top-level entry: {p}")
        if DENIED.search(p):
            problems.append(f"{label}: internal file: {p}")
    return problems


def main(argv):
    problems = []
    tracked = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True).stdout.split()
    if not tracked:
        problems.append("repository: git ls-files returned nothing (not a git checkout?)")
    problems += check(tracked, REPO_ALLOWED, "repository")
    for sdist in argv:
        with tarfile.open(sdist) as tf:
            names = [m.name.split("/", 1)[1] for m in tf.getmembers() if "/" in m.name and m.isfile()]
        if not names:
            problems.append(f"{sdist}: empty archive")
        problems += check(names, SDIST_ALLOWED, sdist)
    if problems:
        print("\n".join(sorted(set(problems))))
        print(f"{len(set(problems))} problem(s)")
        return 1
    print(f"ok: {len(tracked)} tracked files, {len(argv)} sdist(s) checked")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
