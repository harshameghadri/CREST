"""Print failed and errored tests of a JUnit XML report as GitHub Actions error annotations.

Annotations are attached to the check run and can be read in the pull request or through the
REST API (GET /repos/{owner}/{repo}/check-runs/{id}/annotations), without downloading the log.
Usage: python .github/scripts/annotate_failures.py junit.xml [more.xml ...]
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

MAX_ANNOTATIONS = 10   # GitHub shows at most 10 error annotations per step
MAX_MESSAGE = 1500     # characters of the failure message kept in one annotation


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape(text).replace(":", "%3A").replace(",", "%2C")


def failures(paths):
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            yield ("junit", f"could not read {path}: {exc}", "")
            continue
        for case in root.iter("testcase"):
            for tag in ("failure", "error"):
                node = case.find(tag)
                if node is not None:
                    name = f"{case.get('classname', '')}::{case.get('name', '')}"
                    message = (node.get("message") or "").strip()
                    detail = (node.text or "").strip().splitlines()
                    tail = "\n".join(detail[-8:])
                    yield (name, message, tail)


def main(argv) -> int:
    found = list(failures(argv[1:]))
    for name, message, tail in found[:MAX_ANNOTATIONS]:
        body = (message + ("\n" + tail if tail else ""))[:MAX_MESSAGE]
        print(f"::error title={_escape_property('FAILED ' + name)}::{_escape(body)}")
    if len(found) > MAX_ANNOTATIONS:
        rest = ", ".join(n for n, _, _ in found[MAX_ANNOTATIONS:])
        print(f"::error title=more failures::{_escape(rest[:MAX_MESSAGE])}")
    print(f"{len(found)} failed or errored test(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
