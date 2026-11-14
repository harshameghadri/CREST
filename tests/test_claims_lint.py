"""Documentation lint: claims that were withdrawn must not reappear, and the log-log fit
behind any scaling statement is pinned.

The first test scans README.md, CHANGELOG.md, CONTRIBUTING.md, docs/*.md and the changelog fragments in changes/. It needs no build
and no data. The fit tests import ``bench/paper/summarize.py``.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "paper"))

SCANNED = [p for p in (REPO / "README.md", REPO / "CHANGELOG.md", REPO / "CONTRIBUTING.md",
                       *sorted((REPO / "docs").glob("*.md")), *sorted((REPO / "changes").glob("*.md")))
           if p.exists()]

# a speed-up ("12x faster than R", "63-78x faster than R DESeq2", "41-79x faster than pydeseq2")
SPEEDUP_VS_R = r"\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?\s*[×x]\s+faster\s+than\s+(?:single-threaded\s+)?(?:R\b|R DESeq2|pydeseq2)"

WITHDRAWN = {
    "scaling exponent of the form cells^<number>": r"cells\^\s*\d",
    "log-log fit 'over all six datasets'": r"all six datasets",
    "7 s on 4 cores vs R 131 s / pydeseq2 202 s": r"\b7 s on 4 cores|\b131 s\b|\b202 s\b|\b7\.2 s\b",
    "~18x / ~28x DESeq2 speed-ups": r"~\s*(?:18|28|50)\s*[×x]",
    "speed-up of DESeq2 versus R or pydeseq2": SPEEDUP_VS_R,
    "63-78x / 41-79x DESeq2 speed-ups": r"63\s*[–-]\s*78|41\s*[–-]\s*79",
}

# --------------------------------------------------------------------------- fit helper values
ALPHA = 0.8
N_FIT_POINTS = 6
EXACT = 1e-9


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_scanned_files_exist():
    assert (REPO / "README.md").exists() and (REPO / "CHANGELOG.md").exists()
    assert len(SCANNED) >= 6, SCANNED


@pytest.mark.baseline_fails(ref="rc0", raises=AssertionError, match="withdrawn claim")
@pytest.mark.parametrize("claim", sorted(WITHDRAWN))
def test_withdrawn_claims_do_not_reappear(claim):
    pattern = re.compile(WITHDRAWN[claim], re.I)
    hits = [f"{p.relative_to(REPO)}:{i}: {line.strip()[:90]}" for p in SCANNED
            for i, line in enumerate(_text(p).splitlines(), 1) if pattern.search(line)]
    assert not hits, f"withdrawn claim ({claim}) found:\n" + "\n".join(hits)


def test_lint_patterns_match_the_wording_they_replace():
    """The patterns must catch the old sentences (otherwise a silent pattern error passes the lint)."""
    old = {
        r"cells\^\s*\d": "CREST scales as cells^0.74 vs scanpy's cells^0.91",
        r"all six datasets": "log-log fit over all six datasets",
        WITHDRAWN["7 s on 4 cores vs R 131 s / pydeseq2 202 s"]: "takes 7 s on 4 cores (R DESeq2, single-threaded: 131 s",
        WITHDRAWN["~18x / ~28x DESeq2 speed-ups"]: "is ~18× faster than R and ~28× faster than pydeseq2",
        SPEEDUP_VS_R: "63–78× faster than R DESeq2 (single-threaded R vs CREST",
        WITHDRAWN["63-78x / 41-79x DESeq2 speed-ups"]: "41–79× faster than pydeseq2 (all cores)",
    }
    for pattern, text in old.items():
        assert re.search(pattern, text, re.I), (pattern, text)
    assert re.search(SPEEDUP_VS_R, "41-79x faster than pydeseq2")
    assert not re.search(SPEEDUP_VS_R, "10.28× faster than scanpy at 64 threads")


# --------------------------------------------------------------------------- log-log fit
@pytest.fixture(scope="module")
def loglog_fit():
    pytest.importorskip("polars")
    pytest.importorskip("scipy")
    import summarize
    return summarize.loglog_fit


def test_loglog_fit_recovers_a_known_power_law(loglog_fit):
    n = np.logspace(3, 6, N_FIT_POINTS)
    fit = loglog_fit(n, 2e-3 * n**ALPHA)
    assert fit["alpha"] == pytest.approx(ALPHA, abs=EXACT)
    assert fit["n_points"] == N_FIT_POINTS
    assert fit["r2"] == pytest.approx(1.0, abs=EXACT)
    assert fit["alpha_ci95_lo"] <= ALPHA + EXACT and fit["alpha_ci95_hi"] >= ALPHA - EXACT
    assert fit["alpha_ci95_hi"] - fit["alpha_ci95_lo"] < 1e-6        # noise-free: a vanishing interval


def test_loglog_fit_interval_is_wide_with_three_points_and_undefined_with_two(loglog_fit):
    n = np.array([6.8e4, 1e5, 2e5])
    rng = np.random.default_rng(0)
    t = 1e-3 * n**0.8 * np.exp(rng.normal(0, 0.05, size=3))
    fit = loglog_fit(n, t)
    assert fit["n_points"] == 3
    assert (fit["alpha_ci95_hi"] - fit["alpha_ci95_lo"]) > 0.5      # one residual degree of freedom
    two = loglog_fit(n[:2], t[:2])
    assert two["n_points"] == 2 and np.isnan(two["alpha_ci95_lo"])
