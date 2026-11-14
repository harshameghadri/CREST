"""Unit tests for the baseline_fails marker and the skip allowlist (tests/conftest.py).

Each scenario writes a small test module into a scratch directory, installs a copy of
tests/conftest.py there, and runs pytest in a subprocess.
"""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = Path(__file__).with_name("conftest.py").read_text()

HEADER = "import pytest\n\n"


@pytest.fixture
def suite(pytester, monkeypatch):
    monkeypatch.delenv("CREST_BASELINE_REF", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    pytester.makeconftest(CONFTEST)

    def make(body, **extra):
        pytester.makepyfile(test_case=HEADER + body, **extra)
        return pytester

    return make


def run(pytester, *args):
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", *args)


def mark(raises="ValueError", ref="v0.4.0", match=None):
    extra = f", match={match!r}" if match else ""
    return f"@pytest.mark.baseline_fails(ref={ref!r}, raises={raises}{extra})\n"


# --------------------------------------------------------------------------- baseline mode
def test_declared_exception_passes(suite):
    pt = suite(mark("IndexError", match="out of range") + "def test_a():\n    raise IndexError('index 7 is out of range')\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    result.assert_outcomes(passed=1)
    assert result.ret == 0


def test_assertion_error_can_be_declared(suite):
    pt = suite(mark("AssertionError") + "def test_a():\n    assert 1 == 2\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    result.assert_outcomes(passed=1)


def test_tuple_of_exception_classes(suite):
    pt = suite(mark("(KeyError, ZeroDivisionError)") + "def test_a():\n    1 / 0\n")
    run(pt, "--baseline-ref=v0.4.0").assert_outcomes(passed=1)


def test_failed_pytest_raises_block_can_be_declared(suite):
    pt = suite(
        mark("pytest.fail.Exception", match="DID NOT RAISE")
        + "def test_a():\n    with pytest.raises(ValueError):\n        pass\n"
    )
    run(pt, "--baseline-ref=v0.4.0").assert_outcomes(passed=1)


def test_test_that_passes_on_the_baseline_fails(suite):
    pt = suite(mark() + "def test_a():\n    assert True\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    result.assert_outcomes(failed=1)
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*but it passed*"])


def test_wrong_exception_class_fails(suite):
    pt = suite(mark("ValueError") + "def test_a():\n    raise KeyError('x')\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*expected ValueError, got KeyError*"])


def test_wrong_message_fails(suite):
    pt = suite(mark("ValueError", match="needle") + "def test_a():\n    raise ValueError('haystack')\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*does not match*"])


def test_skip_in_body_fails(suite):
    pt = suite(mark() + "def test_a():\n    pytest.skip('not today')\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    result.assert_outcomes(failed=1)
    assert result.ret == 1


def test_skipif_marker_fails(suite):
    pt = suite(mark() + "@pytest.mark.skipif(True, reason='never')\ndef test_a():\n    raise ValueError('unreachable')\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    # a skip decided during setup is reported as an error of the test; the run fails either way
    result.assert_outcomes(errors=1)
    assert result.ret == 1


def test_other_refs_and_unmarked_tests_are_deselected(suite):
    pt = suite(
        mark("RuntimeError", ref="rc0") + "def test_rc0():\n    raise RuntimeError('rc0 only')\n\n"
        + mark("ValueError") + "def test_v040():\n    raise ValueError('boom')\n\n"
        + "def test_unmarked():\n    assert False\n"
    )
    result = run(pt, "--baseline-ref=v0.4.0")
    result.assert_outcomes(passed=1, deselected=2)
    result = run(pt, "--baseline-ref=rc0")
    result.assert_outcomes(passed=1, deselected=2)


def test_summary_line_counts_selected_and_confirmed_tests(suite, monkeypatch):
    pt = suite(
        mark() + "def test_a():\n    raise ValueError('x')\n\n"
        + mark() + "def test_b():\n    pass\n\n"
        + mark("RuntimeError", ref="rc0") + "def test_c():\n    raise RuntimeError('x')\n"
    )
    result = run(pt, "--baseline-ref=v0.4.0")
    result.stdout.fnmatch_lines(["baseline v0.4.0: 2 marked test(s) selected, 1 failed as declared"])
    assert "::notice" not in result.stdout.str()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    result = run(pt, "--baseline-ref=rc0")
    result.stdout.fnmatch_lines(["::notice title=baseline rc0::baseline rc0: 1 marked test(s) selected, 1 failed as declared"])


def test_no_marked_tests_for_the_ref_is_a_pass(suite):
    pt = suite(mark("RuntimeError", ref="rc0") + "def test_rc0():\n    raise RuntimeError('rc0 only')\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    assert result.ret == 0
    result = run(suite("def test_unmarked():\n    assert False\n"), "--baseline-ref=v0.4.0")
    assert result.ret == 0


def test_reference_from_environment(suite, monkeypatch):
    pt = suite(mark() + "def test_a():\n    raise ValueError('x')\n\ndef test_b():\n    assert False\n")
    monkeypatch.setenv("CREST_BASELINE_REF", "v0.4.0")
    result = run(pt)
    result.assert_outcomes(passed=1, deselected=1)


def test_collection_error_fails_the_run(suite):
    pt = suite(mark() + "def test_a():\n    raise ValueError('x')\n", test_broken="import module_that_does_not_exist\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*module_that_does_not_exist*"])


def test_module_skipped_at_collection_fails_the_run(suite):
    pt = suite(
        mark() + "def test_a():\n    raise ValueError('x')\n",
        test_skipped_module="import pytest\npytest.importorskip('module_that_does_not_exist')\n",
    )
    result = run(pt, "--baseline-ref=v0.4.0")
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*skipped at collection*", "*test_skipped_module.py*"])


@pytest.mark.parametrize(
    "marker",
    [
        "@pytest.mark.baseline_fails(raises=ValueError)\n",
        "@pytest.mark.baseline_fails(ref='v0.4.0')\n",
        "@pytest.mark.baseline_fails(ref='v0.4.0', raises='ValueError')\n",
        "@pytest.mark.baseline_fails(ref='v0.4.0', raises=ValueError, match='(')\n",
        "@pytest.mark.baseline_fails(ref='v0.4.0', raises=ValueError, reason='x')\n",
    ],
)
def test_malformed_marker_is_a_usage_error(suite, marker):
    pt = suite(marker + "def test_a():\n    raise ValueError('x')\n")
    result = run(pt, "--baseline-ref=v0.4.0")
    assert result.ret != 0


# --------------------------------------------------------------------------- normal mode
def test_normal_mode_ignores_the_marker(suite):
    pt = suite(
        mark() + "def test_passes():\n    assert True\n\n"
        + mark() + "def test_raises():\n    raise ValueError('real failure')\n\n"
        + mark() + "def test_skips():\n    pytest.skip('x')\n"
    )
    result = run(pt)
    result.assert_outcomes(passed=1, failed=1, skipped=1, deselected=0)


def test_marker_is_registered_under_strict_markers(suite):
    pt = suite(mark() + "def test_a():\n    assert True\n")
    result = run(pt, "--strict-markers")
    result.assert_outcomes(passed=1)


# --------------------------------------------------------------------------- skip allowlist
SKIPPER = "def test_skips():\n    pytest.skip('missing dependency')\n\ndef test_ok():\n    assert True\n"


def test_unlisted_skip_fails_the_run(suite):
    pt = suite(SKIPPER)
    pt.makefile(".txt", skips="# nothing is allowed to skip\n")
    result = run(pt, "--skip-allowlist=skips.txt")
    result.assert_outcomes(passed=1, skipped=1)
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*not on the allowlist*", "test_case.py::test_skips"])


def test_unlisted_skip_is_annotated_in_github_actions(suite, monkeypatch):
    pt = suite(SKIPPER)
    pt.makefile(".txt", skips="")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    result = run(pt, "--skip-allowlist=skips.txt")
    result.stdout.fnmatch_lines(["::error title=skipped test not on the allowlist::test_case.py::test_skips"])


def test_listed_skip_is_accepted(suite):
    pt = suite(SKIPPER)
    pt.makefile(".txt", skips="test_case.py::test_skips  # needs a package that is not installed here\n")
    result = run(pt, "--skip-allowlist=skips.txt")
    assert result.ret == 0


def test_module_level_skip_is_listed_by_path(suite):
    pt = suite("def test_ok():\n    assert True\n", test_mod="import pytest\npytest.importorskip('module_that_does_not_exist')\n")
    pt.makefile(".txt", skips="")
    assert run(pt, "--skip-allowlist=skips.txt").ret == 1
    pt.makefile(".txt", skips="test_mod.py  # optional dependency\n")
    assert run(pt, "--skip-allowlist=skips.txt").ret == 0


def test_expected_failures_are_not_skips(suite):
    pt = suite("@pytest.mark.xfail(reason='known')\ndef test_x():\n    assert False\n")
    pt.makefile(".txt", skips="")
    result = run(pt, "--skip-allowlist=skips.txt")
    assert result.ret == 0


def test_allowlist_entries_need_a_reason(suite):
    pt = suite(SKIPPER)
    pt.makefile(".txt", skips="test_case.py::test_skips\n")
    result = run(pt, "--skip-allowlist=skips.txt")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*needs a reason*"])


def test_without_an_allowlist_skips_are_free(suite):
    assert run(suite(SKIPPER)).ret == 0
