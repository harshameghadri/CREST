"""Shared pytest plugins for the CREST test suite.

Two small plugins live here.

``baseline_fails`` marker
    ``@pytest.mark.baseline_fails(ref="v0.4.0", raises=IndexError, match="out of bounds")``
    documents that a test describes a defect that exists in an earlier build. A normal run
    ignores the marker. With ``--baseline-ref=<ref>`` (or the environment variable
    ``CREST_BASELINE_REF``) the run is inverted for that reference build:

    * every test that does not carry a marker for ``<ref>`` is deselected;
    * a marked test passes only if its body raises ``raises`` (and, when ``match`` is given,
      ``re.search(match, str(exception))`` succeeds);
    * a marked test that passes, raises a different exception, or is skipped fails;
    * collection errors fail the run, and so does a module that is skipped at collection
      time, because tests hidden that way could never be checked;
    * a run in which no test is marked for ``<ref>`` succeeds.

``--skip-allowlist=<file>``
    Fails the run when a test is skipped that is not listed in ``<file>``. Each line is
    ``<node id>  # <reason>``; blank lines and lines starting with ``#`` are ignored. A module
    that is skipped as a whole (``pytest.importorskip`` at module level) is listed by its path.
"""

import os
import re
from pathlib import Path

import pytest

MARKER = "baseline_fails"


# --------------------------------------------------------------------------- options
def pytest_addoption(parser):
    group = parser.getgroup("crest", "CREST test infrastructure")
    group.addoption(
        "--baseline-ref",
        action="store",
        default=os.environ.get("CREST_BASELINE_REF") or None,
        help="run only tests marked baseline_fails for this reference build and require them to fail "
        "as declared (default: $CREST_BASELINE_REF)",
    )
    group.addoption(
        "--skip-allowlist",
        action="store",
        default=None,
        help="file listing the node ids that are allowed to be skipped; any other skip fails the run",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "baseline_fails(ref, raises, match=None): the test describes a defect present in the "
        "reference build `ref`; under --baseline-ref=<ref> it must raise `raises` (matching the "
        "regex `match` against str(exception) when given). Ignored in normal runs.",
    )
    plugin = _CrestPlugin(config)
    config._crest_plugin = plugin
    config.pluginmanager.register(plugin, "crest-test-infrastructure")


# --------------------------------------------------------------------------- helpers
def _parse_mark(mark, nodeid):
    """Return (ref, raises, match) from one baseline_fails mark, validating the arguments."""
    names = ("ref", "raises", "match")
    if len(mark.args) > 3:
        raise pytest.UsageError(f"{nodeid}: baseline_fails takes at most 3 positional arguments")
    values = dict(zip(names, mark.args))
    for key, val in mark.kwargs.items():
        if key not in names:
            raise pytest.UsageError(f"{nodeid}: baseline_fails got an unexpected argument {key!r}")
        if key in values:
            raise pytest.UsageError(f"{nodeid}: baseline_fails got {key!r} twice")
        values[key] = val
    ref, raises, match = values.get("ref"), values.get("raises"), values.get("match")
    if not isinstance(ref, str) or not ref:
        raise pytest.UsageError(f"{nodeid}: baseline_fails needs ref=<non-empty string>")
    classes = raises if isinstance(raises, tuple) else (raises,)
    if not classes or not all(isinstance(c, type) and issubclass(c, BaseException) for c in classes):
        raise pytest.UsageError(f"{nodeid}: baseline_fails needs raises=<exception class or tuple of classes>")
    if match is not None:
        if not isinstance(match, str):
            raise pytest.UsageError(f"{nodeid}: baseline_fails match must be a regular expression string")
        re.compile(match)
    return ref, classes, match


def _spec_for(item, ref):
    """The (ref, raises, match) of the item's marker for `ref`, or None."""
    for mark in item.iter_markers(MARKER):
        spec = _parse_mark(mark, item.nodeid)
        if spec[0] == ref:
            return spec
    return None


def _read_allowlist(path):
    allowed = {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        node, sep, reason = line.partition(" #")
        node, reason = node.strip(), reason.strip()
        if not sep or not reason:
            raise pytest.UsageError(f"{path}:{n}: every entry needs a reason, written as '<node id>  # <reason>'")
        allowed[node] = reason
    return allowed


def _describe(classes):
    return " or ".join(c.__name__ for c in classes)


# --------------------------------------------------------------------------- the plugin
class _CrestPlugin:
    def __init__(self, config):
        self.config = config
        self.ref = config.getoption("--baseline-ref")
        allowlist = config.getoption("--skip-allowlist")
        self.allowed = _read_allowlist(allowlist) if allowlist else None
        self.skipped = set()
        self.collect_skipped = set()

    # -- selection ---------------------------------------------------------------------
    def pytest_collection_modifyitems(self, config, items):
        if not self.ref:
            return
        keep, drop = [], []
        for item in items:
            (keep if _spec_for(item, self.ref) else drop).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
        items[:] = keep

    def pytest_collectreport(self, report):
        if report.skipped:
            self.collect_skipped.add(report.nodeid)

    # -- outcome inversion ---------------------------------------------------------------
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if not self.ref:
            return
        spec = _spec_for(item, self.ref)
        if spec is None:
            return
        _, classes, match = spec
        expected = _describe(classes) + (f" matching {match!r}" if match else "")
        if report.when == "setup" and report.skipped:
            report.outcome = "failed"
            report.longrepr = f"baseline {self.ref}: test was skipped during setup; a skipped test proves nothing"
        elif report.when == "call":
            if report.skipped and not hasattr(report, "wasxfail"):
                report.outcome = "failed"
                report.longrepr = f"baseline {self.ref}: test was skipped; a skipped test proves nothing"
            elif report.passed:
                report.outcome = "failed"
                report.longrepr = (
                    f"baseline {self.ref}: expected the test to fail with {expected}, but it passed"
                )
            elif report.failed and call.excinfo is not None:
                exc = call.excinfo.value
                if not isinstance(exc, classes):
                    report.longrepr = (
                        f"baseline {self.ref}: expected {expected}, got {type(exc).__name__}: {exc}"
                    )
                elif match is not None and re.search(match, str(exc)) is None:
                    report.longrepr = (
                        f"baseline {self.ref}: {type(exc).__name__} was raised but its message "
                        f"{str(exc)!r} does not match {match!r}"
                    )
                else:
                    report.outcome = "passed"
                    report.longrepr = None

    # -- skip bookkeeping ----------------------------------------------------------------
    def pytest_runtest_logreport(self, report):
        if report.skipped and not hasattr(report, "wasxfail"):
            self.skipped.add(report.nodeid)

    def _unexpected_skips(self):
        if self.allowed is None:
            return []
        seen = self.skipped | self.collect_skipped
        return sorted(n for n in seen if n not in self.allowed)

    def pytest_terminal_summary(self, terminalreporter):
        bad = self._unexpected_skips()
        if bad:
            terminalreporter.section("skipped tests that are not on the allowlist", red=True)
            for node in bad:
                terminalreporter.write_line(node)
            terminalreporter.write_line(
                f"{len(bad)} unexpected skip(s); install the missing dependency or list the node id in "
                f"{self.config.getoption('--skip-allowlist')} with a reason"
            )
        if self.ref and self.collect_skipped:
            terminalreporter.section(f"modules skipped at collection (baseline {self.ref})", red=True)
            for node in sorted(self.collect_skipped):
                terminalreporter.write_line(node)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if self.ref and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
            # nothing is marked for this reference build: that is a pass
            session.exitstatus = pytest.ExitCode.OK
            exitstatus = pytest.ExitCode.OK
        failing = bool(self._unexpected_skips()) or bool(self.ref and self.collect_skipped)
        if failing and session.exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def source_root():
    """The repository root, for tests that read repository files or run scripts/.

    $CREST_SOURCE_DIR wins; otherwise the parent of this directory. When the tests run from a
    copy (or from a source distribution, which carries no scripts/), the test is skipped.
    """
    env = os.environ.get("CREST_SOURCE_DIR")
    root = Path(env) if env else Path(__file__).resolve().parents[1]
    if not (root / "Cargo.toml").exists() or not (root / "scripts").is_dir():
        pytest.skip(f"no source checkout at {root} (set CREST_SOURCE_DIR)")
    return root


@pytest.fixture(scope="session")
def load_script(source_root):
    """Import a file from scripts/ as a module: load_script("bump_version")."""
    import importlib.util

    def load(name):
        spec = importlib.util.spec_from_file_location(f"crest_scripts_{name}", source_root / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
