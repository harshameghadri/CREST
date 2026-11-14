"""The version is written in several files; they must agree, and the bump script must keep them so."""

import re
import shutil

import pytest

VERSION_FILES = ["Cargo.toml", "Cargo.lock", "crest/__init__.py", "CITATION.cff",
                 "README.md", "docs/benchmarks.md", "docs/installation.md"]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-(alpha|beta|rc)\.\d+)?$")


@pytest.fixture(scope="module")
def bump(load_script):
    return load_script("bump_version")


def test_every_file_carries_the_same_version(source_root, bump):
    versions = bump.read_versions(source_root)
    assert "Cargo.toml" in versions and "crest/__init__.py" in versions and "CITATION.cff" in versions
    assert any(k.startswith("Cargo.lock") for k in versions), "Cargo.lock has no crest entry"
    assert all(v for v in versions.values()), versions
    assert len({bump.normalize(v) for v in versions.values()}) == 1, versions


def test_cargo_versions_are_semver_and_python_versions_are_pep440(source_root, bump):
    versions = bump.read_versions(source_root)
    for label, value in versions.items():
        if label.startswith("Cargo"):
            assert SEMVER.match(value), f"{label}: {value!r} is not a Cargo version"
        else:
            assert value == bump.to_pep440(value), f"{label}: {value!r} is not in PEP 440 form"


def test_pinned_tags_in_readme_and_docs_equal_the_version_or_are_absent(source_root, bump):
    version = bump.normalize(bump.read_versions(source_root)["Cargo.toml"])
    pins = bump.read_pins(source_root)
    wrong = {k: v for k, v in pins.items() if bump.normalize(v) != version}
    assert not wrong, f"pinned tags differ from {version}: {wrong}"


def test_the_imported_package_reports_the_same_version(source_root, bump):
    crest = pytest.importorskip("crest")
    assert bump.normalize(crest.__version__) == bump.normalize(bump.read_versions(source_root)["Cargo.toml"])


# --------------------------------------------------------------------------- the script itself
@pytest.mark.parametrize(
    "given, pep440, cargo",
    [
        ("0.5.0", "0.5.0", "0.5.0"),
        ("0.5.0rc1", "0.5.0rc1", "0.5.0-rc.1"),
        ("0.5.0-rc.1", "0.5.0rc1", "0.5.0-rc.1"),
        ("v1.2.3a4", "1.2.3a4", "1.2.3-alpha.4"),
        ("1.2.3-beta.2", "1.2.3b2", "1.2.3-beta.2"),
        ("0.6.0.dev3", "0.6.0.dev3", "0.6.0-dev.3"),
    ],
)
def test_version_spellings(bump, given, pep440, cargo):
    assert bump.to_pep440(given) == pep440
    assert bump.to_cargo(given) == cargo
    assert bump.normalize(cargo) == bump.normalize(pep440)


@pytest.mark.parametrize("bad", ["", "1.2", "x.y.z", "1.2.3.4", "1.2.3rc", "1.2.3-foo.1"])
def test_bad_versions_are_rejected(bump, bad):
    with pytest.raises(ValueError):
        bump.parse(bad)


@pytest.fixture
def tree(source_root, tmp_path):
    for rel in VERSION_FILES:
        src = source_root / rel
        if src.exists():
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, tmp_path / rel)
    return tmp_path


@pytest.mark.parametrize("target", ["0.5.0rc1", "0.5.0", "1.0.0-beta.2"])
def test_bump_rewrites_every_location(tree, bump, target):
    changed = bump.bump(tree, target, date="2026-12-01")
    assert {"Cargo.toml", "Cargo.lock", "crest/__init__.py", "CITATION.cff"} <= set(changed)
    versions = bump.read_versions(tree)
    assert {bump.normalize(v) for v in versions.values()} == {bump.normalize(target)}
    assert versions["Cargo.toml"] == bump.to_cargo(target)
    assert versions["crest/__init__.py"] == bump.to_pep440(target)
    assert all(bump.normalize(v) == bump.normalize(target) for v in bump.read_pins(tree).values())
    cff = (tree / "CITATION.cff").read_text()
    assert 'date-released: "2026-12-01"' in cff and f'version: "{bump.to_pep440(target)}"' in cff


def test_bump_changes_nothing_else(tree, bump, source_root):
    bump.bump(tree, "0.5.0", date="2026-12-01")
    old = (source_root / "Cargo.toml").read_text().splitlines()
    new = (tree / "Cargo.toml").read_text().splitlines()
    assert len(old) == len(new)
    assert [a for a, b in zip(old, new) if a != b] == [a for a in old if a.startswith("version")] != []
    # dependency versions in Cargo.lock are untouched
    lock_old = (source_root / "Cargo.lock").read_text().splitlines()
    lock_new = (tree / "Cargo.lock").read_text().splitlines()
    assert len(lock_old) == len(lock_new)
    assert sum(a != b for a, b in zip(lock_old, lock_new)) <= 2


def test_bump_is_idempotent_and_dry_run_writes_nothing(tree, bump):
    before = {rel: (tree / rel).read_text() for rel in VERSION_FILES if (tree / rel).exists()}
    assert bump.bump(tree, "0.5.0", date="2026-12-01", dry_run=True)
    assert before == {rel: (tree / rel).read_text() for rel in before}
    bump.bump(tree, "0.5.0", date="2026-12-01")
    assert bump.bump(tree, "0.5.0", date="2026-12-01") == []


def test_bump_cli_reports_errors(bump, tree, capsys):
    assert bump.main(["not-a-version", "--root", str(tree)]) == 1
    assert "not a version" in capsys.readouterr().err
