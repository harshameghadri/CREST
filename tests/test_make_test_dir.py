"""The scratch-directory helper that CI uses to run the tests against an installed wheel."""

import importlib.util

import pytest


@pytest.fixture(scope="module")
def mtd(source_root):
    path = source_root / ".github" / "scripts" / "make_test_dir.py"
    if not path.exists():
        pytest.skip("no .github/scripts in this source tree")
    spec = importlib.util.spec_from_file_location("crest_make_test_dir", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def test_package_and_rust_sources_are_left_out(mtd, tmp_path):
    tree, dest = tmp_path / "tree", tmp_path / "dest"
    write(tree, {"crest/__init__.py": "", "src/lib.rs": "", "tests/test_a.py": "", "docs/a.md": "x",
                 "pyproject.toml": "", "target/debug/x": "junk", "crest/__pycache__/x.pyc": "junk"})
    mtd.main([str(dest), "--tree", str(tree)])
    got = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert got == ["docs/a.md", "pyproject.toml", "tests/test_a.py"]


def test_overlay_takes_tests_from_the_current_tree(mtd, tmp_path):
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "dest"
    write(old, {"tests/test_old.py": "old", "pyproject.toml": "old", "docs/a.md": "old docs", "crest/x.py": ""})
    write(new, {"tests/test_new.py": "new", "pyproject.toml": "new", "docs/a.md": "new docs"})
    mtd.main([str(dest), "--tree", str(old), "--overlay-from", str(new), "tests", "pyproject.toml"])
    assert (dest / "docs" / "a.md").read_text() == "old docs"
    assert (dest / "pyproject.toml").read_text() == "new"
    assert (dest / "tests" / "test_new.py").exists() and not (dest / "tests" / "test_old.py").exists()
    assert not (dest / "crest").exists()


def test_a_non_empty_destination_is_refused(mtd, tmp_path):
    (tmp_path / "dest").mkdir()
    (tmp_path / "dest" / "keep.txt").write_text("x")
    with pytest.raises(SystemExit):
        mtd.main([str(tmp_path / "dest"), "--tree", str(tmp_path)])
