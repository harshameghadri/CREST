"""scripts/assemble_changelog.py: fragments in changes/ become a CHANGELOG.md section."""

import pytest


@pytest.fixture(scope="module")
def asm(load_script):
    return load_script("assemble_changelog")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "changes").mkdir()
    (tmp_path / "changes" / "README.md").write_text("# not a fragment\n")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 0.4.0 (2026-10-02)\n\n### Changed\n- old entry\n")
    return tmp_path


def fragment(repo, name, text):
    (repo / "changes" / f"{name}.md").write_text(text)


def test_sections_are_grouped_and_inserted_above_the_newest_entry(repo, asm):
    fragment(repo, "fix-a", "- Fixed: `a` no longer fails.\n- Added: a new `b`.\n")
    fragment(repo, "chore-c", "- plain bullet\n  with a continuation line\n")
    section = asm.assemble(repo, "0.5.0", date="2026-12-01")
    text = (repo / "CHANGELOG.md").read_text()
    assert text.startswith("# Changelog\n\n## 0.5.0 (2026-12-01)\n")
    assert text.index("## 0.5.0") < text.index("## 0.4.0")
    assert text.endswith("- old entry\n")
    assert section.index("### Added") < section.index("### Changed") < section.index("### Fixed")
    assert "- `a` no longer fails." in section and "- A new `b`." in section
    assert "- plain bullet\n  with a continuation line" in section


def test_fragments_are_deleted_but_the_readme_stays(repo, asm):
    fragment(repo, "fix-a", "- Fixed: x.\n")
    asm.assemble(repo, "0.5.0", date="2026-12-01")
    assert sorted(p.name for p in (repo / "changes").iterdir()) == ["README.md"]


def test_dry_run_changes_nothing(repo, asm):
    fragment(repo, "fix-a", "- Fixed: x.\n")
    before = (repo / "CHANGELOG.md").read_text()
    section = asm.assemble(repo, "0.5.0", date="2026-12-01", dry_run=True)
    assert "## 0.5.0 (2026-12-01)" in section
    assert (repo / "CHANGELOG.md").read_text() == before and (repo / "changes" / "fix-a.md").exists()


@pytest.mark.parametrize(
    "text",
    ["", "no bullet here\n", "# Heading\n- ok\n", "- 1\n- 2\n- 3\n- 4\n- 5\n", "- Fixed:\n"],
)
def test_malformed_fragments_are_rejected_and_nothing_is_written(repo, asm, text):
    fragment(repo, "bad", text)
    before = (repo / "CHANGELOG.md").read_text()
    with pytest.raises(asm.FragmentError):
        asm.assemble(repo, "0.5.0", date="2026-12-01")
    assert (repo / "CHANGELOG.md").read_text() == before and (repo / "changes" / "bad.md").exists()


def test_existing_version_and_empty_directory_are_rejected(repo, asm):
    with pytest.raises(asm.FragmentError, match="no change fragments"):
        asm.assemble(repo, "0.5.0")
    fragment(repo, "fix-a", "- Fixed: x.\n")
    with pytest.raises(asm.FragmentError, match="already has a section"):
        asm.assemble(repo, "0.4.0")


def test_cli(repo, asm, capsys):
    fragment(repo, "fix-a", "- Fixed: x.\n")
    assert asm.main(["--check", "--root", str(repo)]) == 0
    assert asm.main(["0.5.0", "--date", "2026-12-01", "--root", str(repo)]) == 0
    assert asm.main(["0.5.0", "--root", str(repo)]) == 1  # no fragments left
    assert "no change fragments" in capsys.readouterr().err


def test_the_fragments_in_the_repository_are_valid(source_root, asm):
    for path in asm.fragment_files(source_root / "changes"):
        asm.parse_fragment(path)
