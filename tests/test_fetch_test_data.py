"""scripts/fetch_test_data.py: checksum-verified, cached downloads (exercised with local files)."""

import hashlib

import pytest


@pytest.fixture(scope="module")
def ftd(load_script):
    return load_script("fetch_test_data")


@pytest.fixture
def spec(tmp_path):
    payload = b"crest test payload" * 1000
    src = tmp_path / "origin.bin"
    src.write_bytes(payload)
    return {
        "filename": "data.bin",
        "url": src.as_uri(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }, src


def test_download_verify_and_cache(ftd, spec, tmp_path, monkeypatch):
    spec, src = spec
    cache = tmp_path / "cache"
    path = ftd.fetch("x", cache, spec=spec, log=lambda *_: None)
    assert path == cache / "data.bin" and path.read_bytes() == src.read_bytes()
    # second call must not touch the network
    monkeypatch.setattr(ftd, "download", lambda *a, **k: pytest.fail("downloaded a cached file again"))
    assert ftd.fetch("x", cache, spec=spec, log=lambda *_: None) == path


def test_corrupt_cache_entry_is_replaced(ftd, spec, tmp_path):
    spec, src = spec
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "data.bin").write_bytes(b"truncated")
    path = ftd.fetch("x", tmp_path / "cache", spec=spec, log=lambda *_: None)
    assert path.read_bytes() == src.read_bytes()


def test_checksum_mismatch_raises_and_leaves_no_file(ftd, spec, tmp_path):
    spec, _ = spec
    spec = dict(spec, sha256="0" * 64)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        ftd.fetch("x", tmp_path / "cache", spec=spec, log=lambda *_: None)
    assert not (tmp_path / "cache" / "data.bin").exists()


def test_cache_directory_comes_from_the_environment(ftd, tmp_path, monkeypatch):
    monkeypatch.setenv("CREST_TEST_DATA", str(tmp_path / "from_env"))
    assert ftd.cache_dir() == tmp_path / "from_env"
    assert ftd.cache_dir(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("CREST_TEST_DATA")
    assert ftd.cache_dir().name == "crest-test-data"


def test_registered_checksums_are_well_formed(ftd):
    assert set(ftd.DATASETS) == {"pbmc3k", "pbmc10k"}
    for spec in ftd.DATASETS.values():
        assert len(spec["sha256"]) == 64 and int(spec["sha256"], 16) >= 0
        assert spec["url"].startswith("https://cf.10xgenomics.com/")
