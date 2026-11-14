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


def make_pbmc3k_archive(tmp_path, ftd):
    import tarfile

    src = tmp_path / "src" / "filtered_gene_bc_matrices" / "hg19"
    src.mkdir(parents=True)
    for name in ("matrix.mtx", "genes.tsv", "barcodes.tsv"):
        (src / name).write_text("x\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    with tarfile.open(cache / ftd.DATASETS["pbmc3k"]["filename"], "w:gz") as tf:
        tf.add(tmp_path / "src" / "filtered_gene_bc_matrices", arcname="filtered_gene_bc_matrices")
    return cache


def test_pbmc3k_is_extracted_once_and_exported(ftd, tmp_path):
    cache = make_pbmc3k_archive(tmp_path, ftd)
    mtx = ftd.extract_pbmc3k(cache, log=lambda *_: None)
    assert (mtx / "matrix.mtx").exists() and mtx.name == "hg19"
    (mtx / "matrix.mtx").write_text("kept")
    assert ftd.extract_pbmc3k(cache, log=lambda *_: None) == mtx
    assert (mtx / "matrix.mtx").read_text() == "kept"  # not unpacked a second time
    env = ftd.environment(cache, derive=False, log=lambda *_: None)
    assert env == {"CREST_TEST_DATA": str(cache), "CREST_TEST_PBMC3K": str(mtx)}


def test_archive_with_escaping_paths_is_refused(ftd, tmp_path):
    import io
    import tarfile

    cache = tmp_path / "cache"
    cache.mkdir()
    with tarfile.open(cache / ftd.DATASETS["pbmc3k"]["filename"], "w:gz") as tf:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError, match="unsafe path"):
        ftd.extract_pbmc3k(cache, log=lambda *_: None)
    assert not (tmp_path / "evil.txt").exists()


def test_github_env_file_is_written(ftd, tmp_path, monkeypatch, capsys):
    cache = make_pbmc3k_archive(tmp_path, ftd)
    monkeypatch.setattr(ftd, "fetch", lambda name, dest=None, spec=None, log=print: cache / ftd.DATASETS[name]["filename"])
    env_file = tmp_path / "env.txt"
    env_file.write_text("EXISTING=1\n")
    assert ftd.main(["pbmc3k", "--dest", str(cache), "--github-env", str(env_file), "--print-env"]) == 0
    lines = env_file.read_text().splitlines()
    assert lines[0] == "EXISTING=1"  # appended, not overwritten
    values = dict(line.split("=", 1) for line in lines[1:])
    assert values["CREST_TEST_DATA"] == str(cache)
    assert values["CREST_TEST_PBMC3K"].endswith("hg19")
    assert "CREST_TEST_PBMC_CONNECTIVITIES" not in values  # only with --derive
    assert f"export CREST_TEST_PBMC3K={values['CREST_TEST_PBMC3K']}" in capsys.readouterr().out
