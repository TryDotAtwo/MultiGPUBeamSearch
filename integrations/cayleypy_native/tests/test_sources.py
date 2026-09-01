import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from cayleypy_native import NativeOptions, NativeBackendError, NativeUnavailable
from cayleypy_native import sources


def archive_blob(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, value in entries:
            item = tarfile.TarInfo(name)
            if value is None:
                item.type = tarfile.SYMTYPE
                item.linkname = "../../outside"
                archive.addfile(item)
            else:
                item.size = len(value)
                archive.addfile(item, io.BytesIO(value))
    return stream.getvalue()


@pytest.fixture
def pinned(monkeypatch):
    native = archive_blob([(f"fixture/{p}", b"native") for p in
                           ("CMakeLists.txt", "tools/production_runner.cu", "tools/export_stream1_mlp.py")])
    cutlass = archive_blob([("fixture/include/cutlass/cutlass.h", b"cutlass")])
    payloads = {"native": native, "cutlass": cutlass}
    for name, blob in payloads.items():
        monkeypatch.setattr(sources, f"{name.upper()}_SOURCE", sources.SourceArchive(
            name, "unused/repository", "1" * 40, hashlib.sha256(blob).hexdigest(), "fixture"))
    calls = []

    def download(spec, target):
        calls.append(spec.name)
        target.write_bytes(payloads[spec.name])
    monkeypatch.setattr(sources, "_download", download)
    return calls


def test_explicit_setup_and_offline_reuse_preserve_configuration(tmp_path, pinned):
    options = NativeOptions(cache_dir=tmp_path, devices=(1, 0), build_jobs=3)
    configured = sources.setup_sources(options=options)
    assert configured.source_dir.is_dir() and configured.cutlass_dir.is_dir()
    assert configured.devices == (1, 0) and configured.build_jobs == 3
    assert options.source_dir is None  # Caller config is immutable.
    assert pinned == ["native", "cutlass"]
    assert sources.setup_sources(options=options, offline=True) == configured
    assert pinned == ["native", "cutlass"]


def test_offline_miss_does_not_create_or_download(tmp_path, pinned):
    cache = tmp_path / "absent"
    with pytest.raises(NativeUnavailable, match="not cached"):
        sources.setup_sources(cache_dir=cache, offline=True)
    assert not cache.exists() and not pinned


def test_bad_explicit_path_is_checked_before_any_download(tmp_path, pinned):
    with pytest.raises(NativeUnavailable, match="cutlass_dir"):
        sources.setup_sources(cache_dir=tmp_path, cutlass_dir=tmp_path / "missing")
    assert not pinned


def test_explicit_paths_work_offline(tmp_path, pinned):
    cached = sources.setup_sources(cache_dir=tmp_path)
    pinned.clear()
    configured = sources.setup_sources(cache_dir=tmp_path / "unused", source_dir=cached.source_dir,
                                      cutlass_dir=cached.cutlass_dir, offline=True)
    assert configured.source_dir == cached.source_dir
    assert not pinned and not configured.cache_dir.exists()


@pytest.mark.parametrize("damage", ["modify", "extra", "archive", "missing"])
def test_modified_cache_is_rejected_without_repair_or_download(tmp_path, pinned, damage):
    configured = sources.setup_sources(cache_dir=tmp_path)
    if damage == "modify":
        (configured.source_dir / "CMakeLists.txt").write_text("changed")
    elif damage == "extra":
        (configured.source_dir / "extra.cu").write_text("changed")
    elif damage == "archive":
        (configured.source_dir.parent / "source.tar.gz").write_bytes(b"bad")
    else:
        (configured.source_dir / "CMakeLists.txt").unlink()
    with pytest.raises(NativeBackendError, match="modified|unexpected"):
        sources.setup_sources(cache_dir=tmp_path)
    assert pinned == ["native", "cutlass"]


@pytest.mark.parametrize("member", ["../escape", "/absolute", "fixture/../../escape", "fixture/x\\y", "fixture/C:/x"])
def test_archive_paths_cannot_escape(tmp_path, member):
    archive = tmp_path / "archive.tgz"
    archive.write_bytes(archive_blob([(member, b"bad")]))
    spec = sources.SourceArchive("fake", "unused", "1" * 40, "unused", "fixture")
    with pytest.raises(NativeBackendError, match="unsafe"):
        sources._extract(archive, tmp_path / "tree", spec)
    assert not (tmp_path / "tree").exists()


def test_archive_links_and_duplicate_paths_are_rejected(tmp_path):
    spec = sources.SourceArchive("fake", "unused", "1" * 40, "unused", "fixture")
    for entries in ([('fixture/link', None)], [('fixture/x', b'a'), ('fixture/x', b'b')]):
        archive = tmp_path / "archive.tgz"
        archive.write_bytes(archive_blob(entries))
        with pytest.raises(NativeBackendError):
            sources._extract(archive, tmp_path / "tree", spec)
        assert not (tmp_path / "tree").exists()


def test_download_checksum_mismatch_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong bytes"))
    with pytest.raises(NativeBackendError, match="SHA256"):
        sources._download(sources.NATIVE_SOURCE, tmp_path / "archive")


def test_failed_download_does_not_publish_cache(tmp_path, pinned, monkeypatch):
    def fail(spec, path):
        path.write_bytes(b"partial")
        raise NativeUnavailable("network failed")
    monkeypatch.setattr(sources, "_download", fail)
    with pytest.raises(NativeUnavailable, match="network failed"):
        sources.setup_sources(cache_dir=tmp_path)
    assert list((tmp_path / "sources").iterdir()) == []


def test_setup_rejects_conflicting_runner_choice(tmp_path):
    with pytest.raises(ValueError, match="runner_path"):
        sources.setup_sources(options=NativeOptions(runner_path=tmp_path / "runner"))
