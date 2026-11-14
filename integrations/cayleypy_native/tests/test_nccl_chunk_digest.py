"""Content verification for the experimental session-owned NCCL digest cache."""
import hashlib
import os
from pathlib import Path
import threading
import pytest


@pytest.mark.parametrize("contents", [b"", b"a", b"abcd", b"abcde", b"abcdefghijklmnopq"])
def test_repeated_digest_is_the_conventional_sha256(tmp_path, contents):
    from multigpubeamsearch.nccl_digest import ChunkDigestCache

    library = tmp_path / "libnccl.so.2"
    library.write_bytes(contents)
    cache = ChunkDigestCache(chunk_bytes=4, workers=3)
    expected = hashlib.sha256(contents).hexdigest()
    assert cache.sha256(library) == expected
    assert cache.sha256(library) == expected


def test_same_size_mutation_with_restored_mtime_is_not_a_cache_hit(tmp_path):
    from multigpubeamsearch.nccl_digest import ChunkDigestCache

    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"abcdefghijklmnopq")
    cache = ChunkDigestCache(chunk_bytes=4, workers=3)
    assert cache.sha256(library) == hashlib.sha256(b"abcdefghijklmnopq").hexdigest()
    stat = library.stat()
    library.write_bytes(b"abcdefghijklmnoPq")
    os.utime(library, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cache.sha256(library) == hashlib.sha256(b"abcdefghijklmnoPq").hexdigest()


@pytest.mark.parametrize("contents", [b"a", b"abcdefghijklmnopq--extended"])
def test_size_change_never_reuses_the_old_digest(tmp_path, contents):
    from multigpubeamsearch.nccl_digest import ChunkDigestCache

    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"abcdefghijklmnopq")
    cache = ChunkDigestCache(chunk_bytes=4, workers=2)
    cache.sha256(library)
    library.write_bytes(contents)
    assert cache.sha256(library) == hashlib.sha256(contents).hexdigest()


def test_atomic_replacement_of_library_is_detected(tmp_path):
    from multigpubeamsearch.nccl_digest import ChunkDigestCache

    library = tmp_path / "libnccl.so.2"
    replacement = tmp_path / "replacement"
    library.write_bytes(b"first bytes")
    cache = ChunkDigestCache(chunk_bytes=4, workers=2)
    first = cache.sha256(library)
    replacement.write_bytes(b"other bytes")
    replacement.replace(library)
    assert cache.sha256(library) == hashlib.sha256(b"other bytes").hexdigest()
    assert cache.sha256(library) != first


def test_nccl_discovery_revalidates_library_and_header(tmp_path, monkeypatch):
    from multigpubeamsearch import build
    from multigpubeamsearch.nccl_digest import ChunkDigestCache

    include, libdir = tmp_path / "include", tmp_path / "lib"
    include.mkdir()
    libdir.mkdir()
    header, library = include / "nccl.h", libdir / "libnccl.so.2"
    header.write_bytes(b"header one")
    library.write_bytes(b"test library one")
    monkeypatch.setattr(build, "_nccl_locations", lambda: [(include, libdir)])
    cache = ChunkDigestCache(chunk_bytes=4, workers=2)
    first = build.discover_nccl(digest_cache=cache)
    assert first["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert first["header_sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
    library.write_bytes(b"test library two")
    header.write_bytes(b"header two")
    second = build.discover_nccl(digest_cache=cache)
    assert second["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert second["header_sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
    assert first != second


@pytest.mark.parametrize("changed", ["library", "header"])
def test_opted_in_prepared_runner_rejects_changed_nccl(tmp_path, monkeypatch, changed):
    from types import SimpleNamespace
    from multigpubeamsearch import build
    from multigpubeamsearch.errors import NativeBackendError
    from multigpubeamsearch.nccl_digest import ChunkDigestCache

    include = tmp_path / "include"
    include.mkdir()
    header, library = include / "nccl.h", tmp_path / "libnccl.so.2"
    header.write_bytes(b"header one")
    library.write_bytes(b"library one")
    metadata = {"inference_backend": "cutlass", "nccl": build._nccl_identity(include, library)}
    monkeypatch.setattr(build, "validate_runner", lambda *args: metadata)
    model = SimpleNamespace(backend="mlp")
    options = SimpleNamespace(runner_path=tmp_path / "runner", inference_backend="cutlass")
    cache = ChunkDigestCache(chunk_bytes=4, workers=2)
    assert build.ensure_runner(None, model, options, (86,), tmp_path, nccl_digest_cache=cache)[1] == metadata
    target = library if changed == "library" else header
    old = target.stat()
    target.write_bytes(b"library two" if changed == "library" else b"header two")
    os.utime(target, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(NativeBackendError, match="NCCL.*changed"):
        build.ensure_runner(None, model, options, (86,), tmp_path, nccl_digest_cache=cache)


def test_thread_start_failure_uses_verified_serial_path(tmp_path, monkeypatch):
    from multigpubeamsearch import nccl_digest

    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"abcdefghijklmnopq")
    cache = nccl_digest.ChunkDigestCache(chunk_bytes=4, workers=2)
    cache.sha256(library)

    def no_threads(*args, **kwargs):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(nccl_digest, "ThreadPoolExecutor", no_threads)
    assert cache.sha256(library) == hashlib.sha256(library.read_bytes()).hexdigest()
    library.write_bytes(b"Abcdefghijklmnopq")
    assert cache.sha256(library) == hashlib.sha256(library.read_bytes()).hexdigest()


def test_discovery_rejects_soname_retarget_during_verification(tmp_path, monkeypatch):
    from multigpubeamsearch import build, nccl_digest
    from multigpubeamsearch.errors import NativeUnavailable

    include, libdir = tmp_path / "include", tmp_path / "lib"
    include.mkdir()
    libdir.mkdir()
    (include / "nccl.h").write_bytes(b"header")
    first, second = libdir / "version-one", libdir / "version-two"
    first.write_bytes(b"first library")
    second.write_bytes(b"other library")
    soname = libdir / "libnccl.so"
    try:
        soname.symlink_to(first.name)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    monkeypatch.setattr(build, "_nccl_locations", lambda: [(include, libdir)])
    cache = nccl_digest.ChunkDigestCache(chunk_bytes=4, workers=2)
    build.discover_nccl(digest_cache=cache)
    original = nccl_digest._digest_partition
    changed = threading.Event()
    guard = threading.Lock()

    def retarget_then_read(*args):
        with guard:
            if not changed.is_set():
                replacement = libdir / "replacement-link"
                replacement.symlink_to(second.name)
                replacement.replace(soname)
                changed.set()
        return original(*args)

    monkeypatch.setattr(nccl_digest, "_digest_partition", retarget_then_read)
    with pytest.raises(NativeUnavailable, match="NCCL.*changed"):
        build.discover_nccl(digest_cache=cache)


def test_discovery_rechecks_resolved_soname_target(tmp_path, monkeypatch):
    """Exercise retarget detection even where creating symlinks is restricted."""
    from multigpubeamsearch import build, nccl_digest
    from multigpubeamsearch.errors import NativeUnavailable

    include, libdir = tmp_path / "include", tmp_path / "lib"
    include.mkdir()
    libdir.mkdir()
    (include / "nccl.h").write_bytes(b"header")
    first, second, soname = libdir / "first", libdir / "second", libdir / "libnccl.so"
    first.write_bytes(b"first library")
    second.write_bytes(b"other library")
    soname.write_bytes(b"alias placeholder")
    monkeypatch.setattr(build, "_nccl_locations", lambda: [(include, libdir)])
    original_resolve = Path.resolve
    retargeted = threading.Event()

    def resolve(path, *args, **kwargs):
        if path == soname:
            return original_resolve(second if retargeted.is_set() else first, *args, **kwargs)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    cache = nccl_digest.ChunkDigestCache(chunk_bytes=4, workers=2)
    build.discover_nccl(digest_cache=cache)
    original_partition = nccl_digest._digest_partition

    def retarget_then_read(*args):
        retargeted.set()
        return original_partition(*args)

    monkeypatch.setattr(nccl_digest, "_digest_partition", retarget_then_read)
    with pytest.raises(NativeUnavailable, match="NCCL.*changed"):
        build.discover_nccl(digest_cache=cache)


def test_change_during_parallel_read_fails_and_does_not_cache_a_result(tmp_path, monkeypatch):
    from multigpubeamsearch import nccl_digest

    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"abcdefghijklmnopq")
    cache = nccl_digest.ChunkDigestCache(chunk_bytes=4, workers=2)
    cache.sha256(library)
    original = nccl_digest._digest_partition
    changed = threading.Event()
    guard = threading.Lock()

    def change_then_read(*args):
        with guard:
            if not changed.is_set():
                old = library.stat()
                library.write_bytes(b"Abcdefghijklmnopq")
                os.utime(library, ns=(old.st_atime_ns, old.st_mtime_ns + 2_000_000_000))
                changed.set()
        return original(*args)

    monkeypatch.setattr(nccl_digest, "_digest_partition", change_then_read)
    with pytest.raises(OSError, match="changed"):
        cache.sha256(library)
    monkeypatch.setattr(nccl_digest, "_digest_partition", original)
    assert cache.sha256(library) == hashlib.sha256(b"Abcdefghijklmnopq").hexdigest()


def test_parallel_digest_is_explicitly_opt_in():
    from multigpubeamsearch.persistent import NativeSession

    with NativeSession() as ordinary:
        assert ordinary.nccl_digest_cache is None
    with NativeSession(experimental_parallel_nccl_identity=True) as experimental:
        assert experimental.nccl_digest_cache is not None
    with pytest.raises(ValueError, match="must be a boolean"):
        NativeSession(experimental_parallel_nccl_identity=1)
