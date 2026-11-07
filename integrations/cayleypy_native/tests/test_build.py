"""CPU-only build contract tests; no CUDA compilation or GPU claims."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cayleypy_native.build import (build_key, configure_command, file_sha256, shape_contract,
                                  source_digest, validate_runner)
from cayleypy_native.errors import NativeBackendError, NativeUnavailable


def contract(state_len=88, move_count=24):
    return SimpleNamespace(state_len=state_len, move_count=move_count)


def test_dirty_source_content_and_shapes_change_build_keys(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text("project(example)")
    source = tmp_path / "src"
    source.mkdir()
    code = source / "model.cpp"
    code.write_text("version one")
    base = {"source": source_digest(tmp_path), "shape": shape_contract(contract()), "backend": "mlp", "sm": [75]}
    first = build_key(base)
    code.write_text("version two")
    assert build_key(dict(base, source=source_digest(tmp_path))) != first
    assert build_key(dict(base, shape=shape_contract(contract(72, 18)))) != first
    assert build_key(dict(base, backend="piece_transformer")) != first
    assert build_key(dict(base, sm=[80])) != first
    assert build_key(dict(reversed(list(base.items())))) == first


@pytest.mark.parametrize("logical,physical", [(8, 16), (72, 80), (88, 96), (96, 112), (120, 128), (128, 144)])
def test_shape_reserves_metadata_padding(logical, physical):
    assert shape_contract(contract(logical))["storage_len"] == physical


def test_cmake_dimensions_are_explicit_not_cached_inference(tmp_path):
    command = configure_command(tmp_path, tmp_path / "build", tmp_path / "cutlass", contract(72, 18),
                                "mlp", (75, 86), {"cmake": "cmake", "cxx": "c++", "nvcc": "nvcc"})
    assert "-DBEAM_STATE_LOGICAL_BYTES=72" in command
    assert "-DBEAM_MOVE_COUNT=18" in command
    assert "-DBEAM_STATE_ALIGNMENT=16" in command
    assert "-DBEAM_CUDA_ARCHITECTURES=75;86" in command
    assert command[:2] == ["cmake", "-S"]


def test_pip_nccl_versioned_library_is_selected_and_passed_to_cmake(monkeypatch, tmp_path):
    import cayleypy_native.build as build
    package = tmp_path / "site-packages" / "nvidia" / "nccl"
    include, libdir = package / "include", package / "lib"
    include.mkdir(parents=True)
    libdir.mkdir()
    (include / "nccl.h").write_text("test-only header")
    library = libdir / "libnccl.so.2"
    library.write_bytes(b"test-only library; never executable")
    monkeypatch.setattr(build.sys, "path", [str(tmp_path / "site-packages")])
    monkeypatch.setattr(build.sysconfig, "get_paths", lambda: {})
    monkeypatch.delenv("NCCL_ROOT", raising=False)
    monkeypatch.delenv("NCCL_HOME", raising=False)
    nccl = build.discover_nccl()
    assert nccl["library"] == str(library.resolve())
    assert nccl["header_sha256"] == file_sha256(include / "nccl.h")
    command = configure_command(tmp_path, tmp_path / "build", tmp_path / "cutlass", contract(),
                                "mlp", (75,), {"cmake": "cmake", "cxx": "c++", "nvcc": "nvcc"}, nccl=nccl)
    assert f"-DNCCL_INCLUDE_DIR={include.resolve()}" in command
    assert f"-DNCCL_LIBRARY={library.resolve()}" in command
    assert f"-DCMAKE_BUILD_RPATH={libdir.resolve()}" in command
    before = build_key({"nccl": nccl})
    library.write_bytes(b"test-only changed library")
    assert build_key({"nccl": build.discover_nccl()}) != before
    (include / "nccl.h").write_text("changed test-only header")
    assert build.discover_nccl()["header_sha256"] != nccl["header_sha256"]


def test_nccl_requires_both_header_and_library_without_spawning(monkeypatch, tmp_path):
    import cayleypy_native.build as build
    include, libdir = tmp_path / "include", tmp_path / "lib"
    include.mkdir()
    libdir.mkdir()
    monkeypatch.setattr(build, "_nccl_locations", lambda: [(include, libdir)])
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs: pytest.fail("NCCL discovery must not spawn"))
    (libdir / "libnccl.so.2").write_bytes(b"test-only library")
    with pytest.raises(NativeUnavailable, match="NCCL headers"):
        build.discover_nccl()
    (libdir / "libnccl.so.2").unlink()
    (include / "nccl.h").write_text("test-only header")
    with pytest.raises(NativeUnavailable, match="NCCL headers"):
        build.discover_nccl()


def test_existing_binary_requires_matching_digest_shape_and_backend(tmp_path):
    runner = tmp_path / "runner"
    runner.write_bytes(b"test-only fake native executable")
    runner.chmod(0o755)
    metadata = {"schema_version": 1, "shape": shape_contract(contract()), "backend": "mlp",
                "cuda_architectures": [75], "binary_sha256": file_sha256(runner)}
    sidecar = tmp_path / "native-build.json"
    with pytest.raises(NativeUnavailable, match="metadata"):
        validate_runner(runner, contract(), "mlp", (75,))
    sidecar.write_text(json.dumps(metadata))
    assert validate_runner(runner, contract(), "mlp", (75,)) == metadata
    with pytest.raises(NativeUnavailable, match="shape/backend"):
        validate_runner(runner, contract(96), "mlp", (75,))
    with pytest.raises(NativeUnavailable, match="shape/backend"):
        validate_runner(runner, contract(), "piece_transformer", (75,))
    with pytest.raises(NativeUnavailable, match="architectures"):
        validate_runner(runner, contract(), "mlp", (80,))
    runner.write_bytes(b"modified after sidecar")
    with pytest.raises(NativeBackendError, match="SHA256"):
        validate_runner(runner, contract(), "mlp", (75,))


def test_build_prerequisites_do_not_download_missing_source(tmp_path):
    from cayleypy_native.build import prerequisites
    from cayleypy_native.options import NativeOptions
    with pytest.raises(NativeUnavailable, match="source_dir"):
        prerequisites(NativeOptions(source_dir=tmp_path, cache_dir=tmp_path / "cache"))
    assert list(tmp_path.iterdir()) == []
