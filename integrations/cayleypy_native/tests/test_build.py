"""CPU-only build contract tests; no CUDA compilation or GPU claims."""
import json
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

from cayleypy_native.build import (build_key, configure_command, file_sha256, shape_contract,
                                  source_digest, validate_runner)
from cayleypy_native.errors import NativeBackendError, NativeUnavailable


def contract(state_len=88, move_count=24):
    return SimpleNamespace(state_len=state_len, move_count=move_count)


def native_source_contract():
    path = Path(__file__).resolve().parents[1] / "multigpubeamsearch/native_source_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def prepare_native_source_tree(root):
    manifest, _manifest_path = native_source_contract()
    for directory in manifest["native_fingerprint_roots"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative in manifest["required_files"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("synthetic native input", encoding="utf-8")


def load_build_backend(monkeypatch):
    fake_maturin = types.ModuleType("maturin")
    for name in ("get_requires_for_build_wheel", "get_requires_for_build_sdist",
                 "get_requires_for_build_editable", "prepare_metadata_for_build_wheel",
                 "build_wheel", "build_sdist", "prepare_metadata_for_build_editable",
                 "build_editable"):
        setattr(fake_maturin, name, lambda *args, **kwargs: [])
    monkeypatch.setitem(sys.modules, "maturin", fake_maturin)
    path = Path(__file__).resolve().parents[1] / "build_backend.py"
    spec = importlib.util.spec_from_file_location("mgbs_test_build_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dirty_source_content_and_shapes_change_build_keys(tmp_path):
    prepare_native_source_tree(tmp_path)
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


@pytest.mark.parametrize('relative', ['native/beam_native.h', 'native/beam_native.cpp',
    'rust/beam-runtime/src/lib.rs', 'rust/beam-runtime/Cargo.toml', 'Cargo.toml', 'Cargo.lock'])
def test_abi_and_rust_changes_invalidate_native_cache(tmp_path, relative):
    prepare_native_source_tree(tmp_path)
    (tmp_path/'CMakeLists.txt').write_text('project(example)')
    path=tmp_path/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('first source')
    first=source_digest(tmp_path)
    path.write_text('changed source')
    assert source_digest(tmp_path)!=first


@pytest.mark.parametrize("relative", [
    "third_party/nlohmann/json.hpp",
    "third_party/picosha2/picosha2.h",
])
def test_vendored_headers_invalidate_native_source_fingerprint(tmp_path, relative):
    prepare_native_source_tree(tmp_path)
    (tmp_path / "CMakeLists.txt").write_text("project(example)")
    header = tmp_path / relative
    header.parent.mkdir(parents=True)
    header.write_text("vendor header version one")

    configuration = {"shape": shape_contract(contract()), "backend": "mlp", "sm": [86]}
    first_digest = source_digest(tmp_path)
    first_key = build_key({**configuration, "source_digest": first_digest})
    header.write_text("vendor header version two")

    second_digest = source_digest(tmp_path)
    assert second_digest != first_digest
    assert build_key({**configuration, "source_digest": second_digest}) != first_key


def test_native_source_bundle_rejects_header_omitted_from_bundle_manifest(tmp_path, monkeypatch):
    backend = load_build_backend(monkeypatch)
    repository = Path(__file__).resolve().parents[3]
    contract_path = repository / "integrations/cayleypy_native/multigpubeamsearch/native_source_manifest.json"
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    bundle = tmp_path / "bundle"
    for relative in contract["required_files"]:
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")
    bundle_contract = bundle / "tools/native_source_manifest.json"
    bundle_contract.parent.mkdir(parents=True, exist_ok=True)
    bundle_contract.write_bytes(contract_bytes)
    manifest = {
        relative: hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
        for relative in contract["required_files"]
    }
    manifest["tools/native_source_manifest.json"] = hashlib.sha256(contract_bytes).hexdigest()
    (bundle / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    backend._verify_bundle(bundle)

    header = "third_party/nlohmann/json.hpp"
    manifest.pop(header)
    (bundle / header).unlink()
    (bundle / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="third_party/nlohmann/json.hpp"):
        backend._verify_bundle(bundle)


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
    assert "-DBEAM_EXPERIMENTAL_RADIX128=OFF" in command
    assert "-DBEAM_EXPERIMENTAL_INDEX_SORT=OFF" in command
    assert "-DBEAM_EXPERIMENTAL_STREAM4_SCAN=OFF" in command
    assert "-DBEAM_EXPERIMENTAL_OWNER_RADIX=OFF" in command
    assert "-DBEAM_EXPERIMENTAL_SCORE_BITS=OFF" in command
    assert command[:2] == ["cmake", "-S"]


def test_cutlass_ensemble_enables_libtorch_backbones(tmp_path):
    command = configure_command(tmp_path, tmp_path / 'build', tmp_path / 'cutlass',
        contract(8, 3), 'ensemble', (86,),
        {'cmake': 'cmake', 'cxx': 'c++', 'nvcc': 'nvcc'}, inference_backend='cutlass')
    assert '-DBEAM_ENABLE_LIBTORCH_STREAM1=ON' in command
    assert any(value.startswith('-DCMAKE_PREFIX_PATH=') for value in command)


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


@pytest.mark.parametrize('name',[None,'../outside','/tmp/another-program','unverified-helper'])
def test_calibration_metadata_cannot_select_another_executable(tmp_path,name):
    runner=tmp_path/'runner';runner.write_bytes(b'fake runner');runner.chmod(0o755)
    metadata={'schema_version':1,'shape':shape_contract(contract()),'backend':'mlp',
        'cuda_architectures':[75],'binary_sha256':file_sha256(runner),
        'calibration_binary_name':name,'calibration_binary_sha256':'0'*64}
    (tmp_path/'native-build.json').write_text(json.dumps(metadata))
    with pytest.raises(NativeBackendError,match='name is invalid'):
        validate_runner(runner,contract(),'mlp',(75,))


@pytest.mark.parametrize('name',['stream1_native_mlp_benchmark','stream1_libtorch_mlp_benchmark'])
def test_single_model_calibration_executable_is_bound_to_its_digest(tmp_path,name):
    runner=tmp_path/'runner';runner.write_bytes(b'fake runner');runner.chmod(0o755)
    helper=tmp_path/name;helper.write_bytes(b'fake measured helper')
    metadata={'schema_version':1,'shape':shape_contract(contract()),'backend':'mlp',
        'cuda_architectures':[75],'binary_sha256':file_sha256(runner),
        'calibration_binary_name':name,'calibration_binary_sha256':file_sha256(helper)}
    (tmp_path/'native-build.json').write_text(json.dumps(metadata))
    assert validate_runner(runner,contract(),'mlp',(75,))==metadata
    helper.write_bytes(b'changed helper')
    with pytest.raises(NativeBackendError,match='changed or is missing'):
        validate_runner(runner,contract(),'mlp',(75,))


def test_build_prerequisites_do_not_download_missing_source(tmp_path):
    from cayleypy_native.build import prerequisites
    from cayleypy_native.options import NativeOptions
    with pytest.raises(NativeUnavailable, match="source_dir"):
        prerequisites(NativeOptions(source_dir=tmp_path, cache_dir=tmp_path / "cache"))
    assert list(tmp_path.iterdir()) == []
