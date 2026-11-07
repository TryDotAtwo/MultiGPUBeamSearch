"""Contract tests for the on-demand shared library; no CUDA is loaded here."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from multigpubeamsearch import build
from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.options import NativeOptions


def fixture(tmp_path, monkeypatch):
    from multigpubeamsearch import native_library
    source = tmp_path / 'source'
    (source / 'src').mkdir(parents=True)
    (source / 'CMakeLists.txt').write_text('cmake')
    (source / 'src/model.cpp').write_text('source')
    cutlass = tmp_path / 'cutlass'
    (cutlass / 'include').mkdir(parents=True)
    (cutlass / 'include/header.h').write_text('cutlass')
    directory = tmp_path / 'build'
    directory.mkdir()
    (directory / 'CMakeCache.txt').write_text('configured')
    runner = directory / 'production_runner'
    runner.write_bytes(b'runner')
    runner.chmod(0o755)
    worker = tmp_path / 'beam-runner'
    worker.write_bytes(b'worker')
    worker.chmod(0o755)
    graph = SimpleNamespace(state_len=96, move_count=24)
    model = SimpleNamespace(backend='mlp')
    options = NativeOptions(source_dir=source, cutlass_dir=cutlass, cache_dir=tmp_path / 'cache',
                            inference_backend='cutlass', autotune=False)
    nccl = dict(include_dir='nccl-include', library='nccl-library', header_sha256='a', library_sha256='b')
    programs = dict(cmake='cmake', nvcc='nvcc', cxx='c++')
    toolchain = dict(cmake='cmake version', nvcc='nvcc version', cxx='c++ version')
    metadata = dict(schema_version=1, shape=build.shape_contract(graph), backend='mlp',
                    cuda_architectures=[86], inference_backend='cutlass', binary_sha256=build.file_sha256(runner),
                    source_digest=build.source_digest(source), cutlass_digest=build.source_digest(cutlass, cutlass=True),
                    toolchain=toolchain, nccl=nccl, build_key='b'*64)
    (directory / 'native-build.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(native_library, 'prerequisites', lambda options: programs)
    monkeypatch.setattr(native_library, 'toolchain_identity', lambda programs: toolchain)
    monkeypatch.setattr(native_library, 'discover_nccl', lambda: nccl)
    monkeypatch.setattr(native_library, 'worker_executable', lambda: worker)
    monkeypatch.setattr(native_library, 'child_environment', lambda *args: {})
    calls = []
    def run_build(command, **kwargs):
        calls.append(command)
        if '--build' in command:
            target = command[command.index('--target') + 1]
            (directory / f'lib{target}.so').write_bytes(b'library')
    monkeypatch.setattr(native_library, 'run_process', run_build)
    layout = dict(abi_version=1, state_len=96, storage_len=112, state_alignment=16, move_count=24,
                  candidate_bytes=32, candidate_alignment=32, hash_bytes=16, hash_alignment=16)
    def run_probe(command, **kwargs):
        assert command == [str(worker), '--native-abi-info']
        target = 'beam_native_libtorch' if metadata['inference_backend'] == 'libtorch' else 'beam_native'
        assert kwargs['env']['BEAM_NATIVE_LIBRARY'] == str(directory / f'lib{target}.so')
        return SimpleNamespace(returncode=0, stdout=json.dumps(layout), stderr='')
    monkeypatch.setattr(native_library.subprocess, 'run', run_probe)
    return native_library, graph, model, options, runner, metadata, calls, layout


def test_build_once_reuse_and_detect_library_tampering(tmp_path, monkeypatch):
    native, graph, model, options, runner, metadata, calls, _ = fixture(tmp_path, monkeypatch)
    library, receipt = native.ensure_native_library(graph, model, options, (86,), tmp_path, runner, metadata)
    assert library.name == 'libbeam_native.so' and len(calls) == 2
    assert calls[0][:2] == ['cmake', '-S'] and calls[1][:2] == ['cmake', '--build']
    assert receipt['library_sha256'] == build.file_sha256(library)
    assert native.ensure_native_library(graph, model, options, (86,), tmp_path, runner, metadata) == (library, receipt)
    assert len(calls) == 2
    library.write_bytes(b'changed')
    with pytest.raises(NativeBackendError, match='SHA256'):
        native.ensure_native_library(graph, model, options, (86,), tmp_path, runner, metadata)
    assert len(calls) == 2


def test_wrong_abi_layout_or_changed_source_is_rejected(tmp_path, monkeypatch):
    native, graph, model, options, runner, metadata, calls, layout = fixture(tmp_path, monkeypatch)
    layout['candidate_alignment'] = 16
    with pytest.raises(NativeBackendError, match='layout mismatch'):
        native.ensure_native_library(graph, model, options, (86,), tmp_path, runner, metadata)
    assert not (runner.parent / 'native-library-build.json').exists()
    layout['candidate_alignment'] = 32
    (options.source_dir / 'src/model.cpp').write_text('changed')
    with pytest.raises(NativeBackendError, match='source'):
        native.ensure_native_library(graph, model, options, (86,), tmp_path, runner, metadata)


def test_changed_pytorch_dependency_is_rejected(tmp_path, monkeypatch):
    import torch
    native, graph, model, options, runner, metadata, calls, _ = fixture(tmp_path, monkeypatch)
    metadata['torch'] = dict(version=torch.__version__, cuda=torch.version.cuda,
                             cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                             cmake_prefix=str(torch.utils.cmake_prefix_path))
    (runner.parent / 'native-build.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(torch, '__version__', 'changed-version')
    with pytest.raises(NativeBackendError, match='PyTorch changed'):
        native.ensure_native_library(graph, model, options, (86,), tmp_path, runner, metadata)
    assert not calls


@pytest.mark.parametrize('resolved', ('cutlass', 'libtorch'))
def test_auto_backend_uses_the_verified_runner_resolution(tmp_path, monkeypatch, resolved):
    import torch
    native, graph, model, options, runner, metadata, calls, _ = fixture(tmp_path, monkeypatch)
    metadata['inference_backend'] = resolved
    if resolved == 'libtorch':
        metadata['torch'] = dict(version=torch.__version__, cuda=torch.version.cuda,
                                 cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                                 cmake_prefix=str(torch.utils.cmake_prefix_path))
    (runner.parent / 'native-build.json').write_text(json.dumps(metadata))
    options = replace(options, inference_backend='auto')
    library, receipt = native.ensure_native_library(graph, model, options, (86,), tmp_path, runner, metadata)
    assert library.name == ('libbeam_native_libtorch.so' if resolved == 'libtorch' else 'libbeam_native.so')
    assert receipt['specification']['inference_backend'] == resolved
    assert native.ensure_native_library(graph, model, options, (86,), tmp_path, runner, metadata) == (library, receipt)
    assert len(calls) == 2


def test_explicit_backend_mismatch_is_rejected(tmp_path, monkeypatch):
    native, graph, model, options, runner, metadata, calls, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(NativeBackendError, match='metadata'):
        native.ensure_native_library(graph, model, replace(options, inference_backend='libtorch'),
                                     (86,), tmp_path, runner, metadata)
    assert not calls


@pytest.mark.parametrize('backend', ('auto', 'unexpected', None))
def test_unresolved_runner_backend_is_rejected(tmp_path, monkeypatch, backend):
    native, graph, model, options, runner, metadata, calls, _ = fixture(tmp_path, monkeypatch)
    metadata['inference_backend'] = backend
    (runner.parent / 'native-build.json').write_text(json.dumps(metadata))
    with pytest.raises(NativeBackendError, match='metadata'):
        native.ensure_native_library(graph, model, replace(options, inference_backend='auto'),
                                     (86,), tmp_path, runner, metadata)
    assert not calls
