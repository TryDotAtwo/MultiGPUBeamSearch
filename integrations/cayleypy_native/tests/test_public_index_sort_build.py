"""Public opt-in index sorting must be consistent across native artifacts."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multigpubeamsearch import build
from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.options import NativeOptions
from multigpubeamsearch.rust_control import options_descriptor


def test_public_index_sort_option_is_explicit_and_boolean():
    assert NativeOptions().experimental_index_sort is False
    assert 'experimental_index_sort' not in options_descriptor(NativeOptions())
    selected = NativeOptions(experimental_index_sort=True)
    assert options_descriptor(selected)['experimental_index_sort'] is True
    for value in (1, 'true', None):
        with pytest.raises(ValueError, match='experimental_index_sort'):
            NativeOptions(experimental_index_sort=value)


def test_cmake_index_sort_is_independent_from_other_experiments(tmp_path):
    contract = SimpleNamespace(state_len=96, move_count=24)
    programs = dict(cmake='cmake', cxx='c++', nvcc='nvcc')
    command = build.configure_command(tmp_path, tmp_path / 'build', tmp_path / 'cutlass',
        contract, 'mlp', (86,), programs, experimental_index_sort=True)
    assert '-DBEAM_EXPERIMENTAL_INDEX_SORT=ON' in command
    assert '-DBEAM_EXPERIMENTAL_RADIX128=OFF' in command
    assert '-DBEAM_EXPERIMENTAL_STREAM4_SCAN=OFF' in command
    assert '-DBEAM_EXPERIMENTAL_OWNER_RADIX=OFF' in command
    assert '-DBEAM_EXPERIMENTAL_SCORE_BITS=OFF' in command


def test_managed_builds_separate_index_mode_and_reject_tampered_receipt(tmp_path, monkeypatch):
    source, cutlass = tmp_path / 'source', tmp_path / 'cutlass'
    (source / 'tools').mkdir(parents=True)
    (source / 'CMakeLists.txt').write_text('production_runner')
    (source / 'tools/production_runner.cu').write_text('production runner')
    (cutlass / 'include/cutlass').mkdir(parents=True)
    (cutlass / 'include/cutlass/cutlass.h').write_text('cutlass')
    graph = SimpleNamespace(state_len=96, move_count=24)
    model = SimpleNamespace(backend='mlp')
    programs = dict(cmake='cmake', cxx='c++', nvcc='nvcc')
    nccl = dict(include_dir='nccl-header', library='nccl-library',
                header_sha256='a' * 64, library_sha256='b' * 64)
    monkeypatch.setattr(build, 'prerequisites', lambda options: programs)
    monkeypatch.setattr(build, 'toolchain_identity', lambda programs: {'identity': 'test'})
    monkeypatch.setattr(build, 'discover_nccl', lambda **kwargs: nccl)
    monkeypatch.setattr(build, '_nccl_identity', lambda *args, **kwargs: nccl)
    from multigpubeamsearch import backend
    monkeypatch.setattr(backend, 'child_environment', lambda *args: {})
    commands = []
    def fake_process(command, **kwargs):
        commands.append(command)
        if '--build' in command:
            target_dir = Path(command[command.index('--build') + 1])
            binary = target_dir / 'production_runner'
            binary.write_bytes(str(target_dir).encode())
            binary.chmod(0o755)
    monkeypatch.setattr(backend, 'run_process', fake_process)
    options = NativeOptions(source_dir=source, cutlass_dir=cutlass, cache_dir=tmp_path / 'cache',
                            inference_backend='cutlass', autotune=False)
    off, off_meta = build.ensure_runner(graph, model, options, (86,), tmp_path / 'off')
    on_options = replace(options, experimental_index_sort=True)
    on, on_meta = build.ensure_runner(graph, model, on_options, (86,), tmp_path / 'on')
    assert off != on and off_meta['build_key'] != on_meta['build_key']
    assert off_meta.get('experimental_index_sort', False) is False
    assert on_meta['experimental_index_sort'] is True
    assert '-DBEAM_EXPERIMENTAL_INDEX_SORT=OFF' in commands[0]
    assert '-DBEAM_EXPERIMENTAL_INDEX_SORT=ON' in commands[2]
    assert build.ensure_runner(graph, model, on_options, (86,), tmp_path / 'again')[0] == on
    sidecar = on.parent / 'native-build.json'
    changed = json.loads(sidecar.read_text())
    changed['experimental_index_sort'] = False
    sidecar.write_text(json.dumps(changed))
    with pytest.raises(NativeBackendError, match='index.sort|build key'):
        build.ensure_runner(graph, model, on_options, (86,), tmp_path / 'changed')


def test_external_runner_must_match_explicit_index_mode(tmp_path):
    graph = SimpleNamespace(state_len=96, move_count=24)
    model = SimpleNamespace(backend='mlp')
    runner = tmp_path / 'production_runner'
    runner.write_bytes(b'fake runner')
    runner.chmod(0o755)
    metadata = dict(schema_version=1, shape=build.shape_contract(graph), backend='mlp',
                    cuda_architectures=[86], binary_sha256=build.file_sha256(runner),
                    inference_backend='cutlass')
    (tmp_path / 'native-build.json').write_text(json.dumps(metadata))
    opts = NativeOptions(runner_path=runner, inference_backend='cutlass', experimental_index_sort=True)
    with pytest.raises(NativeBackendError, match='index.sort'):
        build.ensure_runner(graph, model, opts, (86,), tmp_path / 'run')
