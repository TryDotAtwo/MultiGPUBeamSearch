"""The opt-in Rust model snapshot can validate finite bits without a byte copy."""
from types import SimpleNamespace
import json

import pytest
import torch

from multigpubeamsearch import NativeSession, models, rust_control
from multigpubeamsearch.errors import NativeBackendError, NativeUnavailable
from multigpubeamsearch.model_snapshot import ModelSnapshotCache
from multigpubeamsearch.options import NativeOptions


def artifact(path, *, dtype):
    path.mkdir()
    manifest = dict(state_len=4, num_classes=4, hidden1=16, hidden2=8,
                    residual_count=1, output_dim=1, dtype=dtype, normalization='none')
    (path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    sizes = dict(input_weight_hxk=4 * 4 * 16, input_bias=16,
                 hidden_weight_hxk=16 * 8, hidden_bias=8,
                 output_weight_hxk=8, output_bias=1)
    for fc in (1, 2):
        sizes[f'residual0_fc{fc}_weight_hxk'] = 8 * 8
        sizes[f'residual0_fc{fc}_bias'] = 8
    for name, count in sizes.items():
        (path / f'{name}.{dtype}').write_bytes(b'\x00' * (count * 2))
    return path


@pytest.fixture
def contract():
    return SimpleNamespace(state_len=4, num_classes=4, move_count=1, graph_hash='a' * 64)


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_native_model_source_uses_rust_borrowed_bytes_without_torch_buffer_copy(
        tmp_path, monkeypatch, contract, dtype):
    weights = artifact(tmp_path / 'weights', dtype=dtype)
    chunks = []

    def finite(data, encoding):
        assert type(data) is bytes and encoding == dtype and len(data) % 2 == 0
        chunks.append(len(data))
        return True

    class Cache:
        def finite_validator(self): return finite
        def prepare(self, model, contract, run_dir, cache_dir): return model

    monkeypatch.setattr(torch, 'frombuffer', lambda *args, **kwargs: pytest.fail('copied torch buffer'))
    prepared = models.prepare_model(models.NativeModel(weights, contract.graph_hash), contract,
                                    NativeOptions(), tmp_path / 'run', _snapshot_cache=Cache())
    assert chunks and sum(chunks) == sum(prepared.artifact_sizes[1:])
    assert prepared.backend == 'mlp'


@pytest.mark.parametrize('reply', [False, None, 1])
def test_native_model_rejects_invalid_rust_finite_result(tmp_path, contract, reply):
    weights = artifact(tmp_path / 'weights', dtype='fp16')
    class Cache:
        def finite_validator(self): return lambda data, encoding: reply
        def prepare(self, *args): pytest.fail('invalid weights must not be retained')
    with pytest.raises(NativeBackendError):
        models.prepare_model(models.NativeModel(weights, contract.graph_hash), contract,
                             NativeOptions(), tmp_path / 'run', _snapshot_cache=Cache())


def test_session_finite_check_requires_a_separate_explicit_opt_in(monkeypatch):
    function = lambda data, dtype: True
    class Store:
        def prepare_sized_for_session(self, *args): pass
        def clear(self): pass
    module = SimpleNamespace(MlpSnapshotStore=Store)
    module.mlp_finite_16le = function
    monkeypatch.setattr(rust_control, 'extension', lambda: module)
    default = NativeSession(engine='rust', experimental_reuse_model_snapshot=True)
    assert default.model_snapshot_cache.finite_validator() is None
    enabled = NativeSession(engine='rust', experimental_reuse_model_snapshot=True,
                            experimental_rust_mlp_finite=True)
    assert enabled.model_snapshot_cache.finite_validator() is function
    assert enabled.model_snapshot_cache.finite_validator() is function
    default.close()
    enabled.close()


@pytest.mark.parametrize('kwargs', [
    dict(engine='rust', experimental_reuse_model_snapshot=True, experimental_rust_mlp_finite=1),
    dict(engine='rust', experimental_reuse_model_snapshot=False, experimental_rust_mlp_finite=True),
    dict(engine='python', experimental_reuse_model_snapshot=True, experimental_rust_mlp_finite=True),
])
def test_session_rejects_invalid_finite_check_scope(kwargs):
    with pytest.raises(ValueError):
        NativeSession(**kwargs)


def test_explicit_finite_check_rejects_incompatible_owner_and_byte_order(monkeypatch):
    from multigpubeamsearch import model_snapshot
    class OldStore:
        pass
    function = lambda data, dtype: True
    monkeypatch.setattr(rust_control, 'extension', lambda: SimpleNamespace(
        MlpSnapshotStore=OldStore, mlp_finite_16le=function))
    with pytest.raises(NativeUnavailable):
        ModelSnapshotCache(prefer_rust=True, enable_rust_finite=True).finite_validator()
    class Store:
        def prepare_sized_for_session(self, *args): pass
    monkeypatch.setattr(rust_control, 'extension', lambda: SimpleNamespace(
        MlpSnapshotStore=Store, mlp_finite_16le=function))
    monkeypatch.setattr(model_snapshot.sys, 'byteorder', 'big')
    with pytest.raises(NativeUnavailable):
        ModelSnapshotCache(prefer_rust=True, enable_rust_finite=True).finite_validator()


@pytest.mark.parametrize('dtype,infinity', [('fp16', b'\x00\x7c'), ('bf16', b'\x80\x7f')])
def test_python_fallback_rejects_nonfinite_source(tmp_path, contract, dtype, infinity):
    weights = artifact(tmp_path / 'weights', dtype=dtype)
    (weights / f'output_bias.{dtype}').write_bytes(infinity)
    class Cache:
        def finite_validator(self): return None
        def prepare(self, *args): pytest.fail('non-finite weights must not be retained')
    with pytest.raises(NativeBackendError, match='non-finite'):
        models.prepare_model(models.NativeModel(weights, contract.graph_hash), contract,
                             NativeOptions(), tmp_path / 'run', _snapshot_cache=Cache())
