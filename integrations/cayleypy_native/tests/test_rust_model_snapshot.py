"""The session adapter must use the Rust owner only when it is available."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.models import PreparedModel
from multigpubeamsearch.persistent import NativeSession


def test_rust_session_delegates_snapshot_copy_reuse_and_refresh(tmp_path, monkeypatch):
    from multigpubeamsearch import models, rust_control

    source = tmp_path / 'source'
    source.mkdir()
    for name in ('manifest.json', 'a.fp16', 'z.fp16'):
        (source / name).write_bytes(b'fixture')
    model = PreparedModel(source, {'output_dim': 1}, 'mlp', 'b' * 64,
                          ('manifest.json', 'a.fp16', 'z.fp16'), (7, 7, 7))
    contract = SimpleNamespace(graph_hash='a' * 64)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    calls = []
    first_dir = tmp_path / 'run1' / 'weights'
    second_dir = tmp_path / 'run2' / 'weights'

    class Store:
        def prepare_sized_for_session(self, source_dir, cache, run, graph, artifact, tensors, sizes, force_new):
            calls.append((Path(source_dir), Path(cache), Path(run), graph, artifact, tensors, sizes, force_new))
            if len(calls) == 1:
                return str(first_dir), False, 42
            if len(calls) == 2:
                return str(first_dir), True, 0
            return str(second_dir), False, 42

        def clear(self):
            calls.append('clear')

    monkeypatch.setattr(rust_control, 'extension', lambda: SimpleNamespace(MlpSnapshotStore=Store))
    monkeypatch.setattr(models, '_snapshot_artifact', lambda *a, **k: pytest.fail('Python copied weights'))
    monkeypatch.setattr(models, 'verify_prepared_model', lambda *a, **k: pytest.fail('Python revalidated weights'))
    with NativeSession(engine='rust', experimental_reuse_model_snapshot=True) as session:
        cache = session.model_snapshot_cache
        first = cache.prepare(model, contract, tmp_path / 'run1', cache_dir)
        assert cache.last == {'reused': False, 'copied_bytes': 42, 'snapshot_dir': str(first_dir), 'owner': 'rust'}
        assert cache.binds(first, contract)
        second = cache.prepare(model, contract, tmp_path / 'run2', cache_dir)
        assert cache.last['reused'] and cache.needs_fresh_worker_snapshot(second, contract)
        third = cache.refresh_for_worker(second, contract, tmp_path / 'run2')
        assert cache.binds(third, contract) and not cache.last['reused']
    assert [call[-1] for call in calls[:-1]] == [False, False, True]
    assert calls[0][5] == ['a.fp16', 'z.fp16']
    assert calls[0][6] == [7, 7, 7]
    assert calls[-1] == 'clear'


def test_rust_snapshot_failure_is_not_silently_retried_in_python(tmp_path, monkeypatch):
    from multigpubeamsearch import models, rust_control

    source = tmp_path / 'source'
    source.mkdir()
    for name in ('manifest.json', 'a.fp16'):
        (source / name).write_bytes(b'fixture')
    model = PreparedModel(source, {'output_dim': 1}, 'mlp', 'b' * 64,
                          ('manifest.json', 'a.fp16'), (7, 7))
    contract = SimpleNamespace(graph_hash='a' * 64)

    class Store:
        def prepare_sized_for_session(self, *args):
            raise RuntimeError('artifact hash mismatch')
        def clear(self):
            pass

    monkeypatch.setattr(rust_control, 'extension', lambda: SimpleNamespace(MlpSnapshotStore=Store))
    monkeypatch.setattr(models, '_snapshot_artifact', lambda *a, **k: pytest.fail('fallback after Rust failure'))
    with NativeSession(engine='rust', experimental_reuse_model_snapshot=True) as session:
        with pytest.raises(NativeBackendError, match='artifact hash mismatch'):
            session.model_snapshot_cache.prepare(model, contract, tmp_path / 'run', tmp_path)
        assert session.model_snapshot_cache.last is None


def test_old_extension_keeps_the_existing_python_snapshot_path(tmp_path, monkeypatch):
    from multigpubeamsearch import models, rust_control

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_bytes(b'{}')
    (source / 'a.fp16').write_bytes(b'\x00\x00')
    model = PreparedModel(source, {'output_dim': 1}, 'mlp', 'b' * 64,
                          ('manifest.json', 'a.fp16'))
    contract = SimpleNamespace(graph_hash='a' * 64)
    target = tmp_path / 'run' / 'weights'
    monkeypatch.setattr(rust_control, 'extension', lambda: SimpleNamespace())
    monkeypatch.setattr(models, '_snapshot_artifact', lambda *args: PreparedModel(
        target, dict(model.manifest), model.backend, model.artifact_hash, model.artifact_files))
    with NativeSession(engine='rust', experimental_reuse_model_snapshot=True) as session:
        result = session.model_snapshot_cache.prepare(model, contract, tmp_path / 'run', tmp_path)
        assert result.weights_dir == target
        assert session.model_snapshot_cache.last['copied_bytes'] == 4
        assert 'owner' not in session.model_snapshot_cache.last


def test_old_rust_store_without_launch_verification_keeps_python_check(tmp_path, monkeypatch):
    from multigpubeamsearch import models, rust_control
    from multigpubeamsearch.persistent import NativeOutcome

    class OldStore:
        def prepare_for_session(self, *args):
            return str(tmp_path / 'run' / 'weights'), False, 2
        def clear(self):
            pass

    monkeypatch.setattr(rust_control, 'extension', lambda: SimpleNamespace(MlpSnapshotStore=OldStore))
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_bytes(b'{}')
    (source / 'a.fp16').write_bytes(b'\x00\x00')
    model = PreparedModel(source, {}, 'mlp', 'b' * 64, ('manifest.json', 'a.fp16'), (2, 2))
    contract = SimpleNamespace(graph_hash='a' * 64)
    target = tmp_path / 'run' / 'weights'
    target.parent.mkdir()
    monkeypatch.setattr(models, '_snapshot_artifact', lambda *args: PreparedModel(
        target, {}, 'mlp', model.artifact_hash, model.artifact_files, model.artifact_sizes))
    checked = []
    monkeypatch.setattr(models, 'verify_prepared_model', lambda *args: checked.append(args))
    def execute(contract, prepared, options, width, depth, run_dir, devices, *, runtime):
        return NativeOutcome(None, .01, width, run_dir, {'terminal_records': 1})
    with NativeSession(engine='rust', experimental_reuse_model_snapshot=True) as session:
        cache = session.model_snapshot_cache
        prepared = cache.prepare(model, contract, tmp_path / 'run', tmp_path / 'cache')
        assert not cache.verify_for_launch(prepared, contract)
        session.run(execute, contract, prepared, None, 1, 1, tmp_path / 'run', (0,))
    assert checked == [(prepared, contract)]


def test_launch_verification_binds_the_exact_rust_owned_snapshot(tmp_path, monkeypatch):
    from multigpubeamsearch import rust_control

    calls = []
    class Store:
        def prepare_sized_for_session(self, *args):
            return str(tmp_path / 'run' / 'weights'), False, 2
        def verify_snapshot(self, *args):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError('artifact hash mismatch')
        def clear(self):
            pass

    monkeypatch.setattr(rust_control, 'extension', lambda: SimpleNamespace(MlpSnapshotStore=Store))
    source = PreparedModel(tmp_path / 'source', {}, 'mlp', 'b' * 64, ('manifest.json', 'a.fp16'), (2, 2))
    contract = SimpleNamespace(graph_hash='a' * 64, state_len=4, num_classes=4, move_count=2)
    with NativeSession(engine='rust', experimental_reuse_model_snapshot=True) as session:
        cache = session.model_snapshot_cache
        prepared = cache.prepare(source, contract, tmp_path / 'run', tmp_path / 'cache')
        assert cache.verify_for_launch(prepared, contract)
        assert calls[0] == (prepared.weights_dir, contract.graph_hash, prepared.artifact_hash, ['a.fp16'])
        unrelated = PreparedModel(prepared.weights_dir, {}, 'mlp', 'b' * 64, prepared.artifact_files)
        assert not cache.verify_for_launch(unrelated, contract)
        changed = SimpleNamespace(graph_hash=contract.graph_hash, state_len=5, num_classes=4, move_count=2)
        assert not cache.verify_for_launch(prepared, changed)
        with pytest.raises(NativeBackendError, match='changed before launch'):
            cache.verify_for_launch(prepared, contract)
