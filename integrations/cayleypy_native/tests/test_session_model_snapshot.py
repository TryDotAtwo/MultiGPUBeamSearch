"""Session-local model reuse must preserve content and lifetime checks."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from cayleypy import CayleyGraph, PermutationGroups

from multigpubeamsearch import NativeModel, NativeOptions, NativeSession
from multigpubeamsearch import dispatch, models
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.hamming import write_hamming_artifact
from multigpubeamsearch.options import NativeOutcome


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    graph = CayleyGraph(PermutationGroups.lrx(4), device='cpu')
    start = [1, 0, 2, 3]
    contract = GraphContract.from_graph(graph, start)
    source = write_hamming_artifact(contract, tmp_path / 'source')
    model = NativeModel(source, contract.graph_hash)
    monkeypatch.setattr(dispatch, 'runtime_devices', lambda *_: (0,))
    monkeypatch.setattr('multigpubeamsearch.bootstrap.prepare_sources', lambda opts: opts)
    monkeypatch.setattr(dispatch, 'prepare_runtime', lambda *args, **kwargs:
                        SimpleNamespace(build_metadata={'inference_backend': 'cutlass'}))
    def run_native(contract, model, options, width, depth, run_dir, devices, **kwargs):
        # The real session checks the prepared snapshot before submitting work.
        models.verify_prepared_model(model, contract)
        return NativeOutcome(None, 0.01, width, run_dir, {'model_manifest': model.manifest})
    monkeypatch.setattr(dispatch, 'run_native', run_native)
    options = NativeOptions(cache_dir=tmp_path / 'cache', autotune=False)
    def search(session, predictor=model, native_options=options, backend='native'):
        return session.beam_search(graph, backend=backend, native_options=native_options, start_state=start,
                                   predictor=predictor, beam_width=64, max_steps=4)
    return source, model, contract, search


@pytest.mark.parametrize('engine', ['python', 'rust'])
def test_opt_in_reuses_validated_private_snapshot_without_copy(fixture, engine):
    source, _, _, search = fixture
    with NativeSession(engine=engine, experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        second = search(session)
        first_meta, second_meta = (x.native_metadata['model_snapshot'] for x in (first, second))
        expected = sum(path.stat().st_size for path in source.iterdir() if path.is_file())
        assert first_meta['reused'] is False and first_meta['copied_bytes'] == expected
        assert second_meta['reused'] is True and second_meta['copied_bytes'] == 0
        assert first_meta['snapshot_dir'] == second_meta['snapshot_dir']
        assert (Path(first.native_metadata['run_dir']) / 'weights').is_dir()
        assert not (Path(second.native_metadata['run_dir']) / 'weights').exists()
    assert Path(first_meta['snapshot_dir']).is_dir()


def test_source_change_creates_new_snapshot(fixture):
    source, _, _, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        (source / 'output_bias.fp16').write_bytes(b'\x00\x3c')
        second = search(session)
        assert first.native_metadata['model_hash'] != second.native_metadata['model_hash']
        assert second.native_metadata['model_snapshot']['reused'] is False
        assert second.native_metadata['model_snapshot']['copied_bytes'] > 0
        assert first.native_metadata['model_snapshot']['snapshot_dir'] != \
               second.native_metadata['model_snapshot']['snapshot_dir']


def test_expected_hash_is_checked_even_when_a_snapshot_is_cached(fixture):
    source, _, contract, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        pinned = NativeModel(source, contract.graph_hash,
                             expected_artifact_hash=first.native_metadata['model_hash'])
        (source / 'output_bias.fp16').write_bytes(b'\x00\x3c')
        with pytest.raises(NativeBackendError, match='artifact hash changed'):
            search(session, pinned)


def test_source_change_during_reuse_fails_before_launch(fixture, monkeypatch):
    source, _, _, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        search(session)
        original = models._validate_artifact
        checks = []
        def validate(path, contract, backend):
            result = original(path, contract, backend)
            if Path(path) == source and not checks:
                checks.append(True)
                (source / 'output_bias.fp16').write_bytes(b'\x00\x3c')
            return result
        monkeypatch.setattr(models, '_validate_artifact', validate)
        with pytest.raises(NativeBackendError, match='changed before launch'):
            search(session)


def test_result_metadata_mutation_cannot_change_the_cached_manifest(fixture):
    _, _, contract, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        first.native_metadata['model_manifest']['output_dim'] = contract.move_count
        second = search(session)
        assert second.native_metadata['model_snapshot']['reused'] is True
        assert second.native_metadata['model_manifest']['output_dim'] == 1


def test_reuse_validates_source_twice_and_snapshot_once(fixture, monkeypatch):
    source, _, _, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        snapshot = Path(first.native_metadata['model_snapshot']['snapshot_dir'])
        original = models._validate_artifact
        checked = []
        def validate(path, contract, backend):
            checked.append(Path(path))
            return original(path, contract, backend)
        monkeypatch.setattr(models, '_validate_artifact', validate)
        second = search(session)
        assert second.native_metadata['model_snapshot']['reused']
        assert checked.count(source) == 2
        assert checked.count(snapshot) == 1


@pytest.mark.parametrize('damage', ['change', 'missing'])
def test_changed_or_missing_private_snapshot_fails_before_launch(fixture, damage):
    _, _, _, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        snapshot = Path(first.native_metadata['model_snapshot']['snapshot_dir'])
        target = snapshot / 'output_bias.fp16'
        if damage == 'change':
            target.write_bytes(b'\x00\x3c')
        else:
            target.unlink()
        with pytest.raises(NativeBackendError, match='changed before launch|missing or wrong length'):
            search(session)


def test_same_valid_content_from_another_source_reuses_snapshot(fixture, tmp_path):
    source, _, contract, search = fixture
    alternate = tmp_path / 'alternate'
    shutil.copytree(source, alternate)
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        second = search(session, NativeModel(alternate, contract.graph_hash))
        assert first.native_metadata['model_hash'] == second.native_metadata['model_hash']
        assert second.native_metadata['model_snapshot']['reused'] is True


def test_changed_cache_directory_gets_a_new_snapshot(fixture, tmp_path):
    _, model, _, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        options = NativeOptions(cache_dir=tmp_path / 'other-cache', autotune=False)
        second = search(session, model, options)
        assert second.native_metadata['model_snapshot']['reused'] is False
        assert first.native_metadata['model_snapshot']['snapshot_dir'] != \
               second.native_metadata['model_snapshot']['snapshot_dir']


def test_auto_fallback_mode_keeps_per_call_snapshots(fixture):
    _, _, _, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first, second = search(session, backend='auto'), search(session, backend='auto')
        assert 'model_snapshot' not in first.native_metadata
        assert 'model_snapshot' not in second.native_metadata
        assert (Path(first.native_metadata['run_dir']) / 'weights').is_dir()
        assert (Path(second.native_metadata['run_dir']) / 'weights').is_dir()


def test_default_session_still_creates_a_private_snapshot_per_call(fixture):
    _, _, _, search = fixture
    with NativeSession() as session:
        first, second = search(session), search(session)
        assert 'model_snapshot' not in first.native_metadata
        assert (Path(first.native_metadata['run_dir']) / 'weights').is_dir()
        assert (Path(second.native_metadata['run_dir']) / 'weights').is_dir()


@pytest.mark.parametrize('invalid', [1, 'yes', None])
def test_snapshot_opt_in_requires_boolean(invalid):
    with pytest.raises(ValueError, match='experimental_reuse_model_snapshot'):
        NativeSession(experimental_reuse_model_snapshot=invalid)


def test_session_cannot_close_while_model_preparation_owns_snapshot_cache(fixture, monkeypatch):
    _, _, _, search = fixture
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        original = models._validate_artifact
        observed = []
        def validate(*args, **kwargs):
            with pytest.raises(RuntimeError, match='during a search'):
                session.close()
            observed.append(True)
            return original(*args, **kwargs)
        monkeypatch.setattr(models, '_validate_artifact', validate)
        search(session)
        assert observed


def test_session_cannot_close_during_runtime_preparation(fixture, monkeypatch):
    import threading

    _, _, _, search = fixture
    entered, resume = threading.Event(), threading.Event()
    errors = []
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        def prepare(*args, **kwargs):
            entered.set()
            assert resume.wait(timeout=10)
            return SimpleNamespace(build_metadata={'inference_backend': 'cutlass'})
        monkeypatch.setattr(dispatch, 'prepare_runtime', prepare)
        def work():
            try:
                search(session)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=work)
        thread.start()
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(RuntimeError, match='during a search'):
                session.close()
            with pytest.raises(RuntimeError, match='concurrent native session searches'):
                session.run(lambda *args, **kwargs: None, None, None, None, 1, 1, Path('.'), (0,))
        finally:
            resume.set()
            thread.join(timeout=10)
        assert not thread.is_alive() and not errors


@pytest.mark.parametrize('entry', ['prepare_model_snapshot', 'prepare_nccl_identity', 'launch'])
def test_foreign_session_entry_is_rejected_during_public_search(fixture, monkeypatch, entry):
    import threading

    _, _, _, search = fixture
    entered, resume = threading.Event(), threading.Event()
    errors = []
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        def prepare(*args, **kwargs):
            entered.set()
            assert resume.wait(timeout=10)
            return SimpleNamespace(build_metadata={'inference_backend': 'cutlass'})
        monkeypatch.setattr(dispatch, 'prepare_runtime', prepare)
        def work():
            try:search(session)
            except BaseException as error:errors.append(error)
        thread = threading.Thread(target=work)
        thread.start()
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(RuntimeError, match='concurrent native session searches'):
                if entry == 'launch':
                    session.launch(['unused'], contract=None, run_dir=Path('.'), env={}, world=1, timeout=1)
                else:
                    with getattr(session, entry)():
                        pass
        finally:
            resume.set()
            thread.join(timeout=10)
        assert not thread.is_alive() and not errors


def test_incompatible_resident_relaunch_uses_a_fresh_run_snapshot(fixture, monkeypatch):
    from multigpubeamsearch.persistent import current_session

    source, model, _, search = fixture
    launches = []
    class Workers:
        pids = [123]
        sequence = 1
        def close(self):
            pass
    def run(contract, prepared, options, width, depth, run_dir, devices, *, runtime):
        session = current_session()
        def execute(contract, prepared, options, width, depth, run_dir, devices, *, runtime):
            launches.append((Path(prepared.weights_dir), Path(run_dir)))
            session._workers = Workers()
            return NativeOutcome(None, .01, width, run_dir,
                                 {'model_manifest': prepared.manifest, 'requested_beam_width': width,
                                  'profile': {}, 'terminal_records': 1})
        return session.run(execute, contract, prepared, options, width, depth, run_dir, devices, runtime=runtime)
    monkeypatch.setattr(dispatch, 'run_native', run)
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        changed = NativeOptions(cache_dir=source.parent.parent / 'cache', autotune=False, experimental_index_sort=True)
        second = search(session, model, changed)
        assert len(launches) == 2
        assert launches[0][0] == launches[0][1] / 'weights'
        assert launches[1][0] == launches[1][1] / 'weights'
        assert second.native_metadata['model_snapshot']['reused'] is False
        assert second.native_metadata['model_snapshot']['copied_bytes'] > 0
        assert first.native_metadata['model_snapshot']['snapshot_dir'] != second.native_metadata['model_snapshot']['snapshot_dir']


def test_retained_snapshot_modified_during_runtime_preparation_never_reaches_workers(fixture, monkeypatch):
    from multigpubeamsearch.persistent import current_session

    _, _, _, search = fixture
    launches, submits = [], []
    class Workers:
        pids = [123]
        sequence = 1
        def close(self):pass
        def submit(self, *args, **kwargs):
            submits.append(True)
            raise AssertionError('corrupt snapshot reached a resident worker')
    def run(contract, prepared, options, width, depth, run_dir, devices, *, runtime):
        session = current_session()
        def execute(contract, prepared, options, width, depth, run_dir, devices, *, runtime):
            launches.append(True)
            session._workers = Workers()
            return NativeOutcome(None, .01, width, run_dir,
                                 {'model_manifest': prepared.manifest, 'requested_beam_width': width,
                                  'profile': {}, 'terminal_records': 1})
        return session.run(execute, contract, prepared, options, width, depth, run_dir, devices, runtime=runtime)
    monkeypatch.setattr(dispatch, 'run_native', run)
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        first = search(session)
        snapshot = Path(first.native_metadata['model_snapshot']['snapshot_dir'])
        def prepare(*args, **kwargs):
            (snapshot / 'output_bias.fp16').write_bytes(b'\x00\x3c')
            return SimpleNamespace(build_metadata={'inference_backend': 'cutlass'})
        monkeypatch.setattr(dispatch, 'prepare_runtime', prepare)
        with pytest.raises(NativeBackendError, match='changed before launch'):
            search(session)
        assert len(launches) == 1 and not submits


def test_direct_run_never_refreshes_a_different_prepared_model(fixture, tmp_path):
    source, predictor, contract, _ = fixture
    options = NativeOptions(cache_dir=tmp_path / 'cache', autotune=False)
    alternate = tmp_path / 'different-source'
    shutil.copytree(source, alternate)
    (alternate / 'output_bias.fp16').write_bytes(b'\x00\x3c')
    with NativeSession(experimental_reuse_model_snapshot=True) as session:
        with session.prepare_model_snapshot() as cache:
            models.prepare_model(predictor, contract, options, tmp_path / 'run-a', _snapshot_cache=cache)
            models.prepare_model(predictor, contract, options, tmp_path / 'run-a-repeat', _snapshot_cache=cache)
            assert cache.last['reused']
        different = models.prepare_model(NativeModel(alternate, contract.graph_hash), contract,
                                         options, tmp_path / 'run-b')
        class Workers:
            pids = [123]
            sequence = 1
            def close(self):
                pass
        observed = []
        def execute(contract, prepared, options, width, depth, run_dir, devices, *, runtime):
            observed.append(prepared.artifact_hash)
            session._workers = Workers()
            return NativeOutcome(None, .01, width, run_dir,
                                 {'requested_beam_width': width, 'profile': {}, 'terminal_records': 1})
        run_dir = tmp_path / 'direct-run'
        run_dir.mkdir()
        session.run(execute, contract, different, options, 64, 4, run_dir, (0,))
        assert observed == [different.artifact_hash]
