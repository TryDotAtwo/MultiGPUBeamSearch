"""The session must request and check attestation from every native rank."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from multigpubeamsearch import NativeOptions
from multigpubeamsearch.errors import NativeBackendError


@pytest.mark.parametrize('receipt', ['valid', 'missing', 'wrong', 'duplicate'])
@pytest.mark.parametrize('mode', ['cutlass', 'libtorch', 'inactive', 'unrelated'])
def test_experimental_mlp_session_requires_each_loaded_weight_receipt(monkeypatch, tmp_path, receipt, mode):
    from multigpubeamsearch import backend, persistent
    from multigpubeamsearch.build import shape_contract

    graph_hash, model_hash = 'a' * 64, 'b' * 64
    contract = SimpleNamespace(state_len=2, move_count=2, start=(1, 0), center=(0, 1),
        graph_hash=graph_hash, replay=lambda path: tuple(path) == (1,),
        to_puzzle_info=lambda: {'central_state': [0, 1], 'generators': {'m0': [0, 1], 'm1': [1, 0]}})
    model = SimpleNamespace(weights_dir=tmp_path/'weights', backend='mlp', artifact_hash=model_hash,
                            manifest={'dtype': 'fp16', 'output_dim': 1})
    runtime = backend.PreparedRuntime(tmp_path/'runner',
        {'shape': shape_contract(contract), 'backend': 'mlp',
         'inference_backend': 'libtorch' if mode == 'libtorch' else 'cutlass'}, (86,))
    options = NativeOptions(cache_dir=tmp_path, inference_backend='cutlass', autotune=False,
                            touch_bfs_radius=0, report_calibration=False)
    cache = None if mode == 'inactive' else SimpleNamespace(
        last={'reused': False}, binds=lambda model, contract: mode != 'unrelated')
    session = SimpleNamespace(model_snapshot_cache=cache)

    def launch(args, *, run_dir, env, world, **kwargs):
        if mode == 'cutlass':
            assert env.get('BEAM_MLP_GRAPH_HASH') == graph_hash
            assert env.get('BEAM_MLP_ARTIFACT_HASH') == model_hash
        else:
            assert 'BEAM_MLP_GRAPH_HASH' not in env and 'BEAM_MLP_ARTIFACT_HASH' not in env
        if mode == 'libtorch':
            assert env['BEAM_STREAM1_EXECUTOR'] == 'libtorch_eager'
        directory = Path(run_dir)/'resident-session'
        directory.mkdir()
        for rank in range(world):
            if receipt == 'missing' and rank == 1:
                continue
            value = 'c' * 64 if receipt == 'wrong' and rank == 1 else model_hash
            marker = 'BEAM_MLP_ARTIFACT_VERIFIED=' + value + '\n'
            if receipt == 'duplicate' and rank == 1:
                marker += marker
            (directory/f'bootstrap-{rank}.log').write_text(marker)
        (Path(run_dir)/'native.log').write_text(
            'GLOBAL_BEAM_WIDTH_EFFECTIVE=16\n'
            'puzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 solution=m1\n')
        return 0.1

    session.launch = launch
    monkeypatch.setattr(persistent, 'current_session', lambda: session)
    monkeypatch.setattr(backend, 'verify_prepared_model', lambda *args: None)
    monkeypatch.setattr(backend, 'snapshot_runtime_nccl', lambda *args: tmp_path/'libnccl.so.2')
    monkeypatch.setattr(backend, 'snapshot_runtime_runner', lambda *args: tmp_path/'runner')
    monkeypatch.setenv('BEAM_MLP_GRAPH_HASH', 'unrelated-parent-setting')
    monkeypatch.setenv('BEAM_MLP_ARTIFACT_HASH', 'unrelated-parent-setting')

    def execute():
        return backend._run_native_once(contract, model, options, 16, 2, tmp_path/'run', (0, 1),
                                        runtime=runtime)
    if receipt == 'valid' or mode != 'cutlass':
        outcome = execute()
        if mode == 'cutlass':
            assert outcome.metadata['mlp_host_binding'] == {'artifact_hash': model_hash, 'ranks': 2}
        else:
            assert 'mlp_host_binding' not in outcome.metadata
    else:
        with pytest.raises(NativeBackendError, match='MLP.*rank'):
            execute()
