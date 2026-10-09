from types import SimpleNamespace

import pytest

from multigpubeamsearch import pipeline_autotune as module
from multigpubeamsearch.errors import NativeBackendError


def setup_probe(monkeypatch, fail_actual=False, fail_fixture=False):
    calls = []

    class Probe:
        def __init__(self, runner, env, beam, world, directory, fixtures, storage,
                     deadline, verify):
            self.beam, self.deadline = beam, deadline

        def admit(self, env):
            calls.append((self.beam, dict(env)))
            if fail_actual and self.beam == 1000000:
                raise ValueError('memory budget')
            return [dict(frontier_state_capacity=self.beam // 2,
                         GLOBAL_BEAM_WIDTH_EFFECTIVE=self.beam)] * 2

        def measure(self, env):
            raise AssertionError('selector stub must not run hardware')

    def fixture(*args, **kwargs):
        if fail_fixture:
            raise ValueError('finite graph cannot fill frontier')
        return {'legal': True}

    def select(micro, baseline, admit, measure, **kwargs):
        env = dict(baseline, BEAM_B_MICRO=str(micro * 2))
        admit(env)
        return dict(environment=env, workload_parents=8192)

    monkeypatch.setattr(module, 'NativePipelineProbe', Probe)
    monkeypatch.setattr(module, 'write_frontiers', fixture)
    monkeypatch.setattr(module, 'tune_pipeline', select)
    monkeypatch.setattr(module, 'verify_prepared_model', lambda *args: None)
    return calls


def invoke(tmp_path):
    options = SimpleNamespace(calibration_pipeline_seconds=600,
                              calibration_frontier_max_states=8192,
                              calibration_max_batch=1024)
    runtime = SimpleNamespace(build_metadata={'shape': {'storage_len': 112}})
    return module.tune_downstream(None, None, runtime, options, [0, 1],
        1000000, tmp_path, {}, 'runner',
        {'parent_batch': 1024, 'phase': 'inference_verified'})


def test_bounded_measurement_preserves_requested_beam_and_inference(tmp_path, monkeypatch):
    calls = setup_probe(monkeypatch)
    result = invoke(tmp_path)
    assert result['requested_beam_effective'] == 1000000
    assert result['measurement_scope'] == 'bounded_legal_frontier'
    assert result['environment']['BEAM_B_MICRO'] == '2048'
    assert all(env['BEAM_ENSEMBLE_INFERENCE_MICRO'] == '1024' for _, env in calls)
    assert [beam for beam, _ in calls] == [1000000, 8192, 1000000, 8192, 1000000]


def test_real_beam_admission_failure_cannot_be_hidden_by_small_fixture(tmp_path, monkeypatch):
    calls = setup_probe(monkeypatch, fail_actual=True)
    with pytest.raises(NativeBackendError, match='requested beam'):
        invoke(tmp_path)
    assert [beam for beam, _ in calls] == [1000000]


def test_finite_graph_declines_calibration_without_claiming_verification(tmp_path, monkeypatch):
    setup_probe(monkeypatch, fail_fixture=True)
    result = invoke(tmp_path)
    assert result['phase'] == 'not_measured'
    assert result['pipeline_verified'] is False
    assert 'environment' not in result

