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
        return dict(environment=env, workload_parents=1000000)

    monkeypatch.setattr(module, 'NativePipelineProbe', Probe)
    monkeypatch.setattr(module, 'write_frontiers', fixture)
    monkeypatch.setattr(module, 'tune_pipeline', select)
    monkeypatch.setattr(module, 'verify_prepared_model', lambda *args: None)
    return calls


def invoke(tmp_path):
    options = SimpleNamespace(calibration_pipeline_seconds=600,
                              calibration_frontier_max_states=None,
                              calibration_max_batch=1024)
    runtime = SimpleNamespace(build_metadata={'shape': {'storage_len': 112}})
    return module.tune_downstream(None, None, runtime, options, [0, 1],
        1000000, tmp_path, {}, 'runner',
        {'parent_batch': 1024, 'phase': 'inference_verified'})


def test_exact_measurement_preserves_requested_beam_and_inference(tmp_path, monkeypatch):
    calls = setup_probe(monkeypatch)
    result = invoke(tmp_path)
    assert result['requested_beam_effective'] == 1000000
    assert result['measurement_scope'] == 'full_requested_frontier'
    assert result['environment']['BEAM_B_MICRO'] == '2048'
    assert all(env['BEAM_ENSEMBLE_INFERENCE_MICRO'] == '1024' for _, env in calls)
    assert [beam for beam, _ in calls] == [1000000]*4


def test_full_frontier_rejects_slow_component_proxy_winner(tmp_path,monkeypatch):
    from multigpubeamsearch.calibration_stats import Measurement
    from multigpubeamsearch import component_autotune,beam_geometry
    plan=dict(frontier_state_capacity=512,GLOBAL_BEAM_WIDTH_EFFECTIVE=1024,
        SHARD_COUNT=4,STREAM3_RING_SLOTS=2,STREAM4_ACTIVE_SORT_SLOTS=1,
        STREAM4_BATCH_CANDIDATES=1024,STREAM4_BATCH_ALIGNMENT=1024,gpu_budget_bytes=10**9)
    class Probe:
        def __init__(self,*args,**kwargs):pass
        def admit(self,env):return [plan]*2
        def measure(self,env,plans):
            seconds=.01 if env.get('marker')=='base' else .03
            return [Measurement('unused',1024,(seconds,seconds),True,True)]*5
    def components(probe,session,plans,baseline,candidates,**kwargs):
        base=dict(baseline,marker='base');candidate=dict(baseline,marker='candidate')
        return dict(selection='candidate',environment=candidate,
            tested=[dict(name='baseline',plans=plans,environment=base),
                    dict(name='candidate',plans=plans,environment=candidate)])
    monkeypatch.setattr(module,'NativePipelineProbe',Probe)
    monkeypatch.setattr(module,'verify_prepared_model',lambda *a:None)
    monkeypatch.setattr(module,'write_frontiers',lambda *a,**kw:{'legal':True})
    monkeypatch.setattr(component_autotune,'tune_components',components)
    monkeypatch.setattr(beam_geometry,'memory_shortlist',lambda *a,**kw:[])
    result=module.tune_downstream(SimpleNamespace(move_count=3),SimpleNamespace(backend='ensemble'),
        SimpleNamespace(build_metadata={'shape':{'storage_len':32},'component_calibration_protocol':'exact-capacity-v1'}),
        SimpleNamespace(calibration_pipeline_seconds=600,calibration_full_frontier=True,
            calibration_frontier_max_states=None,calibration_max_batch=256),[0,1],1024,tmp_path,{},'runner',
        {'parent_batch':256,'phase':'inference_verified'})
    assert result['selection']=='baseline-full-verified'
    assert result['environment']['marker']=='base'
    assert result['estimate']['profile']=='baseline'
    assert len(result['full_comparison_measurements'])==10


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


def test_native_scalar_inference_parent_batch_maps_to_child_row_budget(tmp_path, monkeypatch):
    calls=setup_probe(monkeypatch)
    def select(micro, baseline, **kwargs):
        assert micro==256 and kwargs['tune_outer'] is False
        assert baseline['BEAM_B_MICRO']=='768'
        return dict(environment=baseline,workload_parents=1000000)
    monkeypatch.setattr(module,'tune_pipeline',select)
    options=SimpleNamespace(calibration_pipeline_seconds=600,
        calibration_frontier_max_states=None,calibration_max_batch=1024)
    runtime=SimpleNamespace(build_metadata={'shape':{'storage_len':16}})
    result=module.tune_downstream(SimpleNamespace(move_count=3),
        SimpleNamespace(backend='mlp',manifest={'output_dim':1}),runtime,options,
        [0,1],1000000,tmp_path,{},'runner',
        {'parent_batch':256,'phase':'inference_verified'})
    assert result['environment']['BEAM_B_MICRO']=='768'
    assert all(env['BEAM_ENSEMBLE_INFERENCE_MICRO']=='256' for _,env in calls)
