import pytest
from multigpubeamsearch.pipeline_probe import parse_plan, parse_depths


def test_native_plan_rejects_conflicting_values():
    assert parse_plan('GLOBAL_BEAM_WIDTH_EFFECTIVE=8192\nB_MICRO=32\n')['B_MICRO']==32
    with pytest.raises(ValueError,match='changes'):
        parse_plan('GLOBAL_BEAM_WIDTH_EFFECTIVE=8192\nB_MICRO=32\nB_MICRO=64')


def test_depth_measurements_require_exact_workload_and_completion():
    text='\n'.join(f'calibration_depth={i} rank=0 parents=4096 depth_sec=1.25 next_frontier_size=4096' for i in range(6))
    assert parse_depths(text,4096,6)==[1.25]*6
    with pytest.raises(ValueError,match='workload'):
        parse_depths(text,8192,6)
    with pytest.raises(ValueError,match='all calibration'):
        parse_depths(text,4096,7)
    with pytest.raises(ValueError,match='reordered'):
        parse_depths(text+'\n'+text.splitlines()[0],4096,7)


def test_nonfinite_depth_time_is_never_a_performance_sample():
    with pytest.raises(ValueError,match='timing'):
        parse_depths('calibration_depth=0 rank=0 parents=4096 depth_sec=nan',4096,1)


@pytest.mark.parametrize('row_budget,outer', [(192,384),(64,384),(192,1152)])
def test_component_receipt_uses_native_parent_batch_not_scalar_row_budget(
        tmp_path,monkeypatch,row_budget,outer):
    import time
    from multigpubeamsearch import pipeline_probe as module
    class Telemetry:
        throttled=False
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def receipt(self):return {'throttled':False}
    monkeypatch.setattr(module,'CalibrationTelemetry',Telemetry)
    row=dict(rank=0,correctness_passed=True,full_step_verified=False,
        shard_capacity=8192,outer_candidates=outer,
        stream3_seconds=[.001]*5,stream4_group_seconds=[.001]*5,union_seconds=[.001]*5)
    class Planner:
        supports_components=True
        def measure_components(self,*args):return [row]
    probe=module.NativePipelineProbe('runner',{},4096,1,tmp_path/'probe',
        [tmp_path/'rank.bin'],16,deadline=time.monotonic()+60,
        verify=lambda *args:True,planning_session=Planner())
    plan={'SHARD_CAPACITY_CANDIDATES':8192,'B_MICRO':64,'STREAM3_RING_SLOTS':2}
    env={'BEAM_B_MICRO':str(row_budget),'BEAM_STREAM3_RING_SLOTS':'2'}
    if outer==384:
        assert probe.measure_components(env,[plan],move_count=3)==[row]
    else:
        with pytest.raises(ValueError,match='contract mismatch'):
            probe.measure_components(env,[plan],move_count=3)


def test_slow_candidate_has_own_timeout_and_cleans_every_rank(tmp_path,monkeypatch):
    import subprocess
    import time
    from multigpubeamsearch import pipeline_probe as module, backend
    launched=[];stopped=[];timeouts=[]
    class Process:
        def __init__(self,*args,**kwargs):launched.append(self)
        def wait(self,timeout):
            timeouts.append(timeout)
            raise subprocess.TimeoutExpired('candidate',timeout)
        def poll(self):return None
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    monkeypatch.setattr(backend,'_stop_process_tree',lambda process:stopped.append(process))
    probe=module.NativePipelineProbe('runner',{},8192,2,tmp_path,
        [tmp_path/'rank0.bin',tmp_path/'rank1.bin'],16,
        deadline=time.monotonic()+600,verify=lambda *args:True)
    probe.fastest_measure_wall_seconds=.2
    with pytest.raises(ValueError,match='timed out'):
        probe._run({},planning=False)
    assert len(launched)==2 and stopped==launched
    assert 0<timeouts[0]<=30

