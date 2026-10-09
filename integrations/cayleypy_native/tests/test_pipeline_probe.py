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

