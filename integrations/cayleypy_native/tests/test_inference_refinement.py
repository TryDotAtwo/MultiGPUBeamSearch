"""Control-path regression: a non-power batch can beat the coarse winner."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from multigpubeamsearch import autotune as module


@pytest.mark.parametrize('cap',[1,17,256,1000,1536,8192,65536])
def test_real_cap_is_measured_without_exceeding_it(cap):
    values=module.coarse_batches(cap)
    assert values[0]==min(256,cap) and cap in values
    assert len(values)==len(set(values))
    assert all(0<x<=cap for x in values)


def test_refinement_adds_non_power_neighbors_and_respects_cap():
    values=module.refinement_batches(1024,1536,{32,64,128,256,512,1024,1536})
    assert {768,896,1152,1280}.issubset(values)
    assert all(0<x<=1536 for x in values)
    assert not set(values)&{32,64,128,256,512,1024,1536}
    assert module.refinement_batches(1,1,{1})==[]


def test_real_tuning_control_selects_refined_batch_with_all_rank_evidence(tmp_path,monkeypatch):
    helper=tmp_path/'stream1_ensemble_benchmark';helper.write_bytes(b'fake helper')
    runtime=SimpleNamespace(runner=tmp_path/'runner',build_metadata={
        'calibration_binary_name':helper.name,'calibration_binary_sha256':module.file_sha256(helper)})
    graph=SimpleNamespace(start=(0,1),generators=((1,0),(0,1)),move_count=2)
    model=SimpleNamespace(backend='ensemble',weights_dir=tmp_path,manifest={
        'ensemble':{'models':[{'weights_dir':'member-0','coefficient':1}]}})
    options=SimpleNamespace(cache_dir=tmp_path/'cache',calibration_max_batch=1024,calibration_seconds=120)
    monkeypatch.setattr(module,'verify_prepared_model',lambda *args:None)
    monkeypatch.setattr(module,'calibration_signature',lambda *args:{'world_size':2,'schema':7,
        'device_indices':[0,1],'gpu_uuids':['GPU-a','GPU-b']})
    class Telemetry:
        throttled=False
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def receipt(self):
            sample=[{'index':0,'uuid':'GPU-a','throttled':False},{'index':1,'uuid':'GPU-b','throttled':False}]
            return {'throttled':False,'errors':[],'samples':[sample,sample]}
    monkeypatch.setattr(module,'CalibrationTelemetry',Telemetry)
    measured=[]
    class Process:
        def __init__(self,command,stdout,**kwargs):
            batch,parents,rank=map(int,command[2:]);measured.append((batch,rank))
            seconds=1. if batch==384 else 2. if batch==512 else 4.
            stdout.write((json.dumps(dict(batch=batch,parents=parents,device=rank,
                seconds=[seconds]*7,correctness_passed=True,numeric_error=0,
                torch_reserved_peak_bytes=1234))+'\n').encode())
            stdout.flush()
        def wait(self,timeout):return 0
        def poll(self):return 0
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    result=module.tune_inference(graph,model,runtime,options,[0,1],8192,tmp_path,{})
    assert result['parent_batch']==384
    assert 384 not in result['coarse_candidates'] and 384 in result['refinement_candidates']
    assert {rank for batch,rank in measured if batch==384}=={0,1}
    assert module.valid_cached_profile(result,result['signature'],1024)


def test_upper_throughput_range_precedes_tiny_capacity_probes():
    values=module.coarse_batches(8192)
    assert values[:2]==[256,8192]
    assert max(values.index(x) for x in (512,1024,2048,4096))<min(values.index(x) for x in (32,64,128))


def test_no_new_probe_when_remaining_budget_cannot_cover_observed_work():
    assert module.probe_time_allowance([])==12.
    assert module.probe_time_allowance([{'seconds':[2.]*7}])==23.
