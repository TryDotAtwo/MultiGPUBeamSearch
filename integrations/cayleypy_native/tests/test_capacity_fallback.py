import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from multigpubeamsearch import autotune, calibration_session, plan_session
from multigpubeamsearch.beam_capacity import CapacityRejected
from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.options import NativeOptions


@pytest.mark.parametrize('capacity_failure',[True,False])
def test_initial_batch_capacity_failure_falls_back_even_after_coarse_budget(tmp_path,monkeypatch,capacity_failure):
    clock=[0.];calls=[]
    monkeypatch.setattr(autotune,'coarse_batches',lambda cap:[8192,16384,4096,2048])
    monkeypatch.setattr(autotune.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(autotune,'verify_prepared_model',lambda *args:None)
    monkeypatch.setattr(autotune,'calibration_signature',lambda *args:{'world_size':2})
    monkeypatch.setattr(autotune,'verified_telemetry',lambda *args:True)
    monkeypatch.setattr(autotune,'free_memory_snapshot',lambda signature:{'gpu0':4000,'gpu1':4000})
    class Telemetry:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def receipt(self):return {'throttled':False}
    monkeypatch.setattr(autotune,'CalibrationTelemetry',Telemetry)
    class Pool:
        starts=2
        def __init__(self,*args):
            assert 'BEAM_ENSEMBLE_INFERENCE_MICRO' not in args[5]
        def close(self):pass
        def measure(self,batch,parents,deadline):
            calls.append(batch)
            if len(calls)==1:
                clock[0]=65.
                return [],['rank 0: CUDA capacity rejected: exited' if capacity_failure else 'rank 0: numerical oracle failed']
            clock[0]=78.
            return [dict(device=r,batch=batch,parents=parents,seconds=[1.]*7,
                correctness_passed=True,numeric_error=0,torch_reserved_peak_bytes=1000) for r in range(2)],[]
    monkeypatch.setattr(calibration_session,'EnsembleProbePool',Pool)
    helper=tmp_path/'helper';helper.write_bytes(b'test-only executable identity')
    runtime=SimpleNamespace(runner=tmp_path/'runner',build_metadata={
        'calibration_binary_name':helper.name,'calibration_binary_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),
        'calibration_protocol':'json-session-v1'})
    model=SimpleNamespace(backend='ensemble',weights_dir=tmp_path,manifest={'ensemble':{'models':[]}})
    contract=SimpleNamespace(start=(1,0),generators=((0,1),(1,0)),move_count=2)
    options=NativeOptions(cache_dir=tmp_path/'cache',calibration_seconds=90,calibration_max_batch=65536)
    environment={'BEAM_ENSEMBLE_INFERENCE_MICRO':'64'}
    args=(contract,model,runtime,options,(0,1),131072,tmp_path,environment)
    if not capacity_failure:
        with pytest.raises(NativeBackendError,match='baseline inference calibration failed'):
            autotune.tune_inference(*args)
        assert calls==[8192]
    else:
        result=autotune.tune_inference(*args)
        assert calls==[8192,4096] and result['parent_batch']==4096
        assert result['records'][0]['parents']==65536
        assert '8192' in result['rejected']
    assert environment['BEAM_ENSEMBLE_INFERENCE_MICRO']=='64'


def test_planner_reserve_initialization_is_capacity_not_protocol_failure(tmp_path,monkeypatch):
    owners=[]
    class Session:
        def __init__(self,command,env,path,deadline):
            self.log=Path(path).open('w');self.log.write('production_runner_error=insufficient blend GPU reserve\n');self.log.flush();self.closed=False;owners.append(self)
        def receive(self):raise RuntimeError('persistent inference probe exited')
        def close(self):self.closed=True;self.log.close()
    monkeypatch.setattr(plan_session,'InferenceSession',Session)
    import time
    with pytest.raises(CapacityRejected,match='measured blend reserve'):
        plan_session.NativePlanSession('runner',{},2,tmp_path/'plan',deadline=time.monotonic()+10)
    assert len(owners)==2 and all(owner.closed for owner in owners)


def test_capacity_limited_cache_retries_when_memory_pressure_lifts(monkeypatch):
    data={'capacity_limited_search':True,'initial_free_memory_mib':{'gpu0':4000,'gpu1':4000}}
    for current,expected in (({'gpu0':3990,'gpu1':4010},True),
            ({'gpu0':8000,'gpu1':4000},False),(None,False)):
        monkeypatch.setattr(autotune,'free_memory_snapshot',lambda signature:current)
        assert autotune.capacity_cache_reusable(data,{}) is expected
    assert autotune.capacity_cache_reusable({}, {})


