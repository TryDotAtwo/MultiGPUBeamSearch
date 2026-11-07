import hashlib,json
import pytest
from pathlib import Path
from multigpubeamsearch.shared_profiles import find_profile


def registry(tmp_path, identity):
    root=tmp_path/'profiles';(root/'records').mkdir(parents=True)
    data=dict(schema=1,compatibility=identity,inference={'parent_batch':512},
        evidence={'status':'verified','source_commit':'a'*40})
    payload=json.dumps(data).encode();digest=hashlib.sha256(payload).hexdigest()
    record=root/'records'/(digest+'.json');record.write_bytes(payload)
    (root/'index.json').write_text(json.dumps(dict(schema=1,profiles=[dict(compatibility=identity,sha256=digest)])))
    return record


def test_matches_exact_workload_without_claiming_local_verification(tmp_path):
    identity=dict(beam=1000000,backend='cutlass',world_size=2)
    registry(tmp_path,identity)
    got=find_profile(tmp_path,identity)
    assert got['inference']['parent_batch']==512
    assert 'cache_hit' not in got and 'pipeline_verified' not in got
    assert find_profile(tmp_path,dict(identity,beam=10000000)) is None
    assert find_profile(tmp_path,dict(identity,backend='libtorch')) is None


def test_modified_or_missing_profile_is_not_used(tmp_path):
    record=registry(tmp_path,{'beam':100})
    record.write_text('{}')
    assert find_profile(tmp_path,{'beam':100}) is None
    record.unlink()
    assert find_profile(tmp_path,{'beam':100}) is None


def test_missing_registry_falls_back(tmp_path):
    assert find_profile(tmp_path,{'beam':100}) is None


def test_pipeline_proposal_cannot_change_inference_or_inject_paths():
    from multigpubeamsearch.shared_profiles import pipeline_environment
    env={'BEAM_B_MICRO':'1024','BEAM_ENSEMBLE_INFERENCE_MICRO':'512',
         'BEAM_STREAM3_RING_SLOTS':'2','BEAM_SHARD_COUNT':'4',
         'BEAM_STREAM4_BATCH_CANDIDATES':'8192','BEAM_STREAM4_ACTIVE_SORT_SLOTS':'2'}
    p={'inference':{'parent_batch':512},'pipeline':{'environment':env}}
    assert pipeline_environment(p,512)==env
    assert pipeline_environment(p,256) is None
    assert pipeline_environment(dict(p,pipeline={'environment':dict(env,BEAM_BLEND_DIR='/tmp/other')}),512) is None
    assert pipeline_environment(dict(p,pipeline={'environment':dict(env,BEAM_SHARD_COUNT='0')}),512) is None


def test_missing_runtime_identity_falls_back_without_gpu_probe():
    from types import SimpleNamespace
    from multigpubeamsearch.shared_profiles import lookup
    assert lookup(None,None,SimpleNamespace(build_metadata={}),SimpleNamespace(),[],1024) is None


@pytest.mark.parametrize("reject_shared", [False, True])
def test_shared_inference_is_measured_once_then_local_cache_wins(tmp_path, monkeypatch, reject_shared):
    from types import SimpleNamespace as NS
    from multigpubeamsearch import autotune as a, calibration_session, shared_profiles
    helper=tmp_path/'stream1_ensemble_benchmark';helper.write_bytes(b'test')
    calls=[]
    class Telemetry:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def receipt(self):return {'throttled':False,'samples':[]}
    class Pool:
        starts=2
        def __init__(self,*args):pass
        def measure(self,batch,parents,deadline):
            calls.append(batch)
            if reject_shared and batch == 512:
                return [], ['CUDA capacity rejected']
            return [dict(batch=batch,device=rank,parents=parents,
                seconds=[.1]*7,torch_reserved_peak_bytes=1024,
                correctness_passed=True,numeric_error=0) for rank in range(2)],[]
        def close(self):pass
    monkeypatch.setattr(a,'verify_prepared_model',lambda *args:None)
    monkeypatch.setattr(a,'file_sha256',lambda p:'hash')
    monkeypatch.setattr(a,'calibration_signature',lambda *args:{'schema':1,'world_size':2})
    monkeypatch.setattr(a,'free_memory_snapshot',lambda *args:{'gpu0':10000,'gpu1':10000})
    monkeypatch.setattr(a,'graph_samples',lambda *args:[])
    monkeypatch.setattr(a,'CalibrationTelemetry',Telemetry)
    monkeypatch.setattr(a,'verified_telemetry',lambda *args:True)
    monkeypatch.setattr(calibration_session,'EnsembleProbePool',Pool)
    monkeypatch.setattr(shared_profiles,'lookup',lambda *args:{'inference':{'parent_batch':512},'registry_sha256':'a'*64})
    runtime=NS(runner=tmp_path/'runner',build_metadata={
        'calibration_binary_sha256':'hash','calibration_protocol':'json-session-v1'})
    options=NS(cache_dir=tmp_path/'cache',calibration_max_batch=1024,calibration_seconds=90)
    model=NS(backend='mlp',weights_dir=tmp_path)
    contract=NS(generators=[])
    first=tmp_path/'first';first.mkdir()
    result=a.tune_inference(contract,model,runtime,options,[0,1],4096,first,{})
    assert calls[0] == 512
    assert result['shared_profile_validated'] is (not reject_shared)
    assert not result['cache_hit']
    if reject_shared:
        assert len(calls) > 1 and result['parent_batch'] != 512
    else:
        assert calls == [512] and result['parent_batch'] == 512
    measured = list(calls)
    assert a.valid_cached_profile(result,result['signature'],1024)
    def forbidden(*args):raise AssertionError('local cache must precede GitHub lookup')
    monkeypatch.setattr(shared_profiles,'lookup',forbidden)
    second=tmp_path/'second';second.mkdir()
    cached=a.tune_inference(contract,model,runtime,options,[0,1],4096,second,{})
    assert cached['cache_hit'] and calls==measured
