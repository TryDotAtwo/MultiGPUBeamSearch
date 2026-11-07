import hashlib,json
from types import SimpleNamespace
from multigpubeamsearch.options import NativeOptions

def test_thin_adapter_passes_validated_model_and_options_to_one_rust_call(monkeypatch,tmp_path):
    from multigpubeamsearch import rust_control
    from multigpubeamsearch import models,autotune,shared_profiles
    helper=tmp_path/'stream1_native_mlp_benchmark';helper.write_bytes(b'helper')
    weights=tmp_path/'weights';weights.mkdir();(weights/'manifest.json').write_text('{}')
    (weights/'blob.fp16').write_bytes(b'weights')
    contract=SimpleNamespace(center=(0,1),generators=((1,0),),graph_hash='a'*64)
    model=SimpleNamespace(weights_dir=weights,backend='mlp',manifest={},artifact_hash='b'*64,
        artifact_files=('manifest.json','blob.fp16'))
    runtime=SimpleNamespace(runner=tmp_path/'runner',build_metadata=dict(calibration_binary_name=helper.name,
        calibration_binary_sha256=hashlib.sha256(b'helper').hexdigest(),calibration_protocol='oneshot'))
    verified=[];monkeypatch.setattr(models,'verify_prepared_model',lambda *args:verified.append(args))
    monkeypatch.setattr(autotune,'calibration_signature',lambda *args:dict(world_size=1,device_indices=[0],gpu_uuids=[]))
    monkeypatch.setattr(shared_profiles,'lookup',lambda *args:None)
    calls=[]
    class Calibration:
        def __init__(self,request):calls.append(json.loads(request))
        def run(self):return json.dumps(dict(phase='inference_verified',parent_batch=8))
    monkeypatch.setattr(rust_control,'extension',lambda:SimpleNamespace(InferenceCalibration=Calibration))
    options=NativeOptions(cache_dir=tmp_path/'cache',calibration_max_batch=32)
    outcome=rust_control.tune_inference(contract,model,runtime,options,(0,),64,tmp_path,{})
    assert outcome['parent_batch']==8 and len(calls)==1 and len(verified)==2
    request=calls[0]
    assert request['options']['calibration_max_batch']==32
    assert request['graph']==dict(center=[0,1],generators=[[1,0]])
    assert len(request['artifacts'])==2 and request['helper']['sha256']==runtime.build_metadata['calibration_binary_sha256']
    assert request['probe_manifest']==dict(weights_dir=str(weights))
    assert not list(tmp_path.glob('calibration*'))
