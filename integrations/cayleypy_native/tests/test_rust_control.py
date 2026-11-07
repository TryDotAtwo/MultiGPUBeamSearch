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


def test_downstream_adapter_only_transfers_model_hardware_and_job_descriptor(monkeypatch,tmp_path):
    from multigpubeamsearch import rust_control,models
    import torch
    runner=tmp_path/'runner';runner.write_bytes(b'runner')
    weights=tmp_path/'weights';weights.mkdir();(weights/'manifest.json').write_text('{}')
    contract=SimpleNamespace(center=(0,1,2),generators=((1,0,2),(0,2,1)),move_count=2)
    model=SimpleNamespace(weights_dir=weights,backend='mlp',manifest={'output_dim':1},artifact_files=('manifest.json',))
    runtime=SimpleNamespace(runner=runner,build_metadata=dict(binary_sha256=hashlib.sha256(b'runner').hexdigest(),
        shape=dict(storage_len=16),frontier_calibration_protocol='natural-growth-v1'))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda dev:SimpleNamespace(multi_processor_count=28,L2_cache_size=3<<20))
    verified=[];monkeypatch.setattr(models,'verify_prepared_model',lambda *args:verified.append(args))
    calls=[]
    class Job:
        def __init__(self,request):calls.append(json.loads(request))
        def run(self):return json.dumps(dict(phase='component_calibrated',pipeline_verified=False))
    monkeypatch.setattr(rust_control,'extension',lambda:SimpleNamespace(DownstreamCalibration=Job))
    inference=dict(parent_batch=16,signature={'bound':'input'})
    result=rust_control.tune_downstream(contract,model,runtime,NativeOptions(cache_dir=tmp_path/'cache'),
        (0,1),65,tmp_path,{'BEAM_TEST_CSV':'/input/test.csv'},runner,inference)
    assert result['pipeline_verified'] is False and len(calls)==1 and len(verified)==2
    request=calls[0]
    assert request['row_factor']==2 and request['physical_bytes']==16 and request['beam_width']==65
    assert request['hardware']==dict(sm_count=28,l2_bytes=3<<20)
    assert request['inference']==inference and request['grow'] is True
    assert request['environment']['BEAM_TEST_CSV']=='/input/test.csv'
    assert not (tmp_path/'pipeline-calibration').exists()
