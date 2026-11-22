"""Typed public configuration and one-job cancellation through the actual PyO3 module."""
import importlib.util,json,sys,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('_beam_runtime',sys.argv[1])
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
assert hasattr(runtime,'InferenceCalibration') and hasattr(runtime,'validate_options')
options=json.loads(runtime.validate_options('{}'))
assert options['calibration_max_batch']==8192
for field,value in [('devices',[0,0]),('build_jobs',True),('unknown',1),('timeout_seconds',-1)]:
    bad=dict(options);bad[field]=value
    try:runtime.validate_options(json.dumps(bad))
    except ValueError:pass
    else:raise AssertionError(field)
with tempfile.TemporaryDirectory() as tmp:
    request=dict(options=dict(options,cache_dir=tmp),graph=dict(center=[0,1],generators=[[1,0]]),
        helper=dict(path='/unused',sha256='a'*64,bytes=None),
        artifacts=[dict(path='/unused-model',sha256='b'*64,bytes=None)],
        probe_manifest=dict(weights_dir='/unused'),ensemble=False,
        signature=dict(world_size=1,device_indices=[0],gpu_uuids=[]),beam_width=32,
        protocol='oneshot',capacity_bootstrap=False,run_dir=tmp,environment={},shared_profile=None)
    owner=runtime.InferenceCalibration(json.dumps(request));owner.cancel()
    try:owner.run()
    except RuntimeError as error:assert 'cancelled' in str(error)
    else:raise AssertionError('cancelled calibration launched')
    assert not list(Path(tmp).iterdir())
print('PYO3_CONFIG_PRE_CANCEL_OK')
