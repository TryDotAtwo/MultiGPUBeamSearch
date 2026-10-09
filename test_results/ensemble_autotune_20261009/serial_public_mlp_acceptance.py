"""Serial public probes with fresh profiles and already verified runner snapshots."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/workspace/results/serial-public-mlp');root.mkdir(exist_ok=False)
cache='/workspace/results/serial-public-mlp-cache'
env=dict(os.environ,PYTHONPATH='/workspace/source/integrations/cayleypy_native')
specs=[('native-one','/workspace/public_native_single_one_gpu_probe.py',
        'public-native-single-one-gpu','public-native-single-one-gpu-v2'),
       ('libtorch-two','/workspace/public_libtorch_single_gpu_probe.py',
        'public-libtorch-single-api','public-libtorch-single-api-v2')]
receipts={}
for name,script,old,new in specs:
    text=Path(script).read_text().replace(old,new).replace(
        '/workspace/results/public-ensemble-api/cache',cache)
    if name=='native-one':
        # Cache substitution must not change the verified input runner path.
        text=text.replace(cache+'/builds/',
            '/workspace/results/public-ensemble-api/cache/builds/')
    else:
        prepared=json.loads(Path('/workspace/results/public-libtorch-single-api/prepared.json').read_text())
        runner=Path(prepared['preparation_dir'])/'bin/production_runner_libtorch_stream1'
        assert runner.is_file()
        text=text.replace("inference_backend='libtorch',",
            "inference_backend='libtorch',runner_path="+repr(str(runner))+",")
    path=root/(name+'.py');path.write_text(text)
    with (root/(name+'.log')).open('w') as log:
        subprocess.run(['/venv/main/bin/python',str(path)],env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    receipt=json.loads(Path('/workspace/results',new,'acceptance.json').read_text())
    assert receipt['replay_valid'] and receipt['path_found']
    assert receipt['metadata']['profile']['inference_calibration']['cache_hit'] is False
    receipts[name]=receipt
(root/'acceptance.json').write_text(json.dumps(receipts,indent=2))
print(json.dumps({key:dict(replay_valid=value['replay_valid'],
    inference_micro=value['metadata']['profile']['inference_calibration']['parent_batch'])
    for key,value in receipts.items()}),flush=True)

