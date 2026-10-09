"""Owned-host acceptance of the unchanged native MLP inference implementation."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path('/workspace/results/native-single-mlp')
inputs=root/'inputs';inputs.mkdir(parents=True,exist_ok=False)
run=Path('/workspace/results/public-ensemble-api/cache/runs/77c2cf8963d1492a821106cdc36a687f')
manifest=json.loads((run/'calibration/inputs/ensemble.json').read_text())
spec={key:manifest[key] for key in ('calibration_states','generators')}
spec['weights_dir']=manifest['models'][0]['weights_dir']
(inputs/'calibration.json').write_text(json.dumps(spec))
binary=Path('/workspace/results/public-ensemble-api/cache/builds/3c758e5bfedf49fe190e89ce7e71fe0102b692c03149bf6f425525e56ef9f3a9/stream1_native_mlp_benchmark')
records=[]
for device in (0,1):
    for batch in (32,256):
        output=subprocess.run([str(binary),str(inputs),str(batch),'8192',str(device)],
            check=True,capture_output=True,text=True,timeout=120)
        (root/f'gpu{device}-batch{batch}.log').write_text(output.stdout+output.stderr)
        rows=[json.loads(line) for line in output.stdout.splitlines() if line.startswith('{')]
        assert len(rows)==1
        record=rows[0]
        assert record['correctness_passed'] is True and record['numeric_error']==0
        assert record['backbone_oracle_max_key_error']==0
        assert record['executor']=='native_cutlass' and len(record['seconds'])==7
        records.append(record)
receipt=dict(records=records,graph='lrx8',model='exact untrained Hamming MLP',
    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    scope='Native scalar-child inference with independent entire-model LibTorch oracle; not public single-model autotuning acceptance')
(root/'acceptance.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)

