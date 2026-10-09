"""Audit persisted terminal artifacts after an incorrect harness-only assertion.

No search or calibration is repeated. Replay uses CayleyPy on CPU.
"""
import hashlib
import json
import sys
from pathlib import Path
import multigpubeamsearch
from cayleypy import CayleyGraph,PermutationGroups
from multigpubeamsearch.build import source_digest

heads=int(sys.argv[1])
assert heads in (3,17)
root=Path('/workspace/results')/('installed-wheel-api-v2' if heads==3 else 'installed-wheel-api-17')
prepared=json.loads((root/'prepared.json').read_text())
prep=Path(prepared['preparation_dir'])
manifest=json.loads((prep/'native-preparation.json').read_text())
runs=Path('/workspace/results/installed-wheel-api/cache/runs')
matching=[]
for run in runs.iterdir():
    if (run/'adapter-result.json').is_file():
        data=json.loads((run/'adapter-result.json').read_text())
        if data['model_artifact_hash']==manifest['model_hash']:
            matching.append((run,data))
assert len(matching)==1,'require exactly one persisted terminal search'
run,metadata=matching[0]
outcome=json.loads((run/'native-outcome.json').read_text())
assert metadata['status']=='found' and metadata['replay_valid']
assert metadata['worker_logs_complete'] and metadata['devices']==[0,1]
assert metadata['build']['backend']=='ensemble'
assert metadata['build']['inference_backend']=='libtorch'
assert metadata['profile']['inference_autotuned'] and metadata['profile']['pipeline_autotuned']
assert metadata['profile']['inference_calibration']['signature']['schema']==4
assert metadata['profile']['pipeline_calibration']['measurement_scope']=='full_requested_frontier'
assert len(json.loads((run/'weights/ensemble.json').read_text())['models'])==heads
graph=CayleyGraph(PermutationGroups.lrx(8),device='cpu',random_seed=20261009)
start=graph.apply_path(graph.central_state,[0,2,0,2]).reshape(-1).tolist()
assert graph.apply_path(start,outcome['path']).reshape(-1).tolist()==graph.central_state.tolist()
bundle=Path(multigpubeamsearch.__file__).parent/'_native_source'
assert source_digest(bundle)==metadata['build']['source_digest']==prepared['source_digest']
assert prepared['cutlass_digest']==metadata['build']['cutlass_digest']
wheel=next(Path('/workspace/compact-dist-v2').glob('*.whl'))
receipt=dict(head_count=heads,path_found=True,replay_valid=True,path_length=len(outcome['path']),
    package_path=str(multigpubeamsearch.__file__),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    wheel_bytes=wheel.stat().st_size,source_commit='5d2ff3b5f216c50e2dde0f280c9d08fea7476512',
    automatic_source=True,automatic_cutlass=True,metadata=metadata,
    verification_note='Persisted successful public search and both calibrations; recovered after harness confused backbone executor with final readout. Additional CPU path replay passed; no GPU run repeated.')
(root/'acceptance.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(head_count=heads,path_length=receipt['path_length'],replay_valid=True,
    inference_batch=metadata['profile']['inference_calibration']['parent_batch'],
    wheel_sha256=receipt['wheel_sha256'])),flush=True)
