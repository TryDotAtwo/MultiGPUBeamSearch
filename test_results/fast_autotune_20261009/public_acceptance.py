import json,time,pathlib,subprocess
from dataclasses import replace
from cayleypy import CayleyGraph,PermutationGroups
from multigpubeamsearch import NativeModel,NativeOptions,beam_search
from multigpubeamsearch.build import build_key
root=pathlib.Path('/workspace/fast-results');d=json.loads((root/'prepared.json').read_text())
bin=pathlib.Path(d['options']['runner_path']).parent
meta=json.loads((bin/'native-build.json').read_text());meta['plan_calibration_protocol']='json-session-v1'
spec={k:v for k,v in meta.items() if k not in ('build_key','binary_sha256','calibration_binary_sha256')}
meta['build_key']=build_key(spec);(bin/'native-build.json').write_text(json.dumps(meta,indent=2))
graph=CayleyGraph(PermutationGroups.lrx(13),device='cuda',random_seed=20261009)
model=NativeModel(**d['model']);opts=dict(d['options']);opts['devices']=tuple(opts['devices'])
options=replace(NativeOptions(**opts),calibration_seconds=45,calibration_pipeline_seconds=150,
 calibration_frontier_max_states=None,calibration_max_batch=2048,touch_bfs_radius=2)
start=graph.apply_path(graph.central_state,[0,2,0,2]).reshape(-1).tolist()
began=time.monotonic()
r=beam_search(graph,start_state=start,predictor=model,native_options=options,backend='native',
 beam_width=8192,max_steps=8,return_path=True)
assert r.path_found
assert graph.apply_path(start,r.path).reshape(-1).tolist()==graph.central_state.tolist()
receipt=dict(public_search=True,replay_valid=True,path_length=r.path_length,
 total_wall_seconds=time.monotonic()-began,metadata=r.native_metadata)
(root/'public-8192.json').write_text(json.dumps(receipt,indent=2,default=str))
profile=r.native_metadata['profile']
print(json.dumps(dict(public_search=True,wall_seconds=receipt['total_wall_seconds'],
 inference=profile['inference_calibration']['calibration_wall_seconds'],
 starts=profile['inference_calibration']['probe_process_starts'],
 pipeline=profile['pipeline_calibration'])),flush=True)
