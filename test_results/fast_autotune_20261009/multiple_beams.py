import json,time,pathlib
from dataclasses import replace
from cayleypy import CayleyGraph,PermutationGroups
from multigpubeamsearch import NativeModel,NativeOptions,beam_search
root=pathlib.Path('/workspace/fast-results');d=json.loads((root/'prepared.json').read_text())
graph=CayleyGraph(PermutationGroups.lrx(13),device='cuda',random_seed=20261009)
model=NativeModel(**d['model']);opts=dict(d['options']);opts['devices']=tuple(opts['devices'])
options=replace(NativeOptions(**opts),calibration_seconds=45,calibration_pipeline_seconds=600,
 calibration_frontier_max_states=None,calibration_max_batch=2048,touch_bfs_radius=2)
start=graph.apply_path(graph.central_state,[0,2,0,2]).reshape(-1).tolist()
for beam in (65536,1048576,'max'):
 began=time.monotonic()
 r=beam_search(graph,start_state=start,predictor=model,native_options=options,backend='native',
  beam_width=beam,max_steps=8,return_path=True)
 assert r.path_found
 assert graph.apply_path(start,r.path).reshape(-1).tolist()==graph.central_state.tolist()
 receipt=dict(requested_beam=beam,public_search=True,replay_valid=True,path_length=r.path_length,
  total_wall_seconds=time.monotonic()-began,metadata=r.native_metadata)
 (root/f'public-{beam}.json').write_text(json.dumps(receipt,indent=2,default=str))
 p=r.native_metadata['profile']
 print(json.dumps(dict(beam=beam,wall_seconds=receipt['total_wall_seconds'],inference_wall=p['inference_calibration']['calibration_wall_seconds'],
  total_calibration_wall=p.get('calibration_total_wall_seconds'),pipeline_phase=p['pipeline_calibration']['phase'],
  matched=p['pipeline_calibration'].get('matched_stream1'),capacity=p.get('maximum_beam'))),flush=True)
