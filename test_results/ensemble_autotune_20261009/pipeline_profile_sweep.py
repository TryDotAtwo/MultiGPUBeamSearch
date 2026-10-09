"""Full-frontier downstream coordinate sweep, inference batch frozen at1024."""
import os,json,pathlib,time,sys,hashlib
root=pathlib.Path('/workspace');sys.path.insert(0,str(root/'source/integrations/cayleypy_native'))
from multigpubeamsearch.pipeline_probe import NativePipelineProbe
from multigpubeamsearch.pipeline_profiles import tune_pipeline
fixture=root/'results/pipeline-baseline';receipt=json.loads((fixture/'fixture-receipt.json').read_text())
for row in receipt['files']:
 path=fixture/f"frontier-rank{row['rank']}.bin"
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
out=root/'results/pipeline-profile-sweep';out.mkdir(exist_ok=True)
env=dict(os.environ,BEAM_BLEND_DIR=str(root/'ensemble-real'),BEAM_GENERATOR_PATH=str(root/'puzzle-ensemble.json'),BEAM_PUZZLE_INFO_JSON=str(root/'puzzle-ensemble.json'),BEAM_TEST_CSV=str(root/'inputs/assets/test.csv'),BEAM_RUNTIME_CONFIG_MODE='auto',BEAM_B_MICRO='1024',BEAM_ENSEMBLE_INFERENCE_MICRO='1024',BEAM_ENSEMBLE_RESERVE_BYTES=str(1293942784+(512<<20)),BEAM_STREAM1_EXECUTOR='libtorch_eager',BEAM_SOLVE_BUCKET_MODE='0',BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_HISTORY_MODE='disk')
readout=[]
for device in (0,1):
 record=json.loads((root/f'results/ensemble-real/gpu{device}.log').read_text())
 assert record['correctness_passed'] and record['numeric_error']==0 and record['readout_oracle_max_key_error']<=2
 readout.append(record)
def verify(texts,plans):
 # Backbone/readout numerical gate and fixture provenance were independently
 # checked above. This gate covers native structural/full-workload completion.
 for text,plan in zip(texts,plans):
  fields={}
  for line in text.splitlines():
   if '=' in line and ' ' not in line:
    key,value=line.split('=',1);fields[key]=value
  if fields.get('completed_depths')!='6' or int(fields.get('last_final_frontier_size','-1'))!=plan['frontier_state_capacity']:return False
  if 'production_runner_error=' in text or 'numeric_error=' in text:return False
 return True
deadline=time.monotonic()+1800
probe=NativePipelineProbe(root/'build-ensemble/production_runner_libtorch_stream1',env,1048576,2,out/'runs',[fixture/f'frontier-rank{i}.bin' for i in (0,1)],112,deadline=deadline,verify=verify,puzzle_id=1000)
result=tune_pipeline(1024,{},admit=probe.admit,measure=probe.measure,deadline=deadline,max_outer=8192,rounds=1)
result['verification_scope']='legal fixture + independently accepted backbone/readout + native structural completion; full arbitrary-graph replay remains pending'
result['inference_readout_gates']=readout
(out/'selection.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'selected':result['environment'],'estimate':result['estimate'],'profiles_measured':len({row['profile'] for row in result['measurements']})}),flush=True)
