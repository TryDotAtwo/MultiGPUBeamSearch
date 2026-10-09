"""Full legal-frontier baseline; fixture generation is outside depth timings."""
import os,json,pathlib,subprocess,uuid,numpy as np,hashlib
root=pathlib.Path('/workspace');out=root/'results/pipeline-baseline';out.mkdir(exist_ok=True)
graph=json.loads((root/'puzzle-ensemble.json').read_text())
moves=np.asarray(graph['moves'],dtype=np.int64)
rng=np.random.default_rng(20261009)
size=1048576
states=np.tile(np.asarray(graph['central_state'],dtype=np.uint8),(size,1))
for step in range(64):
 states=np.take_along_axis(states,moves[rng.integers(0,24,size=size)],axis=1)
unique=len(np.unique(states.view(np.dtype((np.void,96))).reshape(-1)))
if unique!=size:raise RuntimeError(f'fixture duplicates: {size-unique}')
receipt={'global_parents':size,'unique_states':unique,'walk_length':64,'padding_bytes':16,'files':[]}
for rank in range(2):
 padded=np.zeros((size//2,112),dtype=np.uint8);padded[:,:96]=states[rank*(size//2):(rank+1)*(size//2)]
 path=out/f'frontier-rank{rank}.bin';padded.tofile(path)
 receipt['files'].append({'rank':rank,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
(out/'fixture-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
cwd=out/'run';cwd.mkdir(exist_ok=True);(cwd/'test_results').mkdir(exist_ok=True)
env=dict(os.environ,BEAM_NCCL_RUN_ID='pipeline-baseline-'+uuid.uuid4().hex,BEAM_BLEND_DIR=str(root/'ensemble-real'),BEAM_GENERATOR_PATH=str(root/'puzzle-ensemble.json'),BEAM_PUZZLE_INFO_JSON=str(root/'puzzle-ensemble.json'),BEAM_TEST_CSV=str(root/'inputs/assets/test.csv'),BEAM_RUNTIME_CONFIG_MODE='auto',BEAM_B_MICRO='1024',BEAM_ENSEMBLE_INFERENCE_MICRO='1024',BEAM_ENSEMBLE_RESERVE_BYTES=str(1293942784+(512<<20)),BEAM_STREAM1_EXECUTOR='libtorch_eager',BEAM_SOLVE_BUCKET_MODE='0',BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_BENCHMARK_FRONTIER_REPEATS='7',BEAM_HISTORY_MODE='disk',BEAM_HISTORY_DIR=str(out/'history'),BEAM_HISTORY_DISK_PATH=str(out/'history'))
env.pop('BEAM_BENCHMARK_PLAN_ONLY',None)
processes=[]
for rank in range(2):
 rankenv=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',BEAM_BENCHMARK_FRONTIER_FILE=str(out/f'frontier-rank{rank}.bin'))
 log=(out/f'rank{rank}.log').open('w')
 proc=subprocess.Popen([str(root/'build-ensemble/production_runner_libtorch_stream1'),'1000','7',str(size),'2',str(rank)],env=rankenv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 processes.append((rank,proc,log))
try:
 for rank,proc,log in processes:
  code=proc.wait(timeout=900);log.close()
  print(json.dumps({'rank':rank,'exit':code,'tail':(out/f'rank{rank}.log').read_text()[-5000:]}),flush=True)
  if code:raise SystemExit(code)
finally:
 for rank,proc,log in processes:
  if proc.poll() is None:proc.kill();proc.wait()
  log.close()
