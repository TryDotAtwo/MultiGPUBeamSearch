import os,json,pathlib,subprocess,numpy as np,uuid
root=pathlib.Path('/workspace');out=root/'results/pipeline-admission';out.mkdir(exist_ok=True)
graph=json.loads((root/'inputs/assets/p002.json').read_text())
graph['central_state']=np.load(root/'inputs/assets/solved_state.npy').reshape(-1).astype(int).tolist()
(root/'puzzle-ensemble.json').write_text(json.dumps(graph))
env=dict(os.environ,BEAM_NCCL_RUN_ID='pipeline-admission-'+uuid.uuid4().hex,BEAM_BLEND_DIR=str(root/'ensemble-real'),BEAM_GENERATOR_PATH=str(root/'puzzle-ensemble.json'),BEAM_PUZZLE_INFO_JSON=str(root/'puzzle-ensemble.json'),BEAM_TEST_CSV=str(root/'inputs/assets/test.csv'),BEAM_RUNTIME_CONFIG_MODE='auto',BEAM_BENCHMARK_PLAN_ONLY='1',BEAM_B_MICRO='1024',BEAM_ENSEMBLE_INFERENCE_MICRO='1024',BEAM_ENSEMBLE_RESERVE_BYTES=str(1293942784+(512<<20)),BEAM_STREAM1_EXECUTOR='libtorch_eager',BEAM_SOLVE_BUCKET_MODE='0',BEAM_STREAM2_SUFFIX_RADIUS='0')
processes=[]
for rank in range(2):
 rankenv=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
 log=(out/f'rank{rank}.log').open('w')
 proc=subprocess.Popen([str(root/'build-ensemble/production_runner_libtorch_stream1'),'1000','1','1048576','2',str(rank)],env=rankenv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 processes.append((rank,proc,log))
try:
 for rank,proc,log in processes:
  code=proc.wait(timeout=60);log.close()
  print(json.dumps({'rank':rank,'exit':code,'output':(out/f'rank{rank}.log').read_text()[-7000:]}),flush=True)
  if code:raise SystemExit(code)
finally:
 for rank,proc,log in processes:
  if proc.poll() is None:proc.kill();proc.wait()
  log.close()
