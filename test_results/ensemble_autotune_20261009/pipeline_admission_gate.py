import os,json,pathlib,subprocess,numpy as np,uuid
root=pathlib.Path('/workspace');out=root/'results/pipeline-admission';out.mkdir(exist_ok=True)
graph=json.loads((root/'inputs/assets/p002.json').read_text())
graph['central_state']=np.load(root/'inputs/assets/solved_state.npy').reshape(-1).astype(int).tolist()
(root/'puzzle-ensemble.json').write_text(json.dumps(graph))
env=dict(os.environ,BEAM_NCCL_RUN_ID='pipeline-admission-'+uuid.uuid4().hex,BEAM_BLEND_DIR=str(root/'ensemble-real'),BEAM_GENERATOR_PATH=str(root/'puzzle-ensemble.json'),BEAM_PUZZLE_INFO_JSON=str(root/'puzzle-ensemble.json'),BEAM_TEST_CSV=str(root/'inputs/assets/test.csv'),BEAM_RUNTIME_CONFIG_MODE='auto',BEAM_BENCHMARK_PLAN_ONLY='1',BEAM_B_MICRO='1024',BEAM_ENSEMBLE_INFERENCE_MICRO='1024',BEAM_ENSEMBLE_RESERVE_BYTES=str(1293942784+(512<<20)),BEAM_STREAM1_EXECUTOR='libtorch_eager',BEAM_SOLVE_BUCKET_MODE='0',BEAM_STREAM2_SUFFIX_RADIUS='0')
for rank in range(2):
 rankenv=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
 result=subprocess.run([str(root/'build-ensemble/production_runner_libtorch_stream1'),'1000','1','1048576','2',str(rank)],env=rankenv,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
 (out/f'rank{rank}.log').write_text(result.stdout)
 print(json.dumps({'rank':rank,'exit':result.returncode,'output':result.stdout[-7000:]}),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
