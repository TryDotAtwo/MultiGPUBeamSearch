import os,json,pathlib,subprocess,uuid,hashlib
root=pathlib.Path('/workspace');fixture=root/'results/pipeline-baseline';out=root/'results/pipeline-threshold-admission';out.mkdir(exist_ok=True)
receipt=json.loads((fixture/'fixture-receipt.json').read_text());size=receipt['global_parents']
assert receipt['unique_states']==size
for row in receipt['files']:
 path=fixture/f"frontier-rank{row['rank']}.bin"
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
cwd=out/'run';cwd.mkdir(exist_ok=True);(cwd/'test_results').mkdir(exist_ok=True)
env=dict(os.environ,BEAM_NCCL_RUN_ID='pipeline-baseline-'+uuid.uuid4().hex,BEAM_BLEND_DIR=str(root/'ensemble-real'),BEAM_GENERATOR_PATH=str(root/'puzzle-ensemble.json'),BEAM_PUZZLE_INFO_JSON=str(root/'puzzle-ensemble.json'),BEAM_TEST_CSV=str(root/'inputs/assets/test.csv'),BEAM_RUNTIME_CONFIG_MODE='auto',BEAM_B_MICRO='1024',BEAM_ENSEMBLE_INFERENCE_MICRO='1024',BEAM_ENSEMBLE_RESERVE_BYTES=str(1293942784+(512<<20)),BEAM_STREAM1_EXECUTOR='libtorch_eager',BEAM_SOLVE_BUCKET_MODE='0',BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_BENCHMARK_FRONTIER_REPEATS='7',BEAM_HISTORY_MODE='disk',BEAM_HISTORY_DIR=str(out/'history'),BEAM_HISTORY_DISK_PATH=str(out/'history'))
env.pop('BEAM_BENCHMARK_PLAN_ONLY',None)
processes=[]
for rank in range(2):
 rankenv=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',BEAM_BENCHMARK_FRONTIER_FILE=str(fixture/f'frontier-rank{rank}.bin'))
 log=(out/f'rank{rank}.log').open('w')
 proc=subprocess.Popen([str(root/'build-ensemble/production_runner_libtorch_stream1'),'1000','7',str(size),'2',str(rank)],env=rankenv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 processes.append((rank,proc,log))
try:
 for rank,proc,log in processes:
  code=proc.wait(timeout=180);log.close()
  print(json.dumps({'rank':rank,'exit':code,'tail':(out/f'rank{rank}.log').read_text()[-5000:]}),flush=True)
  if code:raise SystemExit(code)
finally:
 for rank,proc,log in processes:
  if proc.poll() is None:proc.kill();proc.wait()
  log.close()
