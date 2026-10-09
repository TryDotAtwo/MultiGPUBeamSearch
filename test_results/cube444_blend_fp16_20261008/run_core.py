import os,json,pathlib,subprocess,time,uuid,re,math,hashlib,statistics,sys
root=pathlib.Path('/tmp/fp16-results/full');root.mkdir(parents=True,exist_ok=True)
exe='/tmp/blend-production/build/production_runner_libtorch_stream1'
cal=json.loads((root/'calibration.json').read_text());reserve=cal['reserve_bytes'];unit=65536
base=os.environ.copy();base.update(BEAM_BLEND_DIR='/tmp/blend-validation/bundle',BEAM_STREAM1_EXECUTOR='libtorch_eager',BEAM_GENERATOR_PATH='/tmp/blend-validation/assets/smoke_graph.json',BEAM_PUZZLE_INFO_JSON='/tmp/blend-validation/assets/smoke_graph.json',BEAM_TEST_CSV='/tmp/blend-validation/assets/test.csv',BEAM_RUNTIME_CONFIG_MODE='manual',BEAM_B_MICRO='256',BEAM_STREAM1_CONCURRENCY='1',BEAM_STREAM3_RING_SLOTS='2',BEAM_RING_COUNT='4',BEAM_SHARD_COUNT='16',BEAM_SHARD_BUFFER_COUNT='2',BEAM_SHARD_CAPACITY_SCALE_PPM='4000000',BEAM_STREAM4_BATCH_ALIGNMENT='256',BEAM_STREAM4_ACTIVE_SORT_SLOTS='2',BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES='65536',BEAM_SOLVED_RESULT_CAPACITY='1024',BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_HISTORY_RAM_BYTES=str(8<<30),BEAM_HISTORY_DISK_BYTES='0',BEAM_HISTORY_MODE='ram',BEAM_HOST_RAM_HEADROOM_BYTES=str(2<<30),BEAM_GPU_HEADROOM_BYTES=str(512<<20),BEAM_BENCHMARK_BLEND_RESERVE_BYTES=str(reserve),BEAM_PREDICT_STATS_VERBOSE='0',NCCL_IB_DISABLE='1',WORLD_SIZE='2',BEAM_DEPTH_LOG_EVERY='1',BEAM_GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS='1')
base['LD_LIBRARY_PATH']='/venv/main/lib/python3.12/site-packages/torch/lib:/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:'+base.get('LD_LIBRARY_PATH','')
def run(beam,name,plan=False,only=False,fixture=None,timeout=5400):
 d=root/name;d.mkdir(exist_ok=False);n=beam//2;cap=max(12288,math.ceil(math.ceil(n/16)*4/256)*256);batch=min(65536,cap)
 env=base.copy();env.update(BEAM_SHARD_CAPACITY_CANDIDATES=str(cap),BEAM_STREAM4_BATCH_CANDIDATES=str(batch),BEAM_STREAM4_TRIGGER_CANDIDATES=str(max(256,batch//2)),BEAM_GLOBAL_SPILL_CAPACITY='0',BEAM_NCCL_RUN_ID=str(uuid.uuid4()),BEAM_NCCL_ID_FILE=str(d/'nccl'),BEAM_HISTORY_DIR=str(d/'history'))
 if plan:env['BEAM_BENCHMARK_PLAN_ONLY']='1'
 if only:env['BEAM_BENCHMARK_TRANSFORMER_ONLY']='1'
 children=[];t=time.monotonic()
 try:
  for rank in (0,1):
   e=env.copy();e.update(RANK=str(rank),LOCAL_RANK=str(rank))
   if fixture:e['BEAM_BENCHMARK_FRONTIER_FILE']=str(fixture/f'rank{rank}.bin')
   (d/f'env{rank}.json').write_text(json.dumps({k:v for k,v in e.items() if k.startswith(('BEAM_','NCCL_')) or k in ['RANK','LOCAL_RANK','WORLD_SIZE']},indent=2))
   f=(d/f'rank{rank}.log').open('w');p=subprocess.Popen([exe,'1000','1',str(beam),'2',str(rank)],env=e,stdout=f,stderr=subprocess.STDOUT);children.append((p,f))
  codes=[]
  while any(p.poll() is None for p,f in children):
   if any(p.poll() not in [None,0] for p,f in children):
    for p,f in children:
     if p.poll() is None:p.terminate()
    break
   if time.monotonic()-t>timeout:raise TimeoutError(name)
   time.sleep(.2)
  codes=[p.wait(timeout=10) for p,f in children]
 finally:
  for p,f in children:
   if p.poll() is None:p.kill();p.wait()
   f.close()
 logs=[(d/f'rank{i}.log').read_text() for i in (0,1)]
 value=dict(beam=beam,only=only,exit_codes=codes,wall_seconds=time.monotonic()-t,depth_seconds=[float(re.search(r'depth_done=0 depth_sec=([0-9.e+-]+)',s).group(1)) for s in logs] if all('depth_done=0' in s for s in logs) else [])
 (d/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps(dict(name=name,**value)),flush=True);return all(x==0 for x in codes),value,logs
