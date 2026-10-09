"""Eight real GPUs: admission, path replay, bounded full-frontier depth sweep."""
import csv, hashlib, json, math, os, pathlib, re, subprocess, sys, time, uuid
ROOT = pathlib.Path('/workspace'); OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
EXE = ROOT / 'runner-frozen'; SOURCE = ROOT / 'source'; WORLD = 8
BEAMS = [1_000_000, 10_000_000, 100_000_000, 268_435_456]
DEADLINE = 1791533319 - 600
EXPECTED = {
 's3.npz': '1239e036f9fc792ffcf177c867aca758a48098966f54c1b4c4427127a68b4dca',
 'mlp_x16.npz': 'bddae95ba86de5a007daf284d6d9f66585a7792a59e4fc4ad26288862590933a',
 'p002.json': 'ec89af45b4b50a0eb0f8c681b39e04cde13f95897791f2008bde60d0bef23d5d',
 'test.csv': 'e9a734df2a63656477c2c6f9920f490188fdfa8a780027f1b91783e82b77419e',
 'solved_state.npy': '27c5b611c5aeb7b841676059b935bddc57aeaaaa130157d281c4c6af7245a941'}

def sha(p):
 with pathlib.Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def fixture_worker(rank, n):
 import torch, numpy as np
 torch.cuda.set_device(rank)
 torch.set_num_threads(1)
 assets = ROOT/'inputs/assets'
 graph=json.loads((assets/'smoke_graph.json').read_text())
 rows={int(x['initial_state_id']):x for x in csv.DictReader((assets/'test.csv').open())}
 initial=list(map(int,rows[1000]['initial_state'].split(',')))
 moves=torch.tensor(graph['moves'],device='cuda',dtype=torch.long)
 gen=torch.Generator(device='cuda').manual_seed(20261009+rank)
 target=ROOT/'fixture'/f'rank{rank}.bin'; target.parent.mkdir(exist_ok=True)
 sample=None; total=0; started=time.monotonic()
 with target.open('wb') as f, torch.inference_mode():
  while total<n:
   count=min(65536,n-total)
   states=torch.tensor(initial,device='cuda',dtype=torch.uint8).repeat(count,1)
   choices=torch.randint(24,(80,count),generator=gen,device='cuda')
   for step in range(80): states=torch.gather(states,1,moves[choices[step]])
   packed=torch.zeros((count,112),device='cuda',dtype=torch.uint8); packed[:,:96]=states
   if sample is None:
    seq=choices[:,0].cpu().tolist(); got=states[0].cpu().tolist(); expected=initial
    for action in seq: expected=[expected[i] for i in graph['moves'][action]]
    assert got==expected
    sample={'moves':seq,'state':got,'independent_replay':True}
   f.write(packed.cpu().numpy().tobytes()); total+=count
 data={'rank':rank,'parents':n,'seed':20261009+rank,'walk_length':80,'sample':sample,
       'bytes':target.stat().st_size,'sha256':sha(target),'seconds':time.monotonic()-started}
 (ROOT/'fixture'/f'rank{rank}.json').write_text(json.dumps(data,indent=2))

def env_for(beam, name, plan=False, fixture=False, near=False):
 effective=math.ceil(beam/(WORLD*16*256))*(WORLD*16*256)
 local=effective//WORLD; cap=math.ceil(math.ceil(local/16)*4/256)*256
 if near: cap=32768
 d=OUT/name; d.mkdir(exist_ok=False)
 e=os.environ.copy();e.update({
  'BEAM_BLEND_DIR':str(ROOT/'inputs/bundle'),'BEAM_STREAM1_EXECUTOR':'libtorch_eager',
  'BEAM_GENERATOR_PATH':str(ROOT/'inputs/assets/smoke_graph.json'),
  'BEAM_PUZZLE_INFO_JSON':str(ROOT/'inputs/assets/smoke_graph.json'),
  'BEAM_TEST_CSV':str(ROOT/'inputs/assets/test.csv'),'BEAM_RUNTIME_CONFIG_MODE':'manual',
  'BEAM_B_MICRO':'256','BEAM_STREAM1_CONCURRENCY':'1','BEAM_STREAM3_RING_SLOTS':'2',
  'BEAM_RING_COUNT':'4','BEAM_SHARD_COUNT':'16','BEAM_SHARD_BUFFER_COUNT':'2',
  'BEAM_SHARD_CAPACITY_SCALE_PPM':'4000000','BEAM_SHARD_CAPACITY_CANDIDATES':str(cap),
  'BEAM_STREAM4_BATCH_ALIGNMENT':'256','BEAM_STREAM4_ACTIVE_SORT_SLOTS':'2',
  'BEAM_STREAM4_BATCH_CANDIDATES':str(min(65536,cap)),
  'BEAM_STREAM4_TRIGGER_CANDIDATES':str(max(256,min(65536,cap)//2)),
  'BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES':'65536','BEAM_GLOBAL_SPILL_CAPACITY':'0',
  'BEAM_SOLVED_RESULT_CAPACITY':'1024','BEAM_SOLVED_NEIGHBORHOOD_RADIUS':'0',
  'BEAM_STREAM2_SUFFIX_RADIUS':'0','BEAM_HISTORY_RAM_BYTES':str(8<<30),
  'BEAM_HISTORY_DISK_BYTES':'0','BEAM_HISTORY_MODE':'ram',
  'BEAM_HOST_RAM_HEADROOM_BYTES':str(2<<30),'BEAM_GPU_HEADROOM_BYTES':str(512<<20),
  'BEAM_HISTORY_DIR':str(d/'history'),'BEAM_PREDICT_STATS_VERBOSE':'0',
  'BEAM_GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS':'1','BEAM_DEPTH_LOG_EVERY':'1',
  'NCCL_IB_DISABLE':'1','WORLD_SIZE':str(WORLD),'BEAM_NCCL_RUN_ID':str(uuid.uuid4()),
  'BEAM_NCCL_ID_FILE':str(d/'nccl-id')})
 e['LD_LIBRARY_PATH']='/venv/main/lib/python3.12/site-packages/torch/lib:/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:'+e.get('LD_LIBRARY_PATH','')
 if plan: e['BEAM_BENCHMARK_PLAN_ONLY']='1'
 return d,e

def run(beam,name,plan=False,fixture=False,near=False,timeout=1500):
 assert sha(EXE)==json.loads((OUT/'build_receipt.json').read_text())['binary_sha256']
 d,e=env_for(beam,name,plan,fixture,near);children=[];started=time.monotonic()
 if near:
  graph=json.loads((ROOT/'inputs/assets/smoke_graph.json').read_text());start=graph['central_state']
  for action in [0,7,13]: start=[start[i] for i in graph['moves'][action]]
  e['BEAM_START_STATE_TEXT']=','.join(map(str,start));(d/'start.json').write_text(json.dumps(start))
 try:
  for rank in range(WORLD):
   rank_env=e.copy();rank_env.update(RANK=str(rank),LOCAL_RANK=str(rank))
   if fixture: rank_env['BEAM_BENCHMARK_FRONTIER_FILE']=str(ROOT/'fixture'/f'rank{rank}.bin')
   (d/f'env{rank}.json').write_text(json.dumps({k:v for k,v in rank_env.items() if k.startswith(('BEAM_','NCCL_')) or k in ('WORLD_SIZE','RANK','LOCAL_RANK')},indent=2))
   f=(d/f'rank{rank}.log').open('w');p=subprocess.Popen([str(EXE),'1000','4' if near else '1',str(beam),str(WORLD),str(rank)],env=rank_env,stdout=f,stderr=subprocess.STDOUT);children.append((p,f))
  while any(p.poll() is None for p,f in children):
   if any(p.poll() not in (None,0) for p,f in children): break
   if time.monotonic()-started>timeout or time.time()>DEADLINE: raise TimeoutError(name)
   time.sleep(.5)
 finally:
  for p,f in children:
   if p.poll() is None: p.terminate()
  for p,f in children:
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   f.close()
 logs=[(d/f'rank{rank}.log').read_text() for rank in range(WORLD)]
 codes=[p.returncode for p,f in children]
 result={'beam':beam,'world':WORLD,'exit_codes':codes,'wall_seconds':time.monotonic()-started,
         'depth_seconds':[float(m.group(1)) for s in logs if (m:=re.search(r'depth_done=0 depth_sec=([0-9.e+-]+)',s))],
         'binary_sha256':sha(EXE)}
 (d/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'run':name,**result}),flush=True)
 return codes==[0]*WORLD,logs,result

def main():
 import numpy as np
 assets=ROOT/'inputs/assets'
 for name,digest in EXPECTED.items(): assert sha(assets/name)==digest,name
 (OUT/'asset_hashes.json').write_text(json.dumps(EXPECTED,indent=2))
 graph=json.loads((assets/'p002.json').read_text());graph['central_state']=np.load(assets/'solved_state.npy',allow_pickle=False).tolist()
 (assets/'smoke_graph.json').write_text(json.dumps(graph))
 sys.path.insert(0,str(SOURCE/'tools'));from export_cube444_blend import export
 export(assets,ROOT/'inputs/bundle')
 subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,power.limit,driver_version','--format=csv'],stdout=(OUT/'hardware.csv').open('w'),check=True)
 if (OUT/'near_goal/result.json').exists():
  prior=json.loads((OUT/'near_goal/result.json').read_text());assert prior['binary_sha256']==sha(EXE)
  passed=prior['exit_codes']==[0]*WORLD;logs=[(OUT/'near_goal'/f'rank{r}.log').read_text() for r in range(WORLD)]
 else:passed,logs,_=run(4096,'near_goal',near=True,timeout=180)
 assert passed,'eight-GPU near-goal run failed'
 graph=json.loads((assets/'smoke_graph.json').read_text());start=json.loads((OUT/'near_goal/start.json').read_text());paths=[]
 for log in logs:
  match=re.search(r'^solution_path=(.*)$',log,re.MULTILINE)
  if match:paths.append(match.group(1).strip())
 assert paths,'no returned solution path'
 for path in paths:
  state=start
  for move in path.split('.'):state=[state[i] for i in graph['moves'][graph['move_names'].index(move)]]
  assert state==graph['central_state']
 (OUT/'path_replay.json').write_text(json.dumps({'status':'PASS','paths':paths},indent=2))
 results=[]
 for beam in BEAMS:
  ok,logs,plan=run(beam,f'plan_{beam}',plan=True,timeout=180)
  if not ok:
   results.append({'beam':beam,'status':'MEMORY_REJECTED','plan':plan});continue
  if time.time()>DEADLINE-1200:
   results.append({'beam':beam,'status':'TIME_BUDGET_NOT_RUN'});continue
  fixture=ROOT/'fixture';fixture.mkdir(exist_ok=True)
  # Explicit test files only; no search result or source is deleted.
  for rank in range(WORLD):
   for suffix in ('.bin','.json'):(fixture/f'rank{rank}{suffix}').unlink(missing_ok=True)
  workers=[subprocess.Popen([sys.executable,__file__,'fixture',str(rank),str(math.ceil(beam/(WORLD*16*256))*16*256)]) for rank in range(WORLD)]
  assert [p.wait(timeout=600) for p in workers]==[0]*WORLD
  manifest=[json.loads((fixture/f'rank{rank}.json').read_text()) for rank in range(WORLD)]
  (OUT/f'fixture_{beam}.json').write_text(json.dumps({'source_index':1000,'requested_beam':beam,'effective_beam':math.ceil(beam/(WORLD*16*256))*(WORLD*16*256),'scope':'80-move seeded legal states; throughput fixture, not search-selected frontier','entries':manifest},indent=2))
  ok,logs,value=run(beam,f'depth_{beam}',fixture=True,timeout=min(1800,max(60,DEADLINE-time.time())))
  results.append({'beam':beam,'status':'PASS' if ok else 'FAILED','plan':plan,'timing':value})
  (OUT/'sweep.json').write_text(json.dumps(results,indent=2))
  assert ok,beam
 (OUT/'sweep.json').write_text(json.dumps(results,indent=2))
 print('SWEEP_COMPLETE',flush=True)

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='fixture':fixture_worker(int(sys.argv[2]),int(sys.argv[3]))
 else:main()

