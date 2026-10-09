import os,json,subprocess,pathlib,time,uuid
root=pathlib.Path('/tmp/fp16-results/solve');root.mkdir(exist_ok=True)
base=os.environ.copy();base.update(json.loads(pathlib.Path('/tmp/smoke_profile.json').read_text()))
base['LD_LIBRARY_PATH']='/venv/main/lib/python3.12/site-packages/torch/lib:/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:'+base.get('LD_LIBRARY_PATH','')
exe='/tmp/blend-production/build/production_runner_libtorch_stream1'
results=[]
for world in (2,):
 run=root/f'world{world}-near-goal-guard';run.mkdir(exist_ok=True)
 env=base.copy();env['BEAM_NCCL_RUN_ID']=str(uuid.uuid4());env.update(WORLD_SIZE=str(world),BEAM_NCCL_ID_FILE=str(run/'nccl-id'),NCCL_IB_DISABLE='1',BEAM_HISTORY_DIR=str(run/'history'),BEAM_HISTORY_DISK_PATH=str(run/'disk'))
 graph=json.loads(pathlib.Path('/tmp/smoke_graph.json').read_text());state=graph['central_state']
 for move in [0,7,13]:state=[state[i] for i in graph['moves'][move]]
 env['BEAM_START_STATE_TEXT']=','.join(map(str,state))
 (run/'start.json').write_text(json.dumps(state))
 children=[];t=time.monotonic()
 try:
  for rank in range(world):
   e=env.copy();e.update(RANK=str(rank),LOCAL_RANK=str(rank))
   f=(run/f'rank{rank}.log').open('w');p=subprocess.Popen([exe,'1000','4','4096',str(world),str(rank)],env=e,stdout=f,stderr=subprocess.STDOUT);children.append((p,f))
  codes=[p.wait(timeout=max(1,180-(time.monotonic()-t))) for p,f in children]
  results.append(dict(world=world,exit_codes=codes,seconds=time.monotonic()-t))
 finally:
  for p,f in children:
   if p.poll() is None:p.kill();p.wait()
   f.close()
 (root/'near_goal_result.json').write_text(json.dumps(results,indent=2))
 if any(c!=0 for c in codes):break
print(json.dumps(results))
