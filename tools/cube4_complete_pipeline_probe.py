"""Remote full-depth 2/8-rank A08/A09 experiment with unchanged S1 policy."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.cayleypy_public.paths import apply_path
from tools.cube4_run_supervisor import stop_group
from tools.beam_smoke_fixture import verify_submission


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def run(root,cohort,build,output,world,sort,prefill,repeat,diagnostics=False,beam=65536,nsys=False):
    output.mkdir()
    pid=956000
    puzzle=json.loads((cohort/'data/puzzle_info.json').read_text())
    names=list(puzzle['generators']);rng=random.Random(2026100503)
    walk=[rng.choice(names) for _ in range(32)]
    state=apply_path(puzzle['central_state'],'.'.join(walk),puzzle['generators'])
    data=output/'data';data.mkdir();save(data/'puzzle_info.json',puzzle)
    with (data/'test.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['initial_state_id','initial_state'])
        writer.writerow([pid,','.join(map(str,state))])
    profile=json.loads((root/'evidence/dual_base_profile.json').read_text())
    profile.update(BEAM_B_MICRO='2048',BEAM_STREAM1_TRANSFORMER_MICRO='1024',
        BEAM_SHARD_CAPACITY_CANDIDATES='131072',BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES='65536',
        BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=str(world*1000000),
        BEAM_STREAM4_BOUNDED_SORT=str(sort),BEAM_STREAM5_PREFILL_BEFORE_WAIT=str(prefill),
        BEAM_HISTORY_RAM_BYTES='134217728',BEAM_HISTORY_DISK_BYTES='134217728',
        BEAM_PREDICT_STATS_VERBOSE='0',BEAM_DEPTH_LOG_EVERY='1')
    if diagnostics:profile['BEAM_DEBUG_PIPELINE_STATS']='1'
    runner=build/'production_runner';sha=hashlib.sha256(runner.read_bytes()).hexdigest()
    env={k:v for k,v in os.environ.items() if not k.startswith(('BEAM_','CUDA_'))}
    env.update(profile,BEAM_WEIGHT_DIR=str(cohort/'weights_fp16'),
        BEAM_GENERATOR_PATH=str(data/'puzzle_info.json'),BEAM_PUZZLE_INFO_JSON=str(data/'puzzle_info.json'),
        BEAM_TEST_CSV=str(data/'test.csv'),BEAM_HISTORY_DIR=str(output/'history'),
        BEAM_HISTORY_DISK_PATH=str(output/'history-disk'),BEAM_PREDICT_STATS_PATH=str(output/'predict_stats'),
        BEAM_NCCL_ID_FILE=str(output/'nccl-id'),BEAM_RANK_STATUS_DIR=str(output/'rank-status'),
        BEAM_NCCL_RUN_ID=str(output))
    if diagnostics:
        env['BEAM_DEBUG_FRONTIER_DUMP_DIR']=str(output/'frontier-dumps')
        (output/'frontier-dumps').mkdir()
    request=dict(schema_version=1,world=world,sort=sort,prefill=prefill,repeat=repeat,diagnostics=diagnostics,nsys=nsys,
        profile=profile,beam=beam,depth_limit=6,runner_sha256=sha,
        initial_state=state,scramble=walk,scope='full_depth_pipeline_diagnostic_not_general_quality')
    save(output/'request.json',request)
    processes,logs=[],[];started=time.monotonic()
    try:
        for rank in range(world):
            log=(output/f'rank{rank}.log').open('xb');logs.append(log)
            command=[str(runner),str(pid),'6',str(beam)]
            if nsys:
                command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                    '--cuda-graph-trace=graph','--force-overwrite=false','-o',str(output/f'rank_{rank}'),*command]
            processes.append(subprocess.Popen(command,cwd=output,
                env=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(world)),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        for process in processes:
            remaining=600-(time.monotonic()-started)
            if remaining<=0 or process.wait(timeout=remaining):raise RuntimeError('pipeline rank failure/timeout')
    finally:
        for process in processes:stop_group(process)
        for log in logs:log.close()
    statuses=[json.loads((output/'rank-status'/f'rank-{rank}.json').read_text()) for rank in range(world)]
    for rank,status in enumerate(statuses):
        if status['rank']!=rank or status['exit_code']!=0 or status['puzzle_id']!=pid:
            raise ValueError('pipeline terminal identity/failure')
        if status['completed_depths']!=6:raise ValueError('missing complete six-depth run')
    if len({status['status'] for status in statuses})!=1:raise ValueError('mixed terminal ranks')
    if statuses[0]['status']=='solved':
        files=list((output/'test_results').glob(f'submit_p{pid}_*.csv'))
        if len(files)!=1:raise ValueError('missing unique solved output')
        replay=verify_submission(files[0],dict(puzzle_id=pid,initial=state,central=puzzle['central_state'],minimum_depth=0),
            dict(move_names=names,moves=list(puzzle['generators'].values())),6)
    else:replay=None
    result=dict(request=request,elapsed_seconds=time.monotonic()-started,status=statuses[0]['status'],
        completed_depths=6,rank_statuses=statuses,replay=replay)
    save(output/'result.json',result)
    print(json.dumps({key:value for key,value in result.items() if key!='request'}),flush=True)


def main():
    def interrupt(signum,frame):
        raise KeyboardInterrupt('pipeline supervision interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupt)
    parser=argparse.ArgumentParser()
    for name in ('root','cohort','build','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--world',type=int,choices=(2,8),required=True)
    parser.add_argument('--sort',type=int,choices=(0,1),required=True)
    parser.add_argument('--prefill',type=int,choices=(0,1),required=True)
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--beam',type=int,choices=(65536,262144),default=65536)
    parser.add_argument('--nsys',action='store_true')
    args=parser.parse_args()
    if os.name!='posix':raise SystemExit('Remote POSIX only')
    run(args.root,args.cohort,args.build,args.output,args.world,args.sort,args.prefill,args.repeat,args.diagnostics,args.beam,args.nsys)

if __name__=='__main__':main()
