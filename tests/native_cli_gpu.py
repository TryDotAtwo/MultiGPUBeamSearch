"""Real Rust CLI and torchrun entry points; exact path replay, no timing claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.backend import parse_terminal


def main():
    parser=argparse.ArgumentParser()
    for name in ('fixture','runner','library','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();fixture=args.fixture.resolve();root=args.out.resolve()
    root.mkdir(parents=True,exist_ok=False)
    puzzle=fixture/'puzzle.json';data=json.loads(puzzle.read_text())
    center=tuple(data['central_state']);moves=tuple(tuple(v) for v in data['generators'].values())
    start=center
    for move in range(3):start=tuple(start[index] for index in moves[move])
    payload=dict(generators_type='PERMUTATION',generators=moves,central_state=center)
    contract=GraphContract(len(center),max(center)+1,moves,tuple(data['generators']),center,start,
                          hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
    csv=root/'test.csv';csv.write_text('initial_state_id,initial_state\n0,"'+','.join(map(str,start))+'"\n')
    results=[]
    for world in (1,2):
        directory=root/f'world-{world}';directory.mkdir()
        env=dict(os.environ)
        env.update(BEAM_GENERATOR_PATH=str(puzzle),BEAM_PUZZLE_INFO_JSON=str(puzzle),
            BEAM_TEST_CSV=str(csv),BEAM_WEIGHT_DIR=str(fixture/'weights'),BEAM_RUNTIME_CONFIG_MODE='auto',
            BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_SOLVE_BUCKET_MODE='0',
            BEAM_HISTORY_MODE='static_hybrid',BEAM_HISTORY_RAM_BYTES=str(16<<20),BEAM_HISTORY_DISK_BYTES=str(1<<20),
            BEAM_HISTORY_DIR=str(directory/'history'),BEAM_NCCL_ID_FILE=str(directory/'nccl-id.bin'),
            BEAM_NCCL_RUN_ID=f'cli-{os.getpid()}-{world}',BEAM_B_MICRO='768',BEAM_SHARD_COUNT='1',
            BEAM_STREAM4_BATCH_CANDIDATES='1024',CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))),
            BEAM_NATIVE_LIBRARY=str(args.library.resolve()))
        for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','BEAM_SESSION_DIR'):
            env.pop(key,None)
        command=[str(args.runner.resolve()),'0','4','64']
        if world==2:
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                '--no-python','--log-dir',str(directory/'torchrun'),'--redirects=3',*command]
        log=directory/'native.log'
        with log.open('w') as output:
            subprocess.run(command,cwd=directory,env=env,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=120)
        if world==2:
            with log.open('a') as output:
                for ranklog in sorted((directory/'torchrun').rglob('stdout.log')):
                    output.write(ranklog.read_text())
        text=log.read_text();assert text.count('history_provider=rust')==world
        path,width,terminal=parse_terminal(log,contract,max_steps=4)
        assert path is not None and len(path)==3 and contract.replay(path)
        assert width==64
        results.append(dict(world=world,path=path,width=width,terminal=terminal))
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('NATIVE_CLI_TORCHRUN_OK')


if __name__=='__main__':main()
