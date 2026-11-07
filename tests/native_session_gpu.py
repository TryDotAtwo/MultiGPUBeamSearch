"""Real CUDA resident reference comparison; launched explicitly on GPU hosts.

The small deterministic permutation graph uses an exact Hamming MLP. This is a
correctness/lifetime fixture, not acceptance of trained-model ranking or speed.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.hamming import write_hamming_artifact
from multigpubeamsearch.resident_transport import ResidentWorkers as ReferenceWorkers
from multigpubeamsearch.backend import parse_terminal


def main():
    parser=argparse.ArgumentParser()
    for name in ('baseline','rust-runner','library','extension','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--history-mode',choices=('ram','disk','static_hybrid'),default='disk')
    parser.add_argument('--executor',choices=('native_cuda_graph','libtorch_eager'),default='native_cuda_graph')
    args=parser.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('_beam_runtime',args.extension)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    center=tuple(i//16 for i in range(96))
    moves=[]
    for i in range(24):
        row=list(range(96));row[i],row[i+24]=row[i+24],row[i];moves.append(tuple(row))
    payload=dict(generators_type='PERMUTATION',generators=moves,central_state=center)
    graph_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    contract=GraphContract(96,6,tuple(moves),tuple('m'+str(i) for i in range(24)),center,center,graph_hash)
    weights=write_hamming_artifact(contract,root)
    puzzle=root/'puzzle.json';puzzle.write_text(json.dumps(contract.to_puzzle_info()))
    csv=root/'test.csv';csv.write_text('initial_state_id,initial_state\n0,"'+','.join(map(str,center))+'"\n')
    starts=[]
    for count in (3,4,5):
        state=center
        for move in range(count):state=tuple(state[index] for index in moves[move])
        starts.append(state)
    results=[]
    for world in (1,2):
        for variant in ('reference','rust'):
            directory=root/f'{variant}-{world}';directory.mkdir()
            env=dict(os.environ)
            env.update(BEAM_GENERATOR_PATH=str(puzzle),BEAM_PUZZLE_INFO_JSON=str(puzzle),
                BEAM_TEST_CSV=str(csv),BEAM_WEIGHT_DIR=str(weights),BEAM_RUNTIME_CONFIG_MODE='auto',
                BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_STREAM2_SUFFIX_RADIUS='0',
                BEAM_SOLVE_BUCKET_MODE='0',BEAM_HISTORY_MODE=args.history_mode,BEAM_RUNTIME_DEPTH_TIMING='1',
                BEAM_HISTORY_RAM_BYTES=str(16<<20),BEAM_HISTORY_DISK_BYTES=str(1<<20),
                BEAM_HISTORY_DIR=str(directory/'history'),BEAM_HISTORY_DISK_PATH=str(directory/'history'),
                BEAM_STREAM1_EXECUTOR=args.executor,
                BEAM_NCCL_ID_FILE=str(directory/'nccl-id.bin'),BEAM_B_MICRO='768',
                BEAM_SHARD_COUNT='1',BEAM_STREAM4_BATCH_CANDIDATES='1024',
                CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))),
                BEAM_NATIVE_LIBRARY=str(args.library.resolve()))
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','BEAM_SESSION_DIR'):
                env.pop(key,None)
            command=[str((args.baseline if variant=='reference' else args.rust_runner).resolve()),'0','4','64']
            began=time.monotonic()
            if variant=='reference':
                group=ReferenceWorkers(command,world=world,directory=directory/'session',cwd=directory,env=env)
            else:
                group=runtime.ResidentWorkers(command,world,directory/'session',directory,env)
            pids=group.pids
            try:
                for sequence,start in enumerate(starts):
                    out=directory/f'job-{sequence}';out.mkdir()
                    elapsed=group.submit(start,out,timeout=120.0)
                    if variant=='rust':assert 'history_provider=rust' in (out/'native.log').read_text()
                    assert group.pids==pids, 'compatible job changed worker PIDs'
                    for rank in range(world):
                        assert f'stream1_executor={args.executor}\n' in (directory/'session'/f'bootstrap-{rank}.log').read_text()
                    task=replace(contract,start=start)
                    path,width,terminal=parse_terminal(out/'native.log',task,max_steps=4)
                    assert width==64, (variant,world,width)
                    if sequence<2:
                        assert path is not None and task.replay(path), (variant,world,path)
                        assert len(path)==sequence+3, (variant,world,path)
                    else:assert path is None, (variant,world,path)
                    results.append(dict(variant=variant,world=world,sequence=sequence,path=path,
                        beam_width=width,elapsed_seconds=elapsed,worker_pids=pids,terminal=terminal))
            finally:group.close()
            if variant=='rust':assert all(code is not None for code in group.exit_codes())
            else:assert all(process.poll() is not None for process in group.processes)
            print(f'{variant} world={world} seconds={time.monotonic()-began:.3f}',flush=True)
    for world in (1,2):
        for sequence in range(3):
            pair=[row for row in results if row['world']==world and row['sequence']==sequence]
            assert pair[0]['path']==pair[1]['path'],pair
    (root/'results.json').write_text(json.dumps(dict(scope='correctness and resident lifetime; no performance promotion',
        graph_hash=graph_hash,history_mode=args.history_mode,executor=args.executor,results=results),indent=2))
    print('NATIVE_SESSION_GPU_OK',flush=True)


if __name__=='__main__':main()
