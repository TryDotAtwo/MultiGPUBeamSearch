"""Rust owns exact admission, all-rank full-step runs and statistical selection."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.calibration_frontier import write_frontiers
from multigpubeamsearch.pipeline_profiles import rank_frontier_counts
from multigpubeamsearch.pipeline_probe import NativePipelineProbe, parse_depths
from multigpubeamsearch.plan_session import NativePlanSession


def main():
    parser=argparse.ArgumentParser()
    for key in ('runner','extension','fixture','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('_beam_runtime',args.extension)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    assert hasattr(native,'PipelineProbe'), 'Rust pipeline ownership is not implemented'
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    fixture=args.fixture.resolve();puzzle=fixture/'puzzle.json';data=json.loads(puzzle.read_text())
    center=tuple(data['central_state']);moves=tuple(tuple(row) for row in data['generators'].values())
    start=center
    for move in range(20):start=tuple(start[i] for i in moves[move])
    contract=GraphContract(96,6,moves,tuple(data['generators']),center,start,hashlib.sha256(puzzle.read_bytes()).hexdigest())
    csv=root/'test.csv';csv.write_text('initial_state_id,initial_state\n0,"'+','.join(map(str,start))+'"\n')
    records=[]
    for world in (1,2):
        directory=root/f'world-{world}';directory.mkdir()
        env=dict(os.environ,BEAM_GENERATOR_PATH=str(puzzle),BEAM_PUZZLE_INFO_JSON=str(puzzle),
            BEAM_TEST_CSV=str(csv),BEAM_WEIGHT_DIR=str(fixture/'weights'),BEAM_RUNTIME_CONFIG_MODE='auto',
            BEAM_B_MICRO='768',BEAM_SHARD_COUNT='1',BEAM_STREAM4_BATCH_CANDIDATES='1024',
            BEAM_ENSEMBLE_INFERENCE_MICRO='32',BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_STREAM2_SUFFIX_RADIUS='0',
            BEAM_SOLVE_BUCKET_MODE='0',CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))))
        profile=dict(BEAM_B_MICRO='768',BEAM_ENSEMBLE_INFERENCE_MICRO='32')
        with NativePlanSession(args.runner.resolve(),env,world,directory/'reference-plans',deadline=time.monotonic()+120) as planner:
            plans=planner.admit(64,profile)
        counts=rank_frontier_counts(plans)
        receipt=write_frontiers(contract,counts,112,directory/'frontiers',deadline=time.monotonic()+60)
        fixtures=[str(directory/'frontiers'/f'frontier-rank-{rank}.bin') for rank in range(world)]
        for grow in (False,True):
            mode='growth' if grow else 'fixture'
            reference=NativePipelineProbe(args.runner.resolve(),env,64,world,directory/f'python-{mode}',
                fixtures,112,deadline=time.monotonic()+120,grow=grow,
                verify=lambda texts,targets: all(len(parse_depths(text,count,6))==6 for text,count in zip(texts,counts)))
            old=reference.measure(profile,plans)
            config=dict(runner=str(args.runner.resolve()),environment=env,beam=64,world=world,
                cwd=str(Path.cwd()),directory=str(directory/f'rust-{mode}'),fixtures=fixtures,
                physical_bytes=112,repeats=6,puzzle_id=0,grow=grow,device_indices=list(range(world)),gpu_uuids=[])
            owner=native.PipelineProbe(json.dumps(config),120.)
            try:
                rows=json.loads(owner.admit(profile))
                owner.finish_planning()
                new=json.loads(owner.measure(profile,json.dumps(rows)))
                assert len(new)==len(old)==5
                assert all(r['parents']==64 and len(r['seconds_by_rank'])==world and r['correctness_passed'] and not r['throttled'] for r in new)
            finally:owner.close()
            records.append(dict(world=world,mode=mode,reference=[r.__dict__ for r in old],rust=new,frontier_receipt=receipt))
        config['directory']=str(directory/'rust-tune')
        owner=native.PipelineProbe(json.dumps(config),120.)
        try:
            selection=json.loads(owner.tune(json.dumps([dict(name='baseline',environment=profile),
                dict(name='outer-double',environment=dict(profile,BEAM_B_MICRO='1536'))]),32))
            assert selection['effective_beam']==64 and selection['world_size']==world
            assert len(selection['measurements'])==10
            assert selection['estimate']['profile'] in ('baseline','outer-double')
            records.append(dict(world=world,selection=selection))
        finally:owner.close()
    (root/'results.json').write_text(json.dumps(dict(scope='Rust pipeline/control correctness; tiny fixture is not a production performance profile',records=records),indent=2))
    print('NATIVE_PIPELINE_GPU_OK',flush=True)


if __name__=='__main__':main()
