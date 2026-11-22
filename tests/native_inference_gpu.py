"""Compare Rust-owned inference probes with existing C++ helpers on real GPUs.

Tiny exact Hamming models test control/quality/lifetime, not trained-model speed.
No inference kernels or numerical policy are replaced by this fixture.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

from multigpubeamsearch.calibration_session import EnsembleProbePool
from multigpubeamsearch.autotune import valid_cached_profile


def reference_once(helper,inputs,batch,parents,world,environment,directory):
    processes=[]
    try:
        for rank in range(world):
            log=(directory/f'batch-{batch}-rank-{rank}.log').open('wb')
            process=subprocess.Popen([str(helper),str(inputs),str(batch),str(parents),str(rank)],
                env=environment,stdout=log,stderr=subprocess.STDOUT)
            processes.append((process,log))
        rows=[]
        for rank,(process,log) in enumerate(processes):
            assert process.wait(timeout=120)==0
            log.close()
            values=[]
            for line in (directory/f'batch-{batch}-rank-{rank}.log').read_text().splitlines():
                try:row=json.loads(line)
                except ValueError:continue
                if 'seconds' in row:values.append(row)
            assert len(values)==1
            rows.extend(values)
        return rows
    finally:
        for process,log in processes:
            if process.poll() is None:process.kill();process.wait()
            log.close()


def main():
    parser=argparse.ArgumentParser()
    for name in ('build','extension','fixture','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('_beam_runtime',args.extension)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    data=json.loads((args.fixture/'puzzle.json').read_text())
    moves=list(data['generators'].values());center=data['central_state'];states=[center]
    for move in moves:states.append([states[-1][index] for index in move])
    inputs=root/'inputs';inputs.mkdir()
    common=dict(calibration_states=states,generators=moves)
    weights=str((args.fixture/'weights').resolve())
    (inputs/'calibration.json').write_text(json.dumps(dict(common,weights_dir=weights)))
    (inputs/'ensemble.json').write_text(json.dumps(dict(common,schema_version=1,state_len=96,output_dim=24,
        score_semantics='distance_q',models=[dict(family='mlp',weights_dir=weights,coefficient=.75),
        dict(family='mlp',weights_dir=weights,coefficient=.25)])))
    results=[]
    for world in (1,2):
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))))
        for backend in ('native_mlp','libtorch_mlp','ensemble'):
            helper=(args.build/f'stream1_{backend}_benchmark').resolve()
            directory=root/f'{backend}-{world}';directory.mkdir()
            ref_dir=directory/'reference';ref_dir.mkdir()
            session=backend=='ensemble'
            reference=EnsembleProbePool(helper,inputs,32,129,world,environment,ref_dir,time.monotonic()+180) if session else None
            config=dict(command=[str(helper)],inputs=str(inputs),capacity=32,parents=129,world=world,
                environment=dict(environment,BEAM_ENSEMBLE_INFERENCE_MICRO='1'),cwd=str(root),
                directory=str(directory/'rust'),protocol='json-session-v1' if session else 'oneshot')
            owner=runtime.InferencePool(json.dumps(config),180.)
            try:
                pids=None
                for batch in (8,32,16):
                    if reference:
                        old,failures=reference.measure(batch,129,time.monotonic()+120);assert not failures,failures
                    else:old=reference_once(helper,inputs,batch,129,world,environment,ref_dir)
                    new=json.loads(owner.measure(batch,129,120.));assert not new['failures'],new
                    assert len(new['rows'])==len(old)==world
                    for before,after in zip(old,new['rows']):
                        for key in ('batch','parents','device','correctness_passed','numeric_error','score_input'):
                            assert before[key]==after[key],(key,before,after)
                        assert after['correctness_passed'] and after['numeric_error']==0 and len(after['seconds'])==7
                    if session:
                        if pids is not None:assert owner.pids==pids
                        pids=owner.pids
                    else:assert owner.pids==[None]*world
                    results.append(dict(backend=backend,world=world,batch=batch,reference=old,rust=new))
                assert owner.starts==(world if session else 3*world)
            finally:
                owner.close()
                if reference:reference.close()
            assert owner.pids==[None]*world
            print(f'INFERENCE_COMPARE_OK {backend} world={world}',flush=True)
        # The entire search/telemetry/statistics loop executes in Rust. Public
        # cache validation must recognize the produced all-rank evidence.
        config.update(command=[str((args.build/'stream1_ensemble_benchmark').resolve())],parents=2048,
            directory=str(root/f'tune-{world}'),protocol='json-session-v1')
        owner=runtime.InferencePool(json.dumps(config),120.)
        options=dict(cap=32,parents=2048,world=world,capacity_bootstrap=False,shared_batch=None,
            device_indices=list(range(world)),gpu_uuids=[])
        try:selection=json.loads(owner.tune(json.dumps(options),120.))
        finally:owner.close()
        assert owner.pids==[None]*world and selection['probe_process_starts']==world
        signature=dict(schema=10,world_size=world,device_indices=list(range(world)),gpu_uuids=[])
        selection['signature']=signature
        assert valid_cached_profile(selection,signature,32),selection
        assert selection['phase']=='inference_verified' and not selection['pipeline_verified']
        (root/f'tune-{world}.json').write_text(json.dumps(selection,indent=2))
        print(f'INFERENCE_TUNE_OK world={world} micro={selection["parent_batch"]}',flush=True)
    (root/'results.json').write_text(json.dumps(dict(scope='inference control and numerical oracle compatibility; no production performance promotion',records=results),indent=2))
    print('NATIVE_INFERENCE_GPU_OK',flush=True)


if __name__=='__main__':main()
