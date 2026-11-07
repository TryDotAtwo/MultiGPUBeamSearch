"""Actual Rust session identity, result ownership and independent path replay."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import threading
import time
from multigpubeamsearch.backend import parse_terminal
from multigpubeamsearch.contracts import GraphContract


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(directory):
    digest=hashlib.sha256()
    for path in sorted(directory.rglob('*')):
        if path.is_file():digest.update(path.relative_to(directory).as_posix().encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def failure_checks(native,args,root,identity,launch,weights,center,moves):
    start=center
    for move in moves:start=tuple(start[i] for i in move)
    identity=json.loads(json.dumps(identity));launch=json.loads(json.dumps(launch))
    identity.update(model_artifact_hash=artifact(weights),beam_width=64,max_steps=32,options=dict(solution_mode='first'))
    identity['runtime_build']['native_library_sha256']=sha(args.library)
    reports=[]
    for mode in ('cancel','failed_rank'):
        directory=root/('failure-'+mode);directory.mkdir()
        wrapper=directory/'rank.py'
        wrapper.write_text("import os,sys\nfrom pathlib import Path\n"
            "rank=os.environ['RANK'];Path(os.environ['BEAM_OWNER_PID_DIR'],f'rank-{rank}.pid').write_text(str(os.getpid()))\n"
            +("if rank=='1':os.environ['BEAM_NATIVE_LIBRARY']='/beam-no-such-library.so'\n" if mode=='failed_rank' else '')
            +"os.execv(sys.argv[1],sys.argv[1:])\n")
        launch['command']=[sys.executable,str(wrapper),str(args.runner.resolve()),'0','32','64']
        launch['cwd']=str(directory)
        launch['environment'].update(BEAM_NATIVE_LIBRARY=str(args.library.resolve()),BEAM_WEIGHT_DIR=str(weights),
            BEAM_STREAM1_EXECUTOR='native_cuda_graph',
            BEAM_OWNER_PID_DIR=str(directory),BEAM_NCCL_ID_FILE=str(directory/'nccl.bin'),BEAM_SOLVE_BUCKET_MODE='0',
            BEAM_HISTORY_DIR=str(directory/'history'),BEAM_HISTORY_DISK_PATH=str(directory/'history'))
        owner=native.SearchSession();errors=[]
        def search():
            try:owner.run(json.dumps(identity),json.dumps(launch),list(start),directory,20.,'{}')
            except RuntimeError as error:errors.append(str(error))
        began=time.monotonic();worker=threading.Thread(target=search);worker.start()
        if mode=='cancel':
            deadline=time.monotonic()+10
            while worker.is_alive() and not (directory/'resident-rank-0.log').exists() and time.monotonic()<deadline:time.sleep(.005)
            owner.cancel()
        worker.join(timeout=25);assert not worker.is_alive()
        owner.close();assert errors and ('cancelled' if mode=='cancel' else 'exited') in errors[0],errors
        pids=[int(path.read_text()) for path in directory.glob('rank-*.pid')]
        assert len(pids)==2 and all(not Path(f'/proc/{pid}').exists() for pid in pids)
        assert owner.closed and owner.pids==[]
        reports.append(dict(mode=mode,error=errors[0],pids=pids,elapsed_seconds=time.monotonic()-began))
    return reports


def main():
    parser=argparse.ArgumentParser()
    for name in ('runner','library','libtorch-library','extension','fixture','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('_beam_runtime',args.extension)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    fixture=args.fixture.resolve();data=json.loads((fixture/'puzzle.json').read_text())
    center=tuple(data['central_state']);moves=tuple(tuple(row) for row in data['generators'].values())
    weights=fixture/'weights';manifest=json.loads((weights/'manifest.json').read_text())
    graph=GraphContract(96,6,moves,tuple(data['generators']),center,center,manifest['graph_hash'])
    alternate=root/'alternate-weights';shutil.copytree(weights,alternate)
    (alternate/'output_bias.fp16').write_bytes(struct.pack('<e',1.))
    starts=[]
    for count in (3,4,5):
        start=center
        for move in range(count):start=tuple(start[i] for i in moves[move])
        starts.append(start)
    records=[]
    for backend,library in [('native',args.library),('libtorch',args.libtorch_library)]:
        for world in (1,2):
            directory=root/f'{backend}-{world}';directory.mkdir()
            identity=dict(graph=dict(center=center,generators=moves),graph_hash=graph.graph_hash,
                model_artifact_hash=artifact(weights),model_backend='mlp',devices=list(range(world)),beam_width=64,max_steps=4,
                runtime_build=dict(binary_sha256=sha(args.runner),native_library_sha256=sha(library),abi_version=1),
                options=dict(solution_mode='first',history='disk'),cuda_visible_devices=','.join(map(str,range(world))),cuda_device_order=None)
            env=dict(os.environ,BEAM_GENERATOR_PATH=str(fixture/'puzzle.json'),BEAM_PUZZLE_INFO_JSON=str(fixture/'puzzle.json'),
                BEAM_TEST_CSV=str(fixture/'test.csv'),BEAM_WEIGHT_DIR=str(weights),BEAM_RUNTIME_CONFIG_MODE='auto',
                BEAM_NATIVE_LIBRARY=str(library.resolve()),BEAM_NCCL_ID_FILE=str(directory/'nccl.bin'),
                BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_SOLVE_BUCKET_MODE='0',
                BEAM_HISTORY_MODE='disk',BEAM_HISTORY_RAM_BYTES=str(16<<20),BEAM_HISTORY_DISK_BYTES=str(1<<20),
                BEAM_HISTORY_DIR=str(directory/'history'),BEAM_HISTORY_DISK_PATH=str(directory/'history'),
                BEAM_B_MICRO='768',BEAM_SHARD_COUNT='1',BEAM_STREAM4_BATCH_CANDIDATES='1024',
                BEAM_STREAM1_EXECUTOR='libtorch_eager' if backend=='libtorch' else 'native_cuda_graph',
                CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))))
            launch=dict(command=[str(args.runner.resolve()),'0','4','64'],
                protocol_command=[str(args.runner.resolve()),'--resident-protocol-info'],environment=env,cwd=str(directory))
            metadata=dict(profile=dict(calibration_total_wall_seconds=3.),build=identity['runtime_build'])
            owner=native.SearchSession();previous=[]
            try:
                cases=[('first',starts[0]),('repeat',starts[1]),('unsolved',starts[2]),
                    ('model-change',starts[0]),('beam-change',starts[0]),('collect',starts[0])]
                for index,(mode,start) in enumerate(cases):
                    if mode=='model-change':
                        identity['model_artifact_hash']=artifact(alternate);env['BEAM_WEIGHT_DIR']=str(alternate)
                    if mode=='beam-change':
                        identity['beam_width']=65;launch['command'][-1]='65'
                    if mode=='collect':
                        identity['options']['solution_mode']='collect'
                        env.update(BEAM_SOLVE_BUCKET_MODE='1',BEAM_SOLVE_BUCKET_EXTRA_DEPTHS='1',
                            BEAM_SOLVE_BUCKET_MAX_SOLUTIONS='7',BEAM_SOLVE_BUCKET_RESULT_TSV='collected-solutions.tsv')
                    out=directory/mode;out.mkdir()
                    result=json.loads(owner.run(json.dumps(identity),json.dumps(launch),list(start),out,120.,json.dumps(metadata)))
                    reference=parse_terminal(out/'native.log',replace(graph,start=start),max_steps=4)
                    expected_path=None if reference[0] is None else list(reference[0])
                    assert result['path']==expected_path and result['effective_beam_width']==reference[1]==identity['beam_width']
                    for key,value in reference[2].items():assert result['metadata'][key]==value,(key,result,reference)
                    assert 'history_provider=rust' in (out/'native.log').read_text()
                    reused=index in (1,2)
                    if not reused:
                        expected='libtorch_eager' if backend=='libtorch' else 'native_cuda_graph'
                        for rank in range(world):
                            assert f'stream1_executor={expected}\n' in (out/'resident-session'/f'bootstrap-{rank}.log').read_text()
                    assert result['metadata']['persistent_session']['reused']==reused
                    assert result['metadata']['model_artifact_hash']==identity['model_artifact_hash']
                    if reused:
                        assert owner.pids==previous
                        assert result['metadata']['profile']['calibration_total_wall_seconds']==0.
                    elif previous:
                        assert owner.pids!=previous
                        assert all(not Path(f'/proc/{pid}').exists() for pid in previous)
                    previous=owner.pids
                    if mode=='unsolved':assert result['path'] is None
                    else:assert result['path'] is not None
                    if mode=='collect':assert 1<=len(result['metadata']['collected_solutions'])<=7
                    stored=json.loads((out/'native-outcome.json').read_text());assert stored==result
                    records.append(dict(backend=backend,world=world,mode=mode,result=result))
                    print(f'SEARCH_OWNER_OK {backend} world={world} mode={mode}',flush=True)
            finally:owner.close()
            assert owner.closed and not owner.pids
            assert all(not Path(f'/proc/{pid}').exists() for pid in previous)
    failures=failure_checks(native,args,root,identity,launch,weights,center,moves)
    (root/'results.json').write_text(json.dumps(dict(scope='Rust resource identity and result correctness; tiny fixture, no speed promotion',records=records,failures=failures),indent=2))
    print('NATIVE_SEARCH_OWNER_GPU_OK',flush=True)


if __name__=='__main__':main()
