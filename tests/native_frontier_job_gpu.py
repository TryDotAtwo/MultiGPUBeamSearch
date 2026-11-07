"""One coarse Rust call owns inference, exact admission and width selection."""
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys,threading,time
from pathlib import Path

def artifact(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    parser=argparse.ArgumentParser()
    for name in ('build','extension','fixture','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--executor',choices=('native_cuda_graph','libtorch_eager'),default='native_cuda_graph')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('_beam_runtime',args.extension)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    assert hasattr(native,'FrontierCalibration'),'Rust frontier job owner is missing'
    import torch
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    fixture=args.fixture.resolve();puzzle=json.loads((fixture/'puzzle.json').read_text())
    graph=dict(center=puzzle['central_state'],generators=list(puzzle['generators'].values()))
    start=graph['center']
    for move in range(20):start=[start[i] for i in graph['generators'][move]]
    csv=root/'test.csv';csv.write_text('initial_state_id,initial_state\n0,"'+','.join(map(str,start))+'"\n')
    runner=args.build/('production_runner_libtorch_stream1' if args.executor=='libtorch_eager' else 'production_runner')
    helper=args.build/('stream1_libtorch_mlp_benchmark' if args.executor=='libtorch_eager' else 'stream1_native_mlp_benchmark')
    files=[artifact(p) for p in sorted((fixture/'weights').rglob('*')) if p.is_file()]
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    records=[]
    for world in (1,2):
        properties=[torch.cuda.get_device_properties(i) for i in range(world)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))),
            BEAM_GENERATOR_PATH=str(fixture/'puzzle.json'),BEAM_PUZZLE_INFO_JSON=str(fixture/'puzzle.json'),
            BEAM_TEST_CSV=str(csv),BEAM_WEIGHT_DIR=str(fixture/'weights'),BEAM_RUNTIME_CONFIG_MODE='auto',
            BEAM_SHARD_COUNT='1',BEAM_STREAM3_RING_SLOTS='2',BEAM_STREAM4_ACTIVE_SORT_SLOTS='1',
            BEAM_STREAM4_BATCH_CANDIDATES='262144',BEAM_STREAM1_EXECUTOR=args.executor)
        request=dict(beam_width=4097,inference=dict(options=dict(cache_dir=str(root/'cache'),
            calibration_max_batch=16,calibration_seconds=60.,calibration_pipeline_seconds=180.,
            calibration_full_frontier=True,calibration_frontier_max_states=8192),
            graph=graph,helper=artifact(helper),artifacts=files,probe_manifest=dict(weights_dir=str(fixture/'weights')),
            ensemble=False,signature=dict(world_size=world,device_indices=list(range(world)),gpu_uuids=uuids[:world]),
            beam_width=4097,protocol='oneshot',capacity_bootstrap=False,run_dir='',environment=env,
            shared_profile=dict(inference=dict(parent_batch=16),registry_sha256='test-only')),
            execution=dict(runner=artifact(runner),row_factor=len(graph['generators']),
                hardware=dict(sm_count=min(p.multi_processor_count for p in properties),l2_bytes=3<<20),
                physical_bytes=112,puzzle_id=0,grow=True,fixtures=[],cwd=str(root),
                gpu_memory_bytes=[p.total_memory for p in properties],frontier_ceiling=8192))
        for mode,attempt in [(4097,'first'),(4097,'hit'),('max','first')]:
            request['beam_width']=mode
            run=root/f'world-{world}-{mode}-{attempt}';run.mkdir()
            request['inference']['run_dir']=str(run)
            result=json.loads(native.FrontierCalibration(json.dumps(request)).run())
            assert result['requested_beam_width']==mode
            assert result['pipeline']['pipeline_verified']
            width=result['effective_beam_width']
            assert result['pipeline']['requested_beam_effective']==width
            if mode!='max':
                assert width==4097 and result['inference']['inference_admission']['requested_beam']==4097
                assert result['pipeline']['cache_hit']==(attempt=='hit')
            else:
                assert 0<width<=8192 and result['capacity']['full_step_verified']
                assert result['working_capacity']['efficiency_verified']
                assert result['capacity']['ceiling']['explicit_state_ceiling']==8192
                assert result['capacity']['inference_refinement']
                assert result['capacity']['full_step_boundary_beam']>=width
                assert result['working_capacity']['comparisons']
            for path in run.rglob('closed.json'):
                receipt=json.loads(path.read_text());assert all(c is not None for c in receipt['exit_codes'])
                assert not any(Path(f'/proc/{pid}').exists() for pid in receipt['pids'])
            full_logs=[p for p in run.rglob('rank-*.log') if 'calibration_saturated_steps=' in p.read_text()]
            if attempt=='first':assert full_logs and all('stream1_executor='+args.executor in p.read_text() for p in full_logs)
            records.append(result)
            print(f'FRONTIER_JOB_OK world={world} mode={mode} attempt={attempt} selected={width}',flush=True)
    # Actual public model/options adapter transfers one complete descriptor.
    from multigpubeamsearch import rust_control
    from multigpubeamsearch.contracts import GraphContract
    from multigpubeamsearch.models import _validate_artifact
    from multigpubeamsearch.options import NativeOptions
    from multigpubeamsearch.backend import PreparedRuntime
    sys.modules['_beam_runtime']=native
    graph_hash=hashlib.sha256(json.dumps(dict(generators_type='PERMUTATION',generators=graph['generators'],central_state=graph['center']),sort_keys=True).encode()).hexdigest()
    contract=GraphContract(96,max(graph['center'])+1,tuple(map(tuple,graph['generators'])),
        tuple(puzzle['generators']),tuple(graph['center']),tuple(start),graph_hash)
    model=_validate_artifact(fixture/'weights',contract,'mlp')
    runtime=PreparedRuntime(runner.resolve(),dict(binary_sha256=artifact(runner)['sha256'],
        calibration_binary_name=helper.name,calibration_binary_sha256=artifact(helper)['sha256'],calibration_protocol='oneshot',
        inference_backend='libtorch' if args.executor=='libtorch_eager' else 'cutlass',
        shape=dict(storage_len=112),frontier_calibration_protocol='natural-growth-v1'),(86,))
    options=NativeOptions(cache_dir=root/'public-cache',calibration_max_batch=16,calibration_seconds=60.,calibration_pipeline_seconds=180.)
    public=[]
    for attempt in ('first','hit'):
        run=root/f'public-{attempt}';run.mkdir()
        result=rust_control.calibrate_frontier(contract,model,runtime,options,(0,1),4097,run,env,runner)
        assert result['effective_beam_width']==4097 and result['pipeline']['pipeline_verified'] is False
        assert result['pipeline']['cache_hit']==(attempt=='hit')
        public.append(result)
    (root/'public.json').write_text(json.dumps(public,indent=2))
    print('PUBLIC_THIN_FRONTIER_OK',flush=True)
    # Independent cancellation token must reach both in-flight inference ranks.
    request=copy.deepcopy(request);request['beam_width']=4097
    block=root/'blocking-helper';block.write_text(f'#!{sys.executable}\nimport os,sys,time\nfrom pathlib import Path\n'
        f'Path({str(root)!r},"blocked-"+sys.argv[4]).write_text(str(os.getpid()))\ntime.sleep(30)\n');block.chmod(0o700)
    request['inference']['helper']=artifact(block);request['inference']['run_dir']=str(root/'cancel')
    (root/'cancel').mkdir();owner=native.FrontierCalibration(json.dumps(request));errors=[]
    def work():
        try:owner.run()
        except RuntimeError as error:errors.append(str(error))
    thread=threading.Thread(target=work);thread.start();deadline=time.monotonic()+20
    while len(list(root.glob('blocked-*')))!=2 and thread.is_alive() and time.monotonic()<deadline:time.sleep(.02)
    assert len(list(root.glob('blocked-*')))==2,errors
    pids=[int(p.read_text()) for p in root.glob('blocked-*')];owner.cancel();thread.join(4)
    assert not thread.is_alive() and errors and not any(Path(f'/proc/{pid}').exists() for pid in pids)
    (root/'results.json').write_text(json.dumps(dict(records=records,cancellation=dict(pids=pids,errors=errors)),indent=2))
    print('NATIVE_FRONTIER_JOB_GPU_OK',flush=True)

if __name__=='__main__':main()
