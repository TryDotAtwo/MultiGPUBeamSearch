"""Coarse Rust downstream jobs, original service oracle and full-step receipts."""
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys,threading,time
from pathlib import Path
from multigpubeamsearch.component_autotune import service_envelope

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
    assert hasattr(native,'DownstreamCalibration'),'Rust downstream coarse owner is missing'
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    runner=args.build/('production_runner_libtorch_stream1' if args.executor=='libtorch_eager' else 'production_runner')
    helper=args.build/('stream1_libtorch_mlp_benchmark' if args.executor=='libtorch_eager' else 'stream1_native_mlp_benchmark')
    fixture=args.fixture.resolve();source=json.loads((fixture/'puzzle.json').read_text())
    graph=dict(center=source['central_state'],generators=list(source['generators'].values()))
    start=graph['center']
    for move in range(20):start=[start[i] for i in graph['generators'][move]]
    csv=root/'test.csv';csv.write_text('initial_state_id,initial_state\n0,"'+','.join(map(str,start))+'"\n')
    files=[artifact(p) for p in sorted((fixture/'weights').rglob('*')) if p.is_file()]
    manifest=json.loads((fixture/'weights'/'manifest.json').read_text())
    row_factor=len(graph['generators']) if manifest['output_dim']==1 else 1
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    records=[]
    for world in (1,2):
        directory=root/f'world-{world}';directory.mkdir()
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))),
            BEAM_GENERATOR_PATH=str(fixture/'puzzle.json'),BEAM_PUZZLE_INFO_JSON=str(fixture/'puzzle.json'),
            BEAM_TEST_CSV=str(csv),BEAM_WEIGHT_DIR=str(fixture/'weights'),BEAM_RUNTIME_CONFIG_MODE='auto',
            BEAM_SHARD_COUNT='1',BEAM_STREAM3_RING_SLOTS='2',BEAM_STREAM4_ACTIVE_SORT_SLOTS='1',
            # This control comparison reserves enough metadata for the whole
            # gross expansion. A separate retained baseline receipt documents
            # overflow with the old 1024-flush/5120-capacity profile at width4096.
            BEAM_STREAM4_BATCH_CANDIDATES='131072',BEAM_STREAM1_EXECUTOR=args.executor)
        inference_dir=directory/'inference';inference_dir.mkdir()
        inf_request=dict(options=dict(cache_dir=str(root/'cache'),calibration_max_batch=16,calibration_seconds=60.),
            graph=graph,helper=artifact(helper),artifacts=files,
            probe_manifest=dict(weights_dir=str(fixture/'weights')),ensemble=False,
            signature=dict(world_size=world,device_indices=list(range(world)),gpu_uuids=uuids[:world]),
            beam_width=4096,protocol='oneshot',capacity_bootstrap=False,run_dir=str(inference_dir),environment=environment,
            shared_profile=dict(inference=dict(parent_batch=16),registry_sha256='test-only'))
        inference=json.loads(native.InferenceCalibration(json.dumps(inf_request)).run())
        request=dict(options=dict(cache_dir=str(root/'cache'),calibration_max_batch=16,calibration_pipeline_seconds=180.),
            graph=graph,runner=artifact(runner),artifacts=files,inference=inference,
            environment=environment,beam_width=4096,row_factor=row_factor,hardware=dict(sm_count=28,l2_bytes=3<<20),
            run_dir='',cwd=str(root),physical_bytes=112,puzzle_id=0,grow=True,fixtures=[])
        for full in (False,True):
            request['options']['calibration_full_frontier']=full
            for attempt in ('first','hit'):
                run=directory/f'{"full" if full else "component"}-{attempt}';run.mkdir();request['run_dir']=str(run)
                result=json.loads(native.DownstreamCalibration(json.dumps(request)).run())
                assert result['cache_hit']==(attempt=='hit'),(world,full,attempt,result['selection'],run)
                assert result['pipeline_verified'] is full
                assert result['requested_beam_width']==result['requested_beam_effective']==4096
                assert result['environment']['BEAM_ENSEMBLE_INFERENCE_MICRO']=='16'
                for trial in result['tested']:
                    if trial['rejected'] is not None:continue
                    reference=service_envelope(trial['rank_measurements'],trial['plans'][0],moves=len(graph['generators']),
                        inference_seconds=inference['estimate']['median'],parent_batch=16)
                    for key,value in reference.items():
                        if isinstance(value,float):assert abs(trial['estimate'][key]-value)<=max(abs(value),1e-300)*1e-12,(key,trial,reference)
                    assert int(trial['environment']['BEAM_B_MICRO'])==trial['plans'][0]['B_MICRO']*row_factor
                if full:
                    assert result['full_grid_complete']
                    rows=result['full_comparison_measurements']
                    for name in result['full_tested']:
                        samples=[r for r in rows if r['profile']==name]
                        assert len(samples)==5 and all(r['parents']==4096 and len(r['seconds_by_rank'])==world and not r['throttled'] for r in samples)
                    if attempt=='first':
                        paths=list((run/'pipeline-calibration/full-steps').glob('*/ranks/rank-*.log'))
                        assert len(paths)==world*len(result['full_tested'])
                        assert all('stream1_executor='+args.executor in p.read_text() for p in paths)
                for p in (run/'pipeline-calibration').rglob('closed.json'):
                    closed=json.loads(p.read_text());assert all(c is not None for c in closed['exit_codes'])
                    assert not any(Path(f'/proc/{pid}').exists() for pid in closed['pids'])
                records.append(dict(world=world,full=full,attempt=attempt,result=result))
                print(f'DOWNSTREAM_JOB_OK world={world} full={full} attempt={attempt}',flush=True)
        if world==2:
            # A forged full-step selection never replaces the measured outcome.
            matches=[p for p in (root/'cache/pipeline-profiles').glob('*-full.json')
                if json.loads(p.read_text())['identity']['tuning']['device_indices']==[0,1]]
            assert len(matches)==1
            corrupted=json.loads(matches[0].read_text());corrupted['selection']['environment']['BEAM_ENSEMBLE_INFERENCE_MICRO']='32'
            matches[0].write_text(json.dumps(corrupted))
            run=directory/'corrupt-full-cache';run.mkdir();request['run_dir']=str(run)
            result=json.loads(native.DownstreamCalibration(json.dumps(request)).run())
            assert not result['cache_hit'] and result['pipeline_verified'] and result['environment']['BEAM_ENSEMBLE_INFERENCE_MICRO']=='16'
            records.append(dict(world=world,attempt='corrupt-full-cache',result=result))
    # Exercise the real public model/options adapter, including descriptor order.
    from multigpubeamsearch import rust_control
    from multigpubeamsearch.contracts import GraphContract
    from multigpubeamsearch.models import _validate_artifact
    from multigpubeamsearch.options import NativeOptions
    from multigpubeamsearch.backend import PreparedRuntime
    sys.modules['_beam_runtime']=native
    graph_hash=hashlib.sha256(json.dumps(dict(generators_type='PERMUTATION',generators=graph['generators'],central_state=graph['center']),sort_keys=True).encode()).hexdigest()
    contract=GraphContract(96,max(graph['center'])+1,tuple(map(tuple,graph['generators'])),
        tuple(source['generators']),tuple(graph['center']),tuple(start),graph_hash)
    model=_validate_artifact(fixture/'weights',contract,'mlp')
    runtime=PreparedRuntime(runner.resolve(),dict(binary_sha256=artifact(runner)['sha256'],
        calibration_binary_name=helper.name,calibration_binary_sha256=artifact(helper)['sha256'],calibration_protocol='oneshot',
        inference_backend='libtorch' if args.executor=='libtorch_eager' else 'cutlass',
        shape=dict(storage_len=112),frontier_calibration_protocol='natural-growth-v1'),(86,))
    options=NativeOptions(cache_dir=root/'public-cache',calibration_max_batch=16,calibration_seconds=45.,calibration_pipeline_seconds=120.)
    public=[]
    for attempt in ('first','hit'):
        run=root/f'public-{attempt}';run.mkdir()
        inference_public=rust_control.tune_inference(contract,model,runtime,options,(0,1),4096,run,environment)
        outcome=rust_control.tune_downstream(contract,model,runtime,options,(0,1),4096,run,environment,runner,inference_public)
        assert outcome['cache_hit']==(attempt=='hit') and not outcome['pipeline_verified']
        public.append(outcome)
    (root/'public.json').write_text(json.dumps(public,indent=2))
    print('PUBLIC_THIN_DOWNSTREAM_OK',flush=True)
    # Cancellation while ranks are still initializing must terminate every rank.
    request=copy.deepcopy(request);request['options']['calibration_full_frontier']=False
    block=root/'blocking-runner';block.write_text(f'#!{sys.executable}\nimport os,time\nfrom pathlib import Path\n'
        f'Path({str(root)!r},"blocked-"+os.environ["RANK"]).write_text(str(os.getpid()))\ntime.sleep(30)\n');block.chmod(0o700)
    run=root/'cancel';run.mkdir();request.update(runner=artifact(block),run_dir=str(run))
    owner=native.DownstreamCalibration(json.dumps(request));errors=[]
    def work():
        try:owner.run()
        except RuntimeError as error:errors.append(str(error))
    thread=threading.Thread(target=work);thread.start();deadline=time.monotonic()+20
    pidfiles=[root/f'blocked-{rank}' for rank in range(2)]
    while not all(p.exists() for p in pidfiles) and thread.is_alive() and time.monotonic()<deadline:time.sleep(.02)
    assert all(p.exists() for p in pidfiles),errors
    pids=[int(p.read_text()) for p in pidfiles];owner.cancel();thread.join(4)
    assert not thread.is_alive() and errors and not any(Path(f'/proc/{pid}').exists() for pid in pids)
    (root/'results.json').write_text(json.dumps(dict(records=records,cancel=dict(pids=pids,errors=errors)),indent=2))
    print('NATIVE_DOWNSTREAM_JOB_GPU_OK',flush=True)

if __name__=='__main__':main()
