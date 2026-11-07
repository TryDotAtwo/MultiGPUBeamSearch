"""Whole Rust calibration ownership, cache validation and cancellation on real GPUs."""
import argparse,copy,hashlib,importlib.util,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
from multigpubeamsearch.autotune import valid_cached_profile

def artifact(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    parser=argparse.ArgumentParser()
    for name in ('build','extension','fixture','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('_beam_runtime',args.extension)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    source=json.loads((args.fixture/'puzzle.json').read_text())
    graph=dict(center=source['central_state'],generators=list(source['generators'].values()))
    weights=root/'weights';shutil.copytree(args.fixture/'weights',weights)
    files=[artifact(p) for p in sorted(weights.rglob('*')) if p.is_file()]
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    results=[]
    for world in (1,2):
        for backend in ('native_mlp','libtorch_mlp','ensemble'):
            directory=root/f'{backend}-{world}';directory.mkdir()
            ensemble=backend=='ensemble'
            manifest=dict(weights_dir=str(weights))
            if ensemble:
                manifest=dict(schema_version=1,state_len=96,output_dim=24,score_semantics='distance_q',models=[
                    dict(family='mlp',weights_dir=str(weights),coefficient=.75),
                    dict(family='mlp',weights_dir=str(weights),coefficient=.25)])
            request=dict(options=dict(cache_dir=str(root/'cache'),calibration_max_batch=32,calibration_seconds=90),
                graph=graph,helper=artifact(args.build/f'stream1_{backend}_benchmark'),artifacts=files,
                probe_manifest=manifest,ensemble=ensemble,
                signature=dict(world_size=world,device_indices=list(range(world)),gpu_uuids=uuids[:world],backend=backend),
                beam_width=4096,protocol='json-session-v1' if ensemble else 'oneshot',capacity_bootstrap=False,
                run_dir=str(directory/'first'),environment=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,range(world)))),
                shared_profile=dict(inference=dict(parent_batch=16),registry_sha256='proposal-only'))
            def run(name):
                request['run_dir']=str(directory/name);Path(request['run_dir']).mkdir()
                return json.loads(runtime.InferenceCalibration(json.dumps(request)).run())
            first=run('first')
            assert not first['cache_hit'] and first['shared_profile_validated']
            assert first['probe_process_starts']==world
            assert valid_cached_profile(first,first['signature'],32),first
            hit=run('hit');assert hit['cache_hit']
            assert not (directory/'hit'/'calibration').exists()
            assert first['records']==hit['records'] and first['estimate']==hit['estimate']
            # Invalid evidence must cause a real fresh measurement, with no reuse
            # of cached timing or a private directory from the previous call.
            matches=[p for p in (root/'cache'/'profiles').glob('*.json')
                if json.loads(p.read_text())['signature']==first['signature']]
            assert len(matches)==1
            corrupt=copy.deepcopy(first);corrupt['records'][0]['numeric_error']=1
            matches[0].write_text(json.dumps(corrupt))
            retest=run('retest');assert not retest['cache_hit'] and retest['probe_process_starts']==world
            assert valid_cached_profile(retest,retest['signature'],32)
            # Content binding is checked even on a cache hit.
            model_file=Path(files[0]['path']);original=model_file.read_bytes()
            try:
                model_file.write_bytes(original+b'x')
                try:run('changed-model')
                except RuntimeError as error:assert 'artifact changed' in str(error)
                else:raise AssertionError('changed model reused cached profile')
            finally:model_file.write_bytes(original)
            results.append(dict(backend=backend,world=world,first=first,hit=hit,retest=retest))
            print(f'INFERENCE_JOB_OK {backend} world={world}',flush=True)
    # Exercise the actual thin package adapter with its graph/model contracts,
    # hardware signature and public NativeOptions dataclass.
    from multigpubeamsearch import rust_control
    from multigpubeamsearch.contracts import GraphContract
    from multigpubeamsearch.models import _validate_artifact
    from multigpubeamsearch.options import NativeOptions
    from multigpubeamsearch.backend import PreparedRuntime
    sys.modules['_beam_runtime']=runtime
    graph_hash=hashlib.sha256(json.dumps(dict(generators_type='PERMUTATION',
        generators=graph['generators'],central_state=graph['center']),sort_keys=True).encode()).hexdigest()
    contract=GraphContract(96,max(graph['center'])+1,tuple(map(tuple,graph['generators'])),
        tuple(source['generators']),tuple(graph['center']),tuple(graph['center']),graph_hash)
    model=_validate_artifact(weights,contract,'mlp')
    helper=artifact(args.build/'stream1_native_mlp_benchmark')
    build=PreparedRuntime((args.build/'production_runner').resolve(),dict(
        binary_sha256=artifact(args.build/'production_runner')['sha256'],inference_backend='cutlass',
        calibration_binary_name=Path(helper['path']).name,calibration_binary_sha256=helper['sha256'],
        calibration_protocol='oneshot'),(86,))
    options=NativeOptions(cache_dir=root/'public-cache',calibration_max_batch=16,calibration_seconds=45.)
    public=[]
    for name in ('public-first','public-hit'):
        directory=root/name;directory.mkdir()
        public.append(rust_control.tune_inference(contract,model,build,options,(0,1),4096,directory,
            dict(os.environ,CUDA_VISIBLE_DEVICES='0,1')))
    assert not public[0]['cache_hit'] and public[1]['cache_hit']
    (root/'public.json').write_text(json.dumps(public,indent=2))
    print('PUBLIC_THIN_INFERENCE_OK',flush=True)
    request=copy.deepcopy(request);directory=root/'cancel';directory.mkdir();request['run_dir']=str(directory)
    helper=root/'blocking-probe.py'
    helper.write_text(f'#!{sys.executable}\nimport sys,os,time\nfrom pathlib import Path\n'
        'root=Path(sys.argv[1]).parent;rank=int(sys.argv[4]);(root/f"rank-{rank}.pid").write_text(str(os.getpid()))\ntime.sleep(30)\n')
    helper.chmod(0o700);request.update(helper=artifact(helper),protocol='oneshot',shared_profile=None)
    owner=runtime.InferenceCalibration(json.dumps(request));errors=[]
    def work():
        try:owner.run()
        except RuntimeError as error:errors.append(str(error))
    thread=threading.Thread(target=work);thread.start();deadline=time.monotonic()+20
    pidfiles=[directory/'calibration'/f'rank-{rank}.pid' for rank in range(2)]
    while not all(p.exists() for p in pidfiles) and thread.is_alive() and time.monotonic()<deadline:time.sleep(.02)
    assert all(p.exists() for p in pidfiles),errors
    pids=[int(p.read_text()) for p in pidfiles];started=time.monotonic();owner.cancel();thread.join(4)
    assert not thread.is_alive() and time.monotonic()-started<4 and errors,errors
    assert not any(Path(f'/proc/{pid}').exists() for pid in pids)
    assert not (directory/'calibration'/'inference-selection.json').exists()
    (root/'results.json').write_text(json.dumps(dict(records=results,cancellation=dict(errors=errors,pids=pids)),indent=2))
    print('NATIVE_INFERENCE_JOB_GPU_OK',flush=True)

if __name__=='__main__':main()
