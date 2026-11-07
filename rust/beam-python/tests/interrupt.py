"""SIGINT must reach a coarse Rust job before its normal deadline."""
import hashlib,importlib.util,json,os,signal,sys,tempfile,threading,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('_beam_runtime',sys.argv[1])
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);helper=root/'wait.py';pidfile=root/'child.pid'
    helper.write_text(f'#!{sys.executable}\nimport os,time\nfrom pathlib import Path\n'
        f'Path({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n')
    helper.chmod(0o700);model=root/'model';model.write_bytes(b'model')
    def artifact(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    request=dict(options=dict(cache_dir=str(root/'cache'),calibration_max_batch=32,calibration_seconds=25),
        graph=dict(center=[0,1],generators=[[1,0]]),helper=artifact(helper),artifacts=[artifact(model)],
        probe_manifest=dict(weights_dir=tmp),ensemble=False,
        signature=dict(world_size=1,device_indices=[0],gpu_uuids=[]),beam_width=32,
        protocol='oneshot',capacity_bootstrap=False,run_dir=tmp,environment=dict(os.environ),shared_profile=None)
    owner=runtime.InferenceCalibration(json.dumps(request));sent=[];finished=threading.Event()
    def interrupt():
        deadline=time.monotonic()+10
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert pidfile.exists()
        sent.append(time.monotonic());os.kill(os.getpid(),signal.SIGINT)
        if not finished.wait(5):owner.cancel()  # bounded rescue for the RED implementation
    thread=threading.Thread(target=interrupt);thread.start()
    try:
        try:owner.run()
        except KeyboardInterrupt:pass
        else:raise AssertionError('SIGINT was lost')
        elapsed=time.monotonic()-sent[0]
        assert elapsed<4,('SIGINT remained blocked in Rust',elapsed)
        assert not Path('/proc/'+pidfile.read_text()).exists()
        assert not list((root/'cache').rglob('*.json'))
    finally:finished.set();owner.cancel();thread.join()
print('PYO3_SIGINT_CLEANUP_OK')

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);worker=root/'rank.py';pidfile=root/'child.pid'
    worker.write_text('import sys,os,json,time\nfrom pathlib import Path\n'
        'if "--resident-protocol-info" in sys.argv:\n'
        ' print(json.dumps(dict(protocol="resident-search-v1")),flush=True);raise SystemExit(0)\n'
        f'Path({str(pidfile)!r}).write_text(str(os.getpid()))\n'
        'directory=Path(os.environ["BEAM_SESSION_DIR"])\n'
        'while not (directory/"stop").exists():time.sleep(.002)\n')
    identity=dict(graph=dict(center=[0,1],generators=[[1,0]]),graph_hash='a'*64,
        model_artifact_hash='b'*64,model_backend='mlp',runtime_build=dict(binary_sha256='c'*64),
        devices=[0],beam_width=64,max_steps=1,options={},cuda_visible_devices=None,cuda_device_order=None)
    command=[sys.executable,str(worker)]
    launch=dict(command=command,protocol_command=command+['--resident-protocol-info'],
        environment=dict(os.environ),cwd=str(root))
    owner=runtime.SearchSession();sent=[];finished=threading.Event()
    def interrupt_search():
        deadline=time.monotonic()+10
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert pidfile.exists()
        sent.append(time.monotonic());os.kill(os.getpid(),signal.SIGINT)
        if not finished.wait(5):owner.cancel()
    thread=threading.Thread(target=interrupt_search);thread.start()
    try:
        try:owner.run(json.dumps(identity),json.dumps(launch),[1,0],root,25.,'{}')
        except KeyboardInterrupt:pass
        else:raise AssertionError('search SIGINT was lost')
        assert time.monotonic()-sent[0]<4
        assert owner.pids==[] and not Path('/proc/'+pidfile.read_text()).exists()
    finally:finished.set();owner.close();thread.join()
print('PYO3_SEARCH_SIGINT_CLEANUP_OK')
