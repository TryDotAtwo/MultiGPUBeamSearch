import sys
from pathlib import Path

import pytest

from multigpubeamsearch.persistent import NativeSession, current_session


def test_context_restores_and_closes():
    session = NativeSession()
    assert current_session() is None
    with session:
        assert current_session() is session
        with pytest.raises(RuntimeError, match='nested'):
            with NativeSession():
                pass
    assert current_session() is None
    assert session.closed
    session.close()
    with pytest.raises(RuntimeError, match='closed'):
        with session:
            pass


def test_explicit_notebook_call_preserves_session_and_restores_context(monkeypatch):
    from multigpubeamsearch import dispatch
    seen=[]
    def search(graph,**kwargs):
        seen.append(current_session())
        return graph
    monkeypatch.setattr(dispatch,'beam_search',search)
    session=NativeSession()
    assert session.beam_search('a')=='a'
    assert session.beam_search('b')=='b'
    assert seen==[session,session] and current_session() is None and not session.closed
    session.close()


def test_forked_owner_cannot_close_parent_workers(monkeypatch):
    import multigpubeamsearch.persistent as module
    session=NativeSession()
    monkeypatch.setattr(module.os,'getpid',lambda:session._owner_pid+1)
    with pytest.raises(RuntimeError,match='Python processes'):
        session.close()
    assert not session.closed


def test_transport_reuses_pids_and_isolates_logs(tmp_path):
    from multigpubeamsearch.resident_transport import ResidentWorkers
    script = tmp_path / 'worker.py'
    script.write_text('''import json,os,time
from pathlib import Path
root=Path(os.environ['BEAM_SESSION_DIR']); rank=int(os.environ['RANK']); seq=0
while not (root/'stop').exists():
 p=root/f'request-{seq}.json'
 if not p.exists(): time.sleep(.01);continue
 row=json.loads(p.read_text());out=Path(row['run_dir'])
 (out/f'resident-rank-{rank}.log').write_text(str(row['start']))
 done=root/f'done-{seq}-{rank}.json';tmp=done.with_suffix('.tmp')
 tmp.write_text(json.dumps({'sequence':seq,'rank':rank}));tmp.replace(done)
 seq+=1
''')
    import os
    workers = ResidentWorkers([sys.executable, str(script)], world=2,
        directory=tmp_path/'session', cwd=tmp_path, env=os.environ)
    try:
        for seq in range(2):
            run = tmp_path / str(seq);run.mkdir()
            workers.submit([seq], run, timeout=5)
            assert (run/'native.log').read_text().count(str([seq])) == 2
            if seq == 0:pids = workers.pids
            else:assert workers.pids == pids
    finally:workers.close()
    assert all(p.poll() is not None for p in workers.processes)


def test_worker_failure_closes_whole_group(tmp_path):
    from multigpubeamsearch.resident_transport import ResidentWorkers
    from multigpubeamsearch.errors import NativeBackendError
    import os
    workers = ResidentWorkers([sys.executable, '-c', 'raise SystemExit(7)'],world=2,
        directory=tmp_path/'session',cwd=tmp_path,env=os.environ)
    with pytest.raises(NativeBackendError):
        workers.submit([0],tmp_path,timeout=5)
    assert all(p.poll() is not None for p in workers.processes)


@pytest.mark.parametrize("world", [1, 2])
def test_backend_reuses_only_compatible_resources_and_replays(tmp_path, monkeypatch, world):
    from types import SimpleNamespace
    from multigpubeamsearch import models
    from multigpubeamsearch.options import NativeOptions, NativeOutcome
    from multigpubeamsearch.errors import NativeBackendError
    monkeypatch.setattr(models, 'verify_prepared_model', lambda *args: None)
    allocations=[]
    class Workers:
        pids=[123]
        sequence=1
        closed=False
        invalid=False
        def close(self):self.closed=True
        def submit(self,start,directory,timeout):
            self.sequence+=1
            (Path(directory)/'native.log').write_text(
                'GLOBAL_BEAM_WIDTH_EFFECTIVE=64\n'
                'puzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 solution=m1\n')
            return .1
    session=NativeSession()
    def execute(*args,**kwargs):
        worker=Workers();allocations.append(worker);session._workers=worker
        return NativeOutcome((1,),.1,64,tmp_path,dict(requested_beam_width=64,profile={},terminal_records=1))
    contract=SimpleNamespace(graph_hash='g',start=(1,0),move_count=2,replay=lambda path:tuple(path)==(1,))
    model=SimpleNamespace(artifact_hash='m',backend='mlp')
    runtime=SimpleNamespace(build_metadata={'binary_sha256':'a'})
    args=(execute,contract,model,NativeOptions(),64,4,tmp_path,tuple(range(world)))
    with session:
        first=session.run(*args,runtime=runtime)
        second=session.run(*args,runtime=runtime)
        assert len(allocations)==1 and second.metadata['persistent_session']['reused']
        model.artifact_hash='changed'
        session.run(*args,runtime=runtime)
        assert len(allocations)==2 and allocations[0].closed
        contract.replay=lambda path:False
        with pytest.raises(NativeBackendError,match='replay'):
            session.run(*args,runtime=runtime)
        assert allocations[-1].closed


def test_loader_failure_preserves_diagnostic_before_starting_workers(tmp_path, monkeypatch):
    import subprocess
    from multigpubeamsearch.errors import NativeBackendError
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(
        a[0], 127, '', 'error while loading shared libraries: libcudart.so.12'))
    with NativeSession() as session:
        with pytest.raises(NativeBackendError, match='exit 127.*libcudart'):
            session.launch(['runner'], contract=None, run_dir=tmp_path, env={}, world=2, timeout=30)
        assert session._workers is None


def test_timeout_terminates_workers(tmp_path):
    from multigpubeamsearch.resident_transport import ResidentWorkers
    from multigpubeamsearch.errors import NativeBackendError
    import os
    workers=ResidentWorkers([sys.executable,'-c','import time;time.sleep(60)'],world=1,
        directory=tmp_path/'session',cwd=tmp_path,env=os.environ)
    with pytest.raises(NativeBackendError,match='timed out'):
        workers.submit([0],tmp_path,timeout=.05)
    assert all(p.poll() is not None for p in workers.processes)
