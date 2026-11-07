"""Run against the exact built cdylib, without installing a wheel or CUDA."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

spec=importlib.util.spec_from_file_location('_beam_runtime',sys.argv[1])
runtime=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    script=root/'worker.py'
    script.write_text('''import os,json,time,pathlib
d=pathlib.Path(os.environ['BEAM_SESSION_DIR']);rank=int(os.environ['RANK']);seq=0
while not (d/'stop').exists():
 p=d/f'request-{seq}.json'
 if not p.exists():time.sleep(.005);continue
 row=json.loads(p.read_text());out=pathlib.Path(row['run_dir'])
 (out/f'resident-rank-{rank}.log').write_text(str(row['start']))
 p=d/f'done-{seq}-{rank}.json';t=p.with_suffix('.tmp')
 t.write_text(json.dumps(dict(sequence=seq,rank=rank)));t.replace(p);seq+=1
''')
    group=runtime.ResidentWorkers([sys.executable,str(script)],2,root/'session',root,dict(os.environ))
    try:
        pids=group.pids
        assert len(pids)==2 and pids[0]!=pids[1]
        for sequence in range(2):
            run=root/str(sequence);run.mkdir()
            assert group.submit([sequence],run,5.0)>=0
            assert (run/'native.log').read_text().count(str([sequence]))==2
            assert group.sequence==sequence+1 and group.pids==pids
        try:group.submit([1],root,float('nan'))
        except ValueError:pass
        else:raise AssertionError('NaN timeout accepted')
    finally:group.close()
    assert group.closed and all(code is not None for code in group.exit_codes())
print('PYO3_RESIDENT_OK')

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    worker=root/'probe.py'
    worker.write_text('import sys,json\nprint(json.dumps({"ready": True}), flush=True)\nfor line in sys.stdin: print(line.strip(),flush=True)\n')
    probe=runtime.ProbeSession([sys.executable,str(worker)],root,dict(os.environ),root/'probe.log',5.)
    try:
        import json
        assert json.loads(probe.receive(2.))=={'ready': True}
        pid=probe.pid
        for value in (3,1,8):
            probe.send_request(json.dumps({'batch':value}))
            assert json.loads(probe.receive(2.))=={'batch':value}
            assert probe.pid==pid
    finally:probe.close()
    assert probe.exit_code() is not None
    command=[sys.executable,str(worker)]
    cohort=runtime.ProbeCohort([command,command],[dict(os.environ)]*2,root,root/'cohort',5.)
    try:
        assert json.loads(cohort.collect(2.))==[{'ready':True}]*2
        cohort.send_requests('[{"batch":1},{"batch":2}]')
        assert json.loads(cohort.collect(2.))==[{'batch':1},{'batch':2}]
    finally:cohort.close()
    assert all(code is not None for code in cohort.exit_codes())
print('PYO3_PROBE_OK')

with tempfile.TemporaryDirectory() as directory:
    import threading,time
    root=Path(directory)
    probe=runtime.ProbeSession([sys.executable,'-c','import time;time.sleep(30)'],root,dict(os.environ),root/'cancel.log',60.)
    failures=[]
    def wait():
        try:probe.receive(30.)
        except RuntimeError as error:failures.append(str(error))
    worker=threading.Thread(target=wait);worker.start();time.sleep(.05)
    began=time.monotonic();probe.cancel();worker.join(timeout=3.)
    assert not worker.is_alive() and time.monotonic()-began<3 and any('cancelled' in text for text in failures)
    probe.close();assert probe.exit_code() is not None
rows=[dict(profile='baseline',parents=64,seconds_by_rank=[.01,.02],correctness_passed=True,memory_admitted=True,throttled=False)]*5
rows+= [dict(row,profile='fast',seconds_by_rank=[.01,.015]) for row in rows]
selection=json.loads(runtime.select_profile(json.dumps(rows),'baseline',.05))
assert selection['estimate']['profile']=='fast'
assert not json.loads(runtime.promote_optimization('[]'))['accepted']
print('PYO3_PLANNING_STATS_CANCEL_OK')
