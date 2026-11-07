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
