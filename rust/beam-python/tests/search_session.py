"""Native result validation and atomic cancellation through the thin adapter."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
spec=importlib.util.spec_from_file_location('_beam_runtime',sys.argv[1])
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
assert hasattr(runtime,'SearchSession'), 'Rust search session adapter is missing'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);worker=root/'worker.py'
    worker.write_text('''import sys,os,json,time
from pathlib import Path
if '--resident-protocol-info' in sys.argv:
 print('{"protocol":"resident-search-v1"}',flush=True);raise SystemExit(0)
directory=Path(os.environ['BEAM_SESSION_DIR'])
while not (directory/'stop').exists():time.sleep(.002)
''')
    identity=dict(graph=dict(center=[0,1],generators=[[1,0]]),graph_hash='a'*64,
        model_artifact_hash='b'*64,model_backend='mlp',runtime_build=dict(binary_sha256='c'*64),
        devices=[0],beam_width=64,max_steps=1,options={},cuda_visible_devices=None,cuda_device_order=None)
    command=[sys.executable,str(worker)]
    launch=dict(command=command,protocol_command=command+['--resident-protocol-info'],environment=dict(os.environ),cwd=str(root))
    owner=runtime.SearchSession();errors=[]
    def search():
        try:owner.run(json.dumps(identity),json.dumps(launch),[1,0],root,30.,'{}')
        except RuntimeError as error:errors.append(str(error))
    task=threading.Thread(target=search);task.start();time.sleep(.1)
    began=time.monotonic();owner.cancel();task.join(timeout=3.)
    assert not task.is_alive() and time.monotonic()-began<3
    assert any('cancelled' in message for message in errors)
    owner.close();assert owner.closed and owner.pids==[]
    owner=runtime.SearchSession()
    missing=root/'must-not-be-created'
    solved=json.loads(owner.run(json.dumps(identity),json.dumps(launch),[0,1],missing,1.,'{}'))
    assert solved['path']==[] and solved['metadata']['already_solved'] and not missing.exists()
    identity['max_steps']=0
    empty=json.loads(owner.run(json.dumps(identity),json.dumps(launch),[1,0],missing,1.,'{}'))
    assert empty['path'] is None and empty['metadata']['budget_exhausted'] and not missing.exists()
    owner.close()
print('PYO3_SEARCH_SESSION_CANCEL_TRIVIAL_OK')
