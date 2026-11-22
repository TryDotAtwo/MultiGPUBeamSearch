"""Real PyO3 ownership and concurrent cancellation without CUDA allocations."""
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
assert hasattr(runtime,'InferencePool'), 'Rust inference pool adapter is missing'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    worker=root/'probe.py'
    worker.write_text('''import sys,json,time
rank=int(sys.argv[4]);print(json.dumps(dict(ready=True,device=rank)),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['batch']==16:time.sleep(30)
 print(json.dumps(dict(batch=r['batch'],parents=r['parents'],device=rank,
  correctness_passed=True,numeric_error=0,seconds=[.1]*7,torch_reserved_peak_bytes=0)),flush=True)
''')
    config=dict(command=[sys.executable,str(worker)],inputs=str(root),capacity=32,parents=64,world=2,
        environment=dict(os.environ),cwd=str(root),directory=str(root/'logs'),protocol='json-session-v1')
    owner=runtime.InferencePool(json.dumps(config),60.)
    try:
        first=json.loads(owner.measure(8,64,5.))
        assert len(first['rows'])==2 and not first['failures']
        pids=owner.pids
        again=json.loads(owner.measure(4,64,5.))
        assert len(again['rows'])==2 and owner.pids==pids and owner.starts==2
        errors=[]
        def measure():
            try:owner.measure(16,64,30.)
            except RuntimeError as error:errors.append(str(error))
        task=threading.Thread(target=measure);task.start();time.sleep(.1)
        began=time.monotonic();owner.cancel();task.join(timeout=3.)
        assert not task.is_alive() and time.monotonic()-began<3
        assert any('cancelled' in message for message in errors)
    finally:owner.close()
    assert owner.pids==[None,None]
print('PYO3_INFERENCE_REUSE_CANCEL_OK')
