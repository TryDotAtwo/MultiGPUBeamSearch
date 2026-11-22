import subprocess,sys
from pathlib import Path
code=r'''
import json,time,subprocess,sys,torch
from pathlib import Path
out=Path('/root/further-result')
while not (out/'gat-component-softmax.json').exists():time.sleep(2)
try:
 for rid,order in [('3',('softmax','tile')),('4',('tile','softmax'))]:
  for variant in order:subprocess.run([sys.executable,'/root/further-paired.py',variant,rid],check=True)
 softmax=torch.load('/root/further-output-softmax.pt',weights_only=True)
 tile=torch.load('/root/further-output-tile.pt',weights_only=True)
 torch.testing.assert_close(tile,softmax,rtol=.03,atol=.02)
 (out/'tile-parity.json').write_text(json.dumps({'max_error':(tile-softmax).abs().max().item(),'bitwise_equal':torch.equal(tile,softmax)}))
 (out/'tile-paired-exit.json').write_text(json.dumps({'returncode':0}))
except Exception as e:
 (out/'tile-paired-exit.json').write_text(json.dumps({'returncode':1,'error':repr(e)}));raise
'''
Path('/root/further-tile-pair-job.py').write_text(code)
with Path('/root/further-tile-pair.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'/root/further-tile-pair-job.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(p.pid)
