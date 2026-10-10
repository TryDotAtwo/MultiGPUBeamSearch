import subprocess,sys
from pathlib import Path
code=r'''
import json,subprocess,sys,time
from pathlib import Path
out=Path('/root/bias-result')
while not (out/'checks-exit.json').exists():time.sleep(2)
assert json.loads((out/'checks-exit.json').read_text())['returncode']==0
for round_id,order in [('1',('baseline','candidate')),('2',('candidate','baseline'))]:
 for variant in order:subprocess.run([sys.executable,'/root/bias-paired.py',variant,round_id],check=True)
(out/'paired-exit.json').write_text(json.dumps({'returncode':0}))
'''
Path('/root/bias-paired-job.py').write_text(code)
with Path('/root/bias-paired.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'/root/bias-paired-job.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print('Paired',p.pid)
