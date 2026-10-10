import subprocess,sys
from pathlib import Path
code=r'''
import json,subprocess,sys,os
from pathlib import Path
out=Path('/root/bias-result')
env=dict(os.environ,PATH='/venv/main/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''))
try:
 for variant in ('baseline','candidate'):
  subprocess.run([sys.executable,'/root/bias-projection.py',variant],env=env,check=True)
 command=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','3',sys.executable,'/root/bias-projection.py','candidate']
 with (out/'sanitizer.log').open('w') as f:
  result=subprocess.run(command,env=dict(env,BIAS_SANITIZER='1'),stdout=f,stderr=subprocess.STDOUT)
 (out/'sanitizer-status.json').write_text(json.dumps({'returncode':result.returncode}))
 if result.returncode:raise RuntimeError('memcheck failed')
 (out/'checks-exit.json').write_text(json.dumps({'returncode':0}))
except Exception as error:
 (out/'checks-exit.json').write_text(json.dumps({'returncode':1,'error':repr(error)}));raise
'''
Path('/root/bias-checks.py').write_text(code)
with Path('/root/bias-checks.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'/root/bias-checks.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print('Checks',p.pid)
