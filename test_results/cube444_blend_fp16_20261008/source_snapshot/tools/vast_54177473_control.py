"""Scoped replacement lifecycle; orchestration only, no local project compute."""
import json
import sys
import time
from pathlib import Path
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
url='https://console.vast.ai/api/v0/instances/54177473/'
action=sys.argv[1] if len(sys.argv)>1 else 'status'
if action not in ('status','watch','attach','stop','destroy'):raise SystemExit('unsupported scoped action')
if action=='watch':
    deadline=1791140135
    while time.time()<deadline:time.sleep(min(30,max(1,deadline-time.time())))
h={'Authorization':'Bearer '+load_key()}
r=requests.get(url,headers=h,timeout=20,allow_redirects=False);r.raise_for_status()
x=r.json();x=x.get('instances',x);x=x[0] if isinstance(x,list) else x
if x is None and action in ('status','watch'):
    print(json.dumps({'id':54177473,'instance_record_absent':True}));raise SystemExit(0)
if not isinstance(x,dict):raise SystemExit('unexpected instance response')
if x.get('id')!=54177473:raise SystemExit('wrong identity')
if action in ('stop','destroy'):
    if x.get('label')!='cube4-audit-owned-proxy-20usd-20261004':
        raise SystemExit('ownership label mismatch')
    if action=='stop':
        r=requests.put(url,headers=h,json={'state':'stopped'},timeout=20,allow_redirects=False)
    else:
        r=requests.delete(url,headers=h,timeout=20,allow_redirects=False)
    r.raise_for_status();j=r.json()
    print(json.dumps({k:j.get(k) for k in ('success','msg','error')}));raise SystemExit(0)
if action=='watch':
    for attempt in range(5):
        try:
            r=requests.put(url,headers=h,json={'state':'stopped'},timeout=20,allow_redirects=False)
            r.raise_for_status()
            if r.json().get('success') is True:raise SystemExit(0)
        except requests.RequestException:pass
        time.sleep(15)
    raise SystemExit(1)
if action=='attach':
    key=(Path.home()/'.ssh/id_rsa.pub').read_text(encoding='utf-8').strip()
    if not key.startswith('ssh-rsa ') or 'PRIVATE KEY' in key:raise SystemExit('wrong public key')
    r=requests.post(url+'ssh/',headers=h,json={'ssh_key':key},timeout=20,allow_redirects=False)
    r.raise_for_status();j=r.json();print(json.dumps({k:j.get(k) for k in ('success','msg','error')}))
print(json.dumps({k:x.get(k) for k in ('id','actual_status','cur_state','intended_status','ssh_host','ssh_port','public_ipaddr','ports','dph_total','num_gpus','gpu_name')}))
