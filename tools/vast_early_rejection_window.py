"""One authorized one-hour remote window; cloud orchestration only."""
import json
import sys
import time
from pathlib import Path
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
receipt=Path('test_results/audit_fixes_2026-09-28/early_rejection_window_20261004.json')
base='https://console.vast.ai/api/v0/'
label='cube4-early-rejection-owned-20261004'
action=sys.argv[1] if len(sys.argv)>1 else 'status'
if action not in ('create','status','watch','attach','stop','destroy'):
    raise SystemExit('unsupported scoped action')
headers={'Authorization':'Bearer '+load_key()}
if action=='create':
    if receipt.exists():raise SystemExit('Existing attempt receipt: reconcile, never repeat')
    record=dict(offer=50492808,label=label,deadline=int(time.time())+3600,
        hourly_rate=.15022222222222223,total_existing_ceiling_usd=20,status='attempt_started')
    receipt.write_text(json.dumps(record),encoding='utf-8')
    r=requests.put(base+'asks/50492808/',headers=headers,json=dict(client_id='me',
        image='vastai/pytorch:cuda-12.8.1-auto',disk=40,runtype='ssh_direc ssh_proxy',
        env={},onstart='true',label=label,cancel_unavail=True),timeout=30,allow_redirects=False)
    r.raise_for_status();result=r.json()
    record.update({k:result.get(k) for k in ('success','new_contract','error','msg')})
    receipt.write_text(json.dumps(record),encoding='utf-8');print(json.dumps(record));raise SystemExit(0)
record=json.loads(receipt.read_text(encoding='utf-8'));owned=record.get('new_contract')
if record.get('success') is not True or type(owned) is not int or record.get('label')!=label:
    raise SystemExit('missing successful owned instance receipt')
url=base+f'instances/{owned}/'
if action=='watch':
    while time.time()<record['deadline']:time.sleep(min(15,max(1,record['deadline']-time.time())))
    for attempt in range(10):
        try:
            r=requests.get(url,headers=headers,timeout=20,allow_redirects=False);r.raise_for_status()
            obj=r.json().get('instances')
            if obj is None:raise SystemExit(0)
            if isinstance(obj,list):obj=obj[0]
            if obj.get('id')!=owned or obj.get('label')!=label:raise SystemExit('wrong ownership')
            r=requests.put(url,headers=headers,json={'state':'stopped'},timeout=20,allow_redirects=False)
            r.raise_for_status()
            if r.json().get('success') is True:raise SystemExit(0)
        except requests.RequestException:pass
        time.sleep(15)
    raise SystemExit('budget stop unsuccessful')
r=requests.get(url,headers=headers,timeout=20,allow_redirects=False);r.raise_for_status()
obj=r.json().get('instances')
if obj is None and action=='status':print(json.dumps({'id':owned,'instance_record_absent':True}));raise SystemExit(0)
if isinstance(obj,list):obj=obj[0]
if not isinstance(obj,dict) or obj.get('id')!=owned or obj.get('label')!=label:
    raise SystemExit('wrong ownership')
if action in ('stop','destroy','attach'):
    if action=='stop':r=requests.put(url,headers=headers,json={'state':'stopped'},timeout=20)
    elif action=='destroy':r=requests.delete(url,headers=headers,timeout=20)
    else:
        key=(Path.home()/'.ssh/id_rsa.pub').read_text(encoding='utf-8').strip()
        if not key.startswith('ssh-rsa ') or 'PRIVATE KEY' in key:raise SystemExit('wrong public key')
        r=requests.post(url+'ssh/',headers=headers,json={'ssh_key':key},timeout=20)
    r.raise_for_status();result=r.json()
    print(json.dumps({k:result.get(k) for k in ('success','msg','error')}));raise SystemExit(0)
print(json.dumps({k:obj.get(k) for k in ('id','actual_status','cur_state','intended_status',
    'ssh_host','ssh_port','ports','dph_total','num_gpus','gpu_name')}))
