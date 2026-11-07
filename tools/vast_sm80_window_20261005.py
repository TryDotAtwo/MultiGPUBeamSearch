"""One owned 2x3060 window under new explicit10USD authority."""
import json
import sys
import time
from pathlib import Path
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
headers={'Authorization':'Bearer '+load_key()}
base='https://console.vast.ai/api/v0/'
receipt=Path('test_results/audit_fixes_2026-09-28/sm80_window_20261005.json')
label='cube4-sm80-owned-new10usd-20261005'
def call(method,path,**kwargs):
    r=requests.request(method,base+path,headers=headers,timeout=30,allow_redirects=False,**kwargs)
    r.raise_for_status()
    return r.json()
action=sys.argv[1]
if action=='create':
    if receipt.exists():raise SystemExit('Existing attempt: reconcile, never create twice')
    if call('GET','instances/').get('instances'):raise SystemExit('Reconcile existing instances first')
    record=dict(offer=19911145,label=label,budget_usd=10,deadline=int(time.time())+7200,
                hourly_rate=.1428148148148148,status='attempt_started')
    receipt.write_text(json.dumps(record),encoding='utf-8')
    result=call('PUT','asks/19911145/',json=dict(client_id='me',image='vastai/pytorch:cuda-12.8.1-auto',
        disk=40,runtype='ssh_direc ssh_proxy',label=label,onstart='true',cancel_unavail=True))
    record.update({k:result.get(k) for k in ('success','new_contract','error','msg')})
    receipt.write_text(json.dumps(record),encoding='utf-8')
    print(json.dumps(record));raise SystemExit
record=json.loads(receipt.read_text(encoding='utf-8'))
owned=record.get('new_contract')
if record.get('success') is not True or type(owned) is not int:raise SystemExit('Unconfirmed owned lease')
path=f'instances/{owned}/'
def status():
    obj=call('GET',path).get('instances')
    if isinstance(obj,list):obj=obj[0] if obj else None
    if obj is not None and (obj.get('id')!=owned or obj.get('label')!=label):raise SystemExit('Ownership mismatch')
    return obj
if action=='watch':
    while time.time()<record['deadline']:time.sleep(min(15,max(1,record['deadline']-time.time())))
    for _ in range(10):
        try:
            if status() is None:raise SystemExit
            if call('PUT',path,json={'state':'stopped'}).get('success') is True:raise SystemExit
        except requests.RequestException:pass
        time.sleep(15)
    raise SystemExit('STOP FAILED')
obj=status()
if action=='status':
    print(json.dumps({k:obj.get(k) for k in ('id','label','actual_status','cur_state','intended_status',
        'ssh_host','ssh_port','dph_total','num_gpus','gpu_name')} if obj else {'absent':True}));raise SystemExit
if obj is None:raise SystemExit('Absent')
if action=='attach':
    key=(Path.home()/'.ssh/id_rsa.pub').read_text(encoding='utf-8').strip()
    if not key.startswith('ssh-rsa ') or 'PRIVATE KEY' in key:raise SystemExit('Invalid public key')
    result=call('POST',path+'ssh/',json={'ssh_key':key})
elif action=='stop':result=call('PUT',path,json={'state':'stopped'})
else:raise SystemExit('Unsupported action')
print(json.dumps({k:result.get(k) for k in ('success','msg','error')}))
