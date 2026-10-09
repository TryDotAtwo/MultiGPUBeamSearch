"""Scoped cloud orchestration under continuing task-wide $20 authority."""
import json,sys,time
from pathlib import Path
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
base='https://console.vast.ai/api/v0/'
headers={'Authorization':'Bearer '+load_key()}
stage=sys.argv[2] if len(sys.argv)>2 else 'ab'
if stage not in ('ab','pipeline','release'):raise SystemExit('Unknown scoped stage')
receipt=Path(f'test_results/audit_fixes_2026-09-28/{stage}_window_20261004.json')
label=f'cube4-{stage}-owned-20261004'
action=sys.argv[1]
def request(method,path,**kwargs):
    r=requests.request(method,base+path,headers=headers,timeout=30,allow_redirects=False,**kwargs)
    r.raise_for_status();return r.json()
if action=='inventory':
    value=request('GET','instances/').get('instances',[])
    print(json.dumps([{k:v.get(k) for k in ('id','label','actual_status','dph_total','num_gpus','gpu_name')} for v in value]));raise SystemExit
if action=='create':
    if receipt.exists():raise SystemExit('Reconcile existing receipt; never retry blind')
    record=dict(offer=45954466,label=label,deadline=int(time.time())+7200,
        hourly_rate=.11777777777777779,total_existing_ceiling_usd=20,
        window_compute_storage_upper_usd=.24,status='attempt_started')
    receipt.write_text(json.dumps(record),encoding='utf-8')
    result=request('PUT','asks/45954466/',json=dict(client_id='me',image='vastai/pytorch:cuda-12.8.1-auto',
        disk=40,runtype='ssh_direc ssh_proxy',env={},onstart='true',label=label,cancel_unavail=True))
    record.update({k:result.get(k) for k in ('success','new_contract','error','msg')})
    receipt.write_text(json.dumps(record),encoding='utf-8');print(json.dumps(record));raise SystemExit
record=json.loads(receipt.read_text(encoding='utf-8'));owned=record.get('new_contract')
if record.get('success') is not True or type(owned) is not int:raise SystemExit('No confirmed owned lease')
path=f'instances/{owned}/'
def status():
    obj=request('GET',path).get('instances')
    if isinstance(obj,list):obj=obj[0] if obj else None
    if obj is not None and (obj.get('id')!=owned or obj.get('label')!=label):raise SystemExit('Ownership mismatch')
    return obj
if action=='watch':
    while time.time()<record['deadline']:time.sleep(min(15,max(1,record['deadline']-time.time())))
    for _ in range(10):
        try:
            if status() is None:raise SystemExit
            if request('PUT',path,json={'state':'stopped'}).get('success') is True:raise SystemExit
        except requests.RequestException:pass
        time.sleep(15)
    raise SystemExit('Stop failed')
obj=status()
if action=='status':
    observed={'id':owned,'instance_record_absent':True} if obj is None else {
        k:obj.get(k) for k in ('id','label','actual_status','cur_state','intended_status',
                             'ssh_host','ssh_port','ports','dph_total','num_gpus','gpu_name')}
    record['last_status_observation']={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                      'instance':observed}
    receipt.write_text(json.dumps(record),encoding='utf-8')
    print(json.dumps(observed));raise SystemExit
if obj is None:raise SystemExit('Already absent')
if action=='stop':result=request('PUT',path,json={'state':'stopped'})
elif action=='destroy':result=request('DELETE',path)
elif action=='attach':
    key=(Path.home()/'.ssh/id_rsa.pub').read_text(encoding='utf-8').strip()
    if not key.startswith('ssh-rsa ') or 'PRIVATE KEY' in key:raise SystemExit('Invalid public key')
    result=request('POST',path+'ssh/',json={'ssh_key':key})
else:raise SystemExit('Unsupported action')
safe_result={k:result.get(k) for k in ('success','msg','error')}
record.setdefault('lifecycle',[]).append(dict(action=action,
    time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),result=safe_result))
receipt.write_text(json.dumps(record),encoding='utf-8')
print(json.dumps(safe_result))
