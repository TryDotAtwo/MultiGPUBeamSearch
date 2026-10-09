"""Exact owned lease control under Ivan's second explicit USD10 authority."""
import json, sys, time
from pathlib import Path
import requests
sys.path.insert(0, r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
BASE='https://console.vast.ai/api/v0/'
HEADERS={'Authorization':'Bearer '+load_key()}
RECEIPT=Path('test_results/audit_fixes_2026-09-28/SECOND_10USD_WINDOW_20261005.json')
LABEL='cube4-admission-second10usd-20261005'
def call(method,path,**kw):
    response=requests.request(method,BASE+path,headers=HEADERS,timeout=30,allow_redirects=False,**kw)
    response.raise_for_status()
    return response.json()
def save(record):
    RECEIPT.write_text(json.dumps(record,indent=2),encoding='utf-8')
action=sys.argv[1]
if action=='inventory':
    items=call('GET','instances/').get('instances',[])
    print(json.dumps([{k:x.get(k) for k in ('id','label','actual_status','dph_total')} for x in items]));sys.exit()
if action=='create':
    if RECEIPT.exists():raise SystemExit('Attempt already recorded; reconcile, never create twice')
    record=dict(offer_id=45120305,machine_id=143816,label=LABEL,budget_usd=10,
        authority='Ivan: 10 баков даю тести на васте',duration_seconds=14400,
        deadline_epoch=int(time.time())+14400,archive_reserve_epoch=int(time.time())+13200,
        quoted_dph_total=.1808888888888889,disk_gb=40,state='attempt_started')
    save(record)
    result=call('PUT','asks/45120305/',json=dict(client_id='me',image='vastai/pytorch:cuda-12.8.1-auto',
        disk=40,runtype='ssh_direc ssh_proxy',label=LABEL,onstart='true',cancel_unavail=True))
    record.update({k:result.get(k) for k in ('success','new_contract','error','msg')})
    record['state']='created' if result.get('success') is True else 'create_unconfirmed'
    save(record);print(json.dumps(record));sys.exit()
record=json.loads(RECEIPT.read_text(encoding='utf-8'))
owned=record.get('new_contract')
if record.get('success') is not True or type(owned) is not int:raise SystemExit('Unconfirmed lease')
path=f'instances/{owned}/'
def status():
    obj=call('GET',path).get('instances')
    if isinstance(obj,list):obj=obj[0] if obj else None
    if obj is not None and (obj.get('id')!=owned or obj.get('label')!=LABEL):raise SystemExit('Ownership mismatch')
    return obj
if action=='watch':
    while time.time()<record['deadline_epoch']:
        time.sleep(min(15,max(1,record['deadline_epoch']-time.time())))
    for attempt in range(10):
        try:
            obj=status()
            if obj is None:sys.exit()
            if call('PUT',path,json={'state':'stopped'}).get('success') is True:
                print(json.dumps({'owned_instance':owned,'deadline_stop':True}),flush=True);sys.exit()
        except requests.RequestException:
            pass
        time.sleep(15)
    raise SystemExit('DEADLINE STOP FAILED')
obj=status()
if action=='status':
    keys=('id','label','actual_status','cur_state','intended_status','ssh_host','ssh_port','dph_total',
          'num_gpus','gpu_name','gpu_ram','cpu_ram','disk_space','inet_up_cost','inet_down_cost')
    print(json.dumps({k:obj.get(k) for k in keys} if obj else {'absent':True}));sys.exit()
if obj is None:raise SystemExit('Owned lease absent')
if action=='attach':
    public=(Path.home()/'.ssh/id_rsa.pub').read_text(encoding='utf-8').strip()
    if not public.startswith('ssh-rsa ') or 'PRIVATE KEY' in public:raise SystemExit('Invalid public key')
    result=call('POST',path+'ssh/',json={'ssh_key':public})
elif action=='stop':result=call('PUT',path,json={'state':'stopped'})
elif action=='destroy':
    if not record.get('archives_verified'):raise SystemExit('Require verified archives before destruction')
    result=call('DELETE',path)
else:raise SystemExit('Unsupported action')
print(json.dumps({k:result.get(k) for k in ('success','msg','error')}))
