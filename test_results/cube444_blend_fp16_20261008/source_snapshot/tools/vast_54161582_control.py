"""Scoped rental orchestration only; never emits credentials."""
import json
import sys
from pathlib import Path
from urllib.parse import urlparse
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key

url='https://console.vast.ai/api/v0/instances/54161582/'
headers={'Authorization':'Bearer '+load_key()}
action=sys.argv[1] if len(sys.argv)>1 else 'status'
if action not in ('status','start','stop','logs'):raise SystemExit('unsupported scoped action')
if action=='logs':
    response=requests.put('https://console.vast.ai/api/v0/instances/request_logs/54161582/',
        headers=headers,json={'tail':'150','filter':'budget|stopguard|stopping|supervisor'},timeout=20,allow_redirects=False)
    response.raise_for_status()
    result=response.json();target=result.get('result_url')
    if not isinstance(target,str) or urlparse(target).scheme!='https':raise SystemExit('missing safe log result URL')
    response=requests.get(target,timeout=20,allow_redirects=False)
    print(json.dumps({'action':'logs','http_status':response.status_code}))
    if response.status_code==200:
        path=Path('test_results/audit_fixes_2026-09-28/vast54161582_restart_filtered.log')
        path.write_text(response.text,encoding='utf-8')
        print(json.dumps({'saved':str(path),'bytes':len(response.content)}))
    raise SystemExit(0)
if action!='status':
    response=requests.put(url,headers=headers,json={'state':'running' if action=='start' else 'stopped'},timeout=20,allow_redirects=False)
    print(json.dumps({'action':action,'http_status':response.status_code}))
    response.raise_for_status()
response=requests.get(url,headers=headers,timeout=20,allow_redirects=False)
response.raise_for_status()
payload=response.json();item=payload.get('instances',payload)
if isinstance(item,list):item=item[0]
if item.get('id')!=54161582:raise SystemExit('unexpected instance identity')
print(json.dumps({k:item.get(k) for k in ('id','actual_status','cur_state','intended_status','num_gpus','gpu_name','dph_total','storage_cost','ssh_host','ssh_port')},ensure_ascii=False))
