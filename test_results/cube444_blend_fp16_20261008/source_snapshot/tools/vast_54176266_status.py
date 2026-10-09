"""Read-only status of this task's authorized replacement."""
import json
import sys
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
r=requests.get('https://console.vast.ai/api/v0/instances/54176266/',headers={'Authorization':'Bearer '+load_key()},timeout=20,allow_redirects=False)
r.raise_for_status();x=r.json();x=x.get('instances',x)
if isinstance(x,list):x=x[0]
if x.get('id')!=54176266:raise SystemExit('wrong instance')
if len(sys.argv)>1 and sys.argv[1]=='delete-empty':
    if x.get('label')!='cube4-audit-owned-20usd-20261004':raise SystemExit('wrong ownership label')
    r=requests.delete('https://console.vast.ai/api/v0/instances/54176266/',
        headers={'Authorization':'Bearer '+load_key()},timeout=20,allow_redirects=False)
    r.raise_for_status();print(json.dumps({'action':'delete-empty','success':r.json().get('success')}))
    raise SystemExit(0)
if len(sys.argv)>1 and sys.argv[1]=='reboot':
    r=requests.put('https://console.vast.ai/api/v0/instances/reboot/54176266/',
        headers={'Authorization':'Bearer '+load_key()},json={},timeout=20,allow_redirects=False)
    r.raise_for_status();print(json.dumps({'action':'reboot','success':r.json().get('success')}))
if len(sys.argv)>1 and sys.argv[1]=='repair-runtime':
    r=requests.put('https://console.vast.ai/api/v0/instances/54176266/',
        headers={'Authorization':'Bearer '+load_key()},json={'runtype':'ssh_proxy'},timeout=20,allow_redirects=False)
    print(json.dumps({'action':'repair-runtime','http_status':r.status_code}))
    r.raise_for_status();result=r.json()
    print(json.dumps({k:result.get(k) for k in ('success','msg','error')}))
print(json.dumps({k:x.get(k) for k in ('id','actual_status','cur_state','intended_status','runtype','ssh_host','ssh_port','public_ipaddr','ports','dph_total','num_gpus','gpu_name')}))
