"""Attach existing user-approved public key only to owned54176266."""
import sys
from pathlib import Path
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
key=(Path.home()/'.ssh/id_rsa.pub').read_text(encoding='utf-8').strip()
if not key.startswith('ssh-rsa ') or 'PRIVATE KEY' in key:raise SystemExit('invalid public key')
r=requests.post('https://console.vast.ai/api/v0/instances/54176266/ssh/',
    headers={'Authorization':'Bearer '+load_key()},json={'ssh_key':key},timeout=20,allow_redirects=False)
r.raise_for_status();result=r.json()
print({'instance':54176266,'http_status':r.status_code,**{k:result.get(k) for k in ('success','msg','error')}})
