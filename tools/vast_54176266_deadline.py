"""Paid rental deadline supervisor only; no local project computation."""
import time
import sys
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
deadline=1791140135  # 2026-10-04 18:55:35 UTC, fixed two-hour limit
while time.time()<deadline:time.sleep(min(30,max(1,deadline-time.time())))
url='https://console.vast.ai/api/v0/instances/54176266/'
for attempt in range(5):
    try:
        h={'Authorization':'Bearer '+load_key()}
        r=requests.get(url,headers=h,timeout=20,allow_redirects=False);r.raise_for_status()
        x=r.json();x=x.get('instances',x);x=x[0] if isinstance(x,list) else x
        if x.get('id')!=54176266:raise RuntimeError('wrong scoped rental')
        r=requests.put(url,headers=h,json={'state':'stopped'},timeout=20,allow_redirects=False)
        r.raise_for_status()
        if r.json().get('success') is True:raise SystemExit(0)
    except requests.RequestException:pass
    time.sleep(15)
raise SystemExit(1)
