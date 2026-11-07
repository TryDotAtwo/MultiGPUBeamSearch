"""Single authorized replacement; an uncertain outcome must not be retried."""
import json
import sys
from pathlib import Path
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
receipt=Path('test_results/audit_fixes_2026-09-28/replacement_20usd_receipt.json')
if receipt.exists():raise SystemExit('Existing attempt receipt: reconcile, never repeat creation')
receipt.write_text(json.dumps({'offer':50734715,'status':'attempt_started','budget_usd':20}),encoding='utf-8')
response=requests.put('https://console.vast.ai/api/v0/asks/50734715/',
    headers={'Authorization':'Bearer '+load_key()},
    json={'client_id':'me','image':'vastai/pytorch:cuda-12.8.1-auto','disk':40,
        'runtype':'ssh','label':'cube4-audit-owned-20usd-20261004','cancel_unavail':True},
    timeout=30,allow_redirects=False)
response.raise_for_status()
result=response.json()
safe={k:result.get(k) for k in ('success','new_contract','error','msg')}
receipt.write_text(json.dumps(safe),encoding='utf-8')
print(json.dumps(safe))
