"""Scoped API acceptance lifecycle; local code only orchestrates Vast requests."""
import json
import sys
import time
from pathlib import Path
import requests
sys.path.insert(0, r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key

ROOT = Path('test_results/cayleypy_api_20261006')
ROOT.mkdir(parents=True, exist_ok=True)
BASE = 'https://console.vast.ai/api/v0/'
HEADERS = {'Authorization': 'Bearer ' + load_key()}
action = sys.argv[1]
world = int(sys.argv[2]) if len(sys.argv) > 2 else 2
if world not in (2, 8): raise SystemExit('Authorized RTX3060 configurations only')
receipt = ROOT / f'lease_{world}.json'
label = f'cayleypy-api-{world}x3060-20usd-20261006'

def call(method, path, **kwargs):
    response = requests.request(method, BASE + path, headers=HEADERS, timeout=25,
                                allow_redirects=False, **kwargs)
    response.raise_for_status()
    return response.json()

def save(row):
    receipt.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')

if action == 'offers':
    query = {'verified': {'eq': True}, 'rentable': {'eq': True}, 'rented': {'eq': False},
             'gpu_name': {'eq': 'RTX 3060'}, 'num_gpus': {'eq': world},
             'cpu_ram': {'gte': 32000}, 'disk_space': {'gte': 50},
             'cuda_max_good': {'gte': 12.8}, 'type': 'on-demand', 'limit': 6,
             'order': [['dph_total', 'asc']], 'allocated_storage': 50}
    keys = ('id', 'gpu_name', 'num_gpus', 'dph_total', 'inet_up_cost', 'inet_down_cost', 'cpu_ram')
    rows = [{k: row.get(k) for k in keys} for row in call('POST', 'bundles/', json=query).get('offers', [])]
    (ROOT / f'offers_{world}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows)); raise SystemExit
if action == 'create':
    if receipt.exists(): raise SystemExit('Reconcile existing receipt before any retry')
    chosen = json.loads((ROOT / f'offers_{world}.json').read_text())[0]
    if chosen['gpu_name'] != 'RTX 3060' or chosen['num_gpus'] != world: raise SystemExit('Hardware mismatch')
    old = Path('test_results/audit_fixes_2026-09-28')
    previous = [json.loads(p.read_text(encoding='utf-8-sig')) for p in old.glob('COMPLETE_AUDIT_LEASE_*_20261005.json')]
    prior_ceiling = sum(float(row['hourly_rate']) * 4 for row in previous)
    current = [json.loads(p.read_text()) for p in ROOT.glob('lease_*.json')]
    ceiling = prior_ceiling + sum(float(row['hourly_rate']) * 4 for row in current) + chosen['dph_total'] * 4
    if ceiling > 16 or chosen['dph_total'] > (.6 if world == 2 else 1): raise SystemExit('Combined budget ceiling exceeded')
    if max(chosen['inet_up_cost'], chosen['inet_down_cost']) > .1: raise SystemExit('Transfer rate above reserve')
    row = dict(label=label, world=world, hourly_rate=chosen['dph_total'], deadline_epoch=int(time.time())+14400,
               prior_four_hour_ceiling=prior_ceiling, combined_four_hour_ceiling=ceiling,
               state='attempt_started', archives_verified=False)
    save(row)
    result = call('PUT', f"asks/{chosen['id']}/", json={'client_id':'me', 'image':'vastai/pytorch:cuda-12.8.1-auto',
                   'disk':50, 'runtype':'ssh_direc ssh_proxy', 'label':label, 'onstart':'true', 'cancel_unavail':True})
    row.update({k:result.get(k) for k in ('success','new_contract','msg','error')}); save(row)
    print(json.dumps(row)); raise SystemExit
row = json.loads(receipt.read_text())
owned = row.get('new_contract')
if row.get('success') is not True or type(owned) is not int: raise SystemExit('Unconfirmed lease')
path = f'instances/{owned}/'
if action == 'watch':
    while time.time() < row['deadline_epoch']:
        time.sleep(min(30, max(1, row['deadline_epoch'] - time.time())))
    for attempt in range(8):
        try:
            current = call('GET', path).get('instances')
            if isinstance(current, list): current = current[0] if current else None
            if not current: raise SystemExit
            if current.get('id') != owned or current.get('label') != label: raise SystemExit('Ownership mismatch')
            if call('PUT', path, json={'state':'stopped'}).get('success') is True: raise SystemExit
        except requests.RequestException:
            pass
        time.sleep(15)
    raise SystemExit('Deadline stop failed')
value = call('GET', path).get('instances')
if isinstance(value, list): value = value[0] if value else None
if value and (value.get('id') != owned or value.get('label') != label): raise SystemExit('Ownership mismatch')
if action == 'status':
    keys = ('id','actual_status','ssh_host','ssh_port','gpu_name','num_gpus','dph_total')
    print(json.dumps({k:value.get(k) for k in keys} if value else {'absent':True})); raise SystemExit
if not value: raise SystemExit('Owned lease absent')
if action == 'attach':
    public = (Path.home()/'.ssh/id_rsa.pub').read_text(encoding='utf-8').strip()
    result = call('POST', path+'ssh/', json={'ssh_key':public})
elif action == 'stop': result = call('PUT', path, json={'state':'stopped'})
elif action == 'destroy':
    if row.get('archives_verified') is not True: raise SystemExit('Save/verify archive first')
    result = call('DELETE', path)
else: raise SystemExit('Unsupported action')
print(json.dumps({k:result.get(k) for k in ('success','error','msg')}))
