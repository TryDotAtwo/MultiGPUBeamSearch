"""Exact owned-lease lifecycle under Ivan's combined USD20 authority."""
import json
import sys
import time
from pathlib import Path
import requests
sys.path.insert(0, r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key

BASE = 'https://console.vast.ai/api/v0/'
HEADERS = {'Authorization': 'Bearer ' + load_key()}
ROOT = Path('test_results/audit_fixes_2026-09-28')
action, world = sys.argv[1], int(sys.argv[2])
if world not in (2, 8): raise SystemExit('Only explicitly requested 2/8 RTX3060')
receipt = ROOT / f'COMPLETE_AUDIT_LEASE_{world}_20261005.json'
label = f'cube4-complete-audit-{world}x3060-20usd-20261005'

def call(method, path, **kwargs):
    response = requests.request(method, BASE + path, headers=HEADERS,
                                timeout=25, allow_redirects=False, **kwargs)
    response.raise_for_status()
    return response.json()

def save(record):
    receipt.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')

if action == 'create':
    if receipt.exists(): raise SystemExit('Existing attempt; reconcile, never duplicate')
    offers = json.loads((ROOT / 'COMPLETE_AUDIT_OFFERS_20261005.json').read_text())
    chosen = offers[str(world)][0]
    if chosen['num_gpus'] != world or chosen['gpu_name'] != 'RTX 3060':
        raise SystemExit('Wrong hardware')
    if chosen['dph_total'] > (0.25 if world == 2 else 1.0):
        raise SystemExit('Rate above bounded window ceiling')
    record = dict(label=label, offer_id=chosen['id'], world=world, budget_combined_usd=20,
        authority='Ivan: доделай всё полностью; Vast 2x3060/8x3060; USD20 суммарно',
        hourly_rate=chosen['dph_total'], disk_gb=50, deadline_epoch=int(time.time()) + 14400,
        archive_reserve_epoch=int(time.time()) + 12600, state='attempt_started')
    save(record)
    result = call('PUT', f"asks/{chosen['id']}/", json=dict(client_id='me',
        image='vastai/pytorch:cuda-12.8.1-auto', disk=50, runtype='ssh_direc ssh_proxy',
        label=label, onstart='true', cancel_unavail=True))
    record.update({key: result.get(key) for key in ('success', 'new_contract', 'error', 'msg')})
    record['state'] = 'created' if result.get('success') is True else 'unconfirmed'
    save(record); print(json.dumps(record)); raise SystemExit

record = json.loads(receipt.read_text())
owned = record.get('new_contract')
if record.get('success') is not True or type(owned) is not int: raise SystemExit('Unconfirmed lease')
path = f'instances/{owned}/'

def status():
    value = call('GET', path).get('instances')
    if isinstance(value, list): value = value[0] if value else None
    if value and (value.get('id') != owned or value.get('label') != label):
        raise SystemExit('Ownership mismatch')
    return value

if action == 'watch':
    while time.time() < record['deadline_epoch']:
        if not status(): raise SystemExit
        time.sleep(min(30, max(1, record['deadline_epoch'] - time.time())))
    for attempt in range(10):
        try:
            if not status(): raise SystemExit
            if call('PUT', path, json={'state': 'stopped'}).get('success') is True:
                print('OWNED_DEADLINE_STOPPED', flush=True); raise SystemExit
        except requests.RequestException: pass
        time.sleep(15)
    raise SystemExit('Owned deadline stop failed')

value = status()
if action == 'status':
    keys = ('id', 'label', 'actual_status', 'ssh_host', 'ssh_port', 'dph_total',
            'gpu_name', 'num_gpus', 'cpu_ram', 'disk_space')
    print(json.dumps({key: value.get(key) for key in keys} if value else {'absent': True}))
    raise SystemExit
if not value: raise SystemExit('Owned lease absent')
if action == 'attach':
    public = (Path.home() / '.ssh/id_rsa.pub').read_text(encoding='utf-8').strip()
    if not public.startswith('ssh-rsa ') or 'PRIVATE KEY' in public: raise SystemExit('Bad public key')
    result = call('POST', path + 'ssh/', json={'ssh_key': public})
elif action == 'stop': result = call('PUT', path, json={'state': 'stopped'})
elif action == 'destroy':
    if record.get('archives_verified') is not True: raise SystemExit('Archive before destroy')
    result = call('DELETE', path)
else: raise SystemExit('Unknown action')
print(json.dumps({key: result.get(key) for key in ('success', 'error', 'msg')}))
