"""Explicit authorized retired-lease cleanup; cloud orchestration, not tests."""
import hashlib
import json
from pathlib import Path
import sys
import time
import requests

sys.path.insert(0, r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key

root = Path('test_results/audit_fixes_2026-09-28')
receipt = root / 'retired_cleanup_20261005.json'
if receipt.exists():
    raise SystemExit('Reconcile existing receipt; no blind mutation retry')
pins = {
    'dims_complete_source_binary_20261004.tar.gz': '708debd73117cb8adff9a5f7506c78f74341ca5f8b3a7230b75124f45d443374',
    'dims_runner_depth8_verified_20261004.tar.gz': '5615fb9af3177487943501e6009bbdf872dc69587dd9dbb32bf612f975e24833',
    'production_runner_sm86_54150123': '4244102f2a18a38b2e95f95f0e93ae0ef7c067217a2e2e2e5f17ba0b39115e44',
    'source_iterator_final_54161582.tar.gz': 'd21e5e482a22ec8b7de958e68466d6db94f54efbb923ae5731629531804233a4',
}
for filename, wanted in pins.items():
    if hashlib.sha256((root / filename).read_bytes()).hexdigest() != wanted:
        raise SystemExit('Custody hash mismatch: ' + filename)
scaling = root / 'scaling_54154002'
raw = ['world1.log', 'world2_rank0.log', 'world2_rank1.log', 'stream1_gpu0.md', 'stream1_gpu1.md']
for filename in raw:
    if not (scaling / filename).is_file() or not (scaling / filename).stat().st_size:
        raise SystemExit('Missing scaling evidence: ' + filename)
for rank in (0, 1):
    if json.loads((scaling / f'world2-status/rank-{rank}.json').read_text()) != dict(rank=rank, puzzle_id=1000, exit_code=0, status='unsolved', completed_depths=8):
        raise SystemExit('Scaling custody status mismatch')
headers = {'Authorization': 'Bearer ' + load_key()}
base = 'https://console.vast.ai/api/v0/'
targets = {54044187: ('Tesla T4', 2), 54150123: ('RTX 3060', 1), 54154002: ('RTX 3060', 2), 54161582: ('RTX 3060', 2)}
record = {'authority': 'User ordered removal of unneeded Cube4 leases; current54201325 retained',
          'custody_pins': pins, 'scaling_raw_sha256': {name: hashlib.sha256((scaling/name).read_bytes()).hexdigest() for name in raw}, 'actions': []}

def request(method, path):
    response = requests.request(method, base + path, headers=headers, timeout=30, allow_redirects=False)
    response.raise_for_status()
    return response.json()

items = request('GET', 'instances/')['instances']
by_id = {item['id']: item for item in items}
for identifier, (gpu, count) in targets.items():
    item = by_id.get(identifier)
    if item is None or item.get('gpu_name') != gpu or item.get('num_gpus') != count or item.get('actual_status') != 'exited':
        raise SystemExit('Target changed; no deletion: ' + str(identifier))
    if item.get('intended_status') not in ('stopped', 'exited'):
        raise SystemExit('Target has active intent: ' + str(identifier))
receipt.write_text(json.dumps(record, indent=2))
for identifier in targets:
    result = request('DELETE', f'instances/{identifier}/')
    record['actions'].append({'id': identifier, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              'result': {k: result.get(k) for k in ('success', 'msg', 'error')}})
    receipt.write_text(json.dumps(record, indent=2))
    if result.get('success') is not True:
        raise SystemExit('Deletion not confirmed; reconcile receipt')
remaining = request('GET', 'instances/')['instances']
record['remaining'] = [{k: item.get(k) for k in ('id', 'label', 'actual_status', 'gpu_name', 'num_gpus', 'dph_total')} for item in remaining]
record['deleted_absent'] = all(item['id'] not in targets for item in remaining)
receipt.write_text(json.dumps(record, indent=2))
print(json.dumps({'actions': record['actions'], 'deleted_absent': record['deleted_absent'], 'remaining': record['remaining']}))
if not record['deleted_absent']:
    raise SystemExit('Provider still reports a deleted target')
