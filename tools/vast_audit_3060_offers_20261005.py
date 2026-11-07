"""Read-only current 2/8 RTX3060 single-node offers for the USD20 audit."""
import json
import sys
from pathlib import Path
import requests

sys.path.insert(0, r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key

headers = {'Authorization': 'Bearer ' + load_key()}
base = 'https://console.vast.ai/api/v0/'
result = {}
for world in (2, 8):
    query = {'verified': {'eq': True}, 'rentable': {'eq': True}, 'rented': {'eq': False},
             'gpu_name': {'eq': 'RTX 3060'}, 'num_gpus': {'eq': world},
             'gpu_ram': {'gte': 12000}, 'cpu_ram': {'gte': 32000},
             'disk_space': {'gte': 50}, 'cuda_max_good': {'gte': 12.8},
             'reliability': {'gte': .98}, 'type': 'on-demand', 'limit': 8,
             'order': [['dph_total', 'asc']], 'allocated_storage': 50}
    if world==8:
        query.pop('reliability')
        query['cuda_max_good']={'gte':12.4}
    response = requests.post(base + 'bundles/', headers=headers, json=query,
                             timeout=25, allow_redirects=False)
    response.raise_for_status()
    keys = ('id', 'machine_id', 'gpu_name', 'num_gpus', 'gpu_ram', 'cpu_ram',
            'cpu_cores_effective', 'disk_space', 'dph_total', 'storage_cost',
            'inet_up_cost', 'inet_down_cost', 'cuda_max_good', 'reliability', 'duration')
    result[str(world)] = [{k: row.get(k) for k in keys}
                         for row in response.json().get('offers', [])
                         if row.get('num_gpus') == world and row.get('gpu_name') == 'RTX 3060']
response = requests.get(base + 'instances/', headers=headers, timeout=25, allow_redirects=False)
response.raise_for_status()
result['existing_instances'] = [{k: row.get(k) for k in ('id', 'label', 'actual_status', 'dph_total')}
                                for row in response.json().get('instances', [])]
destination = Path('test_results/audit_fixes_2026-09-28/COMPLETE_AUDIT_OFFERS_20261005.json')
destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
