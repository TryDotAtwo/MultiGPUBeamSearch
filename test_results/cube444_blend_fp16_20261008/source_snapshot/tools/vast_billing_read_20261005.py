"""Read-only charges probe; never emits credentials or unrelated charge contents."""
import json
import sys
from datetime import datetime, timezone

import requests

sys.path.insert(0, r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key

# Match the official vast-cli show invoices-v1 --charges query encoding.
start = int(datetime(2026, 9, 28, tzinfo=timezone.utc).timestamp())
end = int(datetime.now(timezone.utc).timestamp())
query = {'select_filters': json.dumps({'day': {'gte': start, 'lte': end},
                                     'type': {'in': ['instance']}}),
         'latest_first': 'true', 'limit': '100', 'format': 'tree'}
response = requests.get('https://console.vast.ai/api/v0/charges/',
                        headers={'Authorization': 'Bearer ' + load_key()},
                        params=query, timeout=30, allow_redirects=False)
report = {'endpoint': '/api/v0/charges/', 'http_status': response.status_code,
          'start': start, 'end': end, 'billing_reconciled': False}
if response.status_code == 200:
    data = response.json()
    report.update(response_fields=sorted(data) if isinstance(data, dict) else [],
                  count=data.get('count') if isinstance(data, dict) else None,
                  total=data.get('total') if isinstance(data, dict) else None,
                  has_next_page=bool(data.get('next_token')) if isinstance(data, dict) else None)
    # Establish schema before interpreting amounts or ownership attribution.
    rows = data.get('results', []) if isinstance(data, dict) else []
    report['first_row_fields'] = sorted(rows[0]) if rows and isinstance(rows[0], dict) else []
print(json.dumps(report))
