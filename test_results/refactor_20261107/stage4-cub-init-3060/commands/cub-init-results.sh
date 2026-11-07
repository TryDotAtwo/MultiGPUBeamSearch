python - <<'PY'
import json,statistics
from pathlib import Path
for world in (1,2):
 p=Path(f'/workspace/beam-results/cub-init/session-pairs1/pair-world{world}/paired-results.json')
 d=json.loads(p.read_text()); pairs=d['pairs']
 print('world',world,'median_speedup_pct',d['median_speedup_percent'])
 for key in ('median','wall_seconds'):
  print(key,'control',statistics.median(x['control'][key] for x in pairs),'fixed',statistics.median(x['radix'][key] for x in pairs),
        'paired_change_pct',statistics.median(100*(x['radix'][key]/x['control'][key]-1) for x in pairs))
 print('planned_bytes', [x['estimated_required_device_bytes'] for x in d['plans']['control']], [x['estimated_required_device_bytes'] for x in d['plans']['radix']])
PY
