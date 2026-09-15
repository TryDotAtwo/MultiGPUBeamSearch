import csv
import json
from pathlib import Path
import re
import statistics

root = Path('/workspace/results')
groups = {}
for path in root.glob('compare_*.csv'):
    for row in csv.DictReader(path.open()):
        key = (path.stem, row['mode'], int(row['batch']))
        groups.setdefault(key, []).append(float(row['parents_per_sec']))
out = []
for (source, mode, batch), rates in groups.items():
    out.append(dict(source=source, mode=mode, batch=batch, repeats=len(rates), median=statistics.median(rates), min=min(rates), max=max(rates)))
native = {}
for path in root.glob('compare_native_*.log'):
    match = re.search(r'parents_per_sec=([\d.]+)', path.read_text())
    if match:
        native.setdefault(path.stem.rsplit('_', 1)[0], []).append(float(match[1]))
for name, rates in native.items():
    out.append(dict(source=name, repeats=len(rates), median=statistics.median(rates), min=min(rates), max=max(rates)))
out.sort(key=lambda row: (row['source'], -row['median']))
(root / 'compare_summary.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
