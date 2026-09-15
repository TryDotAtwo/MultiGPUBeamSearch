import json
import re
import statistics
import struct
from pathlib import Path
import numpy as np

root = Path('/workspace/results')
report = []
for prefix in ['combo', 'compact']:
    groups = {}
    for path in sorted(root.glob(prefix + '_m*_c*_*.log')):
        match = re.search(r'parents_per_sec=([\d.]+)', path.read_text())
        if match:
            groups.setdefault(path.stem.rsplit('_', 1)[0], []).append(float(match[1]))
    for tag, rates in groups.items():
        row = dict(tag=tag, repeats=len(rates), median=statistics.median(rates))
        if prefix == 'compact':
            ref = root / (tag.replace('compact_', 'combo_') + '_1.bin')
            out = root / (tag + '_1.bin')
            if ref.exists() and out.exists():
                assert ref.read_bytes()[:24] == out.read_bytes()[:24]
                a = np.fromfile(ref, dtype='<u4', offset=24)
                b = np.fromfile(out, dtype='<u4', offset=24)
                assert a.shape == b.shape
                row['exact_fraction'] = float(np.mean(a == b))
                row['max_key_error'] = int(np.max(np.abs(a.astype('int64') - b.astype('int64'))))
        report.append(row)
report.sort(key=lambda row: -row['median'])
(root / 'sweep_summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
