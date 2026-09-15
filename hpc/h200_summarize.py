"""Summarize raw H200 measurements; reject missing/failed rows explicitly."""
import json
import re
import statistics
import struct
from pathlib import Path
import numpy as np

root = Path('/workspace/results')
pattern = re.compile(r'parents_per_sec=([\d.]+) candidates_per_sec=([\d.]+)')
rows = {}
for path in sorted(root.glob('k_*.log')):
    match = pattern.search(path.read_text())
    key = path.stem.rsplit('_', 1)[0]
    if match:
        parents, candidates = map(float, match.groups())
        assert abs(candidates / parents - 24) < 0.001
        rows.setdefault(key, []).append(parents)
    else:
        print('FAILED', path.name)
baseline_path = root / 'k_baseline_1.bin'
def read_scores(path):
    with path.open('rb') as handle:
        magic, version, lanes, count = struct.unpack('<QIIQ', handle.read(24))
        assert magic == 0x31504D5544533153 and version == 1
        values = np.fromfile(handle, dtype=np.uint32)
        assert values.size == lanes * count
        return values

baseline = read_scores(baseline_path) if baseline_path.exists() else None
report = []
for key, rates in rows.items():
    result = dict(tag=key, repeats=len(rates), median_parents_s=statistics.median(rates), rates=rates)
    dump = root / (key + '_1.bin')
    if baseline is not None and dump.exists():
        values = read_scores(dump)
        assert values.shape == baseline.shape
        error = np.abs(values.astype(np.int64) - baseline.astype(np.int64))
        result.update(exact_fraction=float(np.mean(error == 0)), max_key_error=int(error.max()), mean_key_error=float(error.mean()))
    report.append(result)
report.sort(key=lambda row: -row['median_parents_s'])
print(json.dumps(report, indent=2))
(root / 'kernel_summary.json').write_text(json.dumps(report, indent=2))
for path in sorted(root.glob('overlap_*.log')):
    for line in path.read_text().splitlines():
        if line.startswith('stream_pipeline_benchmark '):
            print(path.name, line)
