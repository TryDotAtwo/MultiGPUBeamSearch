"""Summarize only paired dual-LN runs; keep standalone and pipeline separate."""
import json
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1] if len(sys.argv)>1 else '/workspace/results')
rates = {}
for mode in (0, 1):
    rates[mode] = []
    for p in sorted(root.glob(f'dual_ln_mode{mode}_[1-5].log')):
        m = re.search(r'parents_per_sec=([\d.]+)',p.read_text())
        if m:
            rates[mode].append(float(m[1]))
output = {'isolated': {mode: {'rates': values, 'median': statistics.median(values)}
                       for mode, values in rates.items() if values}}
if rates[0] and len(rates[0]) == len(rates[1]):
    output['isolated']['median_speedup'] = statistics.median(rates[1])/statistics.median(rates[0])
    output['isolated']['paired_ratios'] = [b/a for a,b in zip(rates[0], rates[1])]
output['quality_exact'] = []
output['quality_mismatch'] = []
for p in sorted(root.glob('dual_ln_quality_p*_mode0.bin')):
    q=p.with_name(p.name.replace('_mode0.bin','_mode1.bin'))
    key = 'quality_exact' if q.exists() and p.read_bytes()==q.read_bytes() else 'quality_mismatch'
    output[key].append(p.stem.replace('_mode0',''))
output['quality_pass'] = not output['quality_mismatch'] and len(output['quality_exact']) == 11
output['pipeline'] = {}
for p in sorted(root.glob('dual_ln_pipeline_mode*.log')):
    rows=[]
    for line in p.read_text().splitlines():
        m=re.search(r'depth_done=(\d+) depth_sec=([\d.]+)',line)
        if m: rows.append({'depth':int(m[1]),'seconds':float(m[2])})
    output['pipeline'][p.stem]=rows
text=json.dumps(output,indent=2)
(root/'dual_ln_summary.json').write_text(text)
print(text)
