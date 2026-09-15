"""Summarize Nsight kernel durations separately from the union busy interval."""
import collections
import json
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1] if len(sys.argv) > 1 else '/workspace/results/profile_s1.sqlite')
names = dict(db.execute('select id,value from StringIds'))
columns = [r[1] for r in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')]
print('kernel_columns', columns)
rows = list(db.execute('select start,end,demangledName,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
totals = collections.defaultdict(lambda: [0, 0])
for start, end, name, stream in rows:
    name = names[name]
    if 'layernorm' in name:
        family = 'input+LN' if 'build_input' in name else 'LayerNorm+bias'
    elif 'attention_kernel' in name:
        family = 'Attention'
    elif 'cutlass::' in name:
        family = 'GEMM'
    elif 'zero_padded' in name:
        family = 'padding'
    else:
        family = name[:140]
    totals[family][0] += end - start
    totals[family][1] += 1
total = sum(x[0] for x in totals.values())
for family, (duration, count) in sorted(totals.items(), key=lambda x: -x[1][0]):
    print(f'{family}: {duration / total * 100:.2f}% summed kernel time, {count} calls, avg_us={duration / count / 1000:.3f}')
busy = 0
left, right = rows[0][:2]
gaps = []
for start, end, *_ in rows[1:]:
    if start > right:
        busy += right - left
        gaps.append(start - right)
        left, right = start, end
    else:
        right = max(right, end)
busy += right - left
span = max(r[1] for r in rows) - rows[0][0]
print(json.dumps(dict(kernel_count=len(rows), streams=len(set(r[3] for r in rows)), span_ms=span/1e6, busy_union_ms=busy/1e6, idle_fraction=1-busy/span, max_gap_us=max(gaps, default=0)/1000, summed_kernel_ms=total/1e6)))
