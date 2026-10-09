"""Bounded CPU-only analysis of inspected Nsight SQLite schema; no GPU work."""
import json
import sqlite3
from pathlib import Path

root = Path('/workspace/pipeline_audit_54195079/evidence/nsys_pipeline_54195079')
results = []
for rank in (0, 1):
    db = sqlite3.connect(root / f'rank_{rank}.sqlite')
    rows = db.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
    groups = {}
    for start, end, name in rows:
        group = 'cub' if 'cub::' in name else 'nccl' if 'nccl' in name.lower() else 'other'
        groups[group] = groups.get(group, 0) + end - start
    intervals = db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start').fetchall()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = sorted((b[0] - a[1] for a, b in zip(merged, merged[1:])), reverse=True)
    results.append({'rank': rank, 'kernel_duration_sums_seconds': {k: v / 1e9 for k, v in groups.items()},
                    'trace_activity_span_seconds': (merged[-1][1] - merged[0][0]) / 1e9,
                    'trace_activity_union_seconds': sum(b-a for a,b in merged) / 1e9,
                    'largest_activity_gaps_seconds': [g/1e9 for g in gaps[:10]],
                    'nvtx_event_count': db.execute('SELECT count(*) FROM NVTX_EVENTS').fetchone()[0]})
output = {'scope': 'whole profiled process including setup/warmup/shutdown, not depth-only critical path; graph-level tracing; duration sums overlap', 'ranks': results}
(root / 'timeline_summary.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
