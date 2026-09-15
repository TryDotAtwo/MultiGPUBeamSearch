"""Read-only analysis of the two explicitly mapped dual-H200 Nsight traces.

Stream IDs are trace-specific, checked against creation order and graph counts.
Graph duration is elapsed GPU span, not exclusive kernel time. Overlapping
durations must not be added as wall time. Uses an interior window to omit edges.
"""
import collections
import json
import sqlite3
import statistics
import sys

S1 = {17, 22, 24, 26, 28, 30, 32, 34}
S4 = {36, 37, 38, 39}

def merged(rows, lo, hi):
    out = []
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in rows if b > lo and a < hi):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

def duration(rows):
    return sum(b-a for a, b in rows)

for path in sys.argv[1:]:
    c = sqlite3.connect('file:' + path + '?mode=ro', uri=True)
    rows = list(c.execute('select deviceId,streamId,start,end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE'))
    spans = [(min(a for d,s,a,b in rows if d == gpu), max(b for d,s,a,b in rows if d == gpu)) for gpu in (0,1)]
    lo = max(a for a,b in spans) + 100000000
    hi = min(b for a,b in spans) - 100000000
    result = {'file': path, 'window_seconds': (hi-lo)/1e9, 'gpus': []}
    for gpu in (0,1):
        r = [(s,a,b) for d,s,a,b in rows if d == gpu]
        s1 = [(a,b) for s,a,b in r if s in S1]
        s4 = [(a,b) for s,a,b in r if s in S4]
        inner = [(s,a,b) for s,a,b in r if lo <= a and b <= hi]
        groups = {}
        for label, ids in [('stream12', S1), ('stream3', {19}), ('stream4', S4)]:
            x = [(a,b) for s,a,b in inner if s in ids]
            spans_ms = [(b-a)/1e6 for a,b in x]
            all_x = [(a,b) for s,a,b in r if s in ids]
            groups[label] = {'count': len(x), 'mean_graph_ms': statistics.mean(spans_ms) if x else None,
                             'median_graph_ms': statistics.median(spans_ms) if x else None,
                             'union_fraction': duration(merged(all_x,lo,hi))/(hi-lo),
                             'sum_span_seconds': duration(x)/1e9}
        kernels = list(c.execute('select streamId,start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?', (gpu,)))
        groups['nccl_stream21_union_fraction'] = duration(merged([(a,b) for s,a,b in kernels if s==21],lo,hi))/(hi-lo)
        result['gpus'].append({'gpu':gpu, 'parents_per_second': groups['stream12']['count']*384/((hi-lo)/1e9), 'groups':groups})
    print(json.dumps(result, indent=2))
