"""Observed saturated depths versus a normalized inference-curve reference."""
import math
import re


def observed_depths(text,world,effective_beam,seconds_per_global_parent):
    rank=0;starts={};done={}
    for line in text.splitlines():
        marker=re.fullmatch(r'\[rank (\d+) stdout.log\]',line)
        if marker:rank=int(marker[1]);continue
        if line.startswith('[rank ') or line=='[launcher]':rank=-1;continue
        if not 0<=rank<world:continue
        if line.startswith('runtime_depth_start='):
            fields=dict(p.split('=',1) for p in line.split() if '=' in p)
            key=(int(fields['runtime_depth_start']),rank)
            if key in starts:raise ValueError('duplicate runtime depth start')
            starts[key]=int(fields['frontier_size'])
        if line.startswith('runtime_depth_done='):
            fields=dict(p.split('=',1) for p in line.split() if '=' in p)
            key=(int(fields['runtime_depth_done']),rank)
            if key in done:raise ValueError('duplicate runtime depth completion')
            value=float(fields['depth_sec'])
            if not math.isfinite(value) or value<=0:raise ValueError('invalid complete depth duration')
            done[key]=value
    reports=[]
    for depth in sorted({d for d,r in done}):
        keys=[(depth,r) for r in range(world)]
        if any(k not in starts or k not in done for k in keys):continue
        parents=sum(starts[k] for k in keys)
        if parents!=effective_beam:continue
        full=max(done[k] for k in keys);reference=parents*seconds_per_global_parent
        reports.append({'depth':depth,'parents':parents,'full_step_seconds':full,
            'stream1_reference_seconds':reference,'throughput_loss_fraction':1-reference/full,
            'measurement_scope':'observed full depth versus normalized inference curve; not matched frontier inference'})
    return reports
