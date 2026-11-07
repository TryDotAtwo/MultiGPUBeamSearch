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
            jobs={}
            for name in ('ring_slot_jobs','stream3_jobs','stream4_jobs'):
                if name in fields:
                    count=int(fields[name])
                    if count<0:raise ValueError('invalid native job count')
                    jobs[name]=count
            done[key]=dict(seconds=value,jobs=jobs)
    reports=[]
    for depth in sorted({d for d,r in done}):
        keys=[(depth,r) for r in range(world)]
        if any(k not in starts or k not in done for k in keys):continue
        parents=sum(starts[k] for k in keys)
        if parents!=effective_beam:continue
        full=max(done[k]['seconds'] for k in keys);reference=parents*seconds_per_global_parent
        job_fields=('ring_slot_jobs','stream3_jobs','stream4_jobs')
        job_counts={name:sum(done[k]['jobs'][name] for k in keys)
                    for name in job_fields if all(name in done[k]['jobs'] for k in keys)}
        reports.append({'depth':depth,'parents':parents,'full_step_seconds':full,
            'stream1_reference_seconds':reference,'throughput_loss_fraction':1-reference/full,
            'parents_per_second':parents/full,'native_jobs_per_rank_sum':job_counts,
            'native_job_count_scope':('measured per-rank native dispatch jobs; not individual CUDA kernel launches'
                if len(job_counts)==len(job_fields) else 'native job counts unavailable in this build'),
            'measurement_scope':'observed full depth versus normalized inference curve; not matched frontier inference'})
    return reports
