"""Hardware-derived proposals for the existing component/full-step tuner.

Bounds are a finite experimental domain, not a global optimum certificate.
Inference microbatch is immutable; outer transactions may contain several forwards.
"""
from dataclasses import replace
from .beam_geometry import geometry, round_up, ceil_div
from .profile_heuristics import POLICY

MAX_COARSE=16
MAX_FINE=10
MAX_ROUNDS=2
FULL_PER_ROUND=2


def context(devices, plans, inference):
    import torch
    hardware=[]
    for device in devices:
        p=torch.cuda.get_device_properties(device)
        hardware.append(dict(name=p.name,sm_count=p.multi_processor_count,
            total_memory=p.total_memory,compute=[p.major,p.minor],
            l2_bytes=getattr(p,'L2_cache_size',getattr(p,'l2_cache_size',0))))
    # A bucket prevents trivial allocator jitter invalidating an expensive cache.
    quantum=256*1024**2
    budget=min(p['gpu_budget_bytes'] for p in plans)
    return dict(policy=POLICY,hardware=hardware,budget_bytes=budget//quantum*quantum,
        parent_batch=inference['parent_batch'],reserve_bytes=inference['reserve_bytes'],
        signature=inference.get('signature',{}),
        limits=dict(coarse=MAX_COARSE,fine=MAX_FINE,rounds=MAX_ROUNDS,
                    full_per_round=FULL_PER_ROUND))


def key(env):
    return tuple(sorted(env.items()))


def proposals(shape, baseline, context, *, service=None, fine=False, round_index=0,
              ring_limit=None, seen=()):
    """Generate bounded, memory-pruned neighbors; native admission is mandatory."""
    budget=context['budget_bytes']
    sms=min(p['sm_count'] for p in context['hardware'])
    l2=min(p.get('l2_bytes',0) for p in context['hardware'])
    align=shape.alignment
    outer=int(baseline['BEAM_B_MICRO']);shards=int(baseline['BEAM_SHARD_COUNT'])
    staging=int(baseline['BEAM_STREAM3_RING_SLOTS'])
    lanes=int(baseline['BEAM_STREAM4_ACTIVE_SORT_SLOTS'])
    batch=int(baseline['BEAM_STREAM4_BATCH_CANDIDATES'])
    frozen=int(baseline['BEAM_ENSEMBLE_INFERENCE_MICRO'])
    # Scalar MLP's environment uses child rows; native geometry uses parents.
    row_factor=context.get('outer_rows_per_parent',1)
    local=ceil_div(shape.requested_beam,shape.gpus)
    max_shards=min(128,max(1,ceil_div(local,align)))
    max_lanes=max(1,min(max_shards,ceil_div(sms,16)))
    pool=[]
    def add(label, **updates):pool.append((label,dict(baseline,**{k:str(v) for k,v in updates.items()})))
    if fine:
        step=max(1,shards//(4*(round_index+1)))
        for value in (max(1,shards-step),min(max_shards,shards+step)):
            add('shards-'+str(value),BEAM_SHARD_COUNT=value,
                BEAM_STREAM4_ACTIVE_SORT_SLOTS=min(lanes,value))
        for value in (max(1,lanes-1),min(max_lanes,lanes+1)):
            add('lanes-'+str(value),BEAM_STREAM4_ACTIVE_SORT_SLOTS=value)
        for value in (max(2,staging-1),min(8,staging+1)):
            add('staging-'+str(value),BEAM_STREAM3_RING_SLOTS=value)
        for value in (max(align,round_up(batch*3//4,align)),round_up(batch*5//4,align)):
            add('batch-'+str(value),BEAM_STREAM4_BATCH_CANDIDATES=value)
        # Integral inference batches only; never reduce the winning inference batch.
        quantum=shape.native_b_micro*row_factor
        for value in (max(quantum,round_up(outer*3//4,quantum)),round_up(outer*5//4,quantum)):
            if value<=max(local*row_factor,quantum):add('outer-'+str(value),BEAM_B_MICRO=value)
    else:
        if service:
            add('service',**service)
            add('service-batch',BEAM_STREAM4_BATCH_CANDIDATES=service['BEAM_STREAM4_BATCH_CANDIDATES'])
        # Memory seeds are evaluated over the whole legal shard range, not only
        # divisors of the requested beam. SM seeds provide independent coverage.
        seeds=[]
        for value in range(1,max_shards+1):
            try:g=geometry(shape,value,staging,min(lanes,value),physical_ring_limit=ring_limit)
            except ValueError:continue
            if g['named_memory_lower_bound_bytes']<=budget:
                seeds.append((g['named_memory_lower_bound_bytes'],value))
        seeds.sort()
        shard_values=[v for _,v in seeds[:2]]+[1,min(max_shards,ceil_div(sms,8)),
            min(max_shards,max(1,shards*2)),max(1,shards//2)]
        for value in dict.fromkeys(shard_values):
            add('shards-'+str(value),BEAM_SHARD_COUNT=value,
                BEAM_STREAM4_ACTIVE_SORT_SLOTS=min(lanes,value))
        for value in sorted({1,max_lanes,min(max_lanes,max(1,lanes*2))}):
            add('lanes-'+str(value),BEAM_STREAM4_ACTIVE_SORT_SLOTS=min(value,shards))
        for value in sorted({2, min(8,max(staging, int((service or {}).get('BEAM_STREAM3_RING_SLOTS',staging))))}):
            add('staging-'+str(value),BEAM_STREAM3_RING_SLOTS=value)
        for factor in (2,4):
            if outer*factor<=local*row_factor:
                add('outer-'+str(outer*factor),BEAM_B_MICRO=outer*factor,
                    BEAM_STREAM3_RING_SLOTS=max(2,ceil_div(staging,factor)))
        bulk=round_up(max(1,ceil_div(local*shape.moves,shards*4)),align)
        add('bulk',BEAM_STREAM4_BATCH_CANDIDATES=bulk)
        if l2:add('l2-batch',BEAM_STREAM4_BATCH_CANDIDATES=round_up(max(1,l2//32),align))
    used=set(seen)|{key(baseline)};accepted=[];rejected=[]
    for name,env in pool:
        fingerprint=key(env)
        if fingerprint in used:continue
        used.add(fingerprint)
        try:
            g=geometry(replace(shape,native_b_micro=int(env['BEAM_B_MICRO'])//row_factor),
                int(env['BEAM_SHARD_COUNT']),int(env['BEAM_STREAM3_RING_SLOTS']),
                int(env['BEAM_STREAM4_ACTIVE_SORT_SLOTS']),physical_ring_limit=ring_limit)
            reason=None
            if g['named_memory_lower_bound_bytes']>budget:reason='named memory lower bound exceeds budget'
            elif int(env['BEAM_STREAM4_BATCH_CANDIDATES'])>g['physical_shard_capacity']:reason='flush exceeds capacity'
            if reason:rejected.append(dict(name=name,reason=reason));continue
        except ValueError as error:
            rejected.append(dict(name=name,reason=str(error)));continue
        assert env['BEAM_ENSEMBLE_INFERENCE_MICRO']==str(frozen)
        accepted.append((name,env))
    limit=MAX_FINE if fine else MAX_COARSE
    return accepted[:limit],dict(domain=dict(max_shards=max_shards,max_sort_lanes=max_lanes,
        budget_bytes=budget,local_parents=local,sm_count=sms,l2_bytes=l2),
        generated=len(pool),duplicates=len(pool)-len(accepted)-len(rejected),
        pruned=rejected,trial_limit=limit,truncated=len(accepted)>limit)


def shortlist(rows, *, baseline=True, count=2):
    """Proxy Pareto pruning only; a full step makes the performance decision."""
    valid=[r for r in rows if 'estimate' in r and 'plans' in r]
    def metrics(r):return (r['estimate']['downstream_service_work_seconds'],
        max(p['estimated_required_device_bytes'] for p in r['plans']))
    ordered=sorted(valid,key=metrics)
    pareto=[r for r in ordered if not any(
        metrics(other)[0]<=metrics(r)[0] and metrics(other)[1]<=metrics(r)[1]
        and metrics(other)!=metrics(r) for other in ordered)]
    selected=([next(r for r in valid if r['name']=='baseline')] if baseline else [])
    for row in pareto+ordered:
        if row not in selected:selected.append(row)
        if len(selected)>=count+(1 if baseline else 0):break
    return selected
