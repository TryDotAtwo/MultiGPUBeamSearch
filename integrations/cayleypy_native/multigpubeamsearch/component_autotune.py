"""Bounded exact-capacity service probes; no small-frontier profile transfer."""
from __future__ import annotations
import math
import statistics
import time


def service_envelope(rows, plan, *, moves, inference_seconds, parent_batch):
    """Conservative scheduling proxy, NOT a measured complete depth.

    Sort cost is measured at actual C with actual concurrent lanes. Transport
    interpolation is restricted to its two measured volumes. Gross candidates
    assume no rejection; graph-specific acceptance and interference are unknown.
    """
    n=plan['frontier_state_capacity'];gross=n*moves
    q=rows[0]['outer_candidates'];c=rows[0]['shard_capacity']
    def slow(name):return max(statistics.median(r[name]) for r in rows)
    t1=math.ceil(n/parent_batch)*inference_seconds
    t3=math.ceil(gross/q)*slow('stream3_seconds')
    jobs=rows[0]['sort_jobs_concurrent']
    trigger=min(plan['STREAM4_BATCH_CANDIDATES'],c)
    sort_jobs=plan['SHARD_COUNT']*math.ceil(math.ceil(gross/plan['SHARD_COUNT'])/trigger)
    t4=math.ceil(sort_jobs/jobs)*slow('stream4_group_seconds')
    transport=0.0
    if rows[0]['transport']:
        transport=max(math.ceil(gross/r['transport'][-1]['items'])*
            statistics.median(r['transport'][-1]['seconds']) for r in rows)
    union=plan['SHARD_COUNT']*slow('union_seconds')
    return {'service_envelope_seconds':max(t1,t3,t4,transport)+union,
            'inference_seconds':t1,'stream3_seconds':t3,'stream4_seconds':t4,
            'transport_seconds':transport,'union_seconds':union,
            'scope':'isolated exact-capacity service envelope; not full-step timing'}


def tune_components(probe, session, plans, baseline, candidates, *, moves, inference):
    admitted=[('baseline',dict(baseline),plans)]
    for name,env in candidates[:2]:
        try:
            target=probe.admit(env)
            if target[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']!=plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']:
                continue
            admitted.append((name,env,target))
        except ValueError:
            continue
    if session is not None:session.close()
    probe.planning_session=None
    probe.environment_move_count=moves
    tested=[]
    for name,env,target in admitted:
        if time.monotonic()>=probe.deadline:break
        try:
            rows=probe.measure_components(env,target)
            estimate=service_envelope(rows,target[0],moves=moves,
                inference_seconds=inference['estimate']['median'],parent_batch=inference['parent_batch'])
            tested.append({'name':name,'environment':env,'plans':target,
                           'rank_measurements':rows,'estimate':estimate})
        except ValueError as error:
            if name=='baseline':raise
            tested.append({'name':name,'rejected':str(error)})
    valid=[r for r in tested if 'estimate' in r]
    if not valid:raise ValueError('no verified component profile')
    best=min(valid,key=lambda r:r['estimate']['service_envelope_seconds'])
    # Avoid changing profiles on an insignificant proxy difference.
    first=valid[0]
    if best['estimate']['service_envelope_seconds']>.95*first['estimate']['service_envelope_seconds']:
        best=first
    return {'phase':'component_calibrated','environment':best['environment'],
            'requested_beam_effective':plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
            'workload_parents':plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
            'measurement_scope':'exact_admitted_component_capacities',
            'pipeline_verified':False,'cache_hit':False,'tested':tested,
            'selection':best['name'],'estimate':best['estimate'],
            'search_policy':'baseline plus at most two exact-capacity candidates'}
