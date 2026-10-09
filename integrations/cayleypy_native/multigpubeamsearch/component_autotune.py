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
    # Inference estimate is slowest-rank seconds / GLOBAL parent count.
    t1=n*plan['WORLD_SIZE']*inference_seconds
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


def service_geometry(rows,plan,*,moves,inference):
    """Arrival/service bounds; only native admission can accept the proposal."""
    from .beam_geometry import round_up,staging_from_release_latency
    median=inference['estimate']['median'];world=plan['WORLD_SIZE']
    arrival=moves/(median*world)
    sort=max(statistics.median(r['stream4_group_seconds'])/r['sort_jobs_concurrent'] for r in rows)
    native_floor=math.ceil(plan['frontier_state_capacity']*moves/plan['SHARD_COUNT']/4)
    desired=min(plan['SHARD_CAPACITY_CANDIDATES'],native_floor,max(1,math.ceil(arrival*sort/.8)))
    trigger=round_up(desired,plan['STREAM4_BATCH_ALIGNMENT'])
    release=max(sorted(r['stream3_seconds'])[4]+(
        sorted(r['transport'][-1]['seconds'])[4] if r['transport'] else 0) for r in rows)
    producer=inference['parent_batch']*world*median
    ring=min(8,staging_from_release_latency(release,producer))
    return {'BEAM_STREAM4_BATCH_CANDIDATES':str(trigger),'BEAM_STREAM3_RING_SLOTS':str(ring)}


def tune_components(probe, session, plans, baseline, candidates, *, moves, inference):
    admitted=[('baseline',dict(baseline),plans)]
    if session is not None:session.close()
    probe.planning_session=None
    tested=[]
    samples=[]
    from .calibration_stats import Measurement,select
    for name,env,target in admitted:
        if time.monotonic()>=probe.deadline:break
        try:
            rows=probe.measure_components(env,target,move_count=moves)
            estimate=service_envelope(rows,target[0],moves=moves,
                inference_seconds=inference['estimate']['median'],parent_batch=inference['parent_batch'])
            tested.append({'name':name,'environment':env,'plans':target,
                           'rank_measurements':rows,'estimate':estimate})
            if name=='baseline':
                proposal=service_geometry(rows,target[0],moves=moves,inference=inference)
                proposed=[('service-batch',dict(baseline,
                    BEAM_STREAM4_BATCH_CANDIDATES=proposal['BEAM_STREAM4_BATCH_CANDIDATES']))]
                if candidates:
                    candidate_name,candidate_env=candidates[0]
                    proposed.append((candidate_name,dict(candidate_env,**proposal)))
                else:
                    proposed.append(('service-staging',dict(baseline,**proposal)))
                # Close all native memory owners before measuring another profile.
                from .plan_session import NativePlanSession
                with NativePlanSession(probe.runner,probe.environment,probe.world,
                        probe.directory/'service-plans',deadline=probe.deadline) as planner:
                    for next_name,next_env in proposed:
                        try:
                            next_plans=planner.admit(probe.beam,next_env)
                            if next_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']==plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']:
                                admitted.append((next_name,next_env,next_plans))
                        except ValueError:
                            continue
            for repeat in range(5):
                cohort=[]
                for row in rows:
                    item=dict(row)
                    for key in ('stream3_seconds','stream4_group_seconds','union_seconds'):
                        item[key]=[row[key][repeat]]
                    item['transport']=[dict(t,seconds=[t['seconds'][repeat]]) for t in row['transport']]
                    cohort.append(item)
                value=service_envelope(cohort,target[0],moves=moves,
                    inference_seconds=inference['estimate']['median'],parent_batch=inference['parent_batch'])
                samples.append(Measurement(name,plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
                    (value['service_envelope_seconds'],),True,True))
        except ValueError as error:
            if name=='baseline':raise
            tested.append({'name':name,'rejected':str(error)})
    valid=[r for r in tested if 'estimate' in r]
    if not valid:raise ValueError('no verified component profile')
    winner,rejected=select(samples,baseline='baseline',min_improvement=.05)
    best=next(r for r in valid if r['name']==winner.profile)
    return {'phase':'component_calibrated','environment':best['environment'],
            'requested_beam_effective':plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
            'workload_parents':plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
            'measurement_scope':'exact_admitted_component_capacities',
            'pipeline_verified':False,'cache_hit':False,'tested':tested,
            'selection':best['name'],'estimate':best['estimate'],
            'selection_rejections':rejected,
            'search_policy':'baseline plus at most two exact-capacity candidates'}
