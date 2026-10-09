"""Search downstream parameters while freezing the measured model microbatch."""
from __future__ import annotations


TUNABLE_KEYS=frozenset({'BEAM_B_MICRO','BEAM_STREAM3_RING_SLOTS','BEAM_SHARD_COUNT',
    'BEAM_STREAM4_BATCH_CANDIDATES','BEAM_STREAM4_ACTIVE_SORT_SLOTS'})


def pipeline_candidates(inference_micro, baseline, *, max_outer=65536, tune_outer=True):
    if type(inference_micro) is not int or inference_micro<=0:
        raise ValueError('inference microbatch must be positive')
    if max_outer<inference_micro:raise ValueError('outer capacity cannot truncate model microbatch')
    fixed=dict(baseline,BEAM_ENSEMBLE_INFERENCE_MICRO=str(inference_micro))
    fixed['BEAM_B_MICRO']=str(max(inference_micro,int(fixed.get('BEAM_B_MICRO',inference_micro))))
    candidates=[('baseline',fixed)]
    def add(name,key,value):
        if key not in TUNABLE_KEYS:raise ValueError('attempt to tune a semantic or capacity policy')
        candidates.append((name,dict(fixed,**{key:str(value)})))
    for multiple in ((2,4,8) if tune_outer else ()):
        if inference_micro*multiple<=max_outer:add(f'outer-{multiple}x','BEAM_B_MICRO',inference_micro*multiple)
    for ring in (1,2,4,8):add(f'ring-{ring}','BEAM_STREAM3_RING_SLOTS',ring)
    for shards in (4,8,16,32,64,128):add(f'shards-{shards}','BEAM_SHARD_COUNT',shards)
    for batch in (8192,16384,32768,65536,131072):add(f'sort-batch-{batch}','BEAM_STREAM4_BATCH_CANDIDATES',batch)
    for slots in (1,2,4,8):add(f'sort-slots-{slots}','BEAM_STREAM4_ACTIVE_SORT_SLOTS',slots)
    seen=set();result=[]
    for name,environment in candidates:
        identity=tuple(sorted(environment.items()))
        if identity not in seen:seen.add(identity);result.append((name,environment))
    return result


def validate_rank_plans(plans):
    """Common beam/shard protocol plus exact independent memory admission."""
    if not plans:raise ValueError('no rank plans')
    common={'GLOBAL_BEAM_WIDTH_EFFECTIVE','BEAM_WIDTH_ALIGNMENT','SHARD_COUNT',
            'B_MICRO','WORLD_SIZE','STREAM4_BATCH_ALIGNMENT'}
    for key in common:
        if len({row[key] for row in plans})!=1:raise ValueError('rank plans disagree on '+key)
    world=plans[0]['WORLD_SIZE']
    if type(world) is not int or world<=0 or world!=len(plans):raise ValueError('missing rank admission')
    if any('LOCAL_RANK' in row for row in plans):
        # Native's historical LOCAL_RANK plan field is the communicator rank;
        # CUDA_DEVICE_LOCAL_RANK is the node-local ordinal and may repeat.
        ranks=[row.get('LOCAL_RANK') for row in plans]
        if any(type(rank) is not int for rank in ranks) or set(ranks)!=set(range(world)):
            raise ValueError('duplicate or missing communicator rank admission')
    for row in plans:
        if any(type(row[key]) is not int or row[key]<0 for key in
               ('estimated_required_device_bytes','gpu_budget_bytes')):
            raise ValueError('invalid native memory admission values')
        if row['estimated_required_device_bytes']>row['gpu_budget_bytes']:
            raise ValueError('rank exceeds exact native memory budget')
    return True


def tune_pipeline(inference_micro, baseline, *, admit, measure, deadline,
                  max_outer=65536, rounds=2, tune_outer=True):
    """Coordinate search with native rank admission and full-step measurements.

    Callbacks must observe the actual requested beam. Admission returns native
    plans for every rank; measurement returns correctness-gated Measurement
    records of identical global parent workloads. No plan estimate is a timing.
    """
    import time
    from dataclasses import replace
    from .calibration_stats import select
    if type(rounds) is not int or not 1 <= rounds <= 4:
        raise ValueError('pipeline search rounds must be in [1,4]')
    if not callable(admit) or not callable(measure):
        raise TypeError('native admission and full-step measurement are required')
    fixed = pipeline_candidates(inference_micro, baseline, max_outer=max_outer, tune_outer=tune_outer)[0][1]
    plans = admit(dict(fixed))
    validate_rank_plans(plans)
    effective = plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']
    world = len(plans)
    profiles = {'baseline': fixed}
    samples = [replace(row, profile='baseline') for row in measure(dict(fixed), plans)]
    if not samples or any(len(row.seconds_by_rank) != world for row in samples):
        raise ValueError('baseline measurement is missing ranks')
    workload = samples[0].parents
    if any(row.parents != workload for row in samples):
        raise ValueError('baseline workload changed between repeats')
    winner, rejected = select(samples, baseline='baseline')
    current = fixed
    seen = {tuple(sorted(fixed.items()))}
    for round_index in range(rounds):
        previous = winner.profile
        for name, environment in pipeline_candidates(inference_micro, current, max_outer=max_outer, tune_outer=tune_outer):
            identity = tuple(sorted(environment.items()))
            if identity in seen:
                continue
            if time.monotonic() >= deadline:
                break
            seen.add(identity)
            label = f'round-{round_index}-{name}'
            try:
                plans = admit(dict(environment))
                validate_rank_plans(plans)
                if plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'] != effective:
                    raise ValueError('candidate changes effective beam; timing is not comparable')
                rows = list(measure(dict(environment), plans))
                if not rows or any(row.parents != workload or len(row.seconds_by_rank) != world for row in rows):
                    raise ValueError('candidate changes workload or omits rank timing')
            except (ValueError, MemoryError) as error:
                rejected[label] = str(error)
                continue
            profiles[label] = environment
            samples.extend(replace(row, profile=label) for row in rows)
        winner, statistical_rejections = select(samples, baseline='baseline')
        rejected.update(statistical_rejections)
        current = profiles[winner.profile]
        if winner.profile == previous or time.monotonic() >= deadline:
            break
    return {'environment': current, 'estimate': winner.__dict__,
            'rejected': rejected, 'effective_beam': effective,
            'workload_parents': workload, 'world_size': world,
            'measurements': [row.__dict__ for row in samples]}

