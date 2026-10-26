"""Measured service pressure narrows experiments; it never certifies overlap."""
import math

POLICY = 'service-guided-shortlist-v2'
MAX_COMPONENT_CHALLENGERS = 4


def job_counts(parents, moves, outer_parents, inference_parents, staging, shards, flush):
    """Per-rank/depth schedule proxy, NOT executed CUDA-kernel counts.

    Each inference forward/job can launch multiple kernels; graph replay counts
    are separate. Sort jobs assume gross candidates and balanced shards, before
    rejection; actual threshold/flush counts may differ.
    """
    values=(parents,moves,outer_parents,inference_parents,staging,shards,flush)
    if any(type(v) is not int or v<=0 for v in values):
        raise ValueError('positive integer schedule dimensions required')
    ceil=lambda a,b:(a+b-1)//b
    outer=ceil(parents,outer_parents)
    full,tail=divmod(parents,outer_parents)
    forwards=full*ceil(outer_parents,inference_parents)+ceil(tail,inference_parents)
    return dict(outer_jobs=outer,model_forwards_per_model=forwards,
        stream3_jobs=ceil(outer,staging),
        gross_sort_jobs=shards*ceil(ceil(parents*moves,shards),flush),
        parents=parents,candidates=parents*moves,
        outer_jobs_per_parent=outer/parents,
        scope='analytical per-rank full-step jobs; not CUDA kernels or graph replays')


def narrow(proposals, baseline, estimate, *, plan=None, moves=None):
    """Keep complementary experiments, not a Cartesian hardware sweep.

    At low isolated service pressure, test dispatch amortization and reduced
    concurrency before shard reshaping. Small flushes repeatedly sort the same
    C-sized survivor buffers, so never propose them solely from arrival latency.
    Full-step comparison remains mandatory: service pressure omits interference.
    """
    inference = estimate.get('inference_seconds', 0)
    downstream = estimate.get('downstream_service_work_seconds', -1)
    valid = (math.isfinite(inference) and inference > 0
             and math.isfinite(downstream) and downstream >= 0)
    ratio = downstream / inference if valid else None
    if valid and ratio <= .05:
        regime = 'inference_dominated'
        priority = ('outer-', 'staging-', 'lanes-', 'bulk', 'shards-', 'service')
    elif valid and estimate.get('stream4_seconds', 0) >= downstream / 2:
        regime = 'sort_pressure'
        priority = ('bulk', 'shards-', 'lanes-', 'outer-', 'staging-', 'service')
    else:
        regime = 'transport_or_mixed' if valid else 'unknown'
        priority = ('outer-', 'staging-', 'bulk', 'shards-', 'lanes-', 'service')
    def order(row):
        return next((i for i, prefix in enumerate(priority) if row[0].startswith(prefix)), len(priority))
    selected = []; omitted = []; families = set()
    for name, env in sorted(proposals, key=order):
        if int(env['BEAM_STREAM4_BATCH_CANDIDATES']) < int(baseline['BEAM_STREAM4_BATCH_CANDIDATES']):
            omitted.append(dict(name=name, reason='smaller flush repeats survivor sorting'))
            continue
        family = next((prefix for prefix in priority if name.startswith(prefix)), name)
        if family in families or len(selected) >= MAX_COMPONENT_CHALLENGERS:
            omitted.append(dict(name=name, reason='bounded complementary shortlist'))
            continue
        selected.append((name, env)); families.add(family)
    schedules={}
    if plan is not None:
        row_factor=int(baseline['BEAM_B_MICRO'])//plan['B_MICRO']
        for name,env in [('baseline',baseline)]+selected:
            schedules[name]=job_counts(plan['frontier_state_capacity'],moves,
                int(env['BEAM_B_MICRO'])//row_factor,
                int(env['BEAM_ENSEMBLE_INFERENCE_MICRO']),
                int(env['BEAM_STREAM3_RING_SLOTS']),int(env['BEAM_SHARD_COUNT']),
                int(env['BEAM_STREAM4_BATCH_CANDIDATES']))
    return selected, dict(policy=POLICY, regime=regime, schedule_estimates=schedules,
        isolated_downstream_to_inference=ratio, selected=[n for n, _ in selected],
        omitted=omitted, scope='proposal heuristic only; full-step measurement required')


def full_challengers(trace):
    return 1 if trace.get('regime') == 'inference_dominated' else 2
