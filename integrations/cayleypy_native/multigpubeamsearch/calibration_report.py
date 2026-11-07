"""Matched-workload Stream1 versus complete-depth performance reporting."""
import math
import statistics


def matched_step_report(inference, pipeline):
    """Receipts include global parents, ordered UUIDs, data/model/build hashes.

    Each timing row contains one duration per selected rank. Cohort throughput
    uses the slowest rank, never the sum of independently measured GPU rates.
    """
    identity = ('parents', 'gpu_uuids', 'frontier_sha256', 'model_sha256',
                'build_sha256', 'precision', 'executor')
    for key in identity:
        if key not in inference or key not in pipeline or inference[key] != pipeline[key]:
            raise ValueError('unmatched inference/pipeline workload: ' + key)
    parents = inference['parents']
    if type(parents) is not int or parents <= 0:
        raise ValueError('positive global parent count required')
    world = len(inference['gpu_uuids'])
    if world < 1 or len(set(inference['gpu_uuids'])) != world:
        raise ValueError('unique ordered GPU cohort required')
    def duration(receipt):
        if receipt.get('correctness_passed') is not True:
            raise ValueError('correctness receipt required')
        rows = receipt['seconds_by_rank']
        if len(rows) < 3:
            raise ValueError('at least three matched timing samples required')
        for row in rows:
            if len(row) != world or any(not math.isfinite(x) or x <= 0 for x in row):
                raise ValueError('missing rank or invalid duration')
        return statistics.median(max(row) for row in rows)
    stream1, step = duration(inference), duration(pipeline)
    return dict(parents=parents, world_size=world, measurement_scope='matched_full_frontier',
                stream1_seconds=stream1, full_step_seconds=step,
                stream1_parents_per_second=parents/stream1,
                full_step_parents_per_second=parents/step,
                throughput_loss_fraction=1-stream1/step,
                relative_time_overhead=step/stream1-1)
