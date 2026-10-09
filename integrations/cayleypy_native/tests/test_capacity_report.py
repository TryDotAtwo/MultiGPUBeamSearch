import time
from copy import deepcopy
import pytest
from multigpubeamsearch.beam_capacity import CapacityRejected, find_capacity
from multigpubeamsearch.calibration_report import matched_step_report


def plans(beam):
    return [dict(GLOBAL_BEAM_WIDTH_EFFECTIVE=beam, BEAM_WIDTH_ALIGNMENT=1024,
                 SHARD_COUNT=4, B_MICRO=256, WORLD_SIZE=2,
                 STREAM4_BATCH_ALIGNMENT=1024, LOCAL_RANK=rank,
                 estimated_required_device_bytes=beam, gpu_budget_bytes=8192)
            for rank in range(2)]


def test_capacity_fixed_profiles_and_actual_step():
    def admit(beam, profile):
        if beam > profile['limit']:
            raise CapacityRejected('native capacity exceeded')
        return plans(beam)
    result = find_capacity([{'limit':4096}, {'limit':7168}], admit=admit,
        upper_bound=8192, alignment=1024, deadline=time.monotonic()+10,
        verify_full_step=lambda beam, profile, receipt: beam == 7168 and len(receipt)==2)
    assert result.requested_beam == 7168
    assert result.upper_rejected_beam == 8192
    assert result.search_complete and result.full_step_verified


def test_runtime_error_cannot_be_interpreted_as_capacity():
    def broken(*args):
        raise RuntimeError('NCCL failed')
    with pytest.raises(RuntimeError, match='NCCL'):
        find_capacity([{}], admit=broken, upper_bound=8192, alignment=1024,
                      deadline=time.monotonic()+10)


def receipt(seconds):
    return dict(parents=8192, gpu_uuids=['a','b'], frontier_sha256=['x','y'],
        model_sha256='m', build_sha256='b', precision='fp16/fp32', executor='native',
        correctness_passed=True, seconds_by_rank=[seconds]*3)


def test_report_slowest_rank_and_loss_definition():
    report = matched_step_report(receipt([1,2]), receipt([2,4]))
    assert report['stream1_seconds']==2
    assert report['full_step_seconds']==4
    assert report['throughput_loss_fraction']==.5
    assert report['relative_time_overhead']==1


@pytest.mark.parametrize('key', ['parents','gpu_uuids','frontier_sha256','executor'])
def test_report_rejects_normalized_but_unmatched_workload(key):
    changed=deepcopy(receipt([2,4]));changed[key]='different'
    with pytest.raises(ValueError, match='unmatched'):
        matched_step_report(receipt([1,2]),changed)
