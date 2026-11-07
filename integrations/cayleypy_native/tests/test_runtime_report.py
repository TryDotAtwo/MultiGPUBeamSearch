import pytest
from multigpubeamsearch.runtime_report import observed_depths


def test_only_full_complete_cohort_and_slowest_rank():
    text='[rank 0 stdout.log]\nruntime_depth_start=1 frontier_size=512\nruntime_depth_done=1 depth_sec=2 ring_slot_jobs=4 stream3_jobs=2 stream4_jobs=8\n[rank 1 stdout.log]\nruntime_depth_start=1 frontier_size=512\nruntime_depth_done=1 depth_sec=3 ring_slot_jobs=4 stream3_jobs=3 stream4_jobs=9\n'
    rows=observed_depths(text,2,1024,.001)
    assert rows[0]['full_step_seconds']==3
    assert rows[0]['stream1_reference_seconds']==1.024
    assert 'not matched' in rows[0]['measurement_scope']
    assert rows[0]['parents_per_second']==pytest.approx(1024/3)
    assert rows[0]['native_jobs_per_rank_sum']==dict(ring_slot_jobs=8,stream3_jobs=5,stream4_jobs=17)
    assert 'not individual CUDA kernel launches' in rows[0]['native_job_count_scope']
    assert not observed_depths(text,3,1024,.001)
    assert not observed_depths(text,2,2048,.001)


def test_duplicate_completion_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        observed_depths('runtime_depth_done=0 depth_sec=1\nruntime_depth_done=0 depth_sec=1',1,1024,.001)


def test_invalid_job_counter_is_rejected():
    text='[rank 0 stdout.log]\nruntime_depth_start=0 frontier_size=1024\nruntime_depth_done=0 depth_sec=1 stream3_jobs=-1\n'
    with pytest.raises(ValueError,match='job count'):
        observed_depths(text,1,1024,.001)


def test_old_logs_mark_native_job_counts_unavailable():
    rows=observed_depths('[rank 0 stdout.log]\nruntime_depth_start=0 frontier_size=1024\nruntime_depth_done=0 depth_sec=1\n',1,1024,.001)
    assert not rows[0]['native_jobs_per_rank_sum']
    assert 'unavailable' in rows[0]['native_job_count_scope']
