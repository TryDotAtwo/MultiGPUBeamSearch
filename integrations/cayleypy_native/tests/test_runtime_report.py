import pytest
from multigpubeamsearch.runtime_report import observed_depths


def test_only_full_complete_cohort_and_slowest_rank():
    text='[rank 0 stdout.log]\ndepth_start=1 frontier_size=512\ndepth_done=1 depth_sec=2\n[rank 1 stdout.log]\ndepth_start=1 frontier_size=512\ndepth_done=1 depth_sec=3\n'
    rows=observed_depths(text,2,1024,.001)
    assert rows[0]['full_step_seconds']==3
    assert rows[0]['stream1_reference_seconds']==1.024
    assert 'not matched' in rows[0]['measurement_scope']
    assert not observed_depths(text,3,1024,.001)
    assert not observed_depths(text,2,2048,.001)


def test_duplicate_completion_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        observed_depths('depth_done=0 depth_sec=1\ndepth_done=0 depth_sec=1',1,1024,.001)
