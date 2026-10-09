import pytest
from multigpubeamsearch.pipeline_probe import parse_plan, parse_depths


def test_native_plan_rejects_conflicting_values():
    assert parse_plan('GLOBAL_BEAM_WIDTH_EFFECTIVE=8192\nB_MICRO=32\n')['B_MICRO']==32
    with pytest.raises(ValueError,match='changes'):
        parse_plan('GLOBAL_BEAM_WIDTH_EFFECTIVE=8192\nB_MICRO=32\nB_MICRO=64')


def test_depth_measurements_require_exact_workload_and_completion():
    text='\n'.join(f'calibration_depth={i} rank=0 parents=4096 depth_sec=1.25 next_frontier_size=4096' for i in range(6))
    assert parse_depths(text,4096,6)==[1.25]*6
    with pytest.raises(ValueError,match='workload'):
        parse_depths(text,8192,6)
    with pytest.raises(ValueError,match='all calibration'):
        parse_depths(text,4096,7)
    with pytest.raises(ValueError,match='reordered'):
        parse_depths(text+'\n'+text.splitlines()[0],4096,7)


def test_nonfinite_depth_time_is_never_a_performance_sample():
    with pytest.raises(ValueError,match='timing'):
        parse_depths('calibration_depth=0 rank=0 parents=4096 depth_sec=nan',4096,1)

