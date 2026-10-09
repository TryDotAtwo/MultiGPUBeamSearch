import pytest
from multigpubeamsearch.maximum_beam import capacity_seed
from multigpubeamsearch.beam_capacity import CapacityRejected


def cohort(batch, reserve, seconds):
    return [dict(batch=batch,device=rank,parents=8192,
        seconds=[seconds]*7,torch_reserved_peak_bytes=reserve,
        correctness_passed=True,numeric_error=0) for rank in range(2)]


def test_maximum_width_does_not_bootstrap_with_fastest_largest_cache():
    calibration=dict(signature=dict(world_size=2,max_batch=8192),
        parent_batch=8192,estimate=dict(median=0.5/16384),
        records=cohort(8192,8<<30,.5)+cohort(256,400<<20,.6))
    seed=capacity_seed(calibration)
    assert seed['parent_batch']==256
    assert seed['reserve_bytes']==(400<<20)+(512<<20)
    assert calibration['parent_batch']==8192
    assert seed['capacity_bootstrap']['unconstrained_batch']==8192


def test_incomplete_small_batch_cannot_seed_capacity():
    rows=cohort(32,100<<20,1)[:1]+cohort(256,400<<20,.6)
    seed=capacity_seed(dict(signature=dict(world_size=2),parent_batch=256,
        estimate={},records=rows))
    assert seed['parent_batch']==256


def test_capacity_requires_a_verified_cohort():
    with pytest.raises(CapacityRejected):
        capacity_seed(dict(signature=dict(world_size=2),parent_batch=256,
            estimate={},records=cohort(256,0,.6)[:1]))

