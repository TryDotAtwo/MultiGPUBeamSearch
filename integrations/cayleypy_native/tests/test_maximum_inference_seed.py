import pytest
from multigpubeamsearch.maximum_beam import capacity_seed, refine_maximum
from multigpubeamsearch.beam_capacity import CapacityRejected


def cohort(batch, reserve, seconds):
    return [dict(batch=batch,device=rank,parents=8192,
        seconds=[seconds]*7,torch_reserved_peak_bytes=reserve,
        correctness_passed=True,numeric_error=0) for rank in range(2)]


def test_maximum_frontier_reserves_fastest_verified_batch_first():
    calibration=dict(signature=dict(world_size=2,max_batch=8192),
        parent_batch=8192,estimate=dict(median=0.5/16384),
        records=cohort(8192,8<<30,.5)+cohort(256,400<<20,.6))
    seed=capacity_seed(calibration)
    assert seed['parent_batch']==8192
    assert seed['reserve_bytes']==(8<<30)+(512<<20)
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


def test_faster_exact_width_batch_shrinks_frontier_then_recalibrates():
    calls=[]
    def tune(width, index):
        calls.append(('tune', width))
        return dict(signature={'requested_beam_width':width},
                    parent_batch=8192, reserve_bytes=8<<30)
    def discover(chosen, index):
        calls.append(('discover', chosen['parent_batch']))
        return {'effective_beam':60}
    capacity, chosen=refine_maximum({'effective_beam':100},
        {'parent_batch':32,'reserve_bytes':700<<20},tune=tune,discover=discover)
    assert calls==[('tune',100),('discover',8192),('tune',60)]
    assert capacity['effective_beam']==60 and chosen['parent_batch']==8192
    assert chosen['signature']['requested_beam_width']==60


def test_thermal_invalidated_winner_cannot_expand_frontier():
    discoveries=[]
    def tune(width,index):
        return dict(signature={'requested_beam_width':width},parent_batch=256,
            reserve_bytes=1<<30,rejected={'8192':'GPU telemetry missing, incomplete or throttled'})
    with pytest.raises(RuntimeError,match='cannot confirm inference winner'):
        refine_maximum({'effective_beam':5_000_000},
            {'parent_batch':8192,'reserve_bytes':9<<30},tune=tune,
            discover=lambda *args:discoveries.append(args))
    assert discoveries==[]
