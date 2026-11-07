import pytest
from multigpubeamsearch.beam_geometry import Shape,geometry,memory_shortlist,ring_pool_limit


def test_large_frontier_pool_bound_keeps_logical_capacity():
    shape=Shape(55_050_240,2,64,24,112)
    legacy=geometry(shape,3,2,1)
    bounded=geometry(shape,3,2,1,physical_ring_limit=16)
    assert legacy['ring_count']>5000 and bounded['ring_count']==16
    for field in ('effective_beam','local_parents','logical_shard','physical_shard_capacity',
                  'transaction_candidates','survivor_bytes','histogram_bytes'):
        assert bounded[field]==legacy[field]
    assert bounded['named_memory_lower_bound_bytes']<legacy['named_memory_lower_bound_bytes']
    budget=bounded['named_memory_lower_bound_bytes']
    rows=memory_shortlist(shape,staging_slots=2,sort_slots=1,
        effective_beam=bounded['effective_beam'],budget_bytes=budget,physical_ring_limit=16)
    assert any(row['shards']==3 for row in rows)
    assert not any(row['shards']==3 for row in memory_shortlist(shape,staging_slots=2,
        sort_slots=1,effective_beam=bounded['effective_beam'],budget_bytes=budget))


def test_pool_policy_and_explicit_override():
    assert ring_pool_limit({}) is None
    assert ring_pool_limit({'BEAM_STREAM1_EXECUTOR':'libtorch_eager'})==16
    assert ring_pool_limit({'BEAM_BLEND_DIR':'/weights'})==16
    assert ring_pool_limit({'BEAM_STREAM1_EXECUTOR':'libtorch_eager','BEAM_RING_COUNT_LIMIT':'1'})==1
    for value in ('0','-1','4294967296','oops'):
        with pytest.raises(ValueError):ring_pool_limit({'BEAM_RING_COUNT_LIMIT':value})
