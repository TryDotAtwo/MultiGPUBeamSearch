import pytest
from size_model import Shape,geometry,memory_shortlist,linear_service_batch,staging_from_release_latency

def test_real_cube_1m_geometry_matches_source_integer_equations():
    row=geometry(Shape(1048576,2,1024,24,112,capacity_ppm=4_000_000),4,8)
    assert row['local_parents']==524288 and row['logical_shard']==131072
    assert row['transaction_candidates']==196608 and row['ring_count']==6
    assert row['receive_candidates_per_slot']==393216
    assert row['physical_shard_capacity']==524288
    assert row['physical_stream1_slots']==48

@pytest.mark.parametrize('beam',[1,8192,10_000_000,100_000_000,256_000_000])
def test_shortlist_preserves_exact_effective_frontier(beam):
    rows=memory_shortlist(Shape(beam,8,1024,24,112))
    assert 1<=len(rows)<=3
    assert len({r['effective_beam'] for r in rows})==1
    assert all(r['requested_beam']==beam and r['effective_beam']>=beam for r in rows)
    assert not any(r['native_admission_verified'] or r['performance_verified'] for r in rows)

def test_larger_staging_changes_writable_reserve_floor():
    shape=Shape(1048576,2,1024,24,112)
    small=geometry(shape,64,2);large=geometry(shape,64,8)
    assert large['physical_shard_capacity']>small['physical_shard_capacity']
    assert large['survivor_bytes']>small['survivor_bytes']

def test_histogram_cost_is_per_shard_and_not_per_frontier():
    shape=Shape(1048576,2,1024,24,112)
    assert geometry(shape,16)['histogram_bytes']==16*16*307201
    assert geometry(shape,32)['histogram_bytes']==2*geometry(shape,16)['histogram_bytes']

def test_known_impossible_lower_bound_can_reject_but_never_admit():
    assert memory_shortlist(Shape(256_000_000,8,1024,24,112),budget_bytes=1<<20)==[]

def test_service_rate_equation_and_bottleneck_rejection():
    q=linear_service_batch(1e6,1e-4,1e-7)
    assert 1e6*(1e-4/q+1e-7)<=.8
    with pytest.raises(ValueError,match='bottleneck'):linear_service_batch(1e6,1e-4,1e-6)
    assert staging_from_release_latency(.003,.001)==4

def test_frozen_effective_width_is_never_changed_to_get_more_candidates():
    rows=memory_shortlist(Shape(100_000_000,8,1024,24,112),effective_beam=100007936)
    assert rows and all(r['effective_beam']==100007936 for r in rows)

def test_native_integer_limits_are_not_silently_accepted():
    with pytest.raises(ValueError):Shape(2**64,8,1024,24,112)
    assert memory_shortlist(Shape(2**63,1,1024,24,112))==[]
