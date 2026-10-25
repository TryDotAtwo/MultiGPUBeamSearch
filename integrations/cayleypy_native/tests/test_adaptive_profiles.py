from multigpubeamsearch.adaptive_profiles import proposals, key
from multigpubeamsearch.beam_geometry import Shape,geometry,memory_shortlist


def fixture(beam=1_000_001,sm=48,budget=12*1024**3):
    shape=Shape(beam,2,256,30,160)
    env=dict(BEAM_B_MICRO='256',BEAM_ENSEMBLE_INFERENCE_MICRO='256',
        BEAM_SHARD_COUNT='8',BEAM_STREAM3_RING_SLOTS='4',
        BEAM_STREAM4_ACTIVE_SORT_SLOTS='1',BEAM_STREAM4_BATCH_CANDIDATES='4096')
    context=dict(budget_bytes=budget,hardware=[dict(sm_count=sm,l2_bytes=4*1024**2)])
    return shape,env,context


def test_arbitrary_shards_preserve_exact_width():
    shape,_,_=fixture()
    for shards in (1,3,7,128):
        row=geometry(shape,shards,physical_ring_limit=16)
        assert row['effective_beam']==shape.requested_beam
        assert row['current_frontier_bytes']==500001*160
    assert memory_shortlist(shape,effective_beam=shape.requested_beam,physical_ring_limit=16)


def test_hardware_width_and_budget_change_admissible_domain():
    shape,env,ctx=fixture()
    rows,trace=proposals(shape,env,ctx,ring_limit=16)
    small,small_trace=proposals(*fixture(beam=4097),ring_limit=16)
    fast,fast_trace=proposals(*fixture(sm=132),ring_limit=16)
    low,low_trace=proposals(*fixture(budget=100*1024**2),ring_limit=16)
    assert trace['domain']!=small_trace['domain']!=fast_trace['domain']
    assert {key(e) for _,e in rows}!={key(e) for _,e in small}
    assert {key(e) for _,e in rows}!={key(e) for _,e in fast}
    assert len(low)<len(rows) and low_trace['pruned']
    assert all(e['BEAM_ENSEMBLE_INFERENCE_MICRO']=='256' for _,e in rows)
    assert len({key(e) for _,e in rows})==len(rows)<=16


def test_refinement_follows_measured_center_and_deduplicates():
    shape,env,ctx=fixture()
    first,_=proposals(shape,env,ctx,fine=True,ring_limit=16)
    winner=dict(env,BEAM_SHARD_COUNT='16',BEAM_STREAM4_BATCH_CANDIDATES='8192')
    second,_=proposals(shape,winner,ctx,fine=True,ring_limit=16,seen={key(e) for _,e in first})
    assert second and all(key(e) not in {key(x) for _,x in first} for _,e in second)
    assert any(int(e['BEAM_SHARD_COUNT'])>16 for _,e in second)

def test_scalar_row_budget_does_not_inflate_geometry():
    shape,env,ctx=fixture()
    base,_=proposals(shape,env,ctx,ring_limit=16)
    scalar_env=dict(env,BEAM_B_MICRO=str(256*shape.moves))
    scalar_ctx=dict(ctx,outer_rows_per_parent=shape.moves)
    scalar,_=proposals(shape,scalar_env,scalar_ctx,ring_limit=16)
    normalized=[(n,dict(e,BEAM_B_MICRO=str(int(e['BEAM_B_MICRO'])//shape.moves))) for n,e in scalar]
    assert {key(e) for _,e in normalized}=={key(e) for _,e in base}
