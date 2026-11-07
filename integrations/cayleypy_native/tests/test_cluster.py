import pytest
from multigpubeamsearch.cluster import plan_cluster


@pytest.mark.parametrize('counts',[(1,),(8,),(8,)*16,(2,4,1)])
def test_rank_mapping_and_global_beam(counts):
    plan=plan_cluster(counts,256_000_000)
    assert [x.global_rank for x in plan.ranks]==list(range(sum(counts)))
    assert plan.effective_beam>=plan.requested_beam
    assert plan.effective_beam%plan.alignment==0
    assert plan.local_frontier_capacity*len(plan.ranks)==plan.effective_beam
    assert not plan.hardware_verified
    for rank in plan.ranks:assert rank.local_rank<counts[rank.node_rank]


def test_128_gpu_local_device_is_not_global_rank():
    plan=plan_cluster((8,)*16,268435456,batch_alignment=256)
    assert plan.ranks[127].local_rank==7
    assert plan.ranks[127].node_rank==15
    assert plan.ranks[127].global_rank==127


def test_overflow_and_invalid_counts():
    with pytest.raises(ValueError):plan_cluster((0,),1)
    with pytest.raises(ValueError):plan_cluster((8,)*16,2**64-1)
