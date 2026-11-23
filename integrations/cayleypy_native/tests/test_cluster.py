import pytest
from multigpubeamsearch.cluster import plan_cluster, scheduler_rank


@pytest.mark.parametrize('counts',[(1,),(8,),(8,)*16,(2,4,1)])
def test_rank_mapping_and_global_beam(counts):
    plan=plan_cluster(counts,256_000_000)
    assert [x.global_rank for x in plan.ranks]==list(range(sum(counts)))
    assert plan.effective_beam==256_000_000
    assert plan.local_frontier_capacity==(256_000_000+len(plan.ranks)-1)//len(plan.ranks)
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
    with pytest.raises(ValueError, match='128'):plan_cluster((8,)*16+(1,),1)


def test_exact_top_k_and_physical_storage_for_two_nodes():
    plan = plan_cluster((8, 8), 1_000_003)
    assert plan.effective_beam == 1_000_003
    assert plan.local_frontier_capacity == 62_501
    assert plan.alignment == 262_144


@pytest.mark.parametrize(("rank", "local", "node"), [
    ("11", "4", "1"),
    ("13", "4", "1"),
    ("4", "4", "1"),
])
def test_slurm_global_rank_matches_node_and_local_assignment(rank, local, node):
    env = {
        "SLURM_PROCID": rank,
        "SLURM_LOCALID": local,
        "SLURM_NODEID": node,
        "SLURM_STEP_NUM_TASKS": "14",
        "SLURM_STEP_TASKS_PER_NODE": "8,6",
    }
    with pytest.raises(ValueError, match="rank.*node/local"):
        scheduler_rank(env)



def test_slurm_global_rank_accepts_valid_node_local_mapping():
    rank = scheduler_rank({
        "SLURM_PROCID": "12",
        "SLURM_LOCALID": "4",
        "SLURM_NODEID": "1",
        "SLURM_STEP_NUM_TASKS": "14",
        "SLURM_STEP_TASKS_PER_NODE": "8,6",
    })
    assert (rank.global_rank, rank.node_rank, rank.local_rank, rank.world_size, rank.local_world_size) == (
        12, 1, 4, 14, 6
    )

def test_public_cluster_plan_uses_rust_and_keeps_old_wheel_fallback(monkeypatch):
    from multigpubeamsearch import cluster as cluster_module
    from multigpubeamsearch import rust_control

    request = {
        "gpus_per_node": [2, 4, 1],
        "beam_width": 1_000_003,
        "shard_count": 16,
        "batch_alignment": 1024,
    }
    observed = []
    native_resolver = rust_control.resolve_cluster_plan

    def observe(value):
        observed.append(value)
        return native_resolver(value)

    monkeypatch.setattr(rust_control, "resolve_cluster_plan", observe)
    fallback = cluster_module._resolve_cluster_plan_python

    def forbidden_fallback(*args, **kwargs):
        pytest.fail("installed Rust cluster planner fell back to Python")

    monkeypatch.setattr(cluster_module, "_resolve_cluster_plan_python", forbidden_fallback)
    plan = cluster_module.plan_cluster((2, 4, 1), 1_000_003)
    assert observed == [request]
    assert plan.ranks[0].global_rank == 0
    assert plan.ranks[1].node_rank == 0 and plan.ranks[1].local_rank == 1
    assert plan.ranks[2].node_rank == 1 and plan.ranks[2].local_rank == 0
    assert (plan.alignment, plan.local_frontier_capacity) == (114_688, 142_858)
    assert plan.effective_beam == plan.requested_beam == 1_000_003
    assert not plan.hardware_verified

    monkeypatch.setattr(rust_control, "resolve_cluster_plan", lambda _: None)
    monkeypatch.setattr(cluster_module, "_resolve_cluster_plan_python", fallback)
    assert cluster_module.plan_cluster((2, 4, 1), 1_000_003) == plan
