"""Pure rank/layout planning; hardware acceptance is intentionally separate."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RankPlacement:
    global_rank: int
    node_rank: int
    local_rank: int


@dataclass(frozen=True)
class ClusterPlan:
    ranks: tuple[RankPlacement, ...]
    requested_beam: int
    effective_beam: int
    alignment: int
    local_frontier_capacity: int
    shard_count: int
    hardware_verified: bool = False


def plan_cluster(gpus_per_node, beam_width, *, shard_count=16, batch_alignment=1024):
    """Plan arbitrary rank counts without substituting local CUDA device IDs.

    This is layout arithmetic, not memory admission or a cluster launcher.
    Every native rank must still pass its own exact memory plan, and all ranks
    must agree on the selected shard count and effective global beam.
    """
    counts=tuple(gpus_per_node)
    if not counts or any(type(n) is not int or n<=0 for n in counts):
        raise ValueError('each node must have a positive GPU count')
    for name,value in [('beam_width',beam_width),('shard_count',shard_count),('batch_alignment',batch_alignment)]:
        if type(value) is not int or value<=0:raise ValueError(name+' must be positive')
    world=sum(counts)
    if world>2**31-1:raise ValueError('world size exceeds NCCL signed rank range')
    alignment=world*shard_count*batch_alignment
    effective=((beam_width+alignment-1)//alignment)*alignment
    if effective>=2**64:raise ValueError('aligned beam exceeds uint64')
    ranks=[]
    for node,count in enumerate(counts):
        for local in range(count):ranks.append(RankPlacement(len(ranks),node,local))
    return ClusterPlan(tuple(ranks),beam_width,effective,alignment,effective//world,shard_count)
