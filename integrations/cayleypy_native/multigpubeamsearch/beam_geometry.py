"""Cheap geometry pruning, not native memory admission or a timing predictor.

Equations follow runtime_config.cpp and static_memory.cu. Named allocations
give a LOWER bound; CUB, models, activations, partition/finalization scratch,
alignment and driver reserves must still be supplied by the native planner.
native_b_micro is the native configuration value, NOT necessarily the model
parent batch (scalar MLP's native value includes child rows).
"""
from dataclasses import asdict,dataclass
import math

def ceil_div(a,b):
    return (a+b-1)//b

def round_up(a,b):
    return ceil_div(a,b)*b

@dataclass(frozen=True)
class Shape:
    requested_beam:int
    gpus:int
    native_b_micro:int
    moves:int
    state_bytes:int
    alignment:int=1024
    capacity_ppm:int=1_250_000
    receive_ppm:int=2_000_000

    def __post_init__(self):
        for name,value in asdict(self).items():
            if type(value) is not int or value<=0:raise ValueError(name+' must be a positive integer')
        if self.requested_beam>=2**64 or self.gpus>=2**31:raise ValueError('native integer range exceeded')

def ring_pool_limit(environment):
    """Native eager pool policy; this remains a pruning hint, not admission."""
    value=environment.get('BEAM_RING_COUNT_LIMIT')
    if value is None:
        return 16 if (environment.get('BEAM_STREAM1_EXECUTOR')=='libtorch_eager'
                      or environment.get('BEAM_BLEND_DIR')) else None
    try:limit=int(value)
    except (TypeError,ValueError):raise ValueError('invalid BEAM_RING_COUNT_LIMIT') from None
    if not 1<=limit<2**32:raise ValueError('invalid BEAM_RING_COUNT_LIMIT')
    return limit


def geometry(shape,shards,staging_slots=2,sort_slots=1,*,physical_ring_limit=None):
    for value in (shards,staging_slots,sort_slots):
        if type(value) is not int or value<=0:raise ValueError('positive geometry required')
    effective=round_up(shape.requested_beam,shape.gpus*shards*shape.alignment)
    local=ceil_div(effective,shape.gpus)
    logical=ceil_div(local,shards)
    q=staging_slots*shape.native_b_micro*shape.moves
    rings=ceil_div(logical*staging_slots,q)
    if physical_ring_limit is not None:
        if type(physical_ring_limit) is not int or not 1<=physical_ring_limit<2**32:
            raise ValueError('physical_ring_limit must be a positive uint32')
        rings=min(rings,physical_ring_limit)
    recv=max(q,ceil_div(q*shape.receive_ppm,1_000_000))
    capacity=round_up(max(logical+max(q,recv),
        ceil_div(logical*shape.capacity_ppm,1_000_000)),shape.alignment)
    if effective>=2**64 or q>=2**32 or rings>=2**32 or capacity>=(2**31-1)//2:
        raise ValueError('native/CUB integer range exceeded')
    # Static-memory phase1 prefix; world1 still allocates one send/recv slot.
    network_slots=rings if shape.gpus>1 else 1
    prefix_ring_network=20*rings*q+32*network_slots*(q+recv)
    prefix_sort=124*sort_slots*capacity
    prefix_union=248*capacity
    current_frontier=local*shape.state_bytes
    survivors=64*shards*capacity # two A/B physical copies, CandidateMeta32
    histograms=16*shards*307201 # two uint32 bins on each A/B physical shard
    # Prefix/final layouts overlay; persistent survivors/histograms follow max.
    scratch_named=max(prefix_ring_network+prefix_sort,prefix_union,current_frontier)
    lower=current_frontier+scratch_named+survivors+histograms
    return dict(requested_beam=shape.requested_beam,effective_beam=effective,
        padding=effective-shape.requested_beam,local_parents=local,shards=shards,
        staging_slots=staging_slots,sort_slots=sort_slots,transaction_candidates=q,
        logical_shard=logical,physical_shard_capacity=capacity,ring_count=rings,
        physical_stream1_slots=rings*staging_slots,
        receive_candidates_per_slot=recv,current_frontier_bytes=current_frontier,
        ring_network_bytes=prefix_ring_network,sort_named_bytes=prefix_sort,
        union_named_bytes=prefix_union,survivor_bytes=survivors,
        histogram_bytes=histograms,named_memory_lower_bound_bytes=lower,
        native_admission_verified=False,performance_verified=False)

def memory_shortlist(shape,*,staging_slots=2,sort_slots=1,limit=3,
                     effective_beam=None,budget_bytes=None,physical_ring_limit=None):
    """Cheap memory seed only. Never accept a row as native-admitted/fastest.

    If geometry is not frozen yet, choose a memory seed then restrict all
    comparators to that seed's exact effective frontier. Requested beam is
    retained. Reject ONLY known-impossible rows from the partial memory bound.
    """
    if type(limit) is not int or not 1<=limit<=3:raise ValueError('limit must be1..3')
    rows=[]
    for s in range(1,129):
        try:rows.append(geometry(shape,s,staging_slots,sort_slots,
                                 physical_ring_limit=physical_ring_limit))
        except ValueError:continue
    if budget_bytes is not None:
        rows=[r for r in rows if r['named_memory_lower_bound_bytes']<=budget_bytes]
    if effective_beam is not None:
        rows=[r for r in rows if r['effective_beam']==effective_beam]
    rows.sort(key=lambda r:(r['named_memory_lower_bound_bytes'],r['padding'],r['shards']))
    if not rows:return []
    frozen=rows[0]['effective_beam'] if effective_beam is None else effective_beam
    seed_bound=rows[0]['named_memory_lower_bound_bytes']
    # Only a memory-oriented seed neighborhood. Performance probes may add a
    # larger-memory candidate if the measured service curves justify it.
    return [r for r in rows if r['effective_beam']==frozen and
            r['named_memory_lower_bound_bytes']*5<=seed_bound*6][:limit]

def linear_service_batch(arrival_per_second,launch_seconds,seconds_per_item,
                         *,utilization=.8,alignment=1024):
    """Amortization bound for a measured LINEAR stage, e.g. transport.

    Queue stability with20% margin: lambda*(alpha/Q+beta)<=utilization.
    This is not a MergeSort law or measured full-pipeline overlap throughput.
    """
    vals=(arrival_per_second,launch_seconds,seconds_per_item,utilization)
    if not all(math.isfinite(v) for v in vals):raise ValueError('finite service inputs required')
    if arrival_per_second<=0 or launch_seconds<0 or seconds_per_item<0 or not 0<utilization<1:
        raise ValueError('invalid service inputs')
    if type(alignment) is not int or alignment<=0:raise ValueError('positive alignment required')
    slack=utilization-arrival_per_second*seconds_per_item
    if slack<=0:raise ValueError('consumer throughput bottleneck; larger buffer cannot fix it')
    return round_up(max(1,math.ceil(arrival_per_second*launch_seconds/slack)),alignment)

def staging_from_release_latency(release_p90_seconds,producer_batch_seconds):
    if not all(math.isfinite(x) and x>0 for x in (release_p90_seconds,producer_batch_seconds)):
        raise ValueError('positive measured durations required')
    return max(2,1+math.ceil(release_p90_seconds/producer_batch_seconds))

