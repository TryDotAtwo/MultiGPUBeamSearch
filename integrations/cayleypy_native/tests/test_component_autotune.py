import math
import time
from types import SimpleNamespace
import numpy as np
from multigpubeamsearch.component_autotune import service_envelope,service_geometry
from multigpubeamsearch.calibration_frontier import symmetric_orbit_certificate,write_frontiers


def test_inference_uses_global_parent_normalization():
    row=dict(outer_candidates=64,shard_capacity=256,sort_jobs_concurrent=1,
        stream3_seconds=[.000001]*5,stream4_group_seconds=[.000001]*5,
        union_seconds=[.000001]*5,transport=[])
    plan=dict(frontier_state_capacity=1024,WORLD_SIZE=8,SHARD_COUNT=4,STREAM4_BATCH_CANDIDATES=256)
    result=service_envelope([row]*8,plan,moves=3,inference_seconds=.000001,parent_batch=64)
    assert result['inference_seconds']==1024*8*.000001
    assert result['service_envelope_seconds']>=result['inference_seconds']


def test_connected_conjugates_certificate_rejects_disconnected_edges():
    assert symmetric_orbit_certificate([[1,2,3,0],[1,0,2,3]],4)
    assert symmetric_orbit_certificate([[1,2,3,0],[2,1,0,3]],4) is None
    assert symmetric_orbit_certificate([[1,0,3,2]],4) is None


def test_enumeration_is_unique_reachable_padded_and_excludes_goal(tmp_path):
    generators=[[1,2,3,0],[3,0,1,2],[1,0,2,3]]
    graph=SimpleNamespace(state_len=4,center=[0,1,2,3],generators=generators,graph_hash='test',move_count=3)
    receipt=write_frontiers(graph,[5,7],16,tmp_path/'fixtures',deadline=time.monotonic()+5,max_states=12)
    data=np.vstack([np.fromfile(f['path'],dtype=np.uint8).reshape(-1,16) for f in receipt['files']])
    reachable={tuple(graph.center)};pending=[tuple(graph.center)]
    while pending:
        state=pending.pop()
        for move in generators:
            child=tuple(state[i] for i in move)
            if child not in reachable:reachable.add(child);pending.append(child)
    values=[tuple(r[:4]) for r in data]
    assert len(set(values))==12 and all(v in reachable for v in values)
    assert tuple(graph.center) not in values
    assert not data[:,4:].any()
    assert receipt['reachability_certificate']['proof'].startswith('connected')


def test_service_proposal_is_bounded_by_release_and_native_capacity():
    row=dict(stream4_group_seconds=[.01]*5,sort_jobs_concurrent=2,
        stream3_seconds=[.0002]*5,transport=[{'seconds':[.0003]*5}])
    plan=dict(WORLD_SIZE=2,SHARD_COUNT=4,frontier_state_capacity=1048576,
        SHARD_CAPACITY_CANDIDATES=400000,STREAM4_BATCH_ALIGNMENT=1024)
    result=service_geometry([row]*2,plan,moves=3,
        inference={'estimate':{'median':1e-7},'parent_batch':2048})
    assert 2<=int(result['BEAM_STREAM3_RING_SLOTS'])<=8
    assert int(result['BEAM_STREAM4_BATCH_CANDIDATES'])%1024==0
    assert int(result['BEAM_STREAM4_BATCH_CANDIDATES'])<=plan['SHARD_CAPACITY_CANDIDATES']
