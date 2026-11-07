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


def test_transport_plateau_does_not_hide_extra_sort_work():
    row=dict(outer_candidates=1000,shard_capacity=4000,sort_jobs_concurrent=1,
        stream3_seconds=[.001]*5,stream4_group_seconds=[.1]*5,union_seconds=[.01]*5,
        transport=[dict(items=3000,seconds=[1.0]*5)])
    plan=dict(frontier_state_capacity=1000,WORLD_SIZE=2,SHARD_COUNT=1,STREAM4_BATCH_CANDIDATES=500)
    frequent=service_envelope([row]*2,plan,moves=3,inference_seconds=.0001,parent_batch=100)
    bulk=service_envelope([row]*2,dict(plan,STREAM4_BATCH_CANDIDATES=3000),moves=3,
        inference_seconds=.0001,parent_batch=100)
    assert frequent['service_envelope_seconds']==bulk['service_envelope_seconds']
    assert bulk['service_work_seconds']<frequent['service_work_seconds']


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


def test_outer_and_sort_candidates_keep_exact_frontier_and_inference(monkeypatch):
    from multigpubeamsearch.component_autotune import tune_components
    plan=dict(frontier_state_capacity=8192,GLOBAL_BEAM_WIDTH_EFFECTIVE=16384,
        WORLD_SIZE=2,SHARD_COUNT=4,SHARD_CAPACITY_CANDIDATES=8192,
        STREAM4_BATCH_CANDIDATES=1024,STREAM4_BATCH_ALIGNMENT=1024)
    baseline=dict(BEAM_B_MICRO='256',BEAM_ENSEMBLE_INFERENCE_MICRO='256',
        BEAM_SHARD_COUNT='4',BEAM_STREAM3_RING_SLOTS='4',BEAM_STREAM4_ACTIVE_SORT_SLOTS='1',
        BEAM_STREAM4_BATCH_CANDIDATES='1024')
    requested=[]
    class Planner:
        supports_components=True
        def admit(self,beam,env):
            requested.append((beam,env));return [dict(plan,STREAM4_BATCH_CANDIDATES=int(env['BEAM_STREAM4_BATCH_CANDIDATES']))]*2
    class Probe:
        deadline=time.monotonic()+10;beam=16384
        def measure_components(self,env,plans,move_count):
            return [dict(outer_candidates=int(env['BEAM_B_MICRO'])*int(env['BEAM_STREAM3_RING_SLOTS'])*move_count,
                shard_capacity=8192,sort_jobs_concurrent=int(env['BEAM_STREAM4_ACTIVE_SORT_SLOTS']),
                stream3_seconds=[.00001]*5,stream4_group_seconds=[.00001]*5,
                union_seconds=[.0001]*5,transport=[])]*2
    result=tune_components(Probe(),Planner(),[plan]*2,baseline,[],moves=3,
        inference={'estimate':{'median':1e-7},'parent_batch':256})
    assert any(env['BEAM_B_MICRO']=='512' for _,env in requested)
    assert any(env['BEAM_STREAM4_ACTIVE_SORT_SLOTS']=='2' for _,env in requested)
    assert any(env['BEAM_STREAM4_BATCH_CANDIDATES']=='2048' for _,env in requested)
    outer=next(env for _,env in requested if env['BEAM_B_MICRO']=='512')
    assert outer['BEAM_STREAM3_RING_SLOTS']=='2'
    assert outer['BEAM_STREAM4_BATCH_CANDIDATES']==baseline['BEAM_STREAM4_BATCH_CANDIDATES']
    lanes=next(env for _,env in requested if env['BEAM_STREAM4_ACTIVE_SORT_SLOTS']=='2')
    assert lanes['BEAM_STREAM3_RING_SLOTS']==baseline['BEAM_STREAM3_RING_SLOTS']
    assert lanes['BEAM_STREAM4_BATCH_CANDIDATES']==baseline['BEAM_STREAM4_BATCH_CANDIDATES']
    assert all(beam==16384 and env['BEAM_ENSEMBLE_INFERENCE_MICRO']=='256' for beam,env in requested)
    assert result['workload_parents']==16384


def test_frozen_expensive_inference_cannot_mask_downstream_improvement():
    from multigpubeamsearch.component_autotune import tune_components
    plan=dict(frontier_state_capacity=8192,GLOBAL_BEAM_WIDTH_EFFECTIVE=16384,
        WORLD_SIZE=2,SHARD_COUNT=4,SHARD_CAPACITY_CANDIDATES=8192,
        STREAM4_BATCH_CANDIDATES=1024,STREAM4_BATCH_ALIGNMENT=1024)
    baseline=dict(BEAM_B_MICRO='256',BEAM_ENSEMBLE_INFERENCE_MICRO='256',
        BEAM_SHARD_COUNT='4',BEAM_STREAM3_RING_SLOTS='4',BEAM_STREAM4_ACTIVE_SORT_SLOTS='1',
        BEAM_STREAM4_BATCH_CANDIDATES='1024')
    class Planner:
        supports_components=True
        def admit(self,beam,env):
            assert beam==16384 and env['BEAM_ENSEMBLE_INFERENCE_MICRO']=='256'
            return [dict(plan,STREAM4_BATCH_CANDIDATES=int(env['BEAM_STREAM4_BATCH_CANDIDATES']))]*2
    class Probe:
        deadline=time.monotonic()+10;beam=16384
        def measure_components(self,env,plans,move_count):
            return [dict(outer_candidates=int(env['BEAM_B_MICRO'])*int(env['BEAM_STREAM3_RING_SLOTS'])*move_count,
                shard_capacity=8192,sort_jobs_concurrent=int(env['BEAM_STREAM4_ACTIVE_SORT_SLOTS']),
                stream3_seconds=[.00001]*5,stream4_group_seconds=[.00001]*5,
                union_seconds=[.0001]*5,transport=[])]*2
    results=[]
    for inference_seconds in (1e-7,1.0):
        results.append(tune_components(Probe(),Planner(),[plan]*2,baseline,[],moves=3,
            inference={'estimate':{'median':inference_seconds},'parent_batch':256}))
    assert results[0]['selection']==results[1]['selection']!='baseline'
    for result in results:
        base=next(r for r in result['tested'] if r['name']=='baseline')
        assert result['estimate']['downstream_service_work_seconds']<base['estimate']['downstream_service_work_seconds']
        assert result['environment']['BEAM_ENSEMBLE_INFERENCE_MICRO']=='256'
        assert result['pipeline_verified'] is False
