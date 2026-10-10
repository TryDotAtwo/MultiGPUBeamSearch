import copy
import json

from multigpubeamsearch.pipeline_cache import read_profile,valid_receipt,write_profile,cache_identity


def test_identity_ignores_run_files_but_keeps_capacity_policy(monkeypatch):
    from multigpubeamsearch import pipeline_cache
    monkeypatch.setattr(pipeline_cache,'valid_cached_profile',lambda *args:True)
    inference=dict(signature=dict(world_size=2,requested_beam_width=1024),parent_batch=256,reserve_bytes=1000)
    env=dict(BEAM_TEST_CSV='/run/a.csv',BEAM_PUZZLE_INFO_JSON='/run/a.json',
        BEAM_NCCL_ID_FILE='/run/id',BEAM_NCCL_RUN_ID='a',BEAM_SHARD_CAPACITY_SCALE_PPM='1250000')
    first=cache_identity(inference,env,beam=1024,world=2,full_frontier=False)
    other=dict(env,BEAM_TEST_CSV='/run/b.csv',BEAM_PUZZLE_INFO_JSON='/run/b.json',
        BEAM_NCCL_ID_FILE='/run/bid',BEAM_NCCL_RUN_ID='b')
    assert cache_identity(inference,other,beam=1024,world=2,full_frontier=False)==first
    assert cache_identity(inference,other,beam=2048,world=2,full_frontier=False) is None
    other['BEAM_SHARD_CAPACITY_SCALE_PPM']='1500000'
    assert cache_identity(inference,other,beam=1024,world=2,full_frontier=False)!=first


def receipt():
    identity=dict(parent_batch=256,signature=dict(world_size=2,requested_beam_width=1024))
    env={'BEAM_B_MICRO':'512','BEAM_ENSEMBLE_INFERENCE_MICRO':'256'}
    plans=[dict(WORLD_SIZE=2,LOCAL_RANK=r,GLOBAL_BEAM_WIDTH_EFFECTIVE=1024,
        frontier_state_capacity=512,BEAM_WIDTH_ALIGNMENT=512,SHARD_COUNT=4,B_MICRO=512,
        STREAM4_BATCH_ALIGNMENT=1024,SHARD_CAPACITY_CANDIDATES=2048,
        STREAM3_RING_SLOTS=2,RING_COUNT=16,STREAM4_ACTIVE_SORT_SLOTS=1,STREAM4_BATCH_CANDIDATES=1024,
        estimated_required_device_bytes=100,gpu_budget_bytes=200) for r in range(2)]
    rows=[dict(rank=r,correctness_passed=True,full_step_verified=False,shard_capacity=2048,
        stream3_seconds=[.01]*5,stream4_group_seconds=[.02]*5,union_seconds=[.03]*5,
        transport=[dict(items=1024,seconds=[.04]*5)]) for r in range(2)]
    data=dict(cache_identity=identity,phase='component_calibrated',pipeline_verified=False,
        environment=env,requested_beam_width=1024,requested_beam_effective=1024,
        selection='chosen',tested=[dict(name='chosen',environment=env,plans=plans,rank_measurements=rows)])
    return identity,data,plans


def test_cached_component_requires_current_all_rank_admission(tmp_path):
    identity,data,plans=receipt();path=tmp_path/'cache.json'
    write_profile(path,identity,data);calls=[]
    def admit(env):
        calls.append(env);fresh=copy.deepcopy(plans)
        for p in fresh:p['gpu_budget_bytes']=150
        return fresh
    result=read_profile(path,identity,admit)
    assert len(calls)==1 and result['cache_hit'] and result['pipeline_verified'] is False
    assert result['current_rank_plans'][0]['gpu_budget_bytes']==150
    def rejected(env):raise ValueError('current VRAM cannot admit')
    assert read_profile(path,identity,rejected) is None


def test_corrupt_cohort_and_environment_are_cache_misses(tmp_path):
    identity,data,plans=receipt();path=tmp_path/'cache.json'
    for mutation in (
        lambda x:x['environment'].update(LD_PRELOAD='/bad'),
        lambda x:x['environment'].update(BEAM_B_MICRO='512;false'),
        lambda x:x['environment'].update(BEAM_ENSEMBLE_INFERENCE_MICRO='128'),
        lambda x:x['tested'][0]['rank_measurements'].pop(),
        lambda x:x['tested'][0]['rank_measurements'][1].update(rank=0),
        lambda x:x['tested'][0]['rank_measurements'][0].update(correctness_passed=False),
        lambda x:x['tested'][0]['rank_measurements'][0].update(union_seconds=[float('nan')]*5),
        lambda x:x.update(pipeline_verified=True),
    ):
        corrupt=copy.deepcopy(data);mutation(corrupt)
        assert not valid_receipt(corrupt,identity)
        path.write_text(json.dumps(corrupt))
        assert read_profile(path,identity,lambda env: (_ for _ in ()).throw(AssertionError('invalid cache admitted'))) is None
    path.write_text('{broken');assert read_profile(path,identity,lambda env:plans) is None


def test_width_identity_and_changed_native_geometry_cannot_reuse_receipt(tmp_path):
    identity,data,plans=receipt();path=tmp_path/'cache.json';write_profile(path,identity,data)
    different=copy.deepcopy(identity);different['signature']['requested_beam_width']=4096
    assert read_profile(path,different,lambda env:plans) is None
    for key,value in (('SHARD_CAPACITY_CANDIDATES',4096),('STREAM3_RING_SLOTS',4),('RING_COUNT',8),('B_MICRO',256)):
        fresh=copy.deepcopy(plans)
        for p in fresh:p[key]=value
        assert read_profile(path,identity,lambda env:fresh) is None
