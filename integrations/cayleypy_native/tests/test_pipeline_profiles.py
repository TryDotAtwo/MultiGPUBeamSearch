import pytest
from multigpubeamsearch.pipeline_profiles import pipeline_candidates,validate_rank_plans
from multigpubeamsearch.pipeline_profiles import tune_pipeline
from multigpubeamsearch.calibration_stats import Measurement
import time


def test_inference_is_frozen_and_semantics_preserved():
    base={'BEAM_SHARD_CAPACITY_SCALE_PPM':'4000000','BEAM_GLOBAL_SPILL_SCALE_PPM':'2000000'}
    candidates=pipeline_candidates(1024,base)
    for _,env in candidates:
        assert env['BEAM_ENSEMBLE_INFERENCE_MICRO']=='1024'
        assert int(env['BEAM_B_MICRO'])>=1024
        for key,value in base.items():assert env[key]==value
    assert any(env['BEAM_B_MICRO']=='8192' for _,env in candidates)


def test_native_scalar_row_budget_remains_frozen_during_downstream_tuning():
    candidates=pipeline_candidates(256,{'BEAM_B_MICRO':'768'},tune_outer=False)
    assert len(candidates)>10
    assert all(env['BEAM_B_MICRO']=='768' for _,env in candidates)
    assert not any(name.startswith('outer-') for name,_ in candidates)


def test_128_rank_memory_intersection():
    row={'GLOBAL_BEAM_WIDTH_EFFECTIVE':268435456,'BEAM_WIDTH_ALIGNMENT':524288,
         'SHARD_COUNT':16,'B_MICRO':1024,'WORLD_SIZE':128,'STREAM4_BATCH_ALIGNMENT':256,
         'estimated_required_device_bytes':10,'gpu_budget_bytes':11}
    plans=[dict(row) for _ in range(128)]
    assert validate_rank_plans(plans)
    plans[-1]['gpu_budget_bytes']=9
    with pytest.raises(ValueError,match='memory'):validate_rank_plans(plans)


def test_rank_alignment_cannot_diverge():
    row={'GLOBAL_BEAM_WIDTH_EFFECTIVE':8192,'BEAM_WIDTH_ALIGNMENT':8192,
         'SHARD_COUNT':4,'B_MICRO':32,'WORLD_SIZE':2,'STREAM4_BATCH_ALIGNMENT':1024,
         'estimated_required_device_bytes':10,'gpu_budget_bytes':11}
    other=dict(row,SHARD_COUNT=8)
    with pytest.raises(ValueError,match='SHARD_COUNT'):validate_rank_plans([row,other])


def test_pipeline_uses_slowest_rank_and_keeps_inference_fixed():
    observed=[]
    def admit(env):
        assert env['BEAM_ENSEMBLE_INFERENCE_MICRO']=='32'
        assert env['BEAM_SHARD_CAPACITY_SCALE_PPM']=='4000000'
        row={'GLOBAL_BEAM_WIDTH_EFFECTIVE':65536,'BEAM_WIDTH_ALIGNMENT':1024,
             'SHARD_COUNT':int(env.get('BEAM_SHARD_COUNT',16)),
             'B_MICRO':int(env['BEAM_B_MICRO']),'WORLD_SIZE':2,
             'STREAM4_BATCH_ALIGNMENT':1024,
             'estimated_required_device_bytes':10,'gpu_budget_bytes':11}
        if row['SHARD_COUNT']==128:
            row['GLOBAL_BEAM_WIDTH_EFFECTIVE']=262144
        return [dict(row),dict(row)]
    def measure(env,plans):
        observed.append(dict(env))
        outer=int(env['BEAM_B_MICRO'])
        rings=int(env.get('BEAM_STREAM3_RING_SLOTS',2))
        seconds=1.0 if outer==64 and rings==4 else 1.5 if outer==64 else 2.0
        return [Measurement('',65536,(.1,seconds),True,True) for _ in range(5)]
    result=tune_pipeline(32,{'BEAM_SHARD_CAPACITY_SCALE_PPM':'4000000'},
        admit=admit,measure=measure,deadline=time.monotonic()+60,max_outer=256)
    assert result['environment']['BEAM_B_MICRO']=='64'
    assert result['environment']['BEAM_STREAM3_RING_SLOTS']=='4'
    assert result['estimate']['median']==1.0/65536
    assert any('effective beam' in reason for reason in result['rejected'].values())


def test_missing_rank_measurement_is_rejected():
    row={'GLOBAL_BEAM_WIDTH_EFFECTIVE':8192,'BEAM_WIDTH_ALIGNMENT':1024,
         'SHARD_COUNT':4,'B_MICRO':32,'WORLD_SIZE':2,'STREAM4_BATCH_ALIGNMENT':1024,
         'estimated_required_device_bytes':10,'gpu_budget_bytes':11}
    with pytest.raises(ValueError,match='missing ranks'):
        tune_pipeline(32,{},admit=lambda env:[dict(row),dict(row)],
            measure=lambda env,plans:[Measurement('',8192,(1.0,),True,True)]*5,
            deadline=time.monotonic()+60)


def test_128_rank_cohort_distinguishes_communicator_and_device_ordinals():
    row={'GLOBAL_BEAM_WIDTH_EFFECTIVE':268435456,'BEAM_WIDTH_ALIGNMENT':524288,
         'SHARD_COUNT':16,'B_MICRO':1024,'WORLD_SIZE':128,'STREAM4_BATCH_ALIGNMENT':256,
         'estimated_required_device_bytes':10,'gpu_budget_bytes':11}
    plans=[dict(row,LOCAL_RANK=rank,CUDA_DEVICE_LOCAL_RANK=rank%8) for rank in range(128)]
    assert validate_rank_plans(plans)
    plans[-1]['LOCAL_RANK']=0
    with pytest.raises(ValueError,match='communicator rank'):validate_rank_plans(plans)


def test_negative_memory_estimate_is_not_admission():
    row={'GLOBAL_BEAM_WIDTH_EFFECTIVE':8192,'BEAM_WIDTH_ALIGNMENT':1024,
         'SHARD_COUNT':4,'B_MICRO':32,'WORLD_SIZE':1,'STREAM4_BATCH_ALIGNMENT':1024,
         'estimated_required_device_bytes':-1,'gpu_budget_bytes':11}
    with pytest.raises(ValueError,match='values'):validate_rank_plans([row])

