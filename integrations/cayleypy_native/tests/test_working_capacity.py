import json
from types import SimpleNamespace
import pytest
from multigpubeamsearch.options import NativeOptions
from multigpubeamsearch.working_capacity import maximum_options, assess_efficiency, require_full_step


def test_maximum_requires_full_frontier_without_mutating_fixed_width_options():
    options=NativeOptions()
    assert maximum_options(options).calibration_full_frontier is True
    assert options.calibration_full_frontier is False


def full(beam,seconds_per_parent):
    return dict(phase='pipeline_measured',pipeline_verified=True,
        measurement_scope='full_requested_frontier',requested_beam_effective=beam,
        estimate=dict(median=seconds_per_parent))


def test_allocation_or_component_evidence_cannot_certify_maximum():
    for row in ({},dict(phase='component_calibrated'),full(1024,.01)):
        with pytest.raises(ValueError):require_full_step(2048,row)
    assert require_full_step(2048,full(2048,.01))==.01


def test_efficiency_distinguishes_missing_reference_and_throughput_collapse():
    assert assess_efficiency(4096,.01,None)['efficiency_verified'] is False
    reference=dict(beam=1024,seconds_per_parent=.01)
    good=assess_efficiency(4096,.0105,reference)
    assert good['efficient_working_beam']==4096 and good['near_linear']
    slow=assess_efficiency(4096,.02,reference)
    assert slow['full_step_verified'] and not slow['near_linear']
    assert slow['efficient_working_beam']==1024
    assert slow['tested_beam']==4096  # no silent execution-width change


def test_backend_applies_maximum_verification_policy(tmp_path,monkeypatch):
    from multigpubeamsearch import backend
    class Prepared(Exception):pass
    def prepare(contract,model,options,*args,**kwargs):
        assert options.calibration_full_frontier
        raise Prepared
    monkeypatch.setattr(backend,'prepare_runtime',prepare)
    monkeypatch.setattr(backend,'validate_touch_bfs_contract',lambda *a:1)
    with pytest.raises(Prepared):
        backend.run_native(SimpleNamespace(replay=lambda path:False),None,
            NativeOptions(), 'max',4,tmp_path,[0])


def test_efficiency_reference_requires_same_model_and_gpu(tmp_path,monkeypatch):
    from multigpubeamsearch import pipeline_cache
    from multigpubeamsearch.working_capacity import record_evidence
    monkeypatch.setattr(pipeline_cache,'valid_receipt',lambda *a:True)
    keys=('graph','models','inference_backend','runner','probe','gpu_uuids',
        'world_size','torch','cuda','precision','topology_sha256','hardware_sha256')
    sig={k:k for k in keys};inference=dict(signature=sig,parent_batch=256)
    row=full(1024,.01);row.update(selection='baseline',cache_identity=dict(
        full_frontier=True,parent_batch=256,signature=dict(sig)))
    directory=tmp_path/'pipeline-profiles';directory.mkdir();path=directory/'reference.json'
    path.write_text(json.dumps(row))
    assert record_evidence({'effective_beam':4096},full(4096,.01),inference,tmp_path)['efficiency_verified']
    row['cache_identity']['signature']['models']='different'
    path.write_text(json.dumps(row))
    assert not record_evidence({'effective_beam':4096},full(4096,.01),inference,tmp_path)['efficiency_verified']


def test_working_profile_amortizes_survivor_sort_and_keeps_inference_batch():
    from multigpubeamsearch.maximum_beam import working_profile
    memory=SimpleNamespace(profile={'BEAM_B_MICRO':'1024','BEAM_SHARD_COUNT':'8',
        'BEAM_STREAM4_BATCH_CANDIDATES':'8192'},plans=[{'SHARD_CAPACITY_CANDIDATES':100001}])
    result=working_profile(memory,1024)
    assert result['BEAM_B_MICRO']=='1024' and result['BEAM_SHARD_COUNT']=='8'
    assert int(result['BEAM_STREAM4_BATCH_CANDIDATES'])==100352
    assert memory.profile['BEAM_STREAM4_BATCH_CANDIDATES']=='8192'


def test_working_selection_reduces_only_max_candidate_after_measured_collapse(tmp_path):
    from multigpubeamsearch.working_capacity import choose_working_capacity
    inference=dict(signature={'world_size':2},parent_batch=16)
    widths=[]
    def measure(width,profile):
        widths.append(width);row=full(width,.01);row.update(selection='reference',environment={});return row
    initial=full(4096,.03)
    width,row,evidence=choose_working_capacity({'effective_beam':4096},initial,
        inference,tmp_path,measure=measure)
    assert widths==[2048,1024] and width==2048
    assert evidence['near_linear'] and len(evidence['comparisons'])==2
