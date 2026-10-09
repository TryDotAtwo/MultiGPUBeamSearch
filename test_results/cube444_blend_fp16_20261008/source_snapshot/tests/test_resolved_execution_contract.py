"""Remote validation required; no CUDA execution in synthetic receipt tests."""
from copy import deepcopy
import pytest
from tools.resolved_execution_contract import (
    canonical_resolved_recipe, validate_resolved_receipt, PREFIX, SELECTOR_NAMES)


def inputs():
    expected = dict(device=dict(sm=86), executor='native_cuda_graph',
        outer_microbatch=32, transformer_microbatch=13, lanes=2,
        requested_launch_policies={PREFIX+n:None for n in SELECTOR_NAMES},
        manifest_sha256='a'*64, reference_sha256='b'*64, loaded_tensor_bindings={})
    expected['requested_launch_policies'][PREFIX+'FUSED_INPUT_LAYERNORM']='1'
    wanted = canonical_resolved_recipe(expected)
    receipt = deepcopy(wanted)
    receipt.update({k:deepcopy(expected[k]) for k in ('device','manifest_sha256',
        'reference_sha256','loaded_tensor_bindings','requested_launch_policies')})
    receipt.update(raw_scores_sha256='c'*64, specialized_gemm_observations=[])
    return expected, wanted, receipt, dict(host_kernel_observations=[])


def test_exact_canonical_receipt_is_narrow_diagnostic():
    expected,wanted,receipt,execution=inputs()
    validate_resolved_receipt(receipt,wanted,expected,execution,'c'*64)
    assert receipt['production_admitted'] is False
    assert receipt['kernel_coverage_complete'] is False

@pytest.mark.parametrize('fused',[False,True])
def test_inconsistent_cls_flags_reject_before_scorer(fused,tmp_path,monkeypatch):
    from tools.cube4_production_preflight import collect_fresh_candidate
    expected,_,_,_=inputs()
    expected['requested_launch_policies'][PREFIX+'FUSED_INPUT_LAYERNORM']=str(int(fused))
    expected['requested_launch_policies'][PREFIX+'FINAL_CLS_ATTENTION']='1'
    # Supply a previously valid recipe to ensure the transaction independently
    # recomputes the requested recipe before invoking the scorer.
    old=inputs()[1]
    observed=dict(expected=expected,expected_resolved=old,reference_dir=str(tmp_path),reference_scores=[[0]])
    started=[]
    monkeypatch.setattr('tools.cube4_production_preflight.run_scorer',lambda *a,**k:started.append(True))
    with pytest.raises(ValueError,match='inconsistent final CLS'):
        collect_fresh_candidate([],tmp_path/'run',5,lambda:observed,[[0]])
    assert not started and not (tmp_path/'run').exists()


@pytest.mark.parametrize('mutation', [
    'missing_launch','extra_top','missing_specialized','missing_fp16','bool_schema',
    'int_fp16','int_graph','int_flag','float_policy','policy','stage','swizzle',
    'epilogue','family_order','duplicate_family','extra_family_member',
    'extra_launch','missing_launch_member','attention_tile','ln_grid',
    'stage_skip','sm','executor','hash','manifest','device','specialized',
    'subset_expected','contradictory_expected'])
def test_receipt_mutations_reject(mutation):
    expected,wanted,receipt,execution=inputs()
    if mutation=='missing_launch': del receipt['launch']
    elif mutation=='extra_top': receipt['unobserved']=0
    elif mutation=='missing_specialized': del receipt['specialized_gemm_observations']
    elif mutation=='missing_fp16': del receipt['fp16']
    elif mutation=='bool_schema': receipt['schema_version']=True
    elif mutation=='int_fp16': receipt['fp16']=1
    elif mutation=='int_graph': receipt['graph_executor']=1
    elif mutation=='int_flag': receipt['launch']['flags'][PREFIX+'FUSED_INPUT_LAYERNORM']=1
    elif mutation=='float_policy': receipt['gemm_families'][0]['policy_code']=0.0
    elif mutation in ('policy','stage','swizzle','epilogue'):
        receipt['gemm_families'][0][mutation+'_code']=1
    elif mutation=='family_order': receipt['gemm_families'].reverse()
    elif mutation=='duplicate_family': receipt['gemm_families'][1]=deepcopy(receipt['gemm_families'][0])
    elif mutation=='extra_family_member': receipt['gemm_families'][0]['extra']=0
    elif mutation=='extra_launch': receipt['launch']['extra']=0
    elif mutation=='missing_launch_member': del receipt['launch']['cls_attention_code']
    elif mutation=='attention_tile': receipt['launch']['attention_tile_code']=1
    elif mutation=='ln_grid': receipt['launch']['layernorm_copy_grid']=1
    elif mutation=='stage_skip': receipt['launch']['stage_profile_skip_calls']=0
    elif mutation=='sm': receipt['sm']=90
    elif mutation=='executor': receipt['graph_executor']=False
    elif mutation=='hash': receipt['raw_scores_sha256']='d'*64
    elif mutation=='manifest': receipt['manifest_sha256']='d'*64
    elif mutation=='device': receipt['device']['sm']=80
    elif mutation=='specialized': receipt['specialized_gemm_observations']=[{'family':'linear_bias_strided'}]
    elif mutation=='subset_expected': wanted={'device':expected['device']}
    elif mutation=='contradictory_expected': wanted['launch']['attention_tile_code']=1
    with pytest.raises(ValueError):
        validate_resolved_receipt(receipt,wanted,expected,execution,'c'*64)


@pytest.mark.parametrize('name,value', [('FF1_POLICY','m128n128'),
    ('LAYERNORM_ROWS_POLICY','persistent'),('HOPPER_FF1_EPILOGUE','128x64'),
    ('FF2_EPILOGUE','fused'),('STAGE_PROFILE','1'),('BLOCK51','1'),
    ('FUSED_INPUT_LAYERNORM',True),('ATTENTION_TILE_POLICY','q32k64')])
def test_unsupported_requested_routes_fail_before_receipt(name,value):
    expected,_,_,_=inputs()
    expected['requested_launch_policies'][PREFIX+name]=value
    with pytest.raises(ValueError): canonical_resolved_recipe(expected)


def test_main_cli_observer_requires_resolved_recipe(monkeypatch,tmp_path):
    """Intercept the real CLI observer, not a copy of its construction."""
    import sys
    from tools import cube4_production_preflight as cli
    from tools import resolved_execution_contract as resolved
    monkeypatch.setattr(resolved,'observe_recipe_sources',lambda path:{})
    source=tmp_path/'source';source.mkdir();(source/'cuda').mkdir()
    (source/'cuda/stream1_transformer_policy_snapshot.hpp').write_text(
        '\n'.join('"'+PREFIX+n+'"' for n in SELECTOR_NAMES))
    runner=tmp_path/'runner';runner.touch()
    probe=tmp_path/'probe';probe.touch()
    reference=tmp_path/'reference';reference.mkdir();(reference/'weights_fp16').mkdir()
    monkeypatch.setattr(sys,'argv',['preflight','--runner',str(runner),'--probe',str(probe),
        '--reference-dir',str(reference),'--source-root',str(source),
        '--output-root',str(tmp_path/'out'),'--world-size','1'])
    for key in list(cli.os.environ):
        if key.startswith(('BEAM_','CUDA_')): monkeypatch.delenv(key)
    monkeypatch.setenv('BEAM_STREAM1_EXECUTOR','native_eager')
    from tools import cube4_numeric_gate as numeric, cube4_bundle_contract as bundle
    monkeypatch.setattr(numeric,'read_json',lambda p: (
        {'tensor_files':{}} if p.name=='manifest.json' else {}, 'a'*64))
    monkeypatch.setattr(numeric,'validate_reference',lambda value:[[0.0]*24])
    monkeypatch.setattr(numeric,'validate_checkpoint_binding',lambda *args:None)
    monkeypatch.setattr(bundle,'validate_cube4_weights',lambda path:{})
    from tools import cuda_device_observation
    monkeypatch.setattr(cuda_device_observation,'observe',lambda path:dict(
        visible_devices=dict(devices=[dict(device=0,sm=86,uuid_hex='b'*32)])))
    seen=[]
    def intercept(command,directory,timeout,observer,reference):
        before=observer();seen.append(before)
        assert before['expected_resolved']==canonical_resolved_recipe(before['expected'])
        return dict(production_admitted=False)
    monkeypatch.setattr(cli,'collect_fresh_candidate',intercept)
    assert cli.main()==2
    assert len(seen)==1


@pytest.mark.parametrize('mutation',['changed','missing','oversized','symlink'])
def test_recipe_source_drift_rejected_before_runner(mutation,tmp_path):
    from pathlib import Path
    from tools.resolved_execution_contract import SOURCE_PINS,observe_recipe_sources
    source=Path(__file__).resolve().parents[1]
    for name in SOURCE_PINS:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((source/name).read_bytes())
    target=tmp_path/next(iter(SOURCE_PINS))
    if mutation=='changed': target.write_bytes(target.read_bytes()+b'\n// drift\n')
    elif mutation=='missing': target.unlink()
    elif mutation=='oversized': target.write_bytes(b'x'*(1024*1024+1))
    else:
        target.unlink();target.symlink_to(source/next(iter(SOURCE_PINS)))
    with pytest.raises((ValueError,OSError)): observe_recipe_sources(tmp_path)


def test_recipe_sources_accept_only_pinned_normalized_contents(tmp_path):
    from pathlib import Path
    from tools.resolved_execution_contract import SOURCE_PINS,observe_recipe_sources
    source=Path(__file__).resolve().parents[1]
    for name in SOURCE_PINS:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((source/name).read_bytes().replace(b'\r\n',b'\n'))
    observed=observe_recipe_sources(tmp_path)
    assert {name:observed[name] for name in SOURCE_PINS}==SOURCE_PINS
    assert len(observed['validator_sha256'])==64
