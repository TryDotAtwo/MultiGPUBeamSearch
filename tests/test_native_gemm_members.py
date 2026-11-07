"""Opt-in real captured GEMM Params producer; no independent values/admission."""
import hashlib
import json
import os
from pathlib import Path
import pytest
from collections import Counter
import re
from tools.gemm_residual_values import validate_residual_gemm
from tools.gemm_bias_values import validate_bias_gemm
from tools.gemm_head_values import validate_head_gemm
from tools.full_score_comparison import compare_production_raw_jobs
from tools.gemm_iterator_values import validate_epilogue_iterator


def test_actual_classic_baseline_public_iterator_values():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    geometry=os.environ.get('BEAM_TEST_GEMM_ITERATOR_GEOMETRY')
    if not selected or not geometry:pytest.skip('actual capture and independent compiled geometry required')
    root=Path(selected)
    payload=json.loads((root/'graph_gemm_members.json').read_bytes())
    inventory=json.loads((root/'graph_kernel_inventory.json').read_bytes())
    descriptors=json.loads(Path(geometry).read_bytes())
    assert set(descriptors)=={'regular','broadcast','relu','regular75','broadcast75','relu75'}
    by_symbol={entry['symbol']:entry['descriptor'] for entry in descriptors.values()}
    assert len(by_symbol)==6
    checked=0
    for lane in payload['lanes']:
        for node in lane['kernels']:
            if not node['gemm_signature_known']:continue
            symbol=inventory['functions'][node['function_id']]['mangled_name']
            assert symbol in by_symbol, 'unsupported independent classic baseline signature'
            role=node['pointers']['B']['role']
            if role=='output_weight':rows=24
            else:
                match=re.fullmatch(r'block([0-3])_(attn_qkv|ff1|attn_out|ff2)_weight',role)
                assert match, 'unknown GEMM weight role'
                rows={'attn_qkv':768,'ff1':1024,'attn_out':256,'ff2':256}[match[2]]
            assert node['problem']['n']==rows
            for operand in ('C','D'):
                assert validate_epilogue_iterator(node['epilogue_iterators'][operand],
                    row_elements=rows,descriptor=by_symbol[symbol])
                checked+=1
    assert checked>0
    # Classic unsplit SM75/SM80 only. Receipt is not mandatory admission yet.


def test_actual_gemm_capture_retains_raw_score_quality():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    reference_path=os.environ.get('BEAM_TEST_GEMM_REFERENCE')
    if not selected or not reference_path:pytest.skip('explicit actual capture and trained reference required')
    root=Path(selected)
    execution=json.loads((root/'execution.json').read_bytes())
    raw=json.loads((root/'raw_scores.json').read_bytes())
    reference=json.loads(Path(reference_path).read_bytes())
    result=compare_production_raw_jobs(reference['scores_fp32'],raw,
        outer_microbatch=execution['outer_microbatch'],lanes=execution['lanes'],
        atol=.05,rtol=.001,top_k=4,min_topk_overlap=.75)
    assert result['status']=='pass',result
    assert result['production_quality_accepted'] is False


def test_actual_head_values_and_counts():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    if not selected:pytest.skip('explicit remote actual captured graph required')
    root=Path(selected)
    execution=json.loads((root/'execution.json').read_bytes())
    payload=json.loads((root/'graph_gemm_members.json').read_bytes())
    expected=Counter(min(execution['transformer_microbatch'],execution['outer_microbatch']-offset)
        for offset in range(0,execution['outer_microbatch'],execution['transformer_microbatch']))
    for lane in payload['lanes']:
        actual=Counter()
        for node in lane['kernels']:
            if not node['gemm_signature_known'] or node['pointers']['B']['role']!='output_weight':continue
            batch=node['problem']['m']
            assert validate_head_gemm(node,batch=batch)
            actual[batch]+=1
        assert actual==expected


def test_actual_gemm_scalar_override_pointers_are_observed():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    if not selected:pytest.skip('explicit remote actual captured graph required')
    payload=json.loads((Path(selected)/'graph_gemm_members.json').read_bytes())
    known=0
    for lane in payload['lanes']:
        for node in lane['kernels']:
            if not node['gemm_signature_known']:continue
            wanted={
                'alpha':dict(role='null',offset=0,available_bytes=0),
                'beta':dict(role='null',offset=0,available_bytes=0)}
            if 'Vector' not in node['pointers']:
                wanted.update(alpha_array=dict(role='null',offset=0,available_bytes=0),
                              beta_array=dict(role='null',offset=0,available_bytes=0))
            assert node.get('epilogue_pointers')==wanted
            known+=1
    assert known>0


def test_actual_gemm_epilogue_iterator_strides_are_observed():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    if not selected:pytest.skip('explicit remote actual captured graph required')
    payload=json.loads((Path(selected)/'graph_gemm_members.json').read_bytes())
    known=0
    fields={'stride','increment_row','increment_group','increment_cluster',
        'advance_row','advance_group','advance_cluster','advance_tile'}
    for lane in payload['lanes']:
        for node in lane['kernels']:
            if not node['gemm_signature_known']:continue
            captured=node.get('epilogue_iterators')
            assert isinstance(captured,dict) and set(captured)=={'C','D'}
            for name in ('C','D'):
                assert set(captured[name])==fields
                assert all(type(v) is int for v in captured[name].values())
                assert captured[name]['stride']==node['problem']['n']*2
            known+=1
    assert known>0
    # Other increments are observed, not independently validated by this test.


def test_actual_unsplit_bias_gemm_values_and_counts():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    if not selected:pytest.skip('explicit remote actual captured graph required')
    root=Path(selected)
    execution=json.loads((root/'execution.json').read_bytes())
    payload=json.loads((root/'graph_gemm_members.json').read_bytes())
    seq=int(os.environ['BEAM_TEST_GEMM_PADDED_SEQ'])
    policies=execution['requested_launch_policies']
    assert policies.get('BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV')!='1'
    final=policies.get('BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY')=='1'
    expected=Counter()
    for offset in range(0,execution['outer_microbatch'],execution['transformer_microbatch']):
        batch=min(execution['transformer_microbatch'],execution['outer_microbatch']-offset)
        for layer in range(4):
            expected[(layer,'qkv',batch*seq)]+=1
            expected[(layer,'ff1',batch if final and layer==3 else batch*seq)]+=1
    for lane in payload['lanes']:
        actual=Counter()
        for node in lane['kernels']:
            if not node['gemm_signature_known']:continue
            match=re.fullmatch(r'block([0-3])_(attn_qkv|ff1)_weight',node['pointers']['B']['role'])
            if not match:continue
            layer=int(match[1]);kind='qkv' if match[2]=='attn_qkv' else 'ff1'
            rows=node['problem']['m']
            assert validate_bias_gemm(node,layer=layer,rows=rows,kind=kind)
            actual[(layer,kind,rows)]+=1
        assert actual==expected
    # Values/counts do not attest iterator Params, activation or dependency order.


def test_actual_scorer_publishes_typed_gemm_members():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    if not selected:pytest.skip('explicit authorized remote actual output required')
    root=Path(selected)
    execution_raw=(root/'execution.json').read_bytes()
    inventory_raw=(root/'graph_kernel_inventory.json').read_bytes()
    execution=json.loads(execution_raw);inventory=json.loads(inventory_raw)
    path=root/'graph_gemm_members.json'
    assert path.is_file(), 'real scorer did not publish typed captured GEMM Params'
    payload=json.loads(path.read_bytes())
    assert payload['schema_version']==1
    assert payload['scope']=='captured_known_gemm_members_not_complete_values_or_admission'
    assert payload['production_admitted'] is False
    assert payload['parameter_values_checked'] is False
    assert payload['device']==execution['device']
    assert payload['execution_sha256']==hashlib.sha256(execution_raw).hexdigest()
    assert payload['inventory_sha256']==hashlib.sha256(inventory_raw).hexdigest()
    captures={l['lane']:l['kernels'] for l in inventory['lanes']}
    assert len(payload['lanes'])==execution['lanes']
    for lane in payload['lanes']:
        native=captures[lane['lane']]
        assert len(lane['kernels'])==len(native)
        known=0
        for index,node in enumerate(lane['kernels']):
            assert node['kernel_index']==index and node['function_id']==native[index]['function_id']
            symbol=inventory['functions'][node['function_id']]['mangled_name']
            gemm=symbol.startswith(('_ZN7cutlass6KernelI','_ZN7cutlass7Kernel2I'))
            if gemm:
                assert node['gemm_signature_known'] is True
                assert node['parameter_index']==0
                assert set(node['problem'])=={'m','n','k'}
                assert len(node['pointers'])>=4
                known+=1
            else:
                assert node['gemm_signature_known'] is False and node['parameter_index'] is None
                assert node['pointers']=={} and node['problem']=={}
        assert known>0


def test_actual_residual_gemm_values_and_complete_counts():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    if not selected:pytest.skip('explicit remote actual captured graph required')
    root=Path(selected)
    execution=json.loads((root/'execution.json').read_bytes())
    payload=json.loads((root/'graph_gemm_members.json').read_bytes())
    seq=int(os.environ['BEAM_TEST_GEMM_PADDED_SEQ'])
    final=execution['requested_launch_policies'].get('BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY')=='1'
    expected=Counter()
    for offset in range(0,execution['outer_microbatch'],execution['transformer_microbatch']):
        batch=min(execution['transformer_microbatch'],execution['outer_microbatch']-offset)
        for layer in range(4):
            rows=batch if final and layer==3 else batch*seq
            for kind in ('attn_out','ff2'):expected[(layer,kind,rows)]+=1
    for lane in payload['lanes']:
        actual=Counter()
        for node in lane['kernels']:
            if not node['gemm_signature_known']:continue
            role=node['pointers']['B']['role']
            match=re.fullmatch(r'block([0-3])_(attn_out|ff2)_weight',role)
            if not match:continue
            layer=int(match[1]);kind=match[2];rows=node['problem']['m']
            assert validate_residual_gemm(node,layer=layer,rows=rows,kind=kind,cls=final and layer==3)
            actual[(layer,kind,rows)]+=1
        assert actual==expected
    # This validates values/counts, not which graph dependencies place a weight
    # in its intended transformer layer. Complete ordering admission stays open.
