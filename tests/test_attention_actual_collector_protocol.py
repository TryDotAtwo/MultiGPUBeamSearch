"""Actual capture replayed through subprocess protocol; not fresh native inference."""
import json
import os
from pathlib import Path
import sys
import pytest
from tools.cube4_production_preflight import collect_fresh_candidate
from tools.cube4_numeric_gate import validate_reference


@pytest.mark.parametrize('mode',['valid','missing','device','execution_hash','inventory_hash',
    'stride','short','role','missing_lane','duplicate_node','parameter_promotion',
    'missing_optional','pytorch_rng'] + ['optional_pointer:'+n for n in
        ('attn_bias_ptr','seqstart_q_ptr','seqstart_k_ptr','seqlen_k_ptr')] +
    ['optional_scalar:'+n for n in ('causal_diagonal_offset','num_keys_absolute','bias_strideM',
        'bias_strideH','bias_strideB','use_dropout','dropout_batch_head_rng_offset','dropout_prob')])
def test_actual_attention_collector_protocol(mode,tmp_path):
    selected=os.environ.get('BEAM_TEST_ATTENTION_SCORE_OUTPUT')
    reference_dir=os.environ.get('BEAM_TEST_ATTENTION_REFERENCE_DIR')
    if not selected or not reference_dir:pytest.skip('explicit authorized remote actual artifacts required')
    source=Path(selected)
    execution=json.loads((source/'execution.json').read_bytes())
    keys=('reference_sha256','manifest_sha256','outer_microbatch','transformer_microbatch',
          'lanes','executor','device','loaded_tensor_bindings','requested_launch_policies')
    expected={key:execution[key] for key in keys}
    reference=validate_reference(json.loads((Path(reference_dir)/'reference.json').read_bytes()))
    model=dict(dtype='fp16',num_layers=4,nhead=8,head_dim=32,seq_len=57,d_model=256)
    def observe():
        return dict(expected=expected,reference_dir=reference_dir,reference_scores=reference,
            require_graph_inventory=True,expected_graph_attention_members_profile=dict(
                model=model,padded_seq_len=int(os.environ['BEAM_TEST_ATTENTION_PADDED_SEQ'])))
    code=f'''import pathlib,sys,shutil,json
source=pathlib.Path({str(source)!r}); output=pathlib.Path(sys.argv[3]); output.mkdir()
for name in ('execution.json','raw_scores.json','graph_kernel_inventory.json'):
 shutil.copyfile(source/name,output/name)
p=json.loads((source/'graph_attention_members.json').read_bytes())
mode={mode!r}
node=next(n for n in p['lanes'][0]['kernels'] if n['attention_signature_known'])
if mode=='device':p['device']['uuid_hex']='0'*32
elif mode=='execution_hash':p['execution_sha256']='0'*64
elif mode=='inventory_hash':p['inventory_sha256']='0'*64
elif mode=='stride':node['scalars']['q_strideB']=1
elif mode=='short':node['pointers']['key_ptr']['available_bytes']=1
elif mode=='role':node['pointers']['query_ptr']['role']='lane_context'
elif mode=='missing_lane':p['lanes'].pop()
elif mode=='duplicate_node':p['lanes'][0]['kernels'][1]=p['lanes'][0]['kernels'][0]
elif mode=='parameter_promotion':p['parameter_values_checked']=True
elif mode=='missing_optional':node.pop('optional_scalars')
elif mode=='pytorch_rng':node['pytorch_rng_state_present']=True
elif mode.startswith('optional_pointer:'):
 node['optional_pointers'][mode.split(':')[1]]=dict(role='lane_qkv',offset=0,available_bytes=512)
elif mode.startswith('optional_scalar:'):
 name=mode.split(':')[1]; old=node['optional_scalars'][name]
 node['optional_scalars'][name]=True if type(old) is bool else old+1
if mode!='missing':(output/'graph_attention_members.json').write_text(json.dumps(p))
'''
    command=[sys.executable,'-c',code]
    if mode=='valid':
        result=collect_fresh_candidate(command,tmp_path/'run',10,observe,reference)
        assert result['graph_attention_members_checked'] is True
        assert result['production_admitted'] is False
    else:
        with pytest.raises(ValueError):collect_fresh_candidate(command,tmp_path/'run',10,observe,reference)
