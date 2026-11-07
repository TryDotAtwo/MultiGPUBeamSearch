"""Actual native receipt coverage corruption; no new inference/admission claim."""
import json
import os
from pathlib import Path
from copy import deepcopy
import pytest
from tools.graph_argument_coverage import validate_argument_coverage

def capture():
    path=os.environ.get('BEAM_TEST_COVERAGE_OUTPUT')
    if not path:pytest.skip('explicit actual remote capture required')
    names=('graph_kernel_inventory','graph_pointer_roles','graph_u32_values','graph_embedded_dims',
           'graph_network_members','graph_gemm_members','graph_attention_members')
    return [json.loads((Path(path)/(name+'.json')).read_bytes()) for name in names]

@pytest.mark.parametrize('mutation',['valid','pointer_slot','scalar_slot','dims_slot','network_slot',
    'network_member','gemm_public','attention_public','empty_exclusion','duplicate_node','bool_index','bool_lane','bad_function'])
def test_actual_composed_coverage(mutation):
    data=deepcopy(capture());inv,p,s,d,n,g,a=data;empty=True
    def first(payload,key):return next(x for x in payload['lanes'][0]['kernels'] if x.get(key))
    if mutation=='pointer_slot':first(p,'pointers')['pointers'].pop()
    elif mutation=='scalar_slot':first(s,'u32_values')['u32_values'].pop()
    elif mutation=='dims_slot':first(d,'dims_signature_known')['dims_index']+=1
    elif mutation=='network_slot':first(n,'network_signature_known')['network_index']+=1
    elif mutation=='network_member':first(n,'network_signature_known')['members'].pop()
    elif mutation=='gemm_public':first(g,'gemm_signature_known')['mainloop_iterators']['A'].pop('stride')
    elif mutation=='attention_public':first(a,'attention_signature_known')['scalars'].pop('scale')
    elif mutation=='empty_exclusion':empty=False
    elif mutation=='duplicate_node':p['lanes'][0]['kernels'][1]=p['lanes'][0]['kernels'][0]
    elif mutation=='bool_index':p['lanes'][0]['kernels'][0]['kernel_index']=False
    elif mutation=='bool_lane':p['lanes'][0]['lane']=False
    elif mutation=='bad_function':s['lanes'][0]['kernels'][0]['function_id']=True
    if mutation=='valid':
        report=validate_argument_coverage(*data,empty_arguments_checked=empty)
        assert report['production_admitted'] is False and report['parameter_values_checked'] is False
        assert report['covered_argument_slots']>0
    else:
        with pytest.raises(ValueError):validate_argument_coverage(*data,empty_arguments_checked=empty)
