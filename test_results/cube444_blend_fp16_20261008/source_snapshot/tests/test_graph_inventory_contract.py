"""Structural graph contract tests; never attest actual symbols or replay."""
from copy import deepcopy
import pytest
from tools.graph_inventory_contract import validate_graph_inventory_structure


def fixture():
    return dict(functions=[dict(mangled_name='_Z4testv',attributes=dict(
        binary_version=86,ptx_version=86,max_threads_per_block=1024,num_regs=12,
        static_shared_bytes=0,local_bytes=0,constant_bytes=0,max_dynamic_shared_bytes=49152))],
        lanes=[dict(lane=0,node_count=3,node_types=dict(kernel=1,memcpy=1,memset=1),
            kernels=[dict(function_id=0,grid=[2,1,1],block=[128,1,1],dynamic_shared_bytes=0)])])


def test_valid_structure_remains_not_admission():
    result=validate_graph_inventory_structure(fixture(),lanes=1,sm=86)
    assert result==dict(functions=1,kernel_nodes=1,structure_checked=True,
                       production_admitted=False,kernel_coverage_complete=False)


def test_distinct_used_function_ids_may_share_mangled_symbol():
    # Native T4 graph exposes two host function pointers for the same
    # CUTLASS instantiation; names are descriptive, not unique pointer IDs.
    value=fixture()
    value['functions'].append(deepcopy(value['functions'][0]))
    lane=value['lanes'][0]
    node=deepcopy(lane['kernels'][0]);node['function_id']=1
    lane['kernels'].append(node)
    lane['node_count']+=1;lane['node_types']['kernel']+=1
    result=validate_graph_inventory_structure(value,lanes=1,sm=86)
    assert result['functions']==2 and result['kernel_nodes']==2


@pytest.mark.parametrize('mutation',[
    'missing_functions','empty_functions','missing_lanes','wrong_lane','duplicate_lane',
    'bad_function_id','bool_function_id','missing_attributes','wrong_binary',
    'bad_symbol','duplicate_function','negative_regs','bool_grid','huge_grid',
    'zero_block','too_many_threads','too_much_shared','wrong_count','unknown_node_type'])
def test_invalid_structure_rejected(mutation):
    value=deepcopy(fixture())
    function=value['functions'][0]
    lane=value['lanes'][0]
    kernel=lane['kernels'][0]
    if mutation=='missing_functions': del value['functions']
    if mutation=='empty_functions': value['functions']=[]
    if mutation=='missing_lanes': del value['lanes']
    if mutation=='wrong_lane': lane['lane']=1
    if mutation=='duplicate_lane': value['lanes'].append(deepcopy(lane))
    if mutation=='bad_function_id': kernel['function_id']=1
    if mutation=='bool_function_id': kernel['function_id']=False
    if mutation=='missing_attributes': del function['attributes']
    if mutation=='wrong_binary': function['attributes']['binary_version']=90
    if mutation=='bad_symbol': function['mangled_name']='unknown'
    if mutation=='duplicate_function': value['functions'].append(deepcopy(function))
    if mutation=='negative_regs': function['attributes']['num_regs']=-1
    if mutation=='bool_grid': kernel['grid'][0]=True
    if mutation=='huge_grid': kernel['grid'][1]=65536
    if mutation=='zero_block': kernel['block'][0]=0
    if mutation=='too_many_threads': kernel['block']=[1024,2,1]
    if mutation=='too_much_shared': kernel['dynamic_shared_bytes']=49153
    if mutation=='wrong_count': lane['node_count']=4
    if mutation=='unknown_node_type': lane['node_types']['child_graph']=1
    with pytest.raises(ValueError):
        validate_graph_inventory_structure(value,lanes=1,sm=86)
