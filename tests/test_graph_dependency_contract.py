from copy import deepcopy
import pytest
from test_graph_transfer_contract import fixture


def captured():
    value=fixture();lane=value['lanes'][0]
    lane.update(node_count=8,kernel_node_ids=[7,5,6],
        dependency_edges=[[4,1],[1,7],[7,2],[2,5],[5,0],[0,6],[6,3]])
    for op,node in zip(lane['operations'],[4,1,2,0,3]):op['node_id']=node
    return value


def check(value):
    from tools.graph_transfer_contract import validate_score_graph_dependencies
    return validate_score_graph_dependencies(value,outer=32,inner=13,lanes=1)


def test_actual_edges_not_enumeration_define_copy_order():
    value=captured();value['lanes'][0]['operations'].reverse()
    result=check(value)
    assert result['dependency_order_checked'] is True
    assert result['production_admitted'] is False


@pytest.mark.parametrize('mode', ['missing_edge','parallel','cycle','wrong_node',
    'duplicate_edge','duplicate_operation_node','missing_kernel','unknown_kernel',
    'copy_before_kernel','late_reset','reversed_copies','bool_id','missing_dependencies'])
def test_dependency_mutations_rejected(mode):
    value=captured();lane=value['lanes'][0];edges=lane['dependency_edges'];ops=lane['operations']
    if mode=='missing_edge':edges.pop()
    elif mode=='parallel':edges[-1]=[0,3]
    elif mode=='cycle':edges.append([3,4])
    elif mode=='wrong_node':edges[0][0]=8
    elif mode=='duplicate_edge':edges.append(deepcopy(edges[0]))
    elif mode=='duplicate_operation_node':ops[2]['node_id']=4
    elif mode=='missing_kernel':lane['kernel_node_ids'].pop()
    elif mode=='unknown_kernel':lane['kernel_node_ids'].append(8)
    elif mode=='copy_before_kernel':ops[2]['node_id'],lane['kernel_node_ids'][0]=7,2
    elif mode=='late_reset':ops[0]['node_id'],ops[2]['node_id']=2,4
    elif mode=='reversed_copies':ops[2]['node_id'],ops[3]['node_id']=0,2
    elif mode=='bool_id':ops[0]['node_id']=True
    elif mode=='missing_dependencies':del lane['dependency_edges']
    with pytest.raises(ValueError):check(value)
