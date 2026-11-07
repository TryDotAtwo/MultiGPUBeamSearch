"""Independent transfer contract regression; remote RED/GREEN verified."""
from copy import deepcopy
import pytest


def fixture():
    # Independently hand-derived: outer32, inner13, 24 FP16 logits per parent.
    # Copies preserve chunks13,13,6 at byte offsets0,624,1248.
    return dict(schema_version=1,scope='captured_graph_transfers_not_admission',
        production_admitted=False,lanes=[dict(lane=0,operations=[
            dict(kind='memset',destination='raw_scores',destination_offset=0,
                 bytes=1536,value=0,element_size=1),
            dict(kind='memset',destination='numeric_error',destination_offset=0,
                 bytes=4,value=0,element_size=1),
            dict(kind='memcpy',source='lane_logits',source_offset=0,
                 destination='raw_scores',destination_offset=0,bytes=624,
                 direction='device_to_device'),
            dict(kind='memcpy',source='lane_logits',source_offset=0,
                 destination='raw_scores',destination_offset=624,bytes=624,
                 direction='device_to_device'),
            dict(kind='memcpy',source='lane_logits',source_offset=0,
                 destination='raw_scores',destination_offset=1248,bytes=288,
                 direction='device_to_device')])])


def validate(value,outer=32,inner=13,lanes=1):
    from tools.graph_transfer_contract import validate_score_graph_transfers
    return validate_score_graph_transfers(value,outer=outer,inner=inner,lanes=lanes)


def test_exact_chunk_copies_and_resets_checked_without_admission():
    result=validate(fixture())
    assert result['transfer_arguments_checked'] is True
    assert result['checked_memcpy_nodes']==3
    assert result['checked_memset_nodes']==2
    assert result['production_admitted'] is False


def test_node_enumeration_order_not_confused_with_execution_order():
    value=fixture();value['lanes'][0]['operations'].reverse()
    assert validate(value)['transfer_arguments_checked'] is True


def test_missing_tail_copy_rejected():
    value=fixture();value['lanes'][0]['operations'].pop()
    with pytest.raises(ValueError):validate(value)


@pytest.mark.parametrize('field,bad',[
    ('bytes',624),('destination_offset',0),('source_offset',48),
    ('source','raw_scores'),('destination','lane_logits'),
    ('direction','host_to_device'),('bytes',True),('source_offset',-1)])
def test_corrupted_tail_transfer_rejected(field,bad):
    value=fixture();value['lanes'][0]['operations'][-1][field]=bad
    with pytest.raises(ValueError):validate(value)


@pytest.mark.parametrize('field,bad',[
    ('bytes',1535),('value',1),('element_size',3),
    ('destination','lane_logits'),('destination_offset',1)])
def test_incorrect_raw_reset_rejected(field,bad):
    value=fixture();value['lanes'][0]['operations'][0][field]=bad
    with pytest.raises(ValueError):validate(value)


@pytest.mark.parametrize('element_size',[1,2,4])
def test_equivalent_zero_reset_element_sizes_accept_normalized_byte_span(element_size):
    # CUDA memset width is in elements; zero with1/2/4-byte elements must
    # not reject the same independently verified normalized byte span.
    value=fixture()
    for operation in value['lanes'][0]['operations'][:2]:
        operation['element_size']=element_size
    assert validate(value)['checked_memset_nodes']==2


def test_duplicate_copy_cannot_substitute_missing_chunk():
    value=fixture()
    value['lanes'][0]['operations'][-1]=deepcopy(value['lanes'][0]['operations'][2])
    with pytest.raises(ValueError):validate(value)


def test_unknown_operation_not_ignored():
    value=fixture();value['lanes'][0]['operations'].append(dict(kind='unknown'))
    with pytest.raises(ValueError):validate(value)


def test_second_lane_must_have_its_own_complete_transfer_inventory():
    value=fixture();value['lanes'].append(dict(lane=1,operations=[]))
    with pytest.raises(ValueError):validate(value,lanes=2)


def test_duplicate_lane_rejected():
    value=fixture();value['lanes'].append(deepcopy(value['lanes'][0]))
    with pytest.raises(ValueError):validate(value,lanes=2)


def test_empty_inventory_cannot_pass_vacuously():
    value=fixture();value['lanes']=[]
    with pytest.raises(ValueError):validate(value)


def test_exact_divisible_profile_has_no_extra_tail():
    value=fixture();ops=value['lanes'][0]['operations']
    ops[0]['bytes']=1248;ops.pop()
    result=validate(value,outer=26)
    assert result['checked_memcpy_nodes']==2


def test_small_singleton_profile():
    value=fixture();ops=value['lanes'][0]['operations']
    ops[0]['bytes']=48;ops[2]['bytes']=48;del ops[3:]
    assert validate(value,outer=1,inner=1)['checked_memcpy_nodes']==1


@pytest.mark.parametrize('outer,inner,lanes',[
    (0,13,1),(32,0,1),(32,33,1),(True,13,1),(32,13,0),
    (32,13,True),(2**32,13,1),(32,13,65)])
def test_invalid_profile_bounds_rejected(outer,inner,lanes):
    with pytest.raises(ValueError):validate(fixture(),outer,inner,lanes)
