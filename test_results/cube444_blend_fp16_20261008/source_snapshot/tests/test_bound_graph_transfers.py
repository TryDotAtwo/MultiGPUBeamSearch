"""Transfer sidecars must bind actual execution and inventory bytes."""
from copy import deepcopy
import pytest
from test_graph_transfer_contract import fixture


def bindings():
    expected = dict(outer_microbatch=32, transformer_microbatch=13, lanes=1,
                    executor='native_cuda_graph', device={'sm':75,'uuid_hex':'a'*32})
    hashes = {'execution.json':'b'*64, 'graph_kernel_inventory.json':'c'*64}
    value = fixture()
    value.update(device=deepcopy(expected['device']), execution_sha256='b'*64,
                 inventory_sha256='c'*64)
    return value, expected, hashes


def check(value, expected, hashes):
    from tools.graph_transfer_contract import validate_bound_score_graph_transfers
    return validate_bound_score_graph_transfers(value, expected, hashes)


def test_bound_transfer_checks_remain_unadmitted():
    result = check(*bindings())
    assert result['transfer_arguments_checked'] is True
    assert result['production_admitted'] is False


@pytest.mark.parametrize('field', ['device','execution_sha256','inventory_sha256'])
@pytest.mark.parametrize('mutation', ['missing','wrong'])
def test_missing_or_stale_binding_rejected(field, mutation):
    value, expected, hashes = bindings()
    if mutation == 'missing': del value[field]
    else: value[field] = {'sm':80} if field == 'device' else '0'*64
    with pytest.raises(ValueError): check(value, expected, hashes)


def test_nested_device_type_coercion_is_not_equality():
    value, expected, hashes = bindings()
    value['device']['sm'] = 75.0
    with pytest.raises(ValueError): check(value, expected, hashes)


def test_eager_cannot_use_graph_sidecar():
    value, expected, hashes = bindings(); expected['executor']='native_eager'
    with pytest.raises(ValueError): check(value, expected, hashes)


def test_bound_corrupt_copy_rejected():
    value, expected, hashes = bindings()
    value['lanes'][0]['operations'][-1]['bytes']=48
    with pytest.raises(ValueError): check(value, expected, hashes)
