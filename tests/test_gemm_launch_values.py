"""Reject altered launch geometry, split-K/batching and malformed scalar types."""
from copy import deepcopy
import importlib
import pytest


def validator():
    try:
        module = importlib.import_module('tools.gemm_launch_values')
    except ModuleNotFoundError:
        pytest.fail('independent launch-value gate is missing')
    return module.validate_gemm_launch_values


def captured(broadcast=False):
    values = dict(grid_tiled_shape=dict(m=7, n=12, k=1),
                  swizzle_log_tile=0, gemm_k_size=256, semaphore_null=True)
    if broadcast:
        values.update(mode=0, batch_count=1, batch_stride_D=639744)
    return dict(launch_scalars=values)


@pytest.mark.parametrize('broadcast', [False, True])
def test_valid_single_partition_launch(broadcast):
    assert validator()(captured(broadcast), rows=833, columns=768,
                       reduction=256, tile=(128, 64, 32), broadcast=broadcast)


@pytest.mark.parametrize('field,value', [
    ('swizzle_log_tile', 1), ('swizzle_log_tile', False),
    ('gemm_k_size', 1024), ('gemm_k_size', 256.0),
    ('semaphore_null', False), ('semaphore_null', 1),
    ('mode', 2), ('batch_count', 9), ('batch_stride_D', 4096),
    ('grid_tiled_shape', dict(m=7, n=12, k=2)),
    ('grid_tiled_shape', dict(m=6, n=12, k=1)),
    ('grid_tiled_shape', dict(m=7, n=12, k=True)),
])
def test_reject_altered_launch(field, value):
    node = deepcopy(captured(True))
    node['launch_scalars'][field] = value
    with pytest.raises(ValueError):
        validator()(node, rows=833, columns=768, reduction=256,
                    tile=(128, 64, 32), broadcast=True)


@pytest.mark.parametrize('values', [None, {}, {'unexpected': 1}])
def test_missing_or_unknown_launch_metadata(values):
    with pytest.raises(ValueError):
        validator()(dict(launch_scalars=values), rows=833, columns=768,
                    reduction=256, tile=(128, 64, 32), broadcast=False)
