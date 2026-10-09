"""Next-window RED tests for previously unobserved FMHA Params defaults.

Not yet executed: local project tests are prohibited and paid window is closed.
"""
from copy import deepcopy
import importlib
import pytest


POINTERS = ('attn_bias_ptr', 'seqstart_q_ptr', 'seqstart_k_ptr', 'seqlen_k_ptr')
SCALARS = dict(causal_diagonal_offset=0, num_keys_absolute=0, bias_strideM=0,
               bias_strideH=0, bias_strideB=0, use_dropout=False,
               dropout_batch_head_rng_offset=0, dropout_prob=0.0)


def fixture():
    return dict(optional_pointers={name: dict(role='null', offset=0, available_bytes=0)
                                   for name in POINTERS},
                optional_scalars=deepcopy(SCALARS), pytorch_rng_state_present=False)


def validator():
    try:
        return importlib.import_module('tools.attention_optional_values').validate_attention_optional_members
    except ModuleNotFoundError:
        pytest.fail('independent optional FMHA Params gate missing')


def test_canonical_optional_params():
    assert validator()(fixture()) is True


@pytest.mark.parametrize('pointer', POINTERS)
def test_reject_optional_pointer(pointer):
    node = fixture()
    node['optional_pointers'][pointer] = dict(role='lane_qkv', offset=0, available_bytes=512)
    with pytest.raises(ValueError):
        validator()(node)


@pytest.mark.parametrize('field', list(SCALARS))
def test_reject_optional_scalar(field):
    node = fixture()
    value = SCALARS[field]
    node['optional_scalars'][field] = True if type(value) is bool else value + 1
    with pytest.raises(ValueError):
        validator()(node)


@pytest.mark.parametrize('mutation', ['missing_pointer', 'unknown_scalar', 'int_as_bool',
                                     'bool_as_offset', 'pytorch_rng', 'missing_metadata'])
def test_reject_malformed_optional_metadata(mutation):
    node = fixture()
    if mutation == 'missing_pointer':
        node['optional_pointers'].pop('seqstart_q_ptr')
    elif mutation == 'unknown_scalar':
        node['optional_scalars']['extra'] = 0
    elif mutation == 'int_as_bool':
        node['optional_scalars']['use_dropout'] = 0
    elif mutation == 'bool_as_offset':
        node['optional_pointers']['seqstart_k_ptr']['offset'] = False
    elif mutation == 'pytorch_rng':
        node['pytorch_rng_state_present'] = True
    else:
        node.pop('optional_scalars')
    with pytest.raises(ValueError):
        validator()(node)
