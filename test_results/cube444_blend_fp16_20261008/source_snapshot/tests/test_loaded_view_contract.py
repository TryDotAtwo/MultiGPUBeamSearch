"""Validate observed native view against fixed Cube4 diagnostic expectations."""
import copy
import pytest
from tools.loaded_view_contract import validate_cube4_diagnostic_view


def fixture():
    return dict(state_len=96, num_classes=6, num_pieces=56, max_piece_size=3,
        seq_len=57, padded_seq_len=64, sequence_alignment=16, d_model=256,
        nhead=8, head_dim=32, transformer_layers=4, ff_dim=1024, output_dim=24,
        dtype=0, activation=1,
        fp16_layouts=[dict(block=i, qkv_packed_fp16=False, ff1_packed_fp16=False,
                          ff2_packed_fp16=False) for i in range(4)],
        block_scales=[dict(block=i, qkv_e4m3_scale=0., ff1_e4m3_scale=0.,
                          ff2_e4m3_scale=0.) for i in range(4)])


def test_actual_unpacked_cube4_view_admitted_only_as_diagnostic():
    result = validate_cube4_diagnostic_view(fixture())
    assert result == {'diagnostic_view_matches': True, 'production_quality_accepted': False}


def test_compact57_requires_exact_declared_sequence_layout():
    observed = fixture()
    with pytest.raises(ValueError):
        validate_cube4_diagnostic_view(observed, compact57=True)
    observed.update(padded_seq_len=57, sequence_alignment=1)
    assert validate_cube4_diagnostic_view(observed, compact57=True)['diagnostic_view_matches']
    with pytest.raises(ValueError):
        validate_cube4_diagnostic_view(observed)
    with pytest.raises(ValueError):
        validate_cube4_diagnostic_view(observed, compact57=1)


@pytest.mark.parametrize('field', ['state_len', 'padded_seq_len', 'dtype', 'activation', 'transformer_layers'])
def test_wrong_dimension_or_bool_rejected(field):
    for value in (True, 999):
        observed = fixture()
        observed[field] = value
        with pytest.raises(ValueError):
            validate_cube4_diagnostic_view(observed)


@pytest.mark.parametrize('change', ['extra', 'missing', 'packed', 'scale', 'boolscale', 'nan', 'order', 'count'])
def test_unbound_view_rejected(change):
    observed = copy.deepcopy(fixture())
    if change == 'extra': observed['unbound'] = 1
    elif change == 'missing': del observed['dtype']
    elif change == 'packed': observed['fp16_layouts'][2]['qkv_packed_fp16'] = True
    elif change == 'scale': observed['block_scales'][1]['ff1_e4m3_scale'] = 0.25
    elif change == 'boolscale': observed['block_scales'][0]['ff2_e4m3_scale'] = False
    elif change == 'nan': observed['block_scales'][3]['qkv_e4m3_scale'] = float('nan')
    elif change == 'order': observed['fp16_layouts'].reverse()
    elif change == 'count': observed['block_scales'].pop()
    with pytest.raises(ValueError):
        validate_cube4_diagnostic_view(observed)
