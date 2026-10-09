"""Reject corrupted inputs before GPU startup; values are independent fixtures."""
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.verify_cube4_bundle import verify


@pytest.fixture
def bundle(tmp_path):
    data, weights = tmp_path / 'data', tmp_path / 'weights'
    data.mkdir(); weights.mkdir()
    moves = [f'm{i}' for i in range(24)]
    central = [i // 16 for i in range(96)]
    puzzle = {'central_state': central, 'generators': {m: list(range(96)) for m in moves}}
    model = dict(backend='piece_transformer', state_len=96, move_count=24, output_dim=24,
                 seq_len=57, dtype='fp16', activation='relu', pooling='cls', num_classes=6,
                 num_pieces=56, max_piece_size=3, num_piece_types=3, d_model=256,
                 nhead=8, head_dim=32, num_layers=4, ff_dim=1024, piece_layout='cube4',
                 piece_embed_mode='piece_local', input_embedding='fast_slot_projected', move_names=moves)
    # Exact independent byte counts for the fixed Cube4 architecture.
    sizes = dict(cls_token=512, fast_piece_static=28672, fast_slot_projected=9216,
                 input_ln_beta=512, input_ln_gamma=512, output_bias=48,
                 output_ln_beta=512, output_ln_gamma=512, output_weight_hxk=12288)
    block_sizes = dict(attn_out_bias=512, attn_out_weight_hxk=131072, attn_qkv_bias=1536,
                       attn_qkv_weight_hxk=393216, ff1_bias=2048, ff1_weight_hxk=524288,
                       ff2_bias=512, ff2_weight_hxk=524288, ln1_beta=512, ln1_gamma=512,
                       ln2_beta=512, ln2_gamma=512)
    for block in range(4):
        sizes.update({f'block{block}_{key}': size for key, size in block_sizes.items()})
    for name, size in sizes.items(): (weights / (name + '.fp16')).write_bytes(bytes(size))
    positions, mask, cursor = [], [], 0
    for size in [3] * 8 + [2] * 24 + [1] * 24:
        positions.extend(list(range(cursor, cursor + size)) + [0] * (3 - size))
        mask.extend([1] * size + [0] * (3 - size)); cursor += size
    (weights / 'piece_positions.u16').write_bytes(struct.pack('<168H', *positions))
    (weights / 'piece_mask.u8').write_bytes(bytes(mask))
    (weights / 'piece_types.u8').write_bytes(bytes([0] * 8 + [1] * 24 + [2] * 24))
    model['tensor_files'] = {p.name: {'size_bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in weights.iterdir()}
    (data / 'puzzle_info.json').write_text(json.dumps(puzzle))
    (weights / 'manifest.json').write_text(json.dumps(model))
    with (data / 'test.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['initial_state_id', 'initial_state'])
        w.writerow(['1000', ','.join(map(str, central))])
    return data, weights


def edit_json(path, change):
    value = json.loads(path.read_text()); change(value); path.write_text(json.dumps(value))


def test_complete_bundle_is_accepted(bundle):
    verify(*bundle, '1000')


@pytest.mark.parametrize('row_id', ['18446744073709551616', '9' * 5000], ids=['max-plus-one', 'long-decimal'])
def test_bundle_rejects_uint64_overflow_in_any_csv_row(bundle, row_id):
    data, _ = bundle
    with (data / 'test.csv').open('a', newline='') as stream:
        csv.writer(stream).writerow([row_id, ','.join(str(i // 16) for i in range(96))])
    with pytest.raises(ValueError, match='uint64'):
        verify(*bundle, '1000')


def test_bundle_rejects_requested_uint64_overflow_before_csv_lookup(bundle):
    with pytest.raises(ValueError, match='uint64'):
        verify(*bundle, '18446744073709551616')


@pytest.mark.parametrize('row_id', ['0', '18446744073709551615'])
def test_bundle_accepts_uint64_id_boundaries(bundle, row_id):
    data, _ = bundle
    with (data / 'test.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['initial_state_id', 'initial_state'])
        writer.writerow([row_id, ','.join(str(i // 16) for i in range(96))])
    verify(*bundle, row_id)


def test_weight_admission_returns_observed_tensor_identity(bundle):
    # Losing an actually loaded tensor from the observed identity must fail.
    from tools import cube4_bundle_contract as contract
    _, weights = bundle
    result = contract.validate_cube4_weights(weights)
    expected = json.loads((weights / 'manifest.json').read_text())['tensor_files']
    assert result['tensor_sha256'] == {name: record['sha256'] for name, record in expected.items()}
    assert result['model']['state_len'] == 96


def test_weight_admission_rejects_changed_tensor_with_unchanged_manifest(bundle):
    from tools import cube4_bundle_contract as contract
    _, weights = bundle
    path = weights / 'output_bias.fp16'
    path.write_bytes(b'\x00\x3c' + bytes(46))
    with pytest.raises(ValueError, match='content hash mismatch'):
        contract.validate_cube4_weights(weights)


@pytest.mark.parametrize('value', [-1, 96, 255, True, 1.5, '1'])
def test_generator_invalid_index_rejected(bundle, value):
    data, _ = bundle
    edit_json(data / 'puzzle_info.json', lambda p: p['generators']['m0'].__setitem__(0, value))
    with pytest.raises(ValueError): verify(*bundle, '1000')


def test_generator_not_bijective_rejected(bundle):
    data, _ = bundle
    edit_json(data / 'puzzle_info.json', lambda p: p['generators']['m0'].__setitem__(0, 1))
    with pytest.raises(ValueError): verify(*bundle, '1000')


@pytest.mark.parametrize('value', [-1, 6, 999, True, '0'])
def test_central_invalid_class_rejected(bundle, value):
    data, _ = bundle
    edit_json(data / 'puzzle_info.json', lambda p: p['central_state'].__setitem__(0, value))
    with pytest.raises(ValueError): verify(*bundle, '1000')


@pytest.mark.parametrize('token', ['-1', '6', 'x', '0.0', '1e0', ''])
def test_csv_invalid_class_rejected(bundle, token):
    data, _ = bundle
    state = [str(i // 16) for i in range(96)]; state[0] = token
    with (data / 'test.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['initial_state_id', 'initial_state']); w.writerow(['1000', ','.join(state)])
    with pytest.raises(ValueError): verify(*bundle, '1000')


@pytest.mark.parametrize('field,value', [('d_model', 128), ('num_classes', 96), ('nhead', 4),
                                      ('ff_dim', 512), ('num_layers', 3), ('piece_layout', 'p900')])
def test_wrong_model_shape_rejected(bundle, field, value):
    _, weights = bundle
    edit_json(weights / 'manifest.json', lambda m: m.__setitem__(field, value))
    with pytest.raises(ValueError): verify(*bundle, '1000')


@pytest.mark.parametrize('mutation', ['one_byte', 'same_size_changed', 'missing_hash', 'wrong_declared_size'])
def test_tensor_corruption_rejected(bundle, mutation):
    _, weights = bundle
    path = weights / 'cls_token.fp16'
    if mutation == 'one_byte': path.write_bytes(b'x')
    if mutation == 'same_size_changed': path.write_bytes(b'\x01' + path.read_bytes()[1:])
    if mutation == 'missing_hash':
        edit_json(weights / 'manifest.json', lambda m: m.pop('tensor_files'))
    if mutation == 'wrong_declared_size':
        edit_json(weights / 'manifest.json', lambda m: m['tensor_files']['cls_token.fp16'].__setitem__('size_bytes', 1))
    with pytest.raises(ValueError): verify(*bundle, '1000')


def test_csv_state_multiset_must_match_central(bundle):
    data, _ = bundle
    with (data / 'test.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['initial_state_id', 'initial_state']); w.writerow(['1000', ','.join(['0'] * 96)])
    with pytest.raises(ValueError): verify(*bundle, '1000')


def test_duplicate_json_key_rejected(bundle):
    data, _ = bundle
    p = data / 'puzzle_info.json'; text = p.read_text()
    p.write_text('{"central_state": [],' + text[1:])
    with pytest.raises(ValueError): verify(*bundle, '1000')


@pytest.mark.parametrize('name,content', [('piece_positions.u16', struct.pack('<H', 96)),
                                        ('piece_mask.u8', b'\x02'), ('piece_types.u8', b'\x03'),
                                        ('cls_token.fp16', b'\x00\x7c')])
def test_hash_consistent_invalid_tensor_values_rejected(bundle, name, content):
    _, weights = bundle
    path = weights / name; path.write_bytes(content + path.read_bytes()[len(content):])
    edit_json(weights / 'manifest.json', lambda m: m['tensor_files'][name].__setitem__(
        'sha256', hashlib.sha256(path.read_bytes()).hexdigest()))
    with pytest.raises(ValueError): verify(*bundle, '1000')
