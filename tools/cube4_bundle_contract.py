"""Strict startup contract for the fixed native-FP16 Cube4 model bundle.

Hashes attest consistency with the export manifest, not training quality or
authenticity. Legacy unsealed exports must be regenerated from trusted weights;
validation never silently blesses whatever bytes happen to be present.
"""
from collections import Counter
from array import array
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


MODEL = dict(backend='piece_transformer', state_len=96, move_count=24, output_dim=24,
             seq_len=57, dtype='fp16', activation='relu', pooling='cls', num_classes=6,
             num_pieces=56, max_piece_size=3, num_piece_types=3, d_model=256,
             nhead=8, head_dim=32, num_layers=4, ff_dim=1024, piece_layout='cube4',
             piece_embed_mode='piece_local', input_embedding='fast_slot_projected')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'non-JSON numeric constant: {value}')


def _json(path):
    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain an object')
    return value


def _u8_list(value, length, upper, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{label} must contain exactly {length} integers')
    if any(type(v) is not int or not 0 <= v < upper for v in value):
        raise ValueError(f'{label} must contain unsigned integers in [0,{upper})')
    return value


def cube4_tensor_sizes():
    d, ff = 256, 1024
    sizes = {name + '.fp16': 2 * count for name, count in {
        'cls_token': d, 'fast_piece_static': 56 * d, 'fast_slot_projected': 3 * 6 * d,
        'input_ln_beta': d, 'input_ln_gamma': d, 'output_bias': 24,
        'output_ln_beta': d, 'output_ln_gamma': d, 'output_weight_hxk': d * 24}.items()}
    for block in range(4):
        for name, count in dict(attn_out_bias=d, attn_out_weight_hxk=d*d,
                                attn_qkv_bias=3*d, attn_qkv_weight_hxk=3*d*d,
                                ff1_bias=ff, ff1_weight_hxk=d*ff, ff2_bias=d,
                                ff2_weight_hxk=ff*d, ln1_beta=d, ln1_gamma=d,
                                ln2_beta=d, ln2_gamma=d).items():
            sizes[f'block{block}_{name}.fp16'] = 2 * count
    sizes.update({'piece_positions.u16': 56 * 3 * 2, 'piece_mask.u8': 56 * 3, 'piece_types.u8': 56})
    return sizes


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tensor_values(weights_dir, names):
    for name in names:
        if name.endswith('.fp16'):
            words = array('H')
            words.frombytes((weights_dir / name).read_bytes())
            if sys.byteorder != 'little': words.byteswap()
            if any((word & 0x7c00) == 0x7c00 for word in words):
                raise ValueError(f'{name}: non-finite FP16 value')
    mask = (weights_dir / 'piece_mask.u8').read_bytes()
    if any(value not in (0, 1) for value in mask):
        raise ValueError('piece_mask must contain only 0/1')
    positions = struct.unpack('<168H', (weights_dir / 'piece_positions.u16').read_bytes())
    active = [position for position, keep in zip(positions, mask) if keep]
    if sorted(active) != list(range(96)):
        raise ValueError('active piece_positions must partition all 96 logical positions')
    if any(sum(mask[p:p + 3]) == 0 for p in range(0, 168, 3)):
        raise ValueError('every piece must contain a logical position')
    if any(value >= 3 for value in (weights_dir / 'piece_types.u8').read_bytes()):
        raise ValueError('piece_types must be in [0,3)')


def validate_cube4_weights(weights_dir: Path) -> dict:
    """Validate prepared native tensors and return their observed identities.

    Consistency with a sealed manifest is not authenticity or proof the scorer
    loaded these bytes. Callers must also bind execution and prevent mutation.
    """
    model = _json(weights_dir / 'manifest.json')
    for key, expected in MODEL.items():
        actual = model.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'model {key} must be {expected!r}, got {actual!r}')
    expected_sizes = cube4_tensor_sizes()
    observed = {}
    records = model.get('tensor_files')
    if not isinstance(records, dict) or set(records) != set(expected_sizes):
        raise ValueError('manifest tensor_files must bind every expected tensor; regenerate a trusted export')
    for name, size in expected_sizes.items():
        record, path = records[name], weights_dir / name
        if not isinstance(record, dict) or type(record.get('size_bytes')) is not int or record['size_bytes'] != size:
            raise ValueError(f'{name}: manifest byte size differs from model shape')
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f'{name}: expected exactly {size} bytes')
        digest = record.get('sha256')
        measured = _sha256(path)
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) or measured != digest:
            raise ValueError(f'{name}: content hash mismatch')
        observed[name] = measured
    _validate_tensor_values(weights_dir, expected_sizes)
    return {'model': model, 'tensor_sha256': observed}


def validate_cube4_bundle(data_dir: Path, weights_dir: Path, puzzle_id: str) -> dict:
    puzzle = _json(data_dir / 'puzzle_info.json')
    model = validate_cube4_weights(weights_dir)['model']
    central = _u8_list(puzzle.get('central_state'), 96, 6, 'central_state')
    if Counter(central) != Counter({i: 16 for i in range(6)}):
        raise ValueError('Cube4 central_state must contain 16 stickers of each of 6 classes')
    generators = puzzle.get('generators')
    if not isinstance(generators, dict) or len(generators) != 24:
        raise ValueError('Cube4 requires exactly 24 named generators')
    for name, permutation in generators.items():
        if not name or len(set(_u8_list(permutation, 96, 96, f'generator {name}'))) != 96:
            raise ValueError(f'generator {name} must be a bijection of [0,96)')
    if model.get('move_names') != list(generators):
        raise ValueError('model output order differs from puzzle generators')
    if not re.fullmatch(r'0|[1-9][0-9]*', puzzle_id):
        raise ValueError('puzzle_id must be a canonical unsigned integer')
    if len(puzzle_id) > 20 or (len(puzzle_id) == 20 and puzzle_id > '18446744073709551615'):
        raise ValueError('puzzle_id exceeds uint64')
    selected = None
    seen = set()
    with (data_dir / 'test.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {'initial_state_id', 'initial_state'}.issubset(fields):
            raise ValueError('test.csv requires unique initial_state_id/initial_state columns')
        for row in reader:
            row_id = row.get('initial_state_id')
            if None in row or any(v is None for v in row.values()):
                raise ValueError('test.csv row width differs from its header')
            if not re.fullmatch(r'0|[1-9][0-9]*', row_id or '') or row_id in seen:
                raise ValueError('test.csv puzzle IDs must be unique canonical unsigned integers')
            if len(row_id) > 20 or (len(row_id) == 20 and row_id > '18446744073709551615'):
                raise ValueError('test.csv puzzle ID exceeds uint64')
            seen.add(row_id)
            tokens = row['initial_state'].split(',')
            if len(tokens) != 96 or any(not re.fullmatch(r'[0-5]', t.strip()) for t in tokens):
                raise ValueError(f'puzzle {row_id} requires exactly 96 integer classes in [0,6)')
            state = [int(t) for t in tokens]
            if Counter(state) != Counter(central):
                raise ValueError(f'puzzle {row_id} sticker multiset differs from central_state')
            if row_id == puzzle_id:
                selected = state
    if selected is None:
        raise ValueError(f'puzzle {puzzle_id} not found in test.csv')
    return {'model': model, 'puzzle_id': puzzle_id, 'initial_state': selected,
            'input_sha256': {name: _sha256(data_dir / name) for name in ('puzzle_info.json', 'test.csv')}}
