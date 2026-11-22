"""Admission of Cube555 feature scripts and explicit FP32 ensemble readouts.

This validates the artifact/graph boundary. It does not establish GPU execution
support or prove training provenance from the caller's graph declaration.
"""
import hashlib
import json
from pathlib import Path
import re

from .errors import NativeBackendError

FAMILIES = ('cube555_transformer', 'cube555_mlp')
FILES = {'features.ts', 'output_weight.fp16', 'output_bias.fp32', 'layout.json', 'puzzle_info.json'}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate Cube555 JSON key: '+key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_definition(layout, info, contract):
    if not isinstance(layout, dict) or not isinstance(info, dict):
        raise ValueError('Cube555 layout and puzzle info must be JSON objects')
    def integers(values, count, upper):
        return (isinstance(values, list) and len(values) == count
                and all(type(x) is int and 0 <= x < upper for x in values))
    if (contract.state_len != 150 or contract.num_classes != 150 or contract.move_count != 30
            or tuple(contract.center) != tuple(range(150))):
        raise ValueError('Cube555 requires the identity 150-sticker / 30-move graph')
    if (info.get('central_state') != list(contract.center)
            or not isinstance(info.get('generators'), dict)
            or list(info['generators']) != list(contract.generator_names)
            or list(info['generators'].values()) != [list(g) for g in contract.generators]
            or any(sorted(g) != list(range(150)) for g in info['generators'].values())):
        raise ValueError('Cube555 ordered graph/center mismatch')
    data = dict(generators_type='PERMUTATION', generators=list(info['generators'].values()),
                central_state=info['central_state'])
    if hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() != contract.graph_hash:
        raise ValueError('Cube555 graph hash mismatch')
    if (layout.get('state_size') != 150 or layout.get('num_pieces') != 98
            or layout.get('n_actions') != 30 or layout.get('_move_names') != list(contract.generator_names)):
        raise ValueError('Cube555 feature layout / move order mismatch')
    vocab, types, orbits = layout.get('sticker_vocab'), layout.get('num_piece_types'), layout.get('orbit_sizes')
    if (type(vocab) is not int or not 1 <= vocab <= 150 or type(types) is not int or not 1 <= types <= 98
            or not isinstance(orbits, list) or not orbits
            or any(type(x) is not int or x <= 0 for x in orbits)
            or not integers(layout.get('head_facelet'), 98, 150)
            or len(set(layout['head_facelet'])) != 98
            or not integers(layout.get('sticker_local'), 150, vocab)
            or not integers(layout.get('sticker_orbit'), 150, len(orbits))
            or not integers(layout.get('piece_types'), 98, types)):
        raise ValueError('invalid Cube555 feature layout indices')


def validate_bundle(path, contract, backend):
    import numpy as np
    from .models import PreparedModel
    path = Path(path).resolve()
    try:
        raw = (path/'manifest.json').read_bytes()
        manifest = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(manifest, dict):
            raise ValueError('Cube555 manifest must be an object')
        width = manifest['feature_dim']
        if (backend not in FAMILIES or manifest['family'] != backend
                or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1
                or manifest['dtype'] != 'fp16' or manifest['state_len'] != 150
                or manifest['num_classes'] != 150 or manifest['output_dim'] != 30
                or type(width) is not int or not 8 <= width <= 65536 or width % 8
                or manifest['graph_hash'] != contract.graph_hash
                or manifest['move_names'] != list(contract.generator_names)
                or manifest['score_semantics'] != 'distance_q'
                or manifest['readout_semantics'] != 'fp16_features_fp32_accumulation'
                or manifest['feature_input_dtype'] != 'uint8'
                or manifest['feature_output_dtype'] != 'fp16'
                or manifest['output_weight_layout'] != 'hidden_by_output'
                or manifest['layernorm_eps'] != 1e-5
                or type(manifest['hidden1']) is not int or manifest['hidden1'] != width
                or type(manifest['hidden2']) is not int or manifest['hidden2'] != width
                or type(manifest['residual_count']) is not int or manifest['residual_count'] != 1
                or manifest['normalization'] != 'none' or manifest['planning_descriptor_only'] is not True):
            raise ValueError('Cube555 artifact graph/shape/score contract mismatch')
        for key in ('checkpoint_sha256', 'layout_sha256', 'puzzle_info_sha256'):
            if not isinstance(manifest[key], str) or re.fullmatch('[0-9a-f]{64}', manifest[key]) is None:
                raise ValueError('invalid Cube555 provenance hash')
        bindings = manifest['files']
        if (not isinstance(bindings, dict) or set(bindings) != FILES
                or any(not isinstance(h, str) or re.fullmatch('[0-9a-f]{64}', h) is None
                       for h in bindings.values())):
            raise ValueError('Cube555 bundle requires exact file bindings')
        digest = hashlib.sha256(raw)
        digest.update(contract.graph_hash.encode('ascii'))
        for name in sorted(FILES):
            source = (path/name).resolve()
            if not source.is_relative_to(path):
                raise ValueError('Cube555 artifact escaped snapshot')
            actual = file_hash(source)
            if actual != bindings[name]:
                raise ValueError('Cube555 checksum mismatch: '+name)
            digest.update(name.encode('ascii'))
            digest.update(bytes.fromhex(actual))
        if (bindings['layout.json'] != manifest['layout_sha256']
                or bindings['puzzle_info.json'] != manifest['puzzle_info_sha256']):
            raise ValueError('Cube555 source hash mismatch')
        validate_definition(read_json(path/'layout.json'), read_json(path/'puzzle_info.json'), contract)
        for name, size, dtype in [('output_weight.fp16', width*30, '<f2'), ('output_bias.fp32', 30, '<f4')]:
            if (path/name).stat().st_size != size*np.dtype(dtype).itemsize:
                raise ValueError('Cube555 readout size mismatch: '+name)
            if not np.isfinite(np.fromfile(path/name, dtype=dtype)).all():
                raise ValueError('Cube555 readout contains nonfinite values')
        return PreparedModel(path, manifest, backend, digest.hexdigest(), ('manifest.json', *sorted(FILES)))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        raise NativeBackendError('invalid Cube555 artifact: '+str(error)) from error
