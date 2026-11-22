"""CPU admission for the existing lossless Cube444 s3/mlp_x16 bundle."""
import hashlib
import json
import math
from pathlib import Path
import re

from .errors import NativeBackendError


def tensor_shapes():
    shapes = {'s3/cls_token': (1, 1, 256), 's3/local_value_embedding': (18, 256),
        's3/piece_projection_w': (768, 256), 's3/piece_projection_b': (256,),
        's3/piece_position_embedding': (56, 256), 's3/piece_type_embedding': (3, 256),
        's3/output_layer_w': (256, 24), 's3/output_layer_b': (24,),
        's3/piece_positions': (168,), 's3/piece_mask': (1, 56, 3, 1),
        's3/piece_types': (56,), 'mlp/position_offsets': (96,),
        'mlp/input_w': (576, 1536), 'mlp/input_b': (1536,),
        'mlp/hidden_w': (1536, 2304), 'mlp/hidden_b': (2304,),
        'mlp/out_w': (2304, 24), 'mlp/out_b': (24,)}
    for prefix in ('s3/input_norm', 's3/output_norm'):
        for suffix in ('w', 'b'): shapes[prefix+'_'+suffix] = (256,)
    for prefix, width in (('mlp/input_bn',1536), ('mlp/hidden_bn',2304)):
        for suffix in ('0','1'): shapes[prefix+'/'+suffix] = (width,)
    for block in range(4):
        p = f's3/blocks/{block}/'
        for norm in ('norm1','norm2'):
            for suffix in ('w','b'): shapes[p+norm+'_'+suffix] = (256,)
        for axis in 'qkv':
            shapes[p+'w'+axis] = (256,256); shapes[p+'b'+axis] = (256,)
        for name, width_in, width_out in (('out',256,256), ('ff1',256,1024), ('ff2',1024,256)):
            shapes[p+name+'_w'] = (width_in,width_out); shapes[p+name+'_b'] = (width_out,)
        p = f'mlp/blocks/{block}/'
        for layer in (1,2):
            shapes[p+f'l{layer}_w'] = (2304,2304)
            shapes[p+f'l{layer}_b'] = (2304,)
            for suffix in ('0','1'): shapes[p+f'bn{layer}/'+suffix] = (2304,)
    return shapes


def validate_bundle(path, contract, backend):
    import numpy as np
    from .models import PreparedModel
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate bundle key')
            result[key] = value
        return result
    path = Path(path).resolve()
    try:
        raw = (path/'blend.json').read_bytes()
        manifest = json.loads(raw, object_pairs_hook=unique)
        if (contract.state_len != 96 or contract.move_count != 24 or contract.num_classes != 6
                or manifest['backend'] != 'cube444_q_blend' or manifest['dtype'] != 'fp32'
                or manifest['state_len'] != 96 or manifest['num_classes'] != 6
                or manifest['output_dim'] != 24 or manifest['transformer_weight'] != .6
                or manifest['mlp_weight'] != .4
                or manifest.get('graph_hash', contract.graph_hash) != contract.graph_hash):
            raise ValueError('Cube444 graph or bundle contract mismatch')
        bindings = manifest['files']
        if not isinstance(bindings, dict): raise ValueError('invalid bundle bindings')
        digest = hashlib.sha256(raw); digest.update(contract.graph_hash.encode())
        files = ['blend.json']
        for name, expected in sorted(bindings.items()):
            if (name != 'tensors.tsv' and re.fullmatch(r'[0-9a-f]{64}\.bin', name) is None
                    or not isinstance(expected,str) or re.fullmatch(r'[0-9a-f]{64}',expected) is None):
                raise ValueError('unsafe bundle binding')
            source = (path/name).resolve()
            if not source.is_relative_to(path): raise ValueError('bundle file escaped snapshot')
            sha = hashlib.sha256()
            with source.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''): sha.update(chunk)
            if sha.hexdigest() != expected: raise ValueError('bundle checksum mismatch: '+name)
            digest.update(name.encode()); digest.update(sha.digest()); files.append(name)
        expected_shapes = tensor_shapes(); seen = set(); used = {'tensors.tsv'}
        for line in (path/'tensors.tsv').read_text().splitlines():
            key, name, dtype, dimensions = line.split('\t')
            shape = tuple(int(x) for x in dimensions.split(','))
            integer = key in ('s3/piece_positions','s3/piece_types','mlp/position_offsets')
            if (key in seen or key not in expected_shapes or shape != expected_shapes[key]
                    or dtype != ('int32' if integer else 'float32') or name not in bindings
                    or name in used or name != hashlib.sha256(key.encode()).hexdigest()+'.bin'):
                raise ValueError('invalid Cube444 tensor schema: '+key)
            seen.add(key); used.add(name)
            if (path/name).stat().st_size != math.prod(shape)*4:
                raise ValueError('Cube444 tensor size mismatch: '+key)
            values = np.memmap(path/name, dtype='<i4' if integer else '<f4', mode='r')
            if integer:
                if key == 'mlp/position_offsets': valid = np.array_equal(values,np.arange(96)*6)
                else: valid = bool(((values >= 0) & (values < (96 if key.endswith('positions') else 3))).all())
                if not valid: raise ValueError('invalid Cube444 index tensor: '+key)
            else:
                for offset in range(0,values.size,1 << 20):
                    chunk = values[offset:offset+(1 << 20)]
                    if not np.isfinite(chunk).all() or (np.abs(chunk)>65504).any():
                        raise ValueError('Cube444 tensor overflows FP16: '+key)
            del values
        if seen != set(expected_shapes) or used != set(bindings):
            raise ValueError('missing or unbound Cube444 tensors')
        execution = dict(manifest, dtype='fp16', source_dtype='fp32', graph_hash=contract.graph_hash)
        return PreparedModel(path,execution,backend,digest.hexdigest(),tuple(files))
    except (OSError,ValueError,KeyError,TypeError) as error:
        raise NativeBackendError('invalid Cube444 native artifact: '+str(error)) from error

