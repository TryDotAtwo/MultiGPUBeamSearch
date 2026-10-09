"""Pinned source audit of generic Cube4 input NetworkView consumption only."""
import hashlib
from pathlib import Path
import stat
PINS={'cuda/stream1_transformer.cu':'ac92eb66541d6ca6d0f401d41491094b14759536ac4e72c0141ab5e9aa58d9bc',
      'cuda/stream1.hpp':'b0456951deae151c4a36a2e11150a9248aab176a5cb83b9802771945fab87c41'}
POINTER_MEMBERS=frozenset(('fast_slot_projected','fast_piece_static','cls_token','input_ln_gamma',
    'input_ln_beta','output_ln_gamma','output_ln_beta','blocks','output_weight','output_bias',
    'piece_positions','piece_mask','piece_types'))
COMMON=frozenset(('fast_slot_projected','fast_piece_static','cls_token','piece_positions','piece_mask'))
CONSUMED={'input_graph':COMMON,'fused_input':COMMON|{'input_ln_gamma','input_ln_beta'}}

def observe_network_consumption_sources(source):
    root=Path(source).resolve(strict=True);hashes={}
    for name,wanted in PINS.items():
        path=root/name
        if not stat.S_ISREG(path.lstat().st_mode) or path.resolve(strict=True)!=path or path.stat().st_size>1024*1024:
            raise ValueError('invalid bounded input consumption authority')
        raw=path.read_bytes()
        if len(raw)>1024*1024:raise ValueError('input consumption source exceeds bound')
        digest=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()
        if digest!=wanted:raise ValueError('input NetworkView consumption source changed: '+name)
        hashes[name]=digest
    return {'scope':'pinned_generic_input_network_pointer_consumption_not_full_admission',
        'sha256':hashes,'consumed':{k:sorted(v) for k,v in CONSUMED.items()},
        'unused':{k:sorted(POINTER_MEMBERS-v) for k,v in CONSUMED.items()},
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
