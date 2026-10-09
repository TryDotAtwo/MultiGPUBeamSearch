"""Frozen independent public-member byte layout; never semantic completeness."""
import hashlib
import json
from pathlib import Path
import stat
from tools.cube4_numeric_gate import unique

REFERENCE_SHA256='b20f0f0bd3c83960c82813ceb71c93dcab485b532ed225e40434820c58f206cf'
HEADERS={
    'include/cutlass/tensor_ref.h':'a1f06211976fbf7517aa165474ab733dc49ef0544f140d8543dcd407e01c281e',
    'include/cutlass/layout/matrix.h':'2707e006e1b126fbf288a444e3e0ae6710290cf318920badcefbd17e78476efa',
    'include/cutlass/epilogue/threadblock/predicated_tile_iterator.h':'e19a085bea33559dd76ed10bcf478a3e5c0c2ec963bfbf7c901905e176d20305',
    'include/cutlass/gemm/kernel/gemm.h':'8e293ca5a8b3cf3e891d83a404e16de654998a2a497f8156114c4c7d14a4e237',
    'include/cutlass/gemm/kernel/gemm_with_fused_epilogue.h':'15700ac97cc81f6d4492a49af2b858dfacff2cce362a106f6b8c744b218c5893',
    'include/cutlass/gemm/kernel/params_universal_base.h':'42752a9340fb31af198af1d9d185bfe6434a4b64822ef30767759bd4c2a5cf45',
    'include/cutlass/epilogue/thread/linear_combination.h':'7a2342568f7edb5cb87282c82d08c3c7cd3c150a71e55f6f7bce618ddc1585bd',
    'examples/41_fused_multi_head_attention/kernel_forward.h':'ed88c8932b43c4455d9ba2a9474da6984fa4ab7a87c21a5805cfe3a70c6a128c'}

def bounded_bytes(path):
    path=Path(path)
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size>65536:
        raise ValueError('public layout authority must be a bounded regular file')
    raw=path.read_bytes()
    if len(raw)>65536:raise ValueError('public layout authority exceeds bound')
    return raw

def reference():
    raw=bounded_bytes(Path(__file__).with_name('parameter_public_storage_v1.json'))
    if hashlib.sha256(raw).hexdigest()!=REFERENCE_SHA256:
        raise ValueError('independent public layout reference changed')
    return json.loads(raw,object_pairs_hook=unique)

def validate_table(value,kind):
    wanted=reference()[kind]
    # JSON spelling preserves int/bool and int/float distinctions recursively.
    if json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(wanted,sort_keys=True):
        raise ValueError('compiled public field layout differs from frozen source-bound reference')
    return value

def validate_gemm_public_storage(value):return validate_table(value,'gemm')
def validate_attention_public_storage(value):return validate_table(value,'attention')

def nested_reference():
    raw=bounded_bytes(Path(__file__).with_name('gemm_nested_storage_v1.json'))
    if hashlib.sha256(raw).hexdigest()!='7c5225714ed3a11576bde859f8fed8825ed21389ea3b458dfc2bbb4c3acbe19c':
        raise ValueError('independent nested GEMM reference changed')
    return json.loads(raw,object_pairs_hook=unique)

def validate_nested_gemm_storage(value):
    if json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(nested_reference(),sort_keys=True):
        raise ValueError('nested GEMM storage differs from source-bound independent reference')
    return value

def observe_public_layout_headers(build):
    cache=(Path(build)/'CMakeCache.txt').read_bytes()
    if len(cache)>1024*1024:raise ValueError('public layout CMake cache exceeds bound')
    roots=[line.split('=',1)[1] for line in cache.decode().splitlines() if line.startswith('CUTLASS_DIR:PATH=')]
    if len(roots)!=1:raise ValueError('missing unique public layout CUTLASS binding')
    root=Path(roots[0]).resolve(strict=True);observed={}
    reference()
    nested_reference()
    for name,wanted in HEADERS.items():
        path=root/name
        if path.resolve(strict=True)!=path:raise ValueError('public layout authority path changed')
        digest=hashlib.sha256(bounded_bytes(path)).hexdigest()
        if digest!=wanted:raise ValueError('public parameter declaration drift: '+name)
        observed[name]=digest
    return observed
