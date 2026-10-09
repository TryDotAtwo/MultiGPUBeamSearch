"""Independent unpacked FP16 SM80/86 default host recipe; not kernel admission.

Enum/default authority: cuda/stream1_resolved_gemm.hpp,
stream1_transformer_gemm_policy.hpp and stream1_resolved_launch.hpp.
Expected values never come from resolved_execution.json.
"""
import json
import hashlib
from pathlib import Path
import stat

SOURCE_PINS = {
    'cuda/stream1_loaded_execution_admission.hpp':'9f50cbc7184a83c3a81ca39463028ac62e5c99676472f9d15b7f804ae90337f1',
    'cuda/stream1_execution_contract.hpp':'a5003277cd80217badf40bf4dba976417511b12e2b8c79184d5da83fdad6d5a0',
    'cuda/stream1_resolved_gemm.hpp':'e7d32e4f4a5d37b85f62db855bc5ab93facbb13bdcb2f6b531b20b5062ae62b9',
    'cuda/stream1_resolved_launch.hpp':'c66ade0cedf0237f5372e1d9d31647501f1b17a752bd351b04528fce6dbf4231',
    'cuda/stream1_transformer_gemm_policy.hpp':'39d9e09e7b803fbcf4d42b8cc9a4d92d3146c33303dcd558059266c686cb736a',
    'cuda/stream1_transformer_attention_policy.hpp':'19eab3ea99c70d3dff1204ab92fbcc02f8023d89895b48ef9b1dc0cb1af4df57',
    'cuda/stream1_transformer_layernorm_policy.hpp':'4d1f1e35863ba04ab36cf073a15a6631b036d9f6867f0f5a58fae93de8e10100',
    'cuda/stream1_transformer_hopper_policy.hpp':'803059922863a9fbb2c71a36f4c5702bdf300ffb30d742063c578442addccd07',
    'cuda/stream1_layernorm_execution.hpp':'a040426d5e7ed6b06ea912016905091ea276a1b52ccef125d3f22dbbbba6234d'}


def observe_recipe_sources(source):
    """Version-bound defaults; stable source observations do not attest a binary."""
    root=Path(source).resolve(strict=True)
    observed={}
    for name,wanted in SOURCE_PINS.items():
        path=root/name
        metadata=path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.resolve(strict=True)!=path:
            raise ValueError('resolved recipe source must be a regular file: '+name)
        if metadata.st_size>1024*1024:
            raise ValueError('resolved recipe source exceeds bound')
        raw=path.read_bytes()
        if len(raw)>1024*1024: raise ValueError('resolved recipe source exceeds bound')
        digest=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()
        if digest!=wanted: raise ValueError('independent resolved source version mismatch: '+name)
        observed[name]=digest
    observed['validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return observed

PREFIX = 'BEAM_STREAM1_TRANSFORMER_'
FLAG_NAMES = ('LEGACY_PADDING_ZERO', 'BLOCK51', 'FINAL_CLS_ONLY',
              'FINAL_CLS_ATTENTION', 'FINAL_CLS_SPLIT_QKV',
              'FUSED_INPUT_LAYERNORM', 'DUAL_INPUT_LN', 'STAGE_PROFILE')
SELECTOR_NAMES = FLAG_NAMES + (
    'LAYERNORM_ROWS_POLICY', 'LAYERNORM_PERSISTENT_BLOCKS_PER_SM',
    'FF2_POLICY', 'ATTN_OUT_POLICY', 'FF2_EPILOGUE', 'ATTN_OUT_EPILOGUE',
    'FF2_SWIZZLE', 'ATTN_OUT_SWIZZLE', 'QKV_POLICY', 'QKV_SWIZZLE',
    'HOPPER_FF1_EPILOGUE', 'FF1_POLICY', 'FF1_STAGES', 'FF1_SWIZZLE',
    'STAGE_PROFILE_SKIP_CALLS', 'ATTENTION_MAX_K_POLICY',
    'ATTENTION_TILE_POLICY', 'CLS_ATTENTION_POLICY')


def typed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def canonical_resolved_recipe(expected):
    """Fail closed outside the independently described canonical route."""
    sm = expected['device']['sm']
    if type(sm) is not int or sm not in (80, 86):
        raise ValueError('independent resolved recipe supports SM80/86 only')
    if expected['executor'] not in ('native_cuda_graph', 'native_eager'):
        raise ValueError('unsupported resolved executor')
    selectors = expected['requested_launch_policies']
    if not isinstance(selectors, dict) or set(selectors) != {PREFIX+n for n in SELECTOR_NAMES}:
        raise ValueError('resolved selector inventory mismatch')
    defaults = {name: None for name in SELECTOR_NAMES}
    defaults.update(LAYERNORM_ROWS_POLICY='row', FF2_POLICY='baseline',
        ATTN_OUT_POLICY='baseline', QKV_POLICY='baseline', FF1_POLICY='baseline',
        FF2_EPILOGUE='separate', ATTN_OUT_EPILOGUE='separate',
        FF2_SWIZZLE='1', ATTN_OUT_SWIZZLE='1', QKV_SWIZZLE='1', FF1_SWIZZLE='1',
        HOPPER_FF1_EPILOGUE='auto', FF1_STAGES='3', STAGE_PROFILE_SKIP_CALLS='1',
        ATTENTION_MAX_K_POLICY='padded64', ATTENTION_TILE_POLICY='q64k64',
        CLS_ATTENTION_POLICY='cutlass')
    flags = {}
    for name in SELECTOR_NAMES:
        value = selectors[PREFIX+name]
        if value is not None and type(value) is not str:
            raise ValueError('resolved selector must be null or text: '+name)
        if name in FLAG_NAMES:
            if value not in (None, '', '0', '1'):
                raise ValueError('invalid resolved flag: '+name)
            flags[PREFIX+name] = value == '1'
        elif value not in (None, '', defaults[name]):
            raise ValueError('unsupported independent resolved selector: '+name)
    for name in ('LEGACY_PADDING_ZERO', 'BLOCK51', 'FINAL_CLS_SPLIT_QKV',
                 'DUAL_INPUT_LN', 'STAGE_PROFILE'):
        if flags[PREFIX+name]:
            raise ValueError('unsupported independent resolved route: '+name)
    if flags[PREFIX+'FINAL_CLS_ATTENTION'] and not flags[PREFIX+'FINAL_CLS_ONLY']:
        raise ValueError('inconsistent final CLS execution flags')
    for name in ('outer_microbatch', 'transformer_microbatch', 'lanes'):
        if type(expected[name]) is not int or not 0 < expected[name] <= 65536:
            raise ValueError('invalid resolved batching: '+name)
    if expected['transformer_microbatch'] > expected['outer_microbatch'] or expected['lanes'] > 64:
        raise ValueError('invalid resolved batching bounds')
    return dict(schema_version=1, scope='resolved_host_choices_not_kernel_admission',
        production_admitted=False, kernel_coverage_complete=False, sm=sm,
        outer_microbatch=expected['outer_microbatch'],
        transformer_microbatch=expected['transformer_microbatch'], lanes=expected['lanes'],
        graph_executor=expected['executor']=='native_cuda_graph', fp16=True,
        gemm_families=[dict(family=name, policy_code=0, stage_code=0,
                           swizzle_code=0, epilogue_code=0)
                       for name in ('qkv', 'attention_out', 'ff1', 'ff2')],
        launch=dict(attention_tile_code=0, attention_max_k_code=0, cls_attention_code=0,
            layernorm_rows_code=0, layernorm_copy_grid=0, layernorm_bias_round_grid=0,
            hopper_ff1_epilogue_128x64=False, stage_profile_skip_calls=1, flags=flags))


def validate_resolved_receipt(receipt, wanted, expected, execution, raw_sha256):
    independent = canonical_resolved_recipe(expected)
    if typed(wanted) != typed(independent):
        raise ValueError('incomplete or contradictory independent resolved expectations')
    bindings = ('device', 'manifest_sha256', 'reference_sha256',
                'loaded_tensor_bindings', 'requested_launch_policies')
    keys = set(independent) | set(bindings) | {'raw_scores_sha256', 'specialized_gemm_observations'}
    if not isinstance(receipt, dict) or set(receipt) != keys:
        raise ValueError('resolved candidate exact schema mismatch')
    for key, value in independent.items():
        if typed(receipt[key]) != typed(value):
            raise ValueError('independent resolved binding mismatch: '+key)
    for key in bindings:
        if typed(receipt[key]) != typed(expected[key]):
            raise ValueError('resolved execution binding mismatch: '+key)
    if receipt['raw_scores_sha256'] != raw_sha256:
        raise ValueError('resolved raw-score content mismatch')
    observations = execution.get('host_kernel_observations')
    if not isinstance(observations, list) or any(not isinstance(item, dict) for item in observations):
        raise ValueError('malformed host kernel observations')
    specialized = [item for item in observations if item.get('family') == 'linear_bias_strided']
    if typed(receipt['specialized_gemm_observations']) != typed(specialized):
        raise ValueError('specialized launch observations disagree across receipts')
