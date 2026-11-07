"""Independent FP16 classic-CUTLASS graph GEMM launch expectations, not admission."""
from collections import Counter
import json
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_inventory_contract import validate_graph_inventory_structure

TILES={
    'baseline':((128,64,32),(64,32,32)),
    'm64n128':((64,128,32),(32,64,32)),
    'm128n128':((128,128,32),(64,64,32)),
    'm128n128w64n32':((128,128,32),(64,32,32)),
    'm256n128':((256,128,32),(64,64,32)),
    'm64n64':((64,64,32),(32,32,32)),
    'm32n64':((32,64,32),(32,32,32))}
ALLOWED={'QKV':{'baseline','m64n128','m128n128','m256n128'},
    'ATTN_OUT':{'baseline','m128n128','m64n64'},
    'FF1':{'baseline','m128n128','m128n128w64n32','m64n128'},
    'FF2':{'baseline','m128n128','m64n64'}}


def key(description,grid,block,shared):
    return (json.dumps(description,sort_keys=True,separators=(',',':')),
            tuple(grid),tuple(block),shared)


def expected_graph_gemms(model,execution):
    required=dict(dtype='fp16',activation='relu',d_model=256,ff_dim=1024,
                  num_layers=4,seq_len=57,output_dim=24)
    if any(type(model.get(k)) is not type(v) or model[k]!=v for k,v in required.items()):
        raise ValueError('unsupported independent GEMM model contract')
    sm=execution['device']['sm']
    if execution.get('executor')!='native_cuda_graph' or type(sm) is not int or sm<75 or 75<sm<80:
        raise ValueError('classic FP16 SM75/SM80+ graph GEMM validator required')
    sm75=sm==75
    policies=execution['requested_launch_policies']
    def selected(name,default):
        value=policies.get('BEAM_STREAM1_TRANSFORMER_'+name)
        return default if value is None or value=='' else value
    def flag(name):
        value=selected(name,'0')
        if value not in ('0','1'): raise ValueError('invalid GEMM execution flag')
        return value=='1'
    compact=flag('COMPACT57') if 'BEAM_STREAM1_TRANSFORMER_COMPACT57' in policies else None
    # COMPACT57 is a loaded-layout selector outside the26 launch-key snapshot.
    # Trusted caller must supply its independently validated model padding.
    seq=execution.get('expected_padded_seq_len')
    if type(seq) is not int or seq not in (57,64):
        raise ValueError('missing independent padded sequence expectation')
    if compact is not None and seq!=(57 if compact else 64):
        raise ValueError('padded sequence selector disagreement')
    final=flag('FINAL_CLS_ONLY')
    split=flag('FINAL_CLS_SPLIT_QKV')
    if sm75 and split:raise ValueError('SM75 split-QKV is not compiled')
    if split and (not final or not flag('FINAL_CLS_ATTENTION')):
        raise ValueError('unsupported split-QKV dependency')
    if flag('BLOCK51') or flag('STAGE_PROFILE'):
        raise ValueError('specialized/profiling graph GEMM path not validated')
    for family in ('ATTN_OUT','FF2'):
        if selected(family+'_EPILOGUE','separate')!='separate':
            raise ValueError('fused residual GEMM epilogue not decoded independently')
    outer,inner=execution['outer_microbatch'],execution['transformer_microbatch']
    if any(type(x) is not int or x<=0 for x in (outer,inner)) or inner>outer or outer>65536:
        raise ValueError('invalid GEMM chunk profile')
    expected=Counter()
    def gemm(family,kind,rows,cols,fixed=None):
        policy=selected(family+'_POLICY','baseline') if not fixed else 'baseline'
        if sm75 and (policy!='baseline' or selected(family+'_SWIZZLE','1')!='1'):
            raise ValueError('SM75 non-baseline GEMM profile not independently validated')
        if not fixed and policy not in ALLOWED[family]: raise ValueError('unsupported GEMM family policy')
        tile,warp=TILES[policy] if not fixed else fixed
        stage=selected('FF1_STAGES','2' if sm75 else '3') if family=='FF1' and not fixed else '3'
        if stage not in ('2','3'): raise ValueError('unsupported GEMM stages')
        if sm75 and family=='FF1' and not fixed and stage!='2':raise ValueError('SM75 FF1 uses two stages')
        stages=2 if sm75 or kind=='head' else int(stage)
        swizzle=selected(family+'_SWIZZLE','1') if not fixed else '1'
        if swizzle not in ('1','2','4','8'): raise ValueError('unsupported GEMM swizzle')
        swizzle=int(swizzle)
        m=(rows+tile[0]-1)//tile[0]
        n=(cols+tile[1]-1)//tile[1]
        # Pinned CUTLASS GemmIdentityThreadblockSwizzle::get_log_tile.
        width=8 if swizzle>=8 and n>=6 else 4 if swizzle>=4 and n>=3 else 2 if swizzle>=2 and n>=2 else 1
        grid=(m*width,(n+width-1)//width,1)
        block=((tile[0]//warp[0])*(tile[1]//warp[1])*32,1,1)
        desc=dict(family='gemm_pipelined' if kind=='head' or (sm75 and kind=='gemm_residual') else kind,
            tile=list(tile),warp=list(warp),instruction=[16,8,8] if sm75 or kind=='head' else [16,8,16],
            stages=stages,swizzle=swizzle)
        shared=(tile[0]+tile[1])*tile[2]*2*stages
        expected[key(desc,grid,block,shared)]+=1
    for offset in range(0,outer,inner):
        batch=min(inner,outer-offset)
        rows=batch*seq
        for layer in range(4):
            last=final and layer==3
            if last and split:
                fixed=((128,128,32),(64,64,32))
                gemm('QKV','gemm_bias',batch,256,fixed)
                gemm('QKV','gemm_bias',rows,512,fixed)
            else: gemm('QKV','gemm_bias',rows,768)
            output_rows=batch if last else rows
            gemm('ATTN_OUT','gemm_residual',output_rows,256)
            gemm('FF1','gemm_relu',output_rows,1024)
            gemm('FF2','gemm_residual',output_rows,256)
        gemm('QKV','head',batch,24,((128,64,32),(64,32,32)))
    return expected


def validate_graph_gemm_profile(inventory,model,execution):
    validate_graph_inventory_structure(inventory,lanes=execution['lanes'],sm=execution['device']['sm'])
    descriptions=decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    expected=expected_graph_gemms(model,execution)
    for lane in inventory['lanes']:
        actual=Counter()
        for node in lane['kernels']:
            desc=descriptions[node['function_id']]
            if 'stages' not in desc: continue
            actual[key(desc,node['grid'],node['block'],node['dynamic_shared_bytes'])]+=1
        if actual!=expected: raise ValueError('captured GEMM profile mismatch in lane'+str(lane['lane']))
    return dict(gemm_launches_per_lane=sum(expected.values()),gemm_profile_checked=True,
                production_admitted=False,kernel_coverage_complete=False)
