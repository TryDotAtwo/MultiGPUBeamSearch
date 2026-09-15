"""Offline E4M3FN weights for experimental SM90 native inference.

No training, activation calibration or production qualification. A selected
matrix appears once, as physical column-major KxN bytes plus a scalar scale.
Unselected tensors remain FP16. Runtime must never reinterpret this bundle as
the ordinary FP16 manifest; the schema/backend are deliberately distinct.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def encode_e4m3(x):
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite FP8 input')
    codes=np.arange(127,dtype=np.uint8)
    exp=(codes>>3).astype(np.int32); mant=(codes&7).astype(np.float32)
    values=np.where(exp==0,mant*2**-9,(1+mant/8)*np.exp2(exp-7)).astype(np.float32)
    magnitude=np.minimum(np.abs(x),448.)
    hi=np.searchsorted(values,magnitude,side='left').clip(0,126)
    lo=np.maximum(hi-1,0)
    dl=magnitude-values[lo]; dh=values[hi]-magnitude
    chosen=np.where((dl<dh)|((dl==dh)&((lo&1)==0)),lo,hi).astype(np.uint8)
    return chosen | (np.signbit(x).astype(np.uint8)<<7)


def pack_weight(weight_kxn):
    x=np.asarray(weight_kxn,dtype=np.float32)
    if x.ndim!=2 or not x.size:raise ValueError('Expected nonempty KxN matrix')
    if not np.isfinite(x).all():raise ValueError('Nonfinite weight')
    amax=float(np.abs(x).max())
    scale=float(np.float32(amax/448.)) if amax else 1.
    if scale<=0:raise ValueError('Weight scale underflow')
    return encode_e4m3(x/scale).T.copy().reshape(-1),scale


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source,destination):
    manifest=json.loads((source/'manifest.json').read_text())
    required=dict(backend='piece_transformer',dtype='fp16',d_model=256,ff_dim=1024,
                  num_layers=4,seq_len=57,output_dim=24,activation='relu')
    if any(manifest.get(k)!=v for k,v in required.items()):
        raise ValueError('Only the frozen FP16 Cube4 shape is supported')
    # First boundary campaign: full-token QKV in blocks0-2. Last CLS block and
    # all other operators retain one original FP16 representation.
    selected={f'block{i}_attn_qkv_weight_hxk.fp16':(256,768) for i in range(3)}
    for name,shape in selected.items():
        if (source/name).stat().st_size!=int(np.prod(shape))*2:
            raise ValueError(f'Wrong weight size: {name}')
    destination.mkdir(parents=True,exist_ok=False)
    records={}
    paths=list(source.glob('*.fp16'))+[source/name for name in
        ('piece_positions.u16','piece_mask.u8','piece_types.u8')]
    for path in sorted(paths):
        if path.name in selected:
            shape=selected[path.name]
            packed,scale=pack_weight(np.fromfile(path,dtype='<f2').reshape(shape))
            target=destination/(path.stem+'.e4m3')
            packed.tofile(target)
            records[path.name]=dict(file=target.name,dtype='e4m3fn',shape_kxn=list(shape),
                layout='column_major_kxn',stride_k=1,stride_n=shape[0],
                dequant_scale=scale,scale_dtype='fp32',source_sha256=sha(path),
                sha256=sha(target),bytes=target.stat().st_size)
        else:
            target=destination/path.name;shutil.copyfile(path,target)
            records[path.name]=dict(file=path.name,dtype=path.suffix[1:],source_sha256=sha(path),
                                   sha256=sha(target),bytes=target.stat().st_size)
    result=dict(schema='hopper_native_e4m3_experimental_v1',target='sm90a',
                source_manifest_sha256=sha(source/'manifest.json'),model=manifest,
                activation='relu',accumulation='fp32',gemm_output='fp16',
                activation_calibration='not_performed',quality_qualified=False,
                files=records)
    (destination/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=export(args.source,args.output)
    print(json.dumps(dict(files=len(data['files']),packed=sum(v['dtype']=='e4m3fn' for v in data['files'].values()),
                          manifest_sha256=sha(args.output/'manifest.json'))))
