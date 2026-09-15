"""Separate full Stream1 timing, score-key drift and production depth timing."""
import argparse
import json
from pathlib import Path
import re
import statistics
import struct
import numpy as np

def dump(path):
    raw=path.read_bytes();magic,version,lanes,count=struct.unpack('<QIIQ',raw[:24])
    assert magic==0x31504D5544533153 and version==1 and lanes>0 and count>0
    assert len(raw)==24+lanes*count*4 and count%24==0
    return np.frombuffer(raw,dtype='<u4',offset=24).reshape(-1,24)

def summary(root):
    result={'timing':{},'quality':[],'pipeline':{}}
    pairs=[]
    for mode in ('fp16','fp8'):
        rates=[]
        for rep in range(1,8):
            text=(root/f'native_{mode}_{rep}.log').read_text()
            assert 'stream1_transformer_benchmark_done=1' in text
            rates.append(float(re.search(r'parents_per_sec=([\d.]+)',text)[1]))
        result['timing'][mode]=dict(repeats=7,median=statistics.median(rates),min=min(rates),max=max(rates),rates=rates)
    result['timing']['paired_speedups']=[b/a for a,b in zip(result['timing']['fp16']['rates'],result['timing']['fp8']['rates'])]
    for puzzle in [*range(1,11),1000]:
        a=dump(root/f'quality_p{puzzle}_fp16.bin');b=dump(root/f'quality_p{puzzle}_fp8.bin')
        assert a.shape==b.shape
        delta=(b.astype(np.float64)-a)/1024
        result['quality'].append(dict(puzzle=puzzle,rows=len(a),unique_fp16_score_rows=len(np.unique(a,axis=0)),
            max_abs_score_delta=float(np.abs(delta).max()),rmse_score_delta=float(np.sqrt((delta*delta).mean())),
            argmin_agreement=float(np.mean(a.argmin(1)==b.argmin(1))),
            fp8_choice_in_fp16_minimizers=float(np.mean(np.take_along_axis(a,b.argmin(1)[:,None],axis=1)[:,0]==a.min(1)))))
    for mode in ('fp16','fp8'):
        path=root/f'pipeline_{mode}.log'
        if not path.exists():continue
        text=path.read_text();starts={int(d):int(n) for d,n in re.findall(r'depth_start=(\d+) frontier_size=(\d+)',text)}
        result['pipeline'][mode]=[dict(depth=int(d),parents=starts[int(d)],seconds=float(t),parents_per_second=starts[int(d)]/float(t))
            for d,t in re.findall(r'depth_done=(\d+) depth_sec=([\d.]+)',text)]
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(summary(a.root),indent=2))
