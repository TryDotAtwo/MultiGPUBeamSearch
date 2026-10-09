"""Bounded legal graph frontiers for measuring the complete native pipeline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time


def write_frontiers(contract, rank_counts, storage_len, directory, *, deadline,
                    max_states=1048576, seed=20261009, walk_length=64):
    """Generate exact unique reachable states with native zero padding.

    Reject an unfilled frontier rather than disguising repeated states as full
    work. Small or poorly mixed graphs can therefore decline this calibration;
    that is distinct from refusing their ordinary beam-search execution.
    """
    import numpy as np
    counts=tuple(rank_counts)
    if not counts or any(type(n) is not int or n<=0 for n in counts):
        raise ValueError('rank frontier counts must be positive integers')
    if storage_len<contract.state_len or type(storage_len) is not int:
        raise ValueError('invalid native state storage size')
    total=sum(counts)
    if type(max_states) is not int or max_states<=0 or total>max_states:
        raise ValueError('requested calibration frontier exceeds its explicit state budget')
    if type(walk_length) is not int or walk_length<1:
        raise ValueError('calibration walk length must be positive')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    moves=np.asarray(contract.generators,dtype=np.int64)
    center=np.asarray(contract.center,dtype=np.uint8)
    forbidden={center.tobytes()}
    for move in moves:
        forbidden.add(center[np.argsort(move)].tobytes())
    seen=set();rng=np.random.default_rng(seed);files=[]
    examined=0;attempt_budget=max(total*32,4096)
    for rank,count in enumerate(counts):
        path=directory/f'frontier-rank-{rank}.bin';digest=hashlib.sha256();written=0
        with path.open('xb') as file:
            while written<count:
                if time.monotonic()>=deadline or examined>=attempt_budget:
                    raise ValueError('graph did not yield a full unique calibration frontier within budget')
                batch=min(65536,max(256,count-written),attempt_budget-examined)
                states=np.tile(center,(batch,1))
                lengths=rng.integers(max(1,walk_length//2),walk_length+1,size=batch)
                samples=np.empty_like(states)
                for step in range(walk_length):
                    indices=moves[rng.integers(0,contract.move_count,size=batch)]
                    states=np.take_along_axis(states,indices,axis=1)
                    mask=lengths==step+1;samples[mask]=states[mask]
                    if time.monotonic()>=deadline:
                        raise ValueError('calibration frontier generation timed out')
                examined+=batch;accepted=[]
                for row in samples:
                    value=row.tobytes()
                    if value in seen or value in forbidden:continue
                    seen.add(value);accepted.append(row)
                    if written+len(accepted)==count:break
                if accepted:
                    padded=np.zeros((len(accepted),storage_len),dtype=np.uint8)
                    padded[:,:contract.state_len]=accepted
                    data=padded.tobytes();file.write(data);digest.update(data);written+=len(accepted)
        files.append({'rank':rank,'parents':count,'path':str(path),'sha256':digest.hexdigest(),
                      'bytes':count*storage_len})
    receipt={'schema':1,'graph_hash':contract.graph_hash,'global_parents':total,
             'unique_states':len(seen),'seed':seed,'walk_length':walk_length,
             'walk_length_min':max(1,walk_length//2),
             'state_len':contract.state_len,'storage_len':storage_len,
             'excluded_center_and_predecessors':True,'files':files}
    (directory/'receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt
