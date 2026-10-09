"""Bounded legal graph frontiers for measuring the complete native pipeline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import math


def symmetric_orbit_certificate(generators, state_len):
    """Prove S_n from a full cycle and a transposition's connected conjugates."""
    for cycle in generators:
        orbit=[];p=0
        while p not in orbit:
            orbit.append(p);p=cycle[p]
        if p!=0 or len(orbit)!=state_len:continue
        for swap in generators:
            changed=[i for i,v in enumerate(swap) if i!=v]
            if len(changed)!=2:continue
            a,b=changed
            if swap[a]!=b or swap[b]!=a:continue
            edges=[];adj=[set() for _ in range(state_len)]
            for _ in range(state_len):
                adj[a].add(b);adj[b].add(a);edges.append([a,b]);a,b=cycle[a],cycle[b]
            reached={0};pending=[0]
            while pending:
                for node in adj[pending.pop()]-reached:
                    reached.add(node);pending.append(node)
            if len(reached)==state_len:
                return {'proof':'connected transposition conjugates generate S_n',
                        'cycle':list(cycle),'transposition':list(swap),'edges':edges}
    return None


def _enumerated_frontiers(contract, counts, storage_len, directory, deadline, certificate):
    import numpy as np
    n=contract.state_len;symbols=np.sort(np.asarray(contract.center,dtype=np.uint8))
    forbidden=[np.asarray(contract.center,dtype=np.uint8)]
    forbidden.extend(forbidden[0][np.argsort(move)] for move in contract.generators)
    ceiling=math.factorial(n);cursor=0;files=[];offset=ceiling//3
    stride=min(65537,(2**64-1-offset)//max(1,ceiling-1))
    while math.gcd(stride,ceiling)!=1:stride-=1
    for rank,count in enumerate(counts):
        path=directory/f'frontier-rank-{rank}.bin';digest=hashlib.sha256();written=0
        with path.open('xb') as file:
            while written<count:
                if time.monotonic()>=deadline or cursor>=ceiling:
                    raise ValueError('enumerated legal frontier preparation exceeded budget')
                batch=min(65536,count-written+len(forbidden),ceiling-cursor)
                indices=np.arange(cursor,cursor+batch,dtype=np.uint64);cursor+=batch
                indices=(indices*stride+offset)%ceiling
                available=np.tile(symbols,(batch,1));states=np.empty((batch,n),dtype=np.uint8)
                row=np.arange(batch)
                for pos in range(n):
                    factorial=math.factorial(n-pos-1)
                    digit=(indices//factorial).astype(np.int64);indices%=factorial
                    states[:,pos]=available[row,digit]
                    if pos<n-1:
                        columns=np.arange(n-pos-1)[None,:]
                        available=np.take_along_axis(available,columns+(columns>=digit[:,None]),axis=1)
                keep=np.ones(batch,dtype=bool)
                for value in forbidden:keep &= np.any(states!=value,axis=1)
                states=states[keep][:count-written]
                padded=np.zeros((len(states),storage_len),dtype=np.uint8);padded[:,:n]=states
                data=padded.tobytes();file.write(data);digest.update(data);written+=len(states)
        files.append({'rank':rank,'parents':count,'path':str(path),'sha256':digest.hexdigest(),'bytes':count*storage_len})
    receipt={'schema':2,'graph_hash':contract.graph_hash,'global_parents':sum(counts),
        'unique_states':sum(counts),'state_len':n,'storage_len':storage_len,'files':files,
        'excluded_center_and_predecessors':True,'generator':'distinct factorial permutation ranks',
        'rank_stride':stride,'rank_offset':offset,
        'reachability_certificate':certificate}
    (directory/'receipt.json').write_text(json.dumps(receipt,indent=2));return receipt


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
    if contract.state_len<=20 and len(set(contract.center))==contract.state_len:
        certificate=symmetric_orbit_certificate(contract.generators,contract.state_len)
        if certificate is not None:
            return _enumerated_frontiers(contract,counts,storage_len,directory,deadline,certificate)
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
