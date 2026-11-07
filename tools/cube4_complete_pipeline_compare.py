"""Compare complete global logical frontiers from remote diagnostic runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def observe(directory):
    request=json.loads((directory/'request.json').read_text())
    if not request['diagnostics']:raise ValueError('frontier comparison needs diagnostic build')
    world=request['world'];beam=request['beam'];depths={}
    for rank in range(world):
        paths=sorted((directory/'frontier-dumps').glob(f'depth_*_rank_{rank}.bin'))
        if len(paths)!=6:raise ValueError('missing six complete rank frontiers')
        for path in paths:
            depth=int(re.fullmatch(r'depth_(\d+)_rank_\d+\.bin',path.name)[1])
            raw=path.read_bytes()
            if len(raw)%112:raise ValueError('invalid Cube4 frontier stride')
            states=depths.setdefault(depth,[])
            for offset in range(0,len(raw),112):
                state=raw[offset:offset+112]
                if any(state[96:]) or any(x>5 for x in state[:96]):raise ValueError('padding/class corruption')
                states.append(state[:96])
    result={}
    for depth,states in sorted(depths.items()):
        if len(states)!=len(set(states)):raise ValueError('global logical duplicate')
        if len(states)>beam:raise ValueError('global beam exceeded')
        result[str(depth)]=dict(count=len(states),logical_sha256=hashlib.sha256(b''.join(sorted(states))).hexdigest())
    if sum(value['count']==beam for value in result.values())<2:
        raise ValueError('fewer than two fully saturated depths')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directories',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows={str(directory):observe(directory) for directory in args.directories}
    first=next(iter(rows.values()))
    if any(row!=first for row in rows.values()):raise ValueError('global logical frontier trajectories differ')
    receipt=dict(status='pass',scope='exact_six_depth_logical_frontier_equality_not_score_bytes',runs=rows)
    with args.output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
