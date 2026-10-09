import hashlib
import itertools
from types import SimpleNamespace
import time
import numpy as np
import pytest
from multigpubeamsearch.calibration_frontier import write_frontiers


def graph(size=5):
    return SimpleNamespace(state_len=size,center=tuple(range(size)),
        generators=tuple(tuple(j if j not in (a,b) else b if j==a else a for j in range(size))
                         for a,b in itertools.combinations(range(size),2)),
        move_count=size*(size-1)//2,graph_hash='test-permutation-graph')


def test_frontier_is_reachable_unique_padded_and_reproducible(tmp_path):
    g=graph();receipts=[]
    for name in ('a','b'):
        receipts.append(write_frontiers(g,[20,20],16,tmp_path/name,deadline=time.monotonic()+10))
    assert [row['sha256'] for row in receipts[0]['files']]==[row['sha256'] for row in receipts[1]['files']]
    states=[]
    for row in receipts[0]['files']:
        data=open(row['path'],'rb').read()
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        padded=np.frombuffer(data,dtype=np.uint8).reshape(20,16)
        assert not padded[:,5:].any()
        states.extend(tuple(x) for x in padded[:,:5])
    assert len(set(states))==40
    assert all(sorted(state)==list(g.center) for state in states)


def test_small_graph_declines_unfillable_frontier(tmp_path):
    with pytest.raises(ValueError,match='unique calibration'):
        write_frontiers(graph(2),[4],16,tmp_path/'small',deadline=time.monotonic()+10)


def test_frontier_budget_does_not_silently_shrink_workload(tmp_path):
    with pytest.raises(ValueError,match='state budget'):
        write_frontiers(graph(),[20,20],16,tmp_path/'too-big',deadline=time.monotonic()+10,max_states=32)

