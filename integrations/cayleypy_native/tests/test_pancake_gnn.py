import hashlib
import json
from pathlib import Path
import pytest
import torch
from multigpubeamsearch import PancakeGNN, NeighborConfig, NativeEnsemble, NativeOptions
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.models import prepare_model, verify_prepared_model
from multigpubeamsearch.errors import NativeBackendError, NativeUnavailable

torch.set_num_threads(2)

def contract(n=4):
    identity=tuple(range(n))
    moves=tuple(tuple(reversed(identity[:k]))+identity[k:] for k in range(2,n+1))
    return GraphContract(n,n,moves,tuple(str(k) for k in range(2,n+1)),identity,
                         tuple(reversed(identity)),hashlib.sha256(repr(moves).encode()).hexdigest())

def model():
    return PancakeGNN(4,16,1,0.,NeighborConfig(num_hops=2)).eval()

def test_gnn_export_snapshot_and_tamper_detection(tmp_path):
    original=model()
    prepared=prepare_model(original,contract(),NativeOptions(),tmp_path/'run')
    assert prepared.backend=='ensemble'
    member=prepared.manifest['ensemble']['models'][0]
    assert member['family']=='pancake_gnn'
    path=prepared.weights_dir/member['weights_dir']
    with (path/'backbone.pt').open('rb') as artifact:
        saved=torch.jit.load(artifact)
    states=torch.tensor([[3,2,1,0],[1,0,2,3]])
    with torch.inference_mode():
        torch.testing.assert_close(saved(states),original.half()(states),rtol=0,atol=0)
    verify_prepared_model(prepared,contract())
    with (path/'backbone.pt').open('ab') as f:f.write(b'tamper')
    with pytest.raises(NativeBackendError,match='checksum'):
        verify_prepared_model(prepared,contract())

def test_gnn_blend_arbitrary_member_count(tmp_path):
    prepared=prepare_model(NativeEnsemble([model(),model(),model()],[.2,.3,.5]),
                           contract(),NativeOptions(),tmp_path/'run')
    assert len(prepared.manifest['ensemble']['models'])==3
    verify_prepared_model(prepared,contract())

def test_gnn_rejects_wrong_graph_and_stochastic_sampling(tmp_path):
    c=contract()
    bad=GraphContract(c.state_len,c.num_classes,c.generators[:1],c.generator_names[:1],
                      c.center,c.start,c.graph_hash)
    with pytest.raises(NativeUnavailable,match='prefix flips'):
        prepare_model(model(),bad,NativeOptions(),tmp_path/'run')
    with pytest.raises(ValueError,match='stratified'):
        PancakeGNN(4,neighbors=NeighborConfig(use_policy_sampling=True))

def test_gnn_cap_fails_without_changing_semantics():
    m=PancakeGNN(4,16,1,0.,NeighborConfig(num_hops=2,max_frontier_states=5)).eval()
    with pytest.raises(RuntimeError,match='stochastic'):
        m(torch.arange(4)[None])

def test_python_cutlass_does_not_silently_fallback():
    m=model()
    m.state_encoder.value_convs[0].lin_l.use_cutlass=True
    with pytest.raises(RuntimeError,match=r'native C\+\+ runtime'):
        m(torch.arange(4)[None])

def test_gnn_script_and_batch_invariance():
    m=model()
    s=torch.tensor([[3,2,1,0],[2,0,3,1],[1,0,2,3]])
    with torch.inference_mode():
        torch.testing.assert_close(m(s),torch.jit.script(m)(s),rtol=0,atol=0)
        torch.testing.assert_close(m(s),torch.cat([m(row[None]) for row in s]),rtol=2e-5,atol=2e-6)
