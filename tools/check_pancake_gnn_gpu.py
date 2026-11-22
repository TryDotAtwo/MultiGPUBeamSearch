"""Native projection/loader parity gate. Run on a CUDA host, not the exporter."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from torch.utils.cpp_extension import load

root=Path(__file__).resolve().parents[1]
out=Path(os.environ.get('GNN_TEST_OUTPUT','/tmp/gnn-gate'))
out.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MAX_JOBS','2')
cutlass=Path(os.environ['CUTLASS_DIR'])
assert torch.cuda.is_available(), 'GPU acceptance requires CUDA'
# Load the real C++ schema BEFORE importing the exporter reference registration.
load(name='beam_gnn_native_gate',sources=[str(root/'tools/stream1_gnn_projection_libtorch.cpp'),
    str(root/'cuda/stream1_gnn_projection.cu'),str(root/'tests/stream1_gnn_native_probe.cpp')],
    extra_include_paths=[str(cutlass/'include')],
    extra_cflags=['-O1','-std=c++20','-DBEAM_STATE_LOGICAL_BYTES=4','-DBEAM_MOVE_COUNT=3',
                  '-DBEAM_STATE_PHYSICAL_BYTES=16','-DBEAM_STATE_ALIGNMENT=16'],
    extra_cuda_cflags=['-O2','-DBEAM_HAS_CUTLASS=1'],is_python_module=False,verbose=True)
sys.path.insert(0,str(root/'integrations/cayleypy_native'))
from multigpubeamsearch import PancakeGNN, NeighborConfig
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.gnn_artifacts import export_gnn

torch.manual_seed(105)
torch.set_num_threads(2)
identity=tuple(range(4))
moves=tuple(tuple(reversed(identity[:k]))+identity[k:] for k in range(2,5))
contract=GraphContract(4,4,moves,('2','3','4'),identity,identity,
                        hashlib.sha256(repr(moves).encode()).hexdigest())
model=PancakeGNN(4,32,2,0.,NeighborConfig(num_hops=2,max_frontier_states=24)).eval()
artifact=export_gnn(model,contract,out/'artifact')
states=torch.stack([torch.randperm(4) for _ in range(9)]).cuda()
model=model.half().cuda()
results={}
with torch.inference_mode():
    expected=torch.cat([model.features(states[i:i+4]) for i in range(0,states.size(0),4)])
    actual=torch.ops.multigpubeamsearch_gnn.probe_features(states,str(artifact.weights_dir),False)
    torch.testing.assert_close(actual,expected,rtol=.005,atol=.005)
    results['libtorch_max_error']=(actual-expected).abs().max().item()
    if torch.cuda.get_device_capability()[0]>=8:
        fast=torch.ops.multigpubeamsearch_gnn.probe_features(states,str(artifact.weights_dir),True)
        torch.testing.assert_close(fast,expected,rtol=.02,atol=.01)
        results['cutlass_max_error']=(fast-expected).abs().max().item()
        results['cutlass']='PASS'
    else:
        results['cutlass']='NOT_RUN: requires SM80+'
results.update(gpu=torch.cuda.get_device_name(),libtorch='PASS',
               scope='native GNN features; not full beam search or trained quality')
(out/'result.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))

