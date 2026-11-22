"""Matched large GNN inference comparison; execute only on the rented GPU."""
import hashlib,json,os,sys,time,statistics,types,subprocess
from pathlib import Path
import torch
from torch.utils.cpp_extension import load,_get_build_directory
root=Path(os.environ['GNN_SOURCE']);out=Path('/root/further-baseline-result');out.mkdir(exist_ok=True)
probe=out/'persistent_probe.cpp'
probe.write_text(r'''
#include "stream1_gnn_libtorch.hpp"
#include <torch/library.h>
#include <memory>
namespace {
std::unique_ptr<beam::stream1_libtorch::PancakeGnnNative> models[2];
void setup(const std::string& path,int64_t device) {
 for(int i=0;i<2;++i) models[i]=std::make_unique<beam::stream1_libtorch::PancakeGnnNative>(path,torch::Device(torch::kCUDA,device),i==1);
}
torch::Tensor run(const torch::Tensor& input,bool cutlass) {
 auto& model=*models[cutlass?1:0];
 return torch::linear(model.features(input).to(torch::kFloat32),model.output_weight.to(torch::kFloat32),model.output_bias.to(torch::kFloat32));
}
}
TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_gnn,m) {
 m.def("large_setup(str directory, int device) -> ()");m.impl("large_setup",TORCH_FN(setup));
 m.def("large_run(Tensor input, bool cutlass) -> Tensor");m.impl("large_run",TORCH_FN(run));
}
''')
os.environ.setdefault('MAX_JOBS','2')
load(name='beam_gnn_further_baseline',sources=[str(root/'tools/stream1_gnn_projection_libtorch.cpp'),str(root/'cuda/stream1_gnn_projection.cu'),str(probe)],extra_include_paths=[str(root/'tools'),'/root/cutlass-source/include'],extra_cflags=['-O2','-std=c++20','-DBEAM_STATE_LOGICAL_BYTES=100','-DBEAM_MOVE_COUNT=99','-DBEAM_STATE_PHYSICAL_BYTES=128','-DBEAM_STATE_ALIGNMENT=32'],extra_cuda_cflags=['-O2','-DBEAM_HAS_CUTLASS=1'],is_python_module=False,verbose=True)
sys.path.insert(0,str(root/'integrations/cayleypy_native'))
from multigpubeamsearch import PancakeGNN,NeighborConfig
from multigpubeamsearch.gnn_linear import NativeLinear
from multigpubeamsearch.gnn_artifacts import export_gnn
from multigpubeamsearch.contracts import GraphContract
torch.manual_seed(42);torch.set_num_threads(2)
identity=tuple(range(100));moves=tuple(tuple(reversed(identity[:k]))+identity[k:] for k in range(2,101))
contract=GraphContract(100,100,moves,tuple(str(k) for k in range(2,101)),identity,identity,hashlib.sha256(repr(moves).encode()).hexdigest())
model=PancakeGNN(100,256,2,0.,NeighborConfig(max_neighbors_per_hop=75,num_hops=2,max_frontier_states=100000)).eval()
artifact=export_gnn(model,contract,out/'artifact')
torch.ops.multigpubeamsearch_gnn.large_setup(str(artifact.weights_dir),0)
model=model.cuda().half()
def eager_linear(self,input):return torch.nn.functional.linear(input,self.weight,self.bias)
for module in model.modules():
 if isinstance(module,NativeLinear):module.forward=types.MethodType(eager_linear,module)
def python_run(states):return torch.nn.functional.linear(model.features(states).float(),model.value_head[3].weight.float(),model.value_head[3].bias.float())
methods={'python_pytorch':python_run,'native_libtorch':lambda x:torch.ops.multigpubeamsearch_gnn.large_run(x,False),'native_cutlass':lambda x:torch.ops.multigpubeamsearch_gnn.large_run(x,True)}
binary=Path(_get_build_directory('beam_gnn_further_baseline',False))/'beam_gnn_further_baseline.so'
metadata=dict(source_commit=os.environ['GNN_COMMIT'],binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),arch_flags='8.6',gpu=torch.cuda.get_device_name(),cc=torch.cuda.get_device_capability(),torch=torch.__version__,cuda=torch.version.cuda,driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),model=dict(n=100,d_model=256,layers=2,hops=2,max_neighbors=75,neighbor_counts=model.neighbor_counts,cap=100000,parameters=sum(p.numel() for p in model.parameters())),artifact_hash=artifact.artifact_hash,seed=42,precision='FP16 body + common FP32 scalar readout',scope='one GNN forward including both levels; excludes beam streams, host transfers, loading and compilation',slots=1,streams=1,warmup=2,repeats=5,trained_quality=False)
torch.manual_seed(901)
records=[]
with torch.inference_mode():
 for batch in (3,):
  states=torch.stack([torch.randperm(100,device='cuda') for _ in range(batch)])
  oracle=None;completed=0
  for name in ('python_pytorch','native_cutlass'):
   torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
   try:
    actual=methods[name](states)
    if oracle is None:oracle=actual.detach().clone()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,oracle,rtol=.03,atol=.02)
    error=(actual-oracle).abs().max().item()
    for _ in range(2):actual=methods[name](states)
    torch.cuda.synchronize()
    elapsed=[];gpu_elapsed=[]
    for _ in range(5):
     start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     wall=time.perf_counter();start.record();actual=methods[name](states);end.record();end.synchronize()
     elapsed.append((time.perf_counter()-wall)*1000);gpu_elapsed.append(start.elapsed_time(end))
    record=dict(batch=batch,backend=name,status='PASS',max_error=error,wall_ms=elapsed,gpu_ms=gpu_elapsed,median_ms=statistics.median(elapsed),p95_ms=max(elapsed),items_per_second=batch*1000/statistics.median(elapsed),allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),output_sha=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest())
    completed+=1
    del actual
   except RuntimeError as error:
    if 'out of memory' not in str(error).lower():raise
    record=dict(batch=batch,backend=name,status='OOM',error=str(error)[:400])
    torch.cuda.empty_cache()
   records.append(record);print(json.dumps(record),flush=True)
   (out/'result.json').write_text(json.dumps(dict(metadata=metadata,records=records),indent=2))
  del states,oracle
  if completed<3:break
print('COMPLETED',flush=True)

with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 torch.ops.multigpubeamsearch_gnn.large_run(torch.stack([torch.randperm(100,device='cuda') for _ in range(3)]),True)
 torch.cuda.synchronize()
(out/'profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
