import sys,json,statistics,hashlib,torch
from pathlib import Path
from torch.utils.cpp_extension import _get_build_directory
variant=sys.argv[1]
torch.ops.load_library(str(Path(_get_build_directory('beam_gnn_further_'+variant,False))/('beam_gnn_further_'+variant+'.so')))
torch.manual_seed(644);torch.set_num_threads(2)
n=100;batch=8550;channels=128;nodes=n*batch
base=torch.arange(n-1,device='cuda');allnodes=torch.arange(n,device='cuda')
ix=torch.stack([torch.cat([torch.stack([base,base+1],1).flatten(),allnodes]),torch.cat([torch.stack([base+1,base],1).flatten(),allnodes])])
offset=torch.arange(batch,device='cuda').repeat_interleave(3*n-2)*n
ix=ix.repeat(1,batch)+offset[None,:]
types=torch.cat([torch.tensor([0,1],device='cuda').repeat(n-1),torch.full([n],2,device='cuda')]).repeat(batch)
left=torch.randn(nodes,4,channels,device='cuda',dtype=torch.float16)*.3
right=torch.randn_like(left)*.3
edge=torch.randn(3,4,channels,device='cuda',dtype=torch.float16)*.3
attention=torch.randn(1,4,channels,device='cuda',dtype=torch.float16)*.1
bias=torch.randn(channels,device='cuda',dtype=torch.float16)*.1
with torch.inference_mode():
 for _ in range(5):actual=torch.ops.multigpubeamsearch_gnn.gat(left,right,edge,attention,bias,ix,types)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):actual=torch.ops.multigpubeamsearch_gnn.gat(left,right,edge,attention,bias,ix,types)
 samples=[]
 for _ in range(21):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record();graph.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end))
 digest=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()
r={'variant':variant,'nodes':nodes,'channels':channels,'median_ms':statistics.median(samples),'gpu_ms':samples,'output_sha':digest,'scope':'isolated GAT operator CUDA graph replay; not production beam graph capture'}
Path('/root/further-result/gat-component-'+variant+'.json').write_text(json.dumps(r));print(json.dumps(r))
