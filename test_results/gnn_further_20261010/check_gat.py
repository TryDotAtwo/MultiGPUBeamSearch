import sys,json,torch
from pathlib import Path
from torch.utils.cpp_extension import _get_build_directory
torch.ops.load_library(str(Path(_get_build_directory('beam_gnn_further_softmax',False))/'beam_gnn_further_softmax.so'))
sys.path.insert(0,'/root/further-softmax-source/integrations/cayleypy_native')
from multigpubeamsearch.pancake_gnn import GATv2,DualStreamEncoder
torch.manual_seed(73);records=[]
with torch.inference_mode():
 for n in (4,100):
  batch=2;states=torch.stack([torch.randperm(n,device='cuda') for _ in range(batch)])
  encoder=DualStreamEncoder(n,16,1,0.).cuda()
  offsets=torch.arange(batch,device='cuda').repeat_interleave(3*n-2)*n
  value_edges=encoder.value_edge_index_template.repeat(1,batch)+offsets[None,:]
  values=torch.arange(n,device='cuda')
  position_edges=torch.cat((torch.stack((states[:,:-1],states[:,1:]),-1),torch.stack((states[:,1:],states[:,:-1]),-1),torch.stack((values,values),-1)[None,:,:].expand(batch,-1,-1)),1).reshape(-1,2).t().contiguous()+offsets[None,:]
  for channels in (4,8,16,64,128,256,1024):
   gat=GATv2(channels,channels*2).cuda().half().eval()
   nodes=torch.randn(batch*n,channels,device='cuda',dtype=torch.float16)
   embeddings=torch.randn(3,channels*2,device='cuda',dtype=torch.float16)
   for family,edges,types in [('value',value_edges,encoder.value_edge_type_template.repeat(batch)),('position',position_edges,encoder.pos_edge_type_template.repeat(batch))]:
    expected=gat(nodes,edges,embeddings[types])
    actual=gat.forward_compact(nodes,edges,embeddings,types)
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.01)
    for linear in (gat.lin_l,gat.lin_r,gat.lin_edge):linear.use_cutlass=True
    fast=gat.forward_compact(nodes,edges,embeddings,types)
    torch.testing.assert_close(fast,expected,rtol=.02,atol=.01)
    for linear in (gat.lin_l,gat.lin_r,gat.lin_edge):linear.use_cutlass=False
    records.append(dict(n=n,channels=channels,graph=family,kernel_error=(actual-expected).abs().max().item(),combined_error=(fast-expected).abs().max().item()))
torch.cuda.synchronize()
Path('/root/further-result/gat-parity.json').write_text(json.dumps(records,indent=2))
print(json.dumps(dict(cases=len(records),max_kernel_error=max(r['kernel_error'] for r in records),max_combined_error=max(r['combined_error'] for r in records))))
