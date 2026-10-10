import json,os,statistics,torch
from pathlib import Path
from torch.nn import functional as F
from torch.utils.cpp_extension import _get_build_directory
torch.ops.load_library(str(Path(_get_build_directory('beam_gnn_further_candidate',False))/'beam_gnn_further_candidate.so'))
torch.manual_seed(832);torch.set_num_threads(2)
records=[]
with torch.inference_mode():
 for channels in (4,16,64,128,132,256,512,1024):
  for rows in (1,7,129):
   for pattern in ('random','constant','nearly_constant','large','cancellation'):
    a=torch.randn(rows,channels,device='cuda',dtype=torch.float16)
    r=torch.randn_like(a)*.3
    if pattern=='constant':a.fill_(.5);r.zero_()
    if pattern=='nearly_constant':a=a*.001+.5;r.zero_()
    if pattern=='large':a.mul_(10000);r.mul_(1000)
    if pattern=='cancellation':r=-a;a=a+torch.randn_like(a)*.001
    w=torch.randn(channels,device='cuda',dtype=torch.float16)*.3
    b=torch.randn_like(w)*.2
    expected=F.gelu(F.layer_norm(a+r,(channels,),w,b,1e-5))
    actual=torch.ops.multigpubeamsearch_gnn.residual_norm_gelu(a,r,w,b,1e-5)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,rtol=.003,atol=.002)
    records.append({'shape':[rows,channels],'pattern':pattern,'max_error':(actual-expected).abs().max().item(),'mismatches':int((actual!=expected).sum())})
 timings=[]
 if not os.environ.get('GNN_SANITIZER'):
  for rows,channels in [(4096,128),(855000,128)]:
   a=torch.randn(rows,channels,device='cuda',dtype=torch.float16)
   r=torch.randn_like(a);w=torch.randn(channels,device='cuda',dtype=torch.float16);b=torch.randn_like(w)
   for name,fn in [('aten',lambda:F.gelu(F.layer_norm(a+r,(channels,),w,b,1e-5))),('fused',lambda:torch.ops.multigpubeamsearch_gnn.residual_norm_gelu(a,r,w,b,1e-5))]:
    for _ in range(5):actual=fn()
    samples=[]
    for _ in range(11):
     start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     start.record();actual=fn();end.record();end.synchronize();samples.append(start.elapsed_time(end))
    timings.append({'shape':[rows,channels],'backend':name,'gpu_ms':samples,'median_ms':statistics.median(samples)})
suffix='-sanitizer' if os.environ.get('GNN_SANITIZER') else ''
Path('/root/further-result/norm'+suffix+'.json').write_text(json.dumps({'status':'PASS','cases':records,'max_error':max(x['max_error'] for x in records),'timings':timings}))
print('norm',len(records),'cases PASS',max(x['max_error'] for x in records),flush=True)
