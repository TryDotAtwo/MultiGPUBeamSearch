import sys,json,torch,statistics,os
from pathlib import Path
from torch.utils.cpp_extension import _get_build_directory
variant=sys.argv[1]
torch.ops.load_library(str(Path(_get_build_directory('beam_gnn_bias_'+variant,False))/('beam_gnn_bias_'+variant+'.so')))
torch.manual_seed(311)
records=[];saved={}
reference=Path('/root/bias-projection-reference.pt')
old=torch.load(reference,weights_only=True) if variant=='candidate' else {}
with torch.inference_mode():
 for rows,k,n in [(1,8,8),(3,16,32),(127,128,256),(128,256,128),(129,128,512),(4096,128,512),(8192,256,128)]:
  a=torch.randn(rows,k,device='cuda',dtype=torch.float16)*.1
  w=torch.randn(n,k,device='cuda',dtype=torch.float16)*.1
  for kind in ('none','zero','random','cancellation'):
   b=None if kind=='none' else torch.zeros(n,device='cuda',dtype=torch.float16)
   if kind=='random':b=torch.randn(n,device='cuda',dtype=torch.float16)
   if kind=='cancellation':b=-(a.float()@w.float().t())[0].half()
   actual=torch.ops.multigpubeamsearch_gnn.linear(a,w,b,True)
   key=str((rows,k,n,kind));saved[key]=actual.cpu()
   if variant=='candidate':
    assert torch.equal(saved[key],old[key]),key
   records.append({'shape':[rows,k,n],'bias':kind,'finite':bool(torch.isfinite(actual).all()),'bitwise_equal':variant=='candidate'})
 torch.cuda.synchronize()
 if variant=='baseline':torch.save(saved,reference)
 else:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   torch.ops.multigpubeamsearch_gnn.linear(a,w,b,True);torch.cuda.synchronize()
  names=[event.name for event in prof.events()]
  assert not any('add_bias' in name for name in names),names
  Path('/root/bias-result/projection-profile.json').write_text(json.dumps(names))
timings=[]
with torch.inference_mode():
 for rows,k,n in ([] if os.environ.get('BIAS_SANITIZER') else [(4096,128,512),(100000,128,512),(100000,256,128)]):
  a=torch.randn(rows,k,device='cuda',dtype=torch.float16)*.1
  w=torch.randn(n,k,device='cuda',dtype=torch.float16)*.1
  b=torch.randn(n,device='cuda',dtype=torch.float16)
  for _ in range(3):torch.ops.multigpubeamsearch_gnn.linear(a,w,b,True)
  samples=[]
  for _ in range(9):
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   start.record()
   for _ in range(5):torch.ops.multigpubeamsearch_gnn.linear(a,w,b,True)
   end.record();end.synchronize();samples.append(start.elapsed_time(end)/5)
  timings.append({'shape':[rows,k,n],'gpu_ms':samples,'median_ms':statistics.median(samples)})
suffix='-sanitizer' if os.environ.get('BIAS_SANITIZER') else ''
Path('/root/bias-result/projection-'+variant+suffix+'.json').write_text(json.dumps({'cases':records,'timings':timings,'status':'PASS'}))
print(variant,'28 projection cases PASS')
