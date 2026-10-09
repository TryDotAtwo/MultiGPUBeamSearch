import os,pathlib,json,csv,sys,statistics,time,hashlib
os.environ['MAX_JOBS']='2';os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
import torch,numpy as np
from torch.utils.cpp_extension import load
root=pathlib.Path('/tmp/blend-production');out=pathlib.Path('/tmp/fp16-results');out.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'tools'));from export_cube444_blend import export
bundle=pathlib.Path('/tmp/blend-validation/bundle');assets=pathlib.Path('/tmp/blend-validation/assets')
if not bundle.exists():export(assets,bundle)
ext=load(name='cube_fp16_probe',sources=['/tmp/probe.cpp',str(root/'cuda/stream1_blend_readout.cu')],extra_include_paths=[str(root),str(root/'src'),'/tmp/blend-validation/cutlass/include'],extra_cflags=['-O1'],extra_cuda_cflags=['-O2','-DBEAM_HAS_CUTLASS=1','-DBEAM_STATE_LOGICAL_BYTES=96','-DBEAM_STATE_PHYSICAL_BYTES=112','-DBEAM_STATE_ALIGNMENT=16','-DBEAM_MOVE_COUNT=24'],verbose=True)
rows=list(csv.DictReader((assets/'test.csv').open()))
corpus=np.asarray([list(map(int,r['initial_state'].split(','))) for r in rows[:1024]],np.uint8)
summary=dict(status='RUNNING',warmup=5,repeats=30,precision='FP16 weights/activations; FP32 reductions/accumulation/blend',gpus=[],quality=[],timings=[])
with torch.inference_mode():
 for gpu in (0,1):
  torch.cuda.set_device(gpu);main=torch.cuda.current_stream();other=torch.cuda.Stream();model=ext.Model(str(bundle),gpu);reference=ext.Model(str(bundle),gpu,True)
  summary['gpus'].append(dict(index=gpu,name=torch.cuda.get_device_name(),properties=str(torch.cuda.get_device_properties(gpu))))
  actual=[];oracle=[];fused=[]
  for block in np.array_split(corpus,4):
   parents=torch.tensor(block,device='cuda',dtype=torch.long)
   tr=model.transformer(parents).float();mlp=model.mlp(parents);cls=model.cls(parents);keys,error=model.readout(cls,mlp)
   assert error.item()==0
   # The fused FP32 accumulator is the authoritative FP16 output projection.
   actual.append(keys.float()/1024);oracle.append(.6*reference.transformer(parents)+.4*reference.mlp(parents))
  a=torch.cat(actual);b=torch.cat(oracle);delta=(a-b).abs();top1=(a.argmin(-1)==b.argmin(-1)).float().mean().item()
  ka=set(a.flatten().topk(256,largest=False).indices.tolist());kb=set(b.flatten().topk(256,largest=False).indices.tolist());overlap=len(ka&kb)/256
  q=dict(gpu=gpu,max_abs=delta.max().item(),mean_abs=delta.mean().item(),argmin_agreement=top1,global_top256_overlap=overlap,states=len(corpus))
  summary['quality'].append(q);(out/'progress.json').write_text(json.dumps(summary,indent=2))
  assert q['max_abs']<=.25 and q['mean_abs']<=.025 and top1>=.95 and overlap>=.95, q
  bad=model.cls(parents);bad[0,0]=float('nan');_,error=model.readout(bad,model.mlp(parents));assert error.item()==1
  del reference;torch.cuda.empty_cache()
  for batch in (64,128,256):
   parents=torch.tensor(corpus[:batch],device='cuda',dtype=torch.long);cls=model.cls(parents);mlp=model.mlp(parents)
   ready=torch.cuda.Event();done=torch.cuda.Event()
   def operation(mode):
    if mode=='transformer_head':return model.cls(parents)
    if mode=='mlp_head':return model.mlp(parents)
    if mode=='readout_blend':return model.readout(cls,mlp)
    if mode=='readout_only':return model.readout(cls,mlp,True)
    if mode=='transformer':return model.readout(model.cls(parents),mlp,True)
    if mode=='sequential':return model.readout(model.cls(parents),model.mlp(parents))
    ready.record(main)
    with torch.cuda.stream(other):
     other.wait_event(ready)
     output=model.mlp(parents) if mode=='concurrent' else mlp
     done.record(other)
    tr=model.cls(parents);main.wait_event(done)
    return model.readout(tr,output,mode=='event_control')
   for mode in ('transformer_head','mlp_head','readout_only','readout_blend','transformer','event_control','sequential','concurrent'):
    samples=[];cpu=[]
    for i in range(35):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();t=time.perf_counter();result=operation(mode);cpu.append((time.perf_counter()-t)*1000);end.record();end.synchronize()
     if isinstance(result,list):assert result[1].item()==0
     if i>=5:samples.append(start.elapsed_time(end))
    value=dict(gpu=gpu,batch=batch,mode=mode,median_ms=statistics.median(samples),p95_ms=sorted(samples)[28],cpu_enqueue_median_ms=statistics.median(cpu[5:]),samples_ms=samples)
    summary['timings'].append(value);print(json.dumps({k:v for k,v in value.items() if k!='samples_ms'}),flush=True)
   if batch==256 and gpu==0:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     with torch.profiler.record_function('FP16_BLEND_CONCURRENT'):operation('concurrent');torch.cuda.synchronize()
    prof.export_chrome_trace(str(out/'concurrent_trace.json'))
  del model,parents,cls,mlp;torch.cuda.synchronize();torch.cuda.empty_cache()
summary['status']='PASS';(out/'profile.json').write_text(json.dumps(summary,indent=2));print('PROFILE_PASS',flush=True)
