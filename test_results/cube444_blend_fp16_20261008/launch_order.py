import os,pathlib,json,statistics,csv
os.environ['MAX_JOBS']='2';os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
import torch
from torch.utils.cpp_extension import load
r=pathlib.Path('/tmp/blend-production');out=pathlib.Path('/tmp/fp16-results')
ext=load(name='cube_fp16_probe',sources=['/tmp/probe.cpp',str(r/'cuda/stream1_blend_readout.cu')],extra_include_paths=[str(r),str(r/'src'),'/tmp/blend-validation/cutlass/include'],extra_cflags=['-O1'],extra_cuda_cflags=['-O2','-DBEAM_HAS_CUTLASS=1','-DBEAM_STATE_LOGICAL_BYTES=96','-DBEAM_STATE_PHYSICAL_BYTES=112','-DBEAM_STATE_ALIGNMENT=16','-DBEAM_MOVE_COUNT=24'],verbose=False)
state=list(map(int,list(csv.DictReader(open('/tmp/blend-validation/assets/test.csv')))[1000]['initial_state'].split(',')));results=[]
with torch.inference_mode():
 for gpu in (0,1):
  torch.cuda.set_device(gpu);model=ext.Model('/tmp/blend-validation/bundle',gpu);parents=torch.tensor(state,device='cuda',dtype=torch.long).repeat(256,1);main=torch.cuda.current_stream();other=torch.cuda.Stream();ready=torch.cuda.Event();done=torch.cuda.Event()
  def forward(mode):
   if mode=='sequential':return model.readout(model.cls(parents),model.mlp(parents))
   ready.record(main)
   if mode=='transformer_first':cls=model.cls(parents)
   with torch.cuda.stream(other):other.wait_event(ready);mlp=model.mlp(parents);done.record(other)
   if mode=='mlp_first':cls=model.cls(parents)
   main.wait_event(done);return model.readout(cls,mlp)
  expected=forward('sequential')[0];torch.cuda.synchronize()
  # Alternate order to limit clock drift.
  samples={name:[] for name in ('sequential','mlp_first','transformer_first')}
  for i in range(35):
   for mode in (list(samples) if i%2==0 else list(reversed(samples))):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();keys,error=forward(mode);b.record();b.synchronize();assert error.item()==0 and torch.equal(keys,expected)
    if i>=5:samples[mode].append(a.elapsed_time(b))
  for mode,times in samples.items():results.append(dict(gpu=gpu,mode=mode,median_ms=statistics.median(times),samples_ms=times))
  if gpu==0:
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:forward('transformer_first');torch.cuda.synchronize()
   prof.export_chrome_trace(str(out/'transformer_first_trace.json'))
  del model;torch.cuda.empty_cache()
(out/'launch_order.json').write_text(json.dumps(dict(status='PASS',batch=256,repeats=30,results=results),indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='samples_ms'} for x in results]),flush=True)
