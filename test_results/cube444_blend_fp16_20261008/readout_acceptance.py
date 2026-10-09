import os,pathlib,json
os.environ['MAX_JOBS']='2';os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
import torch
from torch.utils.cpp_extension import load
r=pathlib.Path('/tmp/blend-production')
ext=load(name='cube_fp16_probe',sources=['/tmp/probe.cpp',str(r/'cuda/stream1_blend_readout.cu')],extra_include_paths=[str(r),str(r/'src'),'/tmp/blend-validation/cutlass/include'],extra_cflags=['-O1'],extra_cuda_cflags=['-O2','-DBEAM_HAS_CUTLASS=1','-DBEAM_STATE_LOGICAL_BYTES=96','-DBEAM_STATE_PHYSICAL_BYTES=112','-DBEAM_STATE_ALIGNMENT=16','-DBEAM_MOVE_COUNT=24'],verbose=True)
results=[]
with torch.inference_mode():
 for gpu in (0,1):
  torch.cuda.set_device(gpu);model=ext.Model('/tmp/blend-validation/bundle',gpu)
  for rows in (1,7,31,64,129,255,256):
   torch.manual_seed(8100+rows);parents=torch.randint(0,6,(rows,96),device='cuda');cls=model.cls(parents);mlp=model.mlp(parents)
   assert cls.dtype==torch.float16 and mlp.dtype==torch.float32
   keys,error=model.readout(cls,mlp);reference=model.projection_reference(cls,mlp).clamp(0,300).mul(1024).round().int()
   difference=(keys-reference).abs().max().item();assert error.item()==0 and difference<=1
   results.append(dict(gpu=gpu,rows=rows,max_key_difference=difference))
  del model;torch.cuda.empty_cache()
pathlib.Path('/tmp/fp16-results/readout_acceptance.json').write_text(json.dumps(dict(status='PASS',cases=results),indent=2));print('READOUT_PASS',flush=True)
