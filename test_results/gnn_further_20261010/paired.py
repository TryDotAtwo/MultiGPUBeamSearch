import torch,json,sys,time,statistics
from pathlib import Path
from torch.utils.cpp_extension import _get_build_directory
variant,round_id=sys.argv[1:]
torch.set_num_threads(2)
torch.ops.load_library(str(Path(_get_build_directory('beam_gnn_further_'+variant,False))/('beam_gnn_further_'+variant+'.so')))
torch.ops.multigpubeamsearch_gnn.large_setup('/root/further-'+variant+'-result/artifact/weights',0)
torch.manual_seed(901)
states=torch.stack([torch.randperm(100,device='cuda') for _ in range(3)])
samples=[]
with torch.inference_mode():
 for _ in range(5):actual=torch.ops.multigpubeamsearch_gnn.large_run(states,True)
 for _ in range(21):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record();actual=torch.ops.multigpubeamsearch_gnn.large_run(states,True);end.record();end.synchronize();samples.append(start.elapsed_time(end))
import hashlib
r={'variant':variant,'round':round_id,'gpu_ms':samples,'median_ms':statistics.median(samples),'output_sha':hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()}
torch.save(actual.cpu(),'/root/further-output-'+variant+'.pt')
Path('/root/further-result/paired-'+round_id+'-'+variant+'.json').write_text(json.dumps(r));print(json.dumps(r),flush=True)
