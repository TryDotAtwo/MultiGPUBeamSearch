
import json,os,sys,tarfile,urllib.request
from pathlib import Path
sha='67fbc205814854bda4b98d1d6020b8dce81c18ab'
out=Path('/root/gnn-opt-beam-result');out.mkdir(exist_ok=True)
try:
 root=Path('/root/gnn-opt-beam-source');root.mkdir(exist_ok=True)
 archive=Path('/root/gnn-opt-beam-source.tar.gz')
 urllib.request.urlretrieve('https://codeload.github.com/TryDotAtwo/MultiGPUBeamSearch/tar.gz/'+sha,archive)
 with tarfile.open(archive) as tar:
  for member in tar:
   target=root/Path(*Path(member.name).parts[1:])
   if not target.resolve().is_relative_to(root.resolve()) or not(member.isdir() or member.isfile()):raise RuntimeError('unsafe archive')
   if member.isdir():target.mkdir(parents=True,exist_ok=True)
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,target.open('wb') as dst:dst.write(src.read())
 os.environ['PATH']='/venv/main/bin:/usr/local/cuda/bin:'+os.environ.get('PATH','')
 os.environ['CUDA_HOME']='/usr/local/cuda'
 sys.path.insert(0,str(root/'integrations/cayleypy_native'))
 import torch,cayleypy as cp,multigpubeamsearch as beam
 from multigpubeamsearch import PancakeGNN,NeighborConfig,NativeEnsemble,NativeOptions
 moves=[list(range(k-1,-1,-1))+list(range(k,4)) for k in range(2,5)]
 graph=cp.CayleyGraph(cp.CayleyGraphDef.create(moves),device='cuda')
 models=[]
 for seed in (11,12,13):
  torch.manual_seed(seed)
  models.append(PancakeGNN(4,32,2,neighbors=NeighborConfig(num_hops=2,max_frontier_states=24)).eval())
 records=[]
 for backend in ('cutlass',):
  options=NativeOptions(source_dir=root,cutlass_dir=Path('/root/cutlass-source'),cache_dir=out/'cache',num_gpus=1,inference_backend=backend,autotune=False,build_jobs=2,timeout_seconds=600)
  for name,model in [('single',models[0]),('blend3',NativeEnsemble(models,[.2,.3,.5]))]:
   result=beam.beam_search(graph,start_state=torch.tensor([[3,1,0,2]],device='cuda'),predictor=model,beam_width=1024,max_steps=16,return_path=True,native_options=options)
   assert result.path_found,repr(result)
   records.append(dict(backend=backend,model=name,path_length=int(result.path_length),metadata=result.native_metadata))
   (out/'progress.json').write_text(json.dumps(records,default=str))
 (out/'result.json').write_text(json.dumps(dict(source_commit=sha,gpu=torch.cuda.get_device_name(),records=records,status='PASS',scope='synthetic tiny full beam; not trained quality'),default=str))
except Exception as e:
 (out/'exit.json').write_text(json.dumps(dict(source_commit=sha,error=repr(e),returncode=1)))
 raise
else:
 (out/'exit.json').write_text(json.dumps(dict(source_commit=sha,returncode=0)))
