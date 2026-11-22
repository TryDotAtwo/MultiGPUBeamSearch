import subprocess,sys
from pathlib import Path
code=r'''
import json,os,sys,subprocess,hashlib,shutil
from pathlib import Path
root=Path('/root/further-beam-source')
out=Path('/root/further-final-beam-result');out.mkdir(exist_ok=True)
try:
 # Restore the measured normalization-only CUDA source; all other source files
 # are identical between these two commits. Rebuild only the changed translation unit.
 shutil.copyfile('/root/further-candidate-source/cuda/stream1_gnn_projection.cu',root/'cuda/stream1_gnn_projection.cu')
 os.environ['PATH']='/venv/main/bin:/usr/local/cuda/bin:'+os.environ.get('PATH','')
 os.environ['CUDA_HOME']='/usr/local/cuda'
 sys.path.insert(0,str(root/'integrations/cayleypy_native'))
 from multigpubeamsearch.build import source_digest
 assert source_digest(root)==source_digest(Path('/root/further-candidate-source'))
 dirs=list(Path('/root/further-beam-result/cache').rglob('native-build.json'))
 assert len(dirs)==1,dirs
 metadata_path=dirs[0];build=metadata_path.parent
 with (out/'incremental-build.log').open('w') as log:
  subprocess.run(['cmake','--build',str(build),'--target','production_runner_libtorch_stream1','stream1_ensemble_benchmark','-j','2'],check=True,stdout=log,stderr=subprocess.STDOUT)
 meta=json.loads(metadata_path.read_text())
 meta['source_digest']=source_digest(root)
 meta['source_commit']='b74b6acf599e9a134aa866bbb7bcf61fe01376c0'
 meta['binary_sha256']=hashlib.sha256((build/'production_runner_libtorch_stream1').read_bytes()).hexdigest()
 meta['calibration_binary_sha256']=hashlib.sha256((build/'stream1_ensemble_benchmark').read_bytes()).hexdigest()
 metadata_path.write_text(json.dumps(meta))
 (out/'native-build.json').write_text(json.dumps(meta))
 import torch,cayleypy as cp,multigpubeamsearch as beam
 from multigpubeamsearch import PancakeGNN,NeighborConfig,NativeEnsemble,NativeOptions
 moves=[list(range(k-1,-1,-1))+list(range(k,4)) for k in range(2,5)]
 graph=cp.CayleyGraph(cp.CayleyGraphDef.create(moves),device='cuda')
 models=[]
 for seed in (11,12,13):
  torch.manual_seed(seed);models.append(PancakeGNN(4,32,2,neighbors=NeighborConfig(num_hops=2,max_frontier_states=24)).eval())
 records=[]
 options=NativeOptions(source_dir=root,cutlass_dir=Path('/root/cutlass-source'),runner_path=build/'production_runner_libtorch_stream1',cache_dir=out/'cache',num_gpus=1,inference_backend='cutlass',autotune=False,timeout_seconds=600)
 for name,model in [('single',models[0]),('blend3',NativeEnsemble(models,[.2,.3,.5]))]:
  result=beam.beam_search(graph,start_state=torch.tensor([[3,1,0,2]],device='cuda'),predictor=model,beam_width=1024,max_steps=16,return_path=True,native_options=options)
  assert result.path_found and result.native_metadata['replay_valid']
  records.append({'model':name,'path_length':int(result.path_length),'metadata':result.native_metadata})
 (out/'result.json').write_text(json.dumps({'source_commit':meta['source_commit'],'records':records,'status':'PASS'},default=str))
 (out/'exit.json').write_text(json.dumps({'returncode':0}))
except Exception as e:
 (out/'exit.json').write_text(json.dumps({'returncode':1,'error':repr(e)}));raise
'''
Path('/root/further-final-beam-job.py').write_text(code)
with Path('/root/further-final-beam.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'/root/further-final-beam-job.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(p.pid)
