from pathlib import Path
import json
p=Path(__file__).resolve().parent
src=p.parent/'gnn_opt_20261010'
benchmark=(src/'benchmark.py').read_text()
for variant in ('baseline','candidate'):
 code=benchmark.replace('/root/gnn-opt-result','/root/bias-'+variant+'-result').replace('beam_gnn_opt_comparison','beam_gnn_bias_'+variant)
 # Retain Python numerical oracle, skip unchanged LibTorch timing.
 code=code.replace("('python_pytorch','native_libtorch','native_cutlass')","('python_pytorch','native_cutlass')")
 # Exact cross-process inputs independent of compile/export RNG effects.
 code=code.replace('records=[]','torch.manual_seed(901)\nrecords=[]')
 (p/(variant+'.py')).write_text(code)
driver=r'''
import json,os,subprocess,sys,tarfile,urllib.request
from pathlib import Path
base='28a1ba8c482926b23549cbd066b9b9829b37522e'
candidate='713d7bb2cde3b0e60ad22dd9aa389b5924215528'
out=Path('/root/bias-result');out.mkdir(exist_ok=True)
try:
 for variant,sha in [('baseline',base),('candidate',candidate)]:
  root=Path('/root/bias-'+variant+'-source');root.mkdir(exist_ok=True)
  archive=Path('/root/bias-'+variant+'.tar.gz')
  urllib.request.urlretrieve('https://codeload.github.com/TryDotAtwo/MultiGPUBeamSearch/tar.gz/'+sha,archive)
  with tarfile.open(archive) as tar:
   for member in tar:
    target=root/Path(*Path(member.name).parts[1:])
    if not target.resolve().is_relative_to(root.resolve()) or not(member.isdir() or member.isfile()):raise RuntimeError('unsafe archive')
    if member.isdir():target.mkdir(parents=True,exist_ok=True)
    else:
     target.parent.mkdir(parents=True,exist_ok=True)
     with tar.extractfile(member) as inp,target.open('wb') as dst:dst.write(inp.read())
  if variant=='baseline':
   subprocess.run([sys.executable,'-m','pip','install','-q','cayleypy','ninja'],check=True)
   sys.path.insert(0,str(root/'integrations/cayleypy_native'))
   from multigpubeamsearch.sources import CUTLASS_SOURCE,_download,_extract
   a=Path('/root/cutlass-source.tar.gz');c=Path('/root/cutlass-source')
   _download(CUTLASS_SOURCE,a);_extract(a,c,CUTLASS_SOURCE)
  env=dict(os.environ,GNN_SOURCE=str(root),GNN_COMMIT=sha,TORCH_CUDA_ARCH_LIST='8.6',CUDA_HOME='/usr/local/cuda',MAX_JOBS='2',PATH='/venv/main/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''))
  subprocess.run([sys.executable,'/root/bias-'+variant+'.py'],env=env,check=True)
 (out/'exit.json').write_text(json.dumps({'returncode':0}))
except Exception as error:
 (out/'exit.json').write_text(json.dumps({'returncode':1,'error':repr(error)}));raise
'''
payload="from pathlib import Path\nimport subprocess,sys\n"
for variant in ('baseline','candidate'):
 payload+='Path('+repr('/root/bias-'+variant+'.py')+').write_text('+repr((p/(variant+'.py')).read_text())+')\n'
payload+="Path('/root/bias-job.py').write_text("+repr(driver)+")\nwith Path('/root/bias-task.log').open('ab') as log:\n p=subprocess.Popen([sys.executable,'/root/bias-job.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)\nprint('Started',p.pid)\n"
(p/'start.py').write_text(payload)
