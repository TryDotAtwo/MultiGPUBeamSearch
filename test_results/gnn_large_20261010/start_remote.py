import subprocess,sys
from pathlib import Path
code=r'''
import json,os,sys,tarfile,urllib.request,subprocess
from pathlib import Path
out=Path('/root/gnn-large-result');out.mkdir(exist_ok=True)
sha='83e46be081cae82c071128713bb93c47178f9cf3'
try:
 root=Path('/root/gnn-large-source');root.mkdir(exist_ok=True)
 archive=Path('/root/gnn-large-source.tar.gz')
 urllib.request.urlretrieve('https://codeload.github.com/TryDotAtwo/MultiGPUBeamSearch/tar.gz/'+sha,archive)
 with tarfile.open(archive) as tar:
  for member in tar:
   target=root/Path(*Path(member.name).parts[1:])
   if not target.resolve().is_relative_to(root.resolve()) or not(member.isdir() or member.isfile()):raise RuntimeError('unsafe archive')
   if member.isdir():target.mkdir(parents=True,exist_ok=True)
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,target.open('wb') as dst:dst.write(src.read())
 subprocess.run([sys.executable,'-m','pip','install','-q','cayleypy','ninja'],check=True)
 sys.path.insert(0,str(root/'integrations/cayleypy_native'))
 from multigpubeamsearch.sources import CUTLASS_SOURCE,_download,_extract
 cutlass_archive=Path('/root/cutlass-source.tar.gz');cutlass=Path('/root/cutlass-source')
 _download(CUTLASS_SOURCE,cutlass_archive);_extract(cutlass_archive,cutlass,CUTLASS_SOURCE)
 env=dict(os.environ,GNN_SOURCE=str(root),GNN_COMMIT=sha,TORCH_CUDA_ARCH_LIST='8.6',CUDA_HOME='/usr/local/cuda',MAX_JOBS='2',PATH='/venv/main/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''))
 result=subprocess.run([sys.executable,'/root/gnn-large-benchmark.py'],env=env)
 (out/'exit.json').write_text(json.dumps(dict(returncode=result.returncode,source_commit=sha)))
except Exception as error:
 (out/'exit.json').write_text(json.dumps(dict(error=repr(error),returncode=1)))
 raise
'''
Path('/root/gnn-large-job.py').write_text(code)
with Path('/root/gnn-large-task.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'/root/gnn-large-job.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print('Started',p.pid)
