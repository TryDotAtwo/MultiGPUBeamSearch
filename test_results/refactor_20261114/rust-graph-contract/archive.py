"""Hash and preserve task results, recording omitted rebuildable outputs."""
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import time

workspace=Path('/workspace')
results=workspace/'beam-results/rust-graph-contract'
target=workspace/'rust-graph-contract-evidence.tar.gz'
if target.exists():raise RuntimeError('evidence archive already exists')
files=list(results.rglob('*'))+[workspace/'graph-overlay.tar.gz',workspace/'graph-run.sh',
    workspace/'graph-driver.log',
    workspace/'beam-work/integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so',
    workspace/'beam-work/integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner']
keep_build={'native-build.json','native-library-build.json','CMakeCache.txt','production_runner','libbeam_native.so'}
records,omitted,blobs={},{},{}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
with tarfile.open(target,'w:gz',compresslevel=3) as tar:
    for path in sorted(set(files)):
        if not path.is_file() or path.is_symlink():continue
        name=path.relative_to(workspace).as_posix()
        reason=None
        if 'runtime-libs' in path.parts:reason='rebuildable NCCL files'
        elif '__pycache__' in path.parts or path.suffix=='.pyc':reason='rebuildable interpreter cache'
        elif 'builds' in path.parts and path.name not in keep_build:reason='rebuildable build intermediate'
        if reason:
            omitted[name]={'bytes':path.stat().st_size,'reason':reason}
            continue
        value,size=digest(path),path.stat().st_size
        records[name]={'bytes':size,'sha256':value,'mode':stat.S_IMODE(path.stat().st_mode)}
        if value not in blobs:
            tar.add(path,arcname='blobs/'+value,recursive=False)
            blobs[value]=size
        elif blobs[value]!=size:raise RuntimeError('inconsistent duplicate')
    data=json.dumps(dict(schema=1,created_unix=time.time(),records=records,omitted=omitted,blob_bytes=blobs),sort_keys=True).encode()
    member=tarfile.TarInfo('manifest.json');member.size=len(data);tar.addfile(member,io.BytesIO(data))
verified={};manifest=None
with tarfile.open(target,'r:gz') as tar:
    for member in tar:
        if member.name=='manifest.json':
            manifest=json.load(tar.extractfile(member));continue
        if not member.isfile() or not member.name.startswith('blobs/'):
            raise RuntimeError('unexpected archive member')
        h,count=hashlib.sha256(),0
        stream=tar.extractfile(member)
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block);count+=len(block)
        value=member.name.removeprefix('blobs/')
        if h.hexdigest()!=value or count!=member.size or value in verified:
            raise RuntimeError('archive verification failed')
        verified[value]=count
if manifest is None or manifest['blob_bytes']!=verified:raise RuntimeError('manifest mismatch')
for row in manifest['records'].values():
    if verified[row['sha256']]!=row['bytes']:raise RuntimeError('record mismatch')
receipt=dict(sha256=digest(target),bytes=target.stat().st_size,records=len(records),
    blobs=len(blobs),omitted=len(omitted))
(workspace/'rust-graph-contract-archive.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
