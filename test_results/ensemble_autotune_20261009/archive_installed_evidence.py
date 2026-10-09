"""Archive terminal remote evidence; exclude caches' executable/source trees."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path('/workspace/results')
files=set()
for folder in ('installed-wheel-api-v2','installed-wheel-api-17'):
    acceptance=root/folder/'acceptance.json'
    receipt=json.loads(acceptance.read_text())
    assert receipt['replay_valid'] and receipt['metadata']['profile']['pipeline_autotuned']
    run=Path(receipt['metadata']['run_dir'])
    files.update(p for p in (root/folder).rglob('*') if p.is_file())
    files.update(p for p in run.rglob('*') if p.is_file() and p.name!='production_runner')
    # Frozen runtime libraries are already identified by build-time SHA256.
    files={p for p in files if 'runtime-libs' not in p.parts}
for name in ('compact-wheel-build-v2.log','installed-wheel-full-cpu-v2.log',
             'inference-refinement-cpu.log','compact-source-verification.json',
             'installed-wheel-api-driver.log','installed-wheel-api-v2-driver.log',
             'installed-wheel-api-17-driver.log'):
    files.add(root/name)
cache=root/'installed-wheel-api/cache'
files.update(cache.rglob('native-preparation.json'))
files.update(cache.rglob('native-build.json'))
files.update(cache.rglob('cmake-build.log'))
files.update(cache.rglob('cmake-configure.log'))
records={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,
    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}
manifest=root/'installed-wheel-evidence-manifest.json'
manifest.write_text(json.dumps(records,indent=2))
archive=root/'installed-wheel-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files):tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    tar.add(manifest,arcname=manifest.name,recursive=False)
with tarfile.open(archive,'r:gz') as tar:
    assert len(tar.getmembers())==len(records)+1
    for name,row in records.items():
        content=tar.extractfile(name).read()
        assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
summary=dict(archive=str(archive),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(records),members_verified=True)
(root/'installed-wheel-archive-receipt.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
