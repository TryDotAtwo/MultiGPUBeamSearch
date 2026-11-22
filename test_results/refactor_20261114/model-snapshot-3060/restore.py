"""Restore content-verified compatible build artifacts from the prior archive."""
import hashlib
import json
from pathlib import Path
import tarfile

archive = Path('/workspace/public-index-evidence.tar.gz')
expected = 'd2878a7258afccdad18fa84fb7cbeb3d292036c454d71ccdd21783ac596978fd'
digest = hashlib.sha256()
with archive.open('rb') as source:
    for block in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != expected:
    raise RuntimeError('prior evidence archive digest differs')
workspace = Path('/workspace')
with tarfile.open(archive, 'r:gz') as tar:
    manifest = json.load(tar.extractfile('manifest.json'))
    selected = {name: row for name, row in manifest['records'].items() if
        name.startswith('beam-results/public-index-20261114/cache-cutlass/builds/') or
        name in ('beam-work/integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so',
                 'beam-work/integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner')}
    if len(selected) != 12:
        raise RuntimeError(f'unexpected prior build file count: {len(selected)}')
    for name, row in sorted(selected.items()):
        target = (workspace / name).resolve()
        if not target.is_relative_to(workspace) or target.exists():
            raise RuntimeError(f'unsafe or occupied restoration target: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        data = tar.extractfile('blobs/' + row['sha256']).read()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise RuntimeError(f'prior artifact hash mismatch: {name}')
        target.write_bytes(data)
        target.chmod(row['mode'])
        print(name, row['sha256'], len(data))
print('PRIOR_ARTIFACT_RESTORE_OK')
