"""Restore only the hash-verified test patch into this task's clone."""
import hashlib
import json
from pathlib import Path
import tarfile

archive=Path('/workspace/mlp-binding-overlay.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '73b4ac62d27ea5e0e5af41e302f5b6416efac3f00fca17c7a9c5ddb86a61324e'
root=Path('/workspace/beam-work').resolve()
with tarfile.open(archive,'r:gz') as source:
    manifest=json.load(source.extractfile('manifest.json'))
    assert manifest['base']=='65a1952c265bd60c6d69a7ee90687c0a7d336888'
    for name,row in sorted(manifest['files'].items()):
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise RuntimeError('invalid overlay path')
        data=source.extractfile('files/'+name).read()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise RuntimeError('overlay digest mismatch: '+name)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        path.chmod(row['mode'])
        print(name,row['sha256'],len(data))
print('OVERLAY_VERIFIED')
