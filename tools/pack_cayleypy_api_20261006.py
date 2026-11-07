"""Package bytes and hashes only; no local product execution."""
from pathlib import Path
import hashlib
import io
import json
import tarfile

root = Path.cwd()
destination = root / 'test_results/cayleypy_api_20261006'
files = []
extensions = {'.py','.cu','.cuh','.cpp','.hpp','.h','.cmake','.sh','.json','.txt','.md','.toml'}
for folder in ('cuda','src','tools','tests','cmake','configs','schemas','third_party','integrations/cayleypy_native'):
    files.extend(p for p in (root/folder).rglob('*') if p.is_file() and
                 (p.suffix in extensions or p.name.startswith('LICENSE')) and
                 not any(x in p.parts for x in ('.git','__pycache__','.pytest_cache')))
files.extend(root/p for p in ('CMakeLists.txt','AGENTS.md','ARCHITECTURE_NEED.md'))
manifest = {}
archive = destination/'source.tar.gz'
with tarfile.open(archive, 'w:gz') as output:
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes().replace(b'\r\n',b'\n')
        manifest[relative] = hashlib.sha256(raw).hexdigest()
        info = tarfile.TarInfo(relative); info.size=len(raw); info.mode=0o644
        output.addfile(info, io.BytesIO(raw))
    raw = json.dumps(manifest,sort_keys=True,indent=2).encode()
    info = tarfile.TarInfo('source_manifest.json'); info.size=len(raw)
    output.addfile(info,io.BytesIO(raw))
(destination/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'files':len(manifest),'bytes':archive.stat().st_size,
                  'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
