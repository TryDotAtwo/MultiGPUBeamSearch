"""Compare the retained initial source archive to its stated Git tree."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

if not __debug__:
    raise SystemExit('verification requires Python assertions to be enabled')

here = Path(__file__).resolve().parent
root = here.parents[2]
archive = root / '.local/cache/rust-mlp-snapshot-20261114/base.tar.gz'
manifest = json.loads((here / 'tested-overlay-manifest.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['base_archive_sha256']
raw = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', manifest['base']], cwd=root)
tree = {}
for entry in raw.split(b'\0'):
    if not entry:
        continue
    info, name = entry.split(b'\t', 1)
    mode, kind, oid = info.split(b' ')
    tree[name.decode('utf-8')] = (mode.decode(), kind.decode(), oid.decode())
checked = {}
normalized = []
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
        if member.isdir():
            continue
        name = member.name.removeprefix('./')
        assert name not in checked and name in tree, name
        mode, kind, oid = tree[name]
        if member.isfile():
            assert kind == 'blob' and mode in ('100644', '100755'), name
            data = tar.extractfile(member).read()
        elif member.issym():
            assert kind == 'blob' and mode == '120000', name
            data = member.linkname.encode('utf-8')
        else:
            raise AssertionError(f'unexpected archive entry {name}')
        def blob_sha(blob):
            return hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
        git_hash = blob_sha(data)
        if git_hash != oid and member.isfile() and b'\r\n' in data:
            git_hash = blob_sha(data.replace(b'\r\n', b'\n'))
            if git_hash == oid:
                normalized.append(name)
        assert git_hash == oid, name
        checked[name] = hashlib.sha256(data).hexdigest()
print(json.dumps({'verification': 'PASS', 'base': manifest['base'],
                  'archive_sha256': manifest['base_archive_sha256'],
                  'matching_git_blobs': len(checked),
                  'matched_after_lf_normalization': len(normalized)}, indent=2))
