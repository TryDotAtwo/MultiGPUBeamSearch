"""Verify the full run archive and restore compact, portable evidence."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

if not __debug__:
    raise SystemExit('archive verification requires Python assertions to be enabled')

root = Path(__file__).resolve().parents[3]
archive = root / '.local/cache/rust-mlp-snapshot-20261114/evidence.tar.gz'
out = Path(__file__).resolve().parent
out.mkdir(parents=True, exist_ok=True)
prefix = 'beam-results/rust-mlp-snapshot-20261114/'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

archive_sha = digest(archive)
with tarfile.open(archive, 'r:gz') as tar:
    source_manifest = tar.extractfile('manifest.json').read()
manifest = json.loads(source_manifest)
source_manifest_sha = hashlib.sha256(source_manifest).hexdigest()
records = manifest['records']
selected = {}
receipts = {}
for name, row in records.items():
    if name.startswith(prefix):
        rel = name[len(prefix):]
        if '/' not in rel or rel == 'gpu/results.json':
            selected[name] = row
        if '/resident-session/bootstrap-' in name and name.endswith('.log'):
            receipts[name] = row
needed = {row['sha256'] for row in (*selected.values(), *receipts.values())}
content, verified = {}, {}
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
        if member.name == 'manifest.json':
            continue
        assert member.isfile() and member.name.startswith('blobs/')
        name = member.name.removeprefix('blobs/')
        h, total, chunks = hashlib.sha256(), 0, []
        stream = tar.extractfile(member)
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
            total += len(block)
            if name in needed:
                chunks.append(block)
        assert h.hexdigest() == name and total == member.size and name not in verified
        verified[name] = total
        if chunks:
            content[name] = b''.join(chunks)
        elif name in needed:
            content[name] = b''
assert verified == manifest['blob_bytes']
for row in records.values():
    assert verified[row['sha256']] == row['bytes']
for name, row in selected.items():
    # An earlier malformed shell script created a filename ending in CR.
    # Preserve it under a portable name; the original remains in the archive.
    target = out / name[len(prefix):].replace('\r', '_CR')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content[row['sha256']])
portable = out / 'bootstrap-receipts.tar.gz'
evidence_names = ('gpu/results.json', 'status.txt', 'python-tests.log', 'cargo-tests.log',
                  'pyo3-test.log', 'mock-tests.log', 'inputs-after.txt')
evidence = {prefix + name: selected[prefix + name] for name in evidence_names}
receipt_manifest = json.dumps({'schema': 1, 'source_archive_sha256': archive_sha,
                               'source_manifest_sha256': source_manifest_sha,
                               'records': receipts, 'evidence': evidence}, sort_keys=True).encode()
with tarfile.open(portable, 'w:gz') as tar:
    info = tarfile.TarInfo('manifest.json')
    info.size = len(receipt_manifest)
    tar.addfile(info, io.BytesIO(receipt_manifest))
    info = tarfile.TarInfo('source-manifest.json')
    info.size = len(source_manifest)
    tar.addfile(info, io.BytesIO(source_manifest))
    for sha in sorted({row['sha256'] for row in (*receipts.values(), *evidence.values())}):
        data = content[sha]
        info = tarfile.TarInfo('blobs/' + sha)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
report = {'archive_sha256': archive_sha, 'source_manifest_sha256': source_manifest_sha,
          'records': len(records), 'blobs': len(verified),
          'omitted': len(manifest['omitted']), 'bootstrap_records': len(receipts),
          'bootstrap_sha256': digest(portable)}
(out / 'archive-verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
