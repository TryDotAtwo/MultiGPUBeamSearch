"""Check all archived blobs and retain compact source-bound test evidence."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

archive, expected = Path(sys.argv[1]), sys.argv[2]
workspace = Path(__file__).resolve().parents[3]
out = workspace / 'test_results/refactor_20261114/rust-mlp-verify-3060'
prefix = 'beam-results/rust-mlp-verify-20261114/'
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha(archive.read_bytes()) == expected, 'downloaded archive hash mismatch'
verified, manifest = {}, None
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
        if member.name == 'manifest.json':
            manifest = json.load(tar.extractfile(member))
            continue
        assert member.isfile() and member.name.startswith('blobs/')
        digest = member.name.removeprefix('blobs/')
        h, size = hashlib.sha256(), 0
        stream = tar.extractfile(member)
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
            size += len(block)
        assert h.hexdigest() == digest and size == member.size and digest not in verified
        verified[digest] = size
assert manifest is not None and manifest['blob_bytes'] == verified
for row in manifest['records'].values():
    assert verified[row['sha256']] == row['bytes']
selected_names = {'status.txt', 'inputs-before.txt', 'inputs-after.txt', 'gpus.csv', 'software.txt',
                  'cargo-tests.log', 'cargo-build.log', 'rust-binaries.sha256', 'pyo3-test.log',
                  'python-tests.log', 'gpu.log', 'gpu/results.json', 'gpu/partial.json', 'pip.log', 'rustup.log'}
selected = {name: row for name, row in manifest['records'].items()
            if name.startswith(prefix) and name[len(prefix):] in selected_names}
bootstrap = {name: row for name, row in manifest['records'].items()
             if '/resident-session/bootstrap-' in name and name.endswith('.log')}
out.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive, 'r:gz') as source:
    source_manifest_bytes = source.extractfile('manifest.json').read()
    for name, row in selected.items():
        target = (out / name[len(prefix):]).resolve()
        assert target.is_relative_to(out.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.extractfile('blobs/' + row['sha256']).read()
        assert sha(data) == row['sha256'] and len(data) == row['bytes']
        target.write_bytes(data)
    portable = out / 'bootstrap-receipts.tar.gz'
    with tarfile.open(portable, 'w:gz', compresslevel=6) as dest:
        for digest in sorted({row['sha256'] for row in bootstrap.values()}):
            data = source.extractfile('blobs/' + digest).read()
            assert sha(data) == digest
            member = tarfile.TarInfo('blobs/' + digest)
            member.size = len(data)
            dest.addfile(member, io.BytesIO(data))
        member = tarfile.TarInfo('source-manifest.json')
        member.size = len(source_manifest_bytes)
        dest.addfile(member, io.BytesIO(source_manifest_bytes))
        data = json.dumps({'source_archive_sha256': expected,
                           'source_manifest_sha256': sha(source_manifest_bytes),
                           'records': bootstrap, 'evidence': selected},
                          sort_keys=True).encode()
        member = tarfile.TarInfo('manifest.json')
        member.size = len(data)
        dest.addfile(member, io.BytesIO(data))
report = {'archive_sha256': expected, 'archive_bytes': archive.stat().st_size,
          'records': len(manifest['records']), 'blobs': len(verified), 'omitted': len(manifest['omitted']),
          'extracted': sorted(name[len(prefix):] for name in selected),
          'bootstrap_records': len(bootstrap), 'bootstrap_sha256': sha(portable.read_bytes()),
          'source_manifest_sha256': sha(source_manifest_bytes)}
(out / 'archive-verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
