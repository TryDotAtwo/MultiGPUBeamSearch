"""Verify source-pinned CNOT borrowed expansion and compatibility evidence."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

here = Path(__file__).resolve().parent
root = here.parents[2]
commit = sys.argv[1] if len(sys.argv) > 1 else None
base = '3d029d303b683e5882a1e4163d496a45248f96d7'
legacy = '4846dd9d222cfb09a0df4d1acd556ff81dda7ac2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(name):
    if commit:
        return subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
    return (root / name).read_bytes().replace(b'\r\n', b'\n')


def archive(name, digest):
    path = here / name
    assert sha(path.read_bytes()) == digest, name
    return tarfile.open(path, 'r:gz')


with archive('cnot-borrowed-expansion-evidence.tar.gz',
             '4d592c17064a78ba189e60ba34fe3a034ee8204103fe87af017e2845228921d3') as tar:
    def read(name):
        return tar.extractfile('cnot-borrowed-expansion-results/' + name).read()
    manifest = json.loads(read('source-manifest.json'))
    assert manifest['base_commit'] == base and len(manifest['overlays']) == 6
    for name, digest in manifest['overlays'].items():
        if name != 'docs/plans/2026-11-17-cnot-borrowed-expansion.md':
            assert sha(source(name)) == digest, name
    if commit:
        changed = subprocess.check_output(['git', 'diff', '--name-only', base, commit],
                                          cwd=root, text=True).splitlines()
        for name in changed:
            assert (name in manifest['overlays'] or name == 'ARCHITECTURE_NEED.md'
                    or name == 'memory/CHANGELOG.md' or name.startswith('docs/')
                    or name.startswith('test_results/')), name
    assert b'RED_CONFIRMED\n' in read('status.txt')
    assert b'RUST_WORKSPACE\nEXIT_CODE=101\n' in read('status.txt')
    assert b'undefined symbol: beam_cnot_domain_create_device' in read('red.log')
    assert b'assertion failed: !old.supports_device_owners()' in read('rust-owners.log')
    assert b'borrowed_expansion_and_device_only_context ... ok' in read('rust-expansion.log')
    assert b'test result: FAILED' not in read('rust-workspace.log')
    assert read('gpus.csv').count(b'NVIDIA GeForce RTX 3060') == 2
    for name, count in (('domain.log', 5), ('owners.log', 4), ('expansion.log', 6)):
        log = read(name)
        assert f'Ran {count} tests'.encode() in log and b'\nOK\n' in log, name
    for name in ('memcheck-expansion.log', 'memcheck-owners.log'):
        log = read(name)
        assert b'LEAK SUMMARY: 0 bytes leaked' in log and b'ERROR SUMMARY: 0 errors' in log, name
    assert b'RACECHECK SUMMARY: 0 hazards' in read('racecheck-expansion.log')
    before = read('source-before.txt')
    libraries = read('libraries-sha256.txt')
    private_manifest = json.loads(read('private-manifest.json'))

with archive('cnot-borrowed-expansion-continue-evidence.tar.gz',
             'a521221ff41b7e637372e3bfc006e7f6e07a91735af3aaae94dd5a4aacfd37e8') as tar:
    def read_final(name):
        return tar.extractfile('cnot-borrowed-expansion-continue-results/' + name).read()
    assert json.loads(read_final('source-manifest.json')) == manifest
    assert json.loads(read_final('private-manifest.json')) == private_manifest
    assert read_final('source-before.txt') == read_final('source-after.txt')
    assert read_final('source-before.txt').splitlines()[0] == before.strip()
    assert read_final('libraries-sha256.txt') == libraries
    assert read_final('libraries-verified.txt').count(b': OK') == 2
    assert b'COMPLETE\nEXIT_CODE=0\n' in read_final('status.txt')
    assert b'borrowed_device_stream_and_older_library_compatibility ... ok' in read_final('rust-owners.log')
    log = read_final('rust-memcheck.log')
    assert b'borrowed_expansion_and_device_only_context ... ok' in log
    assert b'LEAK SUMMARY: 0 bytes leaked' in log and b'ERROR SUMMARY: 0 errors' in log
    assert b'835 passed, 10 skipped' in read_final('python-tests.log')
    assert b'75;80;90;120' in read_final('run.sh')
    assert b'Linking CUDA shared library libbeam_cnot_owned_domain.so' in read_final('cross-build.log')
    old_manifest = json.loads(read_final('legacy-manifest.json'))
    assert old_manifest['commit'] == legacy
    old_archive = subprocess.check_output(['git', 'archive', '--format=tar', legacy, '--',
                                           *old_manifest['files']], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(old_archive)) as baseline:
        old_files = {m.name: sha(baseline.extractfile(m).read()) for m in baseline if m.isfile()}
    assert old_files == old_manifest['files']
    for world in (1, 2):
        rows = [json.loads(read_final(f'receipts/gpu-{world}/rank-{rank}.json'))
                for rank in range(world)]
        assert all(row['world'] == world and len(row['cases']) == 5 for row in rows)
        assert all(row['cases'] == rows[0]['cases'] for row in rows)
        checkpoint = [json.loads(read_final(f'receipts/checkpoint-{world}/rank-{rank}.json'))
                      for rank in range(world)]
        assert all(row['world'] == world and len(row['results']) == 2 for row in checkpoint)
        for row in checkpoint:
            assert row['checkpoint_sha256'] == checkpoint[0]['checkpoint_sha256']
            assert row['results'] == checkpoint[0]['results']
        for rank in range(world):
            assert f'CNOT_RUST_EQUIVALENCE_OK rank={rank} world={world}'.encode() in read_final(f'gpu-{world}.log')
            assert f'CNOT_RUST_CHECKPOINT_OK rank={rank} world={world}'.encode() in read_final(f'checkpoint-{world}.log')

for name, instance in (('release-failed-host.json', 61001341), ('release.json', 61002149)):
    receipt = json.loads((here / name).read_text())
    assert receipt['instance'] == instance and receipt['provider_absence_verified'] is True
failed = json.loads((here / 'failed-host-status.json').read_text())
assert failed['instance'] == 61001341 and 'exit code: 2' in failed['message_excerpt']
print('verified borrowed CNOT expansion, source, multi-GPU and checkpoint regression, sanitizer and releases')
