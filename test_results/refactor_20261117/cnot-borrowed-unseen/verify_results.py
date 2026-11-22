"""Verify source, CUDA sanitizer and release evidence for borrowed CNOT selection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

here = Path(__file__).resolve().parent
root = here.parents[2]
commit = sys.argv[1] if len(sys.argv) > 1 else None
base = '0aab5c5111cf0f9f8c7ab823ee8570d5f9dbc8eb'


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


with archive('cnot-borrowed-unseen-evidence.tar.gz',
             'c7a987fe6417b59776163168b9c94511e7cd33517741f49edfc9d5327cfed852') as tar:
    def read(name):
        return tar.extractfile('cnot-borrowed-unseen-results/' + name).read()
    original = json.loads(read('source-manifest.json'))
    assert original['base_commit'] == base and len(original['overlays']) == 6
    assert b'RED_CONFIRMED\n' in read('status.txt')
    assert b'MEMCHECK_UNSEEN\nEXIT_CODE=97\n' in read('status.txt')
    assert b'undefined symbol: beam_cnot_domain_unseen_scratch_bytes' in read('red.log')
    for name, count in (('domain.log', 5), ('owners.log', 4), ('expansion.log', 6), ('unseen.log', 3)):
        log = read(name)
        assert f'Ran {count} tests'.encode() in log and b'\nOK\n' in log, name
    assert read('gpus.csv').count(b'NVIDIA GeForce RTX 3060') == 2
    failure = read('memcheck.log')
    assert b'Ran 3 tests' in failure and b'\nOK\n' in failure
    assert b'LEAK SUMMARY: 8 bytes leaked in 2 allocations' in failure
    assert b'_local_scalar_dense_cuda' in failure and b'cudaHostAlloc' in failure
    assert b'Invalid __global__' not in failure
    libraries = read('libraries-sha256.txt')

with archive('cnot-borrowed-unseen-continue-evidence.tar.gz',
             'a095d04f7c66bf8e7d5aeff263dae9f7375ad4053decf9cec04efd91d5831b22') as tar:
    def read_final(name):
        return tar.extractfile('cnot-borrowed-unseen-continue-results/' + name).read()
    final = json.loads(read_final('source-manifest.json'))
    assert json.loads(read_final('previous-manifest.json')) == original
    assert final['baseline'] == original['baseline']
    assert final['base_commit'] == base
    changed_test = 'tests/cnot_borrowed_unseen_test.py'
    for name, digest in original['overlays'].items():
        if name == changed_test:
            assert final['overlays'][name] != digest
        else:
            assert final['overlays'][name] == digest
    for name, digest in final['overlays'].items():
        if name != 'docs/plans/2026-11-17-cnot-borrowed-unseen.md':
            assert sha(source(name)) == digest, name
    assert read_final('source-before.txt') == read_final('source-after.txt')
    assert read_final('libraries-verified.txt').count(b': OK') == 2
    assert b'COMPLETE\nEXIT_CODE=0\n' in read_final('status.txt')
    assert b'Ran 3 tests' in read_final('unseen.log') and b'\nOK\n' in read_final('unseen.log')
    for name in ('memcheck.log', 'rust-memcheck.log'):
        log = read_final(name)
        assert b'LEAK SUMMARY: 0 bytes leaked' in log and b'ERROR SUMMARY: 0 errors' in log, name
    assert b'RACECHECK SUMMARY: 0 hazards' in read_final('racecheck.log')
    assert b'borrowed_unseen_and_older_library_compatibility ... ok' in read_final('rust-unseen.log')
    assert b'test result: FAILED' not in read_final('rust-workspace.log')
    assert b'75;80;90;120' in read_final('continue.sh')
    assert b'Linking CUDA shared library libbeam_cnot_owned_domain.so' in read_final('cross-build.log')

if commit:
    changed = subprocess.check_output(['git', 'diff', '--name-only', base, commit],
                                      cwd=root, text=True).splitlines()
    for name in changed:
        assert (name in final['overlays'] or name == 'ARCHITECTURE_NEED.md'
                or name == 'memory/CHANGELOG.md' or name.startswith('docs/')
                or name.startswith('test_results/')), name
for name, instance in (('release-ssh-failure.json', 61005252), ('release.json', 61005582)):
    receipt = json.loads((here / name).read_text())
    assert receipt['instance'] == instance and receipt['provider_absence_verified'] is True
print('verified standalone CNOT unseen source, tests, sanitizer and Vast releases')
