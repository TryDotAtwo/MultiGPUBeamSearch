"""Verify source-pinned standalone CNOT CUDA domain and regression evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

here = Path(__file__).resolve().parent
root = here.parents[2]
commit = sys.argv[1] if len(sys.argv) > 1 else None
base = 'ac9a571388a0440d406fc6a05cefdd48821b2531'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source(name):
    if commit:
        return subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
    return (root / name).read_bytes()


def normalized(raw):
    return raw.replace(b'\r\n', b'\n')


def open_archive(path, digest):
    assert sha(path.read_bytes()) == digest, path
    return tarfile.open(path, 'r:gz')


with open_archive(here / 'cnot-owned-domain5-evidence.tar.gz',
                  '8ffc4825d5dcb69d2054a7f4e853c3980be51396ccbbe2659d7b8fbb6fc12a6d') as tar:
    def read(name):
        return tar.extractfile('cnot-owned-domain5-results/' + name).read()

    manifest = json.loads(read('source-manifest.json'))
    assert manifest['base_commit'] == base
    assert len(manifest['overlays']) == 8
    cmake = gzip.decompress((here / 'tested-cmake.bin').read_bytes())
    assert sha(cmake) == manifest['overlays']['CMakeLists.txt']
    assert normalized(cmake) == normalized(source('CMakeLists.txt'))
    # The plan text was updated after the run. The final Python test includes
    # one additional overflow assertion and is checked against the extra run.
    for name, digest in manifest['overlays'].items():
        if name in ('CMakeLists.txt', 'docs/plans/2026-11-17-cnot-owned-domain.md',
                    'tests/cnot_owned_domain_test.py'):
            continue
        assert sha(source(name)) == digest, name
    if commit:
        changed = subprocess.check_output(['git', 'diff', '--name-only', base, commit],
                                          cwd=root, text=True).splitlines()
        for name in changed:
            assert (name in manifest['overlays'] or name == 'ARCHITECTURE_NEED.md'
                    or name == 'memory/CHANGELOG.md' or name.startswith('docs/')
                    or name.startswith('test_results/')), name
    assert read('source-before.txt') == read('source-after.txt')
    assert b'RED_CONFIRMED\n' in read('status.txt')
    assert b'COMPLETE\nEXIT_CODE=0\n' in read('status.txt')
    assert b'undefined symbol: beam_cnot_domain_abi_version' in read('red.log')
    assert b'55 passed' in read('cnot-regression.log')
    assert b'833 passed, 12 skipped' in read('python-tests.log')
    assert b'owned_domain_survives_library_handle_and_reuses_resources ... ok' in read('rust-gpu.log')
    assert b'test result: FAILED' not in read('rust-workspace.log')
    assert b'75;80;90;120' in read('run.sh')
    assert b'Linking CUDA shared library libbeam_cnot_owned_domain.so' in read('cross-build.log')
    assert read('gpus.csv').count(b'NVIDIA GeForce RTX 3060') == 2
    for tool, marker in (('memcheck.log', b'ERROR SUMMARY: 0 errors'),
                         ('racecheck.log', b'RACECHECK SUMMARY: 0 hazards')):
        assert marker in read(tool)
    for world in (1, 2):
        rows = [json.loads(read(f'receipts/gpu-{world}/rank-{rank}.json')) for rank in range(world)]
        assert all(row['world'] == world and len(row['cases']) == 5 for row in rows)
        assert all(row['cases'] == rows[0]['cases'] for row in rows)
        for rank in range(world):
            assert f'CNOT_RUST_EQUIVALENCE_OK rank={rank} world={world}'.encode() in read(f'gpu-{world}.log')
    library_hash = read('library-sha256.txt').split()[0]

extra_path = here.parent / 'cnot-owned-domain6-extra/cnot-owned-domain6-extra-evidence.tar.gz'
with open_archive(extra_path, 'd21656ea740676d5f02e6f5f9e8dbdf7070f2936a4bd92a23baead8461dfffea') as tar:
    def read_extra(name):
        return tar.extractfile('cnot-owned-domain6-extra-results/' + name).read()

    assert b'COMPLETE\nEXIT_CODE=0\n' in read_extra('status.txt')
    assert normalized(source('tests/cnot_owned_domain_test.py')) == normalized(read_extra('test-source.py'))
    assert library_hash == read_extra('library-sha256.txt').split()[0]
    assert b'Ran 5 tests' in read_extra('gpu.log') and b'\nOK\n' in read_extra('gpu.log')
    assert b'ERROR SUMMARY: 0 errors' in read_extra('memcheck.log')
    assert b'LEAK SUMMARY: 0 bytes leaked' in read_extra('memcheck.log')
    assert b'RACECHECK SUMMARY: 0 hazards' in read_extra('racecheck.log')
    assert b'OK (skipped=1)' in read_extra('optional-skip.log')

release = json.loads((here / 'release.json').read_text())
assert release['instance'] == 60993089 and release['provider_absence_verified'] is True
print('verified standalone CNOT domain, source, two-GPU regression, sanitizer and lease release')
