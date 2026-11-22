"""Verify the source-pinned standalone CNOT borrowed-device evidence."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

here = Path(__file__).resolve().parent
root = here.parents[2]
commit = sys.argv[1] if len(sys.argv) > 1 else None
base = '4846dd9d222cfb09a0df4d1acd556ff81dda7ac2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(name):
    if commit:
        return subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
    return (root / name).read_bytes().replace(b'\r\n', b'\n')


def archive(name, expected):
    path = here / name
    assert sha(path.read_bytes()) == expected, name
    return tarfile.open(path, 'r:gz')


with archive('cnot-borrowed-device-evidence.tar.gz',
             '7c67567c795a19c56cffdae5a9851c2954c741d187c302d27996663f765ef62e') as tar:
    def read_first(name):
        return tar.extractfile('cnot-borrowed-device-results/' + name).read()
    assert b'MEMCHECK\nEXIT_CODE=97\n' in read_first('status.txt')
    log = read_first('memcheck.log').decode('utf-8')
    leaks = re.split(r'(?=^========= Leaked )', log, flags=re.M)[1:]
    assert len(leaks) == 6 and all('CUDACachingAllocator' in chunk for chunk in leaks)
    assert 'ERROR SUMMARY: 6 errors' in log

with archive('cnot-borrowed-device2-evidence.tar.gz',
             'da2e0cc82b4b5380316a93b3f7b72c0d40ac21c6c9bf13fd442b05f76464948d') as tar:
    def read_second(name):
        return tar.extractfile('cnot-borrowed-device2-results/' + name).read()
    assert b'RACECHECK\nEXIT_CODE=1\n' in read_second('status.txt')
    assert b'FAILED (failures=2)' in read_second('racecheck.log')
    assert b'RACECHECK SUMMARY: 0 hazards' in read_second('racecheck.log')
    assert b'LEAK SUMMARY: 0 bytes leaked' in read_second('memcheck.log')

with archive('diagnostics-evidence.tar.gz',
             '1c150845fced849e259713e03f29bcbfec9717fb89bf27af4be9fdec9711a974') as tar:
    def diagnostic(name):
        return tar.extractfile('diagnostics/cnot-borrowed-device-' + name + '.log').read()
    assert b'frame: test_previous_device_restored' in diagnostic('referrer')
    assert b'LEAK SUMMARY: 0 bytes leaked' in diagnostic('torch-control')
    assert b'ERROR SUMMARY: 0 errors' in diagnostic('memcheck-cleanup-tests')
    assert b'RACECHECK SUMMARY: 0 hazards' in diagnostic('racecheck-cleanup-tests')

with archive('cnot-borrowed-device3-evidence.tar.gz',
             '3c95e8fe05612e8dfc05e90131a606a2b0120433c6f0d97429996bf000988881') as tar:
    def read(name):
        return tar.extractfile('cnot-borrowed-device3-results/' + name).read()
    manifest = json.loads(read('source-manifest.json'))
    assert manifest['base_commit'] == base and len(manifest['overlays']) == 7
    for name, digest in manifest['overlays'].items():
        # The plan status and only these two interface comments changed after
        # execution. Reconstruct the tested source before comparing its hash.
        if name == 'docs/plans/2026-11-17-cnot-borrowed-device.md':
            continue
        data = source(name)
        if name == 'native/cnot_owned_domain.h':
            newer = (b' * guarantees physical allocation bounds, accessibility from the domain CUDA\n'
                     b' * context, and lifetime through the call. */')
            older = b' * guarantees physical allocation bounds and lifetime through the call. */'
            assert data.count(newer) == 1
            data = data.replace(newer, older)
        if name == 'rust/beam-native-sys/src/cnot_domain.rs':
            newer = (b"    /// allocations on this domain's device, accessible from its CUDA context.\n"
                     b'    /// Their used regions must not overlap. `stream` must be a valid stream in\n'
                     b'    /// the same CUDA context (null is the default stream), with every input\n'
                     b'    /// producer ordered before it.')
            older = (b"    /// allocations on this domain's device. Their used regions must not\n"
                     b'    /// overlap. `stream` must be a valid stream in the same CUDA context (null\n'
                     b'    /// is the default stream), with every input producer ordered before it.')
            assert data.count(newer) == 1
            data = data.replace(newer, older)
        assert sha(data) == digest, name
    if commit:
        changed = subprocess.check_output(['git', 'diff', '--name-only', base, commit],
                                          cwd=root, text=True).splitlines()
        for name in changed:
            assert (name in manifest['overlays'] or name == 'ARCHITECTURE_NEED.md'
                    or name == 'memory/CHANGELOG.md' or name.startswith('docs/')
                    or name.startswith('test_results/')), name
    assert b'RED_CONFIRMED\n' in read('status.txt')
    assert b'RUST_DEVICE_MEMCHECK\nEXIT_CODE=1\n' in read('status.txt')
    assert b'-maxdepth 1 -type f -executable' in read('run.sh')
    assert b'undefined symbol: beam_cnot_domain_device_abi_version' in read('red.log')
    assert read('gpus.csv').count(b'NVIDIA GeForce RTX 3060') == 2
    assert b'Ran 4 tests' in read('borrowed.log') and b'\nOK\n' in read('borrowed.log')
    assert b'Ran 5 tests' in read('domain.log') and b'\nOK\n' in read('domain.log')
    for name in ('memcheck.log', 'native-memcheck.log'):
        assert b'ERROR SUMMARY: 0 errors' in read(name)
        assert b'LEAK SUMMARY: 0 bytes leaked' in read(name)
    assert b'RACECHECK SUMMARY: 0 hazards' in read('racecheck.log')
    assert b'test result: FAILED' not in read('rust-workspace.log')
    assert b'owned_domain_survives_library_handle_and_reuses_resources ... ok' in read('rust-domain.log')
    assert b'borrowed_device_stream_and_older_library_compatibility ... ok' in read('rust-borrowed.log')
    before = read('source-before.txt')
    libraries = read('libraries-sha256.txt')

with archive('cnot-borrowed-device4-evidence.tar.gz',
             '3ca7b1a8bce8fc7fb7c9b9efcb0c768d4c1c462e1252d4d1ef49a81e747abbeb') as tar:
    def read_final(name):
        return tar.extractfile('cnot-borrowed-device4-results/' + name).read()
    assert json.loads(read_final('source-manifest.json')) == manifest
    assert read_final('source-before.txt') == before == read_final('source-after.txt')
    assert read_final('libraries-sha256.txt') == libraries
    assert b'COMPLETE\nEXIT_CODE=0\n' in read_final('status.txt')
    assert b'ERROR SUMMARY: 0 errors' in read_final('rust-memcheck.log')
    assert b'LEAK SUMMARY: 0 bytes leaked' in read_final('rust-memcheck.log')
    assert b'borrowed_device_stream_and_older_library_compatibility ... ok' in read_final('rust-memcheck.log')
    assert b'833 passed, 12 skipped' in read_final('python-tests.log')
    assert b'75;80;90;120' in read_final('run.sh')
    assert b'Linking CUDA shared library libbeam_cnot_owned_domain.so' in read_final('cross-build.log')
    for world in (1, 2):
        rows = [json.loads(read_final(f'receipts/gpu-{world}/rank-{rank}.json'))
                for rank in range(world)]
        assert all(row['world'] == world and len(row['cases']) == 5 for row in rows)
        assert all(row['cases'] == rows[0]['cases'] for row in rows)
        for rank in range(world):
            marker = f'CNOT_RUST_EQUIVALENCE_OK rank={rank} world={world}'.encode()
            assert marker in read_final(f'gpu-{world}.log')

release = json.loads((here / 'release.json').read_text())
assert release['instance'] == 60996988 and release['provider_absence_verified'] is True
print('verified borrowed CNOT device owner, source, two-GPU regression, sanitizer and lease release')
