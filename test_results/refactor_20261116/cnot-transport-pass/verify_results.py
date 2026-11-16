"""Verify the source-pinned CNOT Stream 5 transport evidence."""
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
base = 'b68d37ce3e539cfc50a74167d7cc7e33054c228c'


def checked_archive(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return tarfile.open(path, 'r:gz')


with checked_archive(here / 'cnot-transport-pass-evidence.tar.gz',
                     '13669543b557fde0f18002836f87dc8431d1e07df148a7e3618b6f4670baa46a') as tar:
    def read(name):
        return tar.extractfile('cnot-transport-pass-results/' + name).read()

    manifest = json.loads(read('source-manifest.json'))
    assert manifest['base_commit'] == base
    assert len(manifest['overlays']) == 15
    for name, digest in manifest['overlays'].items():
        raw = (subprocess.check_output(['git', 'show', commit + ':' + name], cwd=root)
               if commit else (root / name).read_bytes())
        assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == digest, name
    if commit:
        changed = subprocess.check_output(['git', 'diff', '--name-only', base, commit],
                                          cwd=root, text=True).splitlines()
        for name in changed:
            assert (name in manifest['overlays'] or name == 'ARCHITECTURE_NEED.md'
                    or name == 'memory/CHANGELOG.md' or name.startswith('docs/')
                    or name.startswith('test_results/')), name
    assert read('source-before.txt') == read('source-after.txt')
    assert b'COMPLETE\nEXIT_CODE=0\n' in read('status.txt')
    for filename, pattern in (('cnot-cpu.log', rb'\b43 passed\b'),
                              ('cnot-cuda.log', rb'\b62 passed\b'),
                              ('python-tests.log', rb'\b783 passed, 10 skipped\b')):
        assert re.search(pattern, read(filename)), filename
    rust = read('rust-tests.log')
    assert b'exchanges_counts_before_rows_and_votes_before_each' in rust
    assert b'rejects_receive_overflow_before_payload_exchange' in rust
    for rank in range(2):
        assert f'CNOT_RUST_STREAM5_DEVICE_OK rank={rank} rounds=3'.encode() in read('stream5-nccl.log')
        assert f'CNOT_RUST_STREAM5_EXCHANGE_OK rank={rank} cases=3'.encode() in read('transport-nccl.log')
        assert f'CNOT_RADIX_NCCL_OK rank={rank} cases=3'.encode() in read('radix-nccl.log')
        assert f'CNOT_NATIVE_GOAL_NCCL_OK rank={rank} cases=2'.encode() in read('goal-native-nccl.log')
        assert f'CNOT_GOAL_COLLECTIVE_OK rank={rank} world=2'.encode() in read('goal-legacy-nccl.log')
        legacy = json.loads(read(f'receipts/goal-legacy/rank-{rank}.json'))
        assert legacy['rank'] == rank and legacy['world'] == 2
        assert len(legacy['faults']) == 2 and all(case['reused'] for case in legacy['faults'])
    for world in (1, 2):
        gpu = [json.loads(read(f'receipts/gpu-{world}/rank-{rank}.json')) for rank in range(world)]
        checkpoint = [json.loads(read(f'receipts/checkpoint-{world}/rank-{rank}.json')) for rank in range(world)]
        assert all(row['world'] == world and len(row['cases']) == 5 for row in gpu)
        assert all(row['cases'] == gpu[0]['cases'] for row in gpu)
        if world == 2:
            assert all(row['reused_after_failure'] and row['rust_transport_calls'] > 0
                       and row['fallback_legacy_calls'] > 0 for row in gpu)
        for row in checkpoint:
            assert row['world'] == world
            assert row['checkpoint_sha256'] == '92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4'
            qhat, proposal = row['results']
            assert qhat['result']['gates'] == [5, 7, 0] and qhat['replay']
            assert 'feature F02 rejected' in proposal['error']
        assert all(row['results'] == checkpoint[0]['results'] for row in checkpoint)

with checked_archive(here.parent / 'cnot-transport-red/artifacts/cnot-transport-red-evidence.tar.gz',
                     'cf065125ff90e37bb322992bc1161ae180bc76215edc33c62b4fd89ecdce4a1f') as tar:
    assert b'RED_CONFIRMED\nCOMPLETE\nEXIT_CODE=0\n' in tar.extractfile('cnot-transport-red-results/status.txt').read()
    assert b"object has no attribute 'exchange'" in tar.extractfile('cnot-transport-red-results/cnot-transport-red.log').read()
    assert b'unresolved import' in tar.extractfile('cnot-transport-red-results/cnot-exchange-red.log').read()

with checked_archive(here.parent / 'cnot-transport-green/initial-failure/cnot-transport-green-evidence.tar.gz',
                     'c9cb3d50cf8cb534ac28d0f61e8fedfd37f6e36aabca154380c4ac3d38e8b713') as tar:
    assert b"invalid data type 'bytes'" in tar.extractfile('cnot-transport-green-results/gpu-2.log').read()

with checked_archive(here.parent / 'cnot-transport-final/cnot-transport-final-evidence.tar.gz',
                     'a0687c98fc835ceb2f67d7cab01310a2c9ff12a77a1594f6f1a76673a6cd497c') as tar:
    assert b"'NoneType' object is not iterable" in tar.extractfile('cnot-transport-final-results/gpu-2.log').read()

with checked_archive(here.parent / 'cnot-transport-sanitizer/cnot-transport-sanitizer-evidence.tar.gz',
                     '0e3d5aac8d065e254849138c604b6b3aebe7f2e147c91fe7218cfcd323f5995d') as tar:
    prefix = 'cnot-transport-sanitizer-results/'
    assert b'EXIT_CODE=1\n' in tar.extractfile(prefix + 'status.txt').read()
    transport = tar.extractfile(prefix + 'transport.log').read()
    for rank in range(2):
        assert f'CNOT_RUST_STREAM5_EXCHANGE_OK rank={rank} cases=3'.encode() in transport
    logs = [tar.extractfile(m).read() for m in tar.getmembers() if '/memcheck-' in m.name]
    assert len(logs) == 2
    assert all(b'cudaErrorNoKernelImageForDevice' in log and b'ncclInitKernelsForDevice' in log for log in logs)

with checked_archive(here.parent / 'cnot-transport-sanitizer-filtered/cnot-transport-sanitizer-filtered-evidence.tar.gz',
                     'd8757cdb74a9c34a68d056d1865a5fe6137d4fbda8d8908c4349202a3d02fcc1') as tar:
    prefix = 'cnot-transport-sanitizer-filtered-results/'
    def read_filtered(name):
        return tar.extractfile(prefix + name).read()
    assert json.loads(read_filtered('source-manifest.json')) == manifest
    assert read_filtered('source-before.txt') == read_filtered('source-after.txt')
    assert b'--report-api-errors no' in read_filtered('run.sh')
    assert b'COMPLETE\nEXIT_CODE=0\n' in read_filtered('status.txt')
    for rank in range(2):
        assert f'CNOT_RUST_STREAM5_EXCHANGE_OK rank={rank} cases=3'.encode() in read_filtered('transport.log')
    logs = [tar.extractfile(m).read() for m in tar.getmembers() if '/memcheck-' in m.name]
    assert len(logs) == 2 and all(b'ERROR SUMMARY: 0 errors' in log for log in logs)

release = json.loads((here / 'release.json').read_text())
assert release['instance'] == 60969024 and release['provider_absence_verified'] is True
print('verified source overlays, CNOT transport, prior attempts, GPU evidence and lease release')
