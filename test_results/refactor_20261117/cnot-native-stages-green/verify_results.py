"""Verify the source-pinned CNOT Rust stage-ownership migration evidence."""
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
base = '46d4780a863c30c77a003850d9bf60194e8a9dd1'


def checked_archive(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return tarfile.open(path, 'r:gz')


with checked_archive(here / 'cnot-native-stages-green-evidence.tar.gz',
                     'c23496d1f4b9a138d43c60a4e42e32513454fa321ac68f9a6de9c9f7fcfde0f4') as tar:
    def read(name):
        return tar.extractfile('cnot-native-stages-green-results/' + name).read()

    manifest = json.loads(read('source-manifest.json'))
    assert manifest['base_commit'] == base and len(manifest['overlays']) == 6
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
    assert b'CNOT_NATIVE_INTERFACE\n' in read('status.txt')
    assert b'COMPLETE\nEXIT_CODE=0\n' in read('status.txt')
    assert b'assert hasattr(extension(), "run_cnot_native")' in read('run.sh')
    assert b'CARGO_TARGET_DIR=/workspace/target-native-stages-green' in read('run.sh')
    for filename, pattern in (('cnot-cpu.log', rb'\b45 passed\b'),
                              ('cnot-cuda.log', rb'\b62 passed\b'),
                              ('python-tests.log', rb'\b785 passed, 10 skipped\b')):
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
                       and row['fallback_legacy_calls'] > 0 and row['fallback_stage_calls'] > 0
                       for row in gpu)
        for row in checkpoint:
            assert row['world'] == world
            assert row['checkpoint_sha256'] == '92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4'
            qhat, proposal = row['results']
            assert qhat['result']['gates'] == [5, 7, 0] and qhat['replay']
            assert 'feature F02 rejected' in proposal['error']
        assert all(row['results'] == checkpoint[0]['results'] for row in checkpoint)

with checked_archive(here.parent / 'cnot-native-stages-red/cnot-native-stages-red-evidence.tar.gz',
                     'a6b4900070eb103b3e01e1fc832a9a9ff2d4dec53f7e92abf4fc8e014ed85b68') as tar:
    prefix = 'cnot-native-stages-red-results/'
    assert b'RED_CONFIRMED\nCOMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    log = tar.extractfile(prefix + 'red.log').read()
    assert b'2 failed' in log and b'legacy Python CNOT stage owner was instantiated' in log

with checked_archive(here.parent / 'cnot-native-stages-smoke/cnot-native-stages-smoke-evidence.tar.gz',
                     '97f45a7010ac499d8cdd2b7b30d126419b22874a04d3cb48b8c80789c000d652') as tar:
    prefix = 'cnot-native-stages-smoke-results/'
    assert b'EXIT_CODE=1\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'legacy Python CNOT stage owner was instantiated' in tar.extractfile(prefix + 'cnot-cpu.log').read()

with checked_archive(here.parent / 'cnot-native-stages-smoke2/cnot-native-stages-smoke2-evidence.tar.gz',
                     '9397010aada861373a8a0ef5c497dae4276c6d292cd4a3a0a2307be3b50b2858') as tar:
    prefix = 'cnot-native-stages-smoke2-results/'
    assert b'COMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'11 passed' in tar.extractfile(prefix + 'cnot-cpu.log').read()

with checked_archive(here.parent / 'cnot-native-stages-memcheck/cnot-native-stages-memcheck-evidence.tar.gz',
                     '063170681726dae827c068abcf2bbb0a1af93db7e8b4314016dc9018f23436fb') as tar:
    prefix = 'cnot-native-stages-memcheck-results/'
    def read_mem(name):
        return tar.extractfile(prefix + name).read()
    assert json.loads(read_mem('source-manifest.json')) == manifest
    assert read_mem('source-before.txt') == read_mem('source-after.txt')
    assert b'--report-api-errors no' in read_mem('run.sh')
    assert b'COMPLETE\nEXIT_CODE=0\n' in read_mem('status.txt')
    for rank in range(2):
        assert f'CNOT_RUST_EQUIVALENCE_OK rank={rank} world=2'.encode() in read_mem('search.log')
    logs = [tar.extractfile(m).read() for m in tar.getmembers() if '/memcheck-' in m.name]
    assert len(logs) == 2 and all(b'ERROR SUMMARY: 0 errors' in log for log in logs)

release = json.loads((here / 'release.json').read_text())
assert release['instance'] == 60975486 and release['provider_absence_verified'] is True
print('verified source overlays, Rust CNOT stages, search, NCCL, memcheck and lease release')
