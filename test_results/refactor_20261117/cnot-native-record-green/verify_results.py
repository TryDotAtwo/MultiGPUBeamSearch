"""Verify source-pinned CNOT record-exchange migration and multi-GPU evidence."""
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
base = '4a16113cd376ed99fcdda2afa341d3270e6d78df'


def archive(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return tarfile.open(path, 'r:gz')


with archive(here / 'cnot-native-record-green-evidence.tar.gz', 'a87983773e11cf73a8227c7502d80a8e633a15cd06521991bf120f156459b3cb') as tar:
    def read(name):
        return tar.extractfile('cnot-native-record-green-results/' + name).read()

    manifest = json.loads(read('source-manifest.json'))
    assert manifest['base_commit'] == base and len(manifest['overlays']) == 7
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
    assert b'CARGO_TARGET_DIR=/workspace/target-native-record-green' in read('run.sh')
    for filename, pattern in (('cnot-cpu.log', rb'\b95 passed\b'),
                              ('cnot-cuda.log', rb'\b62 passed\b'),
                              ('python-tests.log', rb'\b835 passed, 10 skipped\b')):
        assert re.search(pattern, read(filename)), filename
    for rank in range(2):
        assert f'CNOT_NATIVE_RECORD_EXCHANGE_OK rank={rank} scenarios=3 cases=4'.encode() in read('record-native-nccl.log')
        assert f'CNOT_RUST_CHECKED_NCCL_OK rank={rank} cases=3'.encode() in read('checked-native-nccl.log')
        assert f'CNOT_RUST_STREAM5_DEVICE_OK rank={rank} rounds=3'.encode() in read('stream5-nccl.log')
        assert f'CNOT_RUST_STREAM5_EXCHANGE_OK rank={rank} cases=3'.encode() in read('transport-nccl.log')
        assert f'CNOT_RADIX_NCCL_OK rank={rank} cases=3'.encode() in read('radix-nccl.log')
        assert f'CNOT_NATIVE_GOAL_NCCL_OK rank={rank} cases=2 mixed_operators=1'.encode() in read('goal-native-nccl.log')
        assert f'CNOT_GOAL_COLLECTIVE_OK rank={rank} world=2'.encode() in read('goal-legacy-nccl.log')
    for world in (1, 2):
        gpu = [json.loads(read(f'receipts/gpu-{world}/rank-{rank}.json')) for rank in range(world)]
        checkpoint = [json.loads(read(f'receipts/checkpoint-{world}/rank-{rank}.json')) for rank in range(world)]
        assert all(row['world'] == world and len(row['cases']) == 5 for row in gpu)
        assert all(row['cases'] == gpu[0]['cases'] for row in gpu)
        assert all(row['batch_fallbacks'] == ['expand', 'distinct', 'both', 'mixed'] for row in gpu)
        if world == 2:
            assert all(row['reused_after_failure'] and row['rust_transport_calls'] > 0
                       and row['fallback_legacy_calls'] > 0 and row['fallback_stage_calls'] > 0
                       and row['fallback_selection_calls'] > 0 for row in gpu)
        for row in checkpoint:
            assert row['world'] == world
            assert row['checkpoint_sha256'] == '92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4'
            qhat, proposal = row['results']
            assert qhat['result']['gates'] == [5, 7, 0] and qhat['replay']
            assert 'feature F02 rejected' in proposal['error']
        assert all(row['results'] == checkpoint[0]['results'] for row in checkpoint)

with archive(here.parent / 'cnot-native-record-red/cnot-native-record-red-evidence.tar.gz',
             '0ab959bdc9ab236dc9da2da1e90f44c26267b55cf4138b3a9ae70bca42ec610f') as tar:
    prefix = 'cnot-native-record-red-results/'
    assert b'RED_CONFIRMED\nCOMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    log = tar.extractfile(prefix + 'red.log').read()
    assert b'3 failed' in log and b'legacy Python CNOT record exchange was called' in log
    assert b'matching Rust CNOT record exchange is required' in log

with archive(here.parent / 'cnot-native-record-smoke/cnot-native-record-smoke-evidence.tar.gz',
             '672c19ce8a8a8f05c1add9f44593cb09d82854b4e6b7c1b4f4107187db333fec') as tar:
    prefix = 'cnot-native-record-smoke-results/'
    assert b'COMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'32 passed' in tar.extractfile(prefix + 'smoke.log').read()
    smoke_manifest = json.loads(tar.extractfile(prefix + 'source-manifest.json').read())
    assert smoke_manifest['overlays'] == manifest['overlays']
    assert tar.extractfile(prefix + 'source-before.txt').read() == tar.extractfile(prefix + 'source-after.txt').read()

release = json.loads((here / 'release.json').read_text())
assert release['instance'] == 60990077 and release['provider_absence_verified'] is True
unreachable = json.loads((here.parent / 'cnot-native-record-red/unreachable-release.json').read_text())
assert unreachable['instance'] == 60989817 and unreachable['provider_absence_verified'] is True
print('verified pinned CNOT record exchange, NCCL, checkpoint and lease release')
