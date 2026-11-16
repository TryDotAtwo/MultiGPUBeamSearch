"""Verify source-pinned Rust CNOT stage coordination and multi-rank evidence."""
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
base = '1fa662aa12b4c5676613c3c2e74ecba2de5593f1'


def archive(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return tarfile.open(path, 'r:gz')


with archive(here / 'cnot-native-checked-green-evidence.tar.gz',
             'eb1b2a57f0b3ea6d34c4635dc45671cdcb0372c8ca8886f0a102bd01108f2dd6') as tar:
    def read(name):
        return tar.extractfile('cnot-native-checked-green-results/' + name).read()

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
    assert b'CARGO_TARGET_DIR=/workspace/target-native-checked-green' in read('run.sh')
    assert b'assert hasattr(extension(), "run_cnot_native_checked")' in read('run.sh')
    for filename, pattern in (('cnot-cpu.log', rb'\b80 passed\b'),
                              ('cnot-cuda.log', rb'\b62 passed\b'),
                              ('python-tests.log', rb'\b820 passed, 10 skipped\b')):
        assert re.search(pattern, read(filename)), filename
    for rank in range(2):
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

with archive(here.parent / 'cnot-native-checked-red/cnot-native-checked-red-evidence.tar.gz',
             'f75e7c236a7d211df01b1aed88e90835a2edc203515c814104f7f0fd4df57b55') as tar:
    prefix = 'cnot-native-checked-red-results/'
    assert b'RED_CONFIRMED\nCOMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    log = tar.extractfile(prefix + 'red.log').read()
    assert b'3 failed' in log and b'legacy Python CNOT stage coordinator was called' in log
    assert b'matching Rust CNOT stage error coordinator is required' in log

with archive(here.parent / 'cnot-native-checked-smoke/cnot-native-checked-smoke-evidence.tar.gz',
             'ccbd47bfe18813ca9779f491a59e556550f96bebf87ff5bc6fbfbea22f961717') as tar:
    prefix = 'cnot-native-checked-smoke-results/'
    assert b'COMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'23 passed' in tar.extractfile(prefix + 'smoke.log').read()
    smoke_manifest = json.loads(tar.extractfile(prefix + 'source-manifest.json').read())
    assert smoke_manifest['overlays'] == manifest['overlays']
    assert tar.extractfile(prefix + 'source-before.txt').read() == tar.extractfile(prefix + 'source-after.txt').read()

release = json.loads((here / 'release.json').read_text())
assert release['instance'] == 60986684 and release['provider_absence_verified'] is True
print('verified source overlays, Rust CNOT stage coordination, NCCL and lease release')
