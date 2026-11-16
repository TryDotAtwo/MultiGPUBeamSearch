"""Verify source-pinned CNOT Rust selection ownership evidence."""
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
base = 'f79469d3cca4374a5cd9b16bc2f739ec82bea620'


def checked_archive(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return tarfile.open(path, 'r:gz')


with checked_archive(here / 'cnot-native-selection-green2-evidence.tar.gz',
                     '9632ca233ac491485d086497549affeec04e7232c40f900d70a54acb286a82ff') as tar:
    def read(name):
        return tar.extractfile('cnot-native-selection-green2-results/' + name).read()

    manifest = json.loads(read('source-manifest.json'))
    assert manifest['base_commit'] == base and len(manifest['overlays']) == 8
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
    assert b'assert hasattr(extension(), "run_cnot_native_selection")' in read('run.sh')
    assert b'CARGO_TARGET_DIR=/workspace/target-native-selection-green2' in read('run.sh')
    for filename, pattern in (('cnot-cpu.log', rb'\b53 passed\b'),
                              ('cnot-cuda.log', rb'\b62 passed\b'),
                              ('python-tests.log', rb'\b793 passed, 10 skipped\b')):
        assert re.search(pattern, read(filename)), filename
    rust = read('rust-tests.log')
    assert b'exchanges_counts_before_rows_and_votes_before_each' in rust
    assert b'score_passes_then_word_tie_break_and_exact_boundary' in rust
    for rank in range(2):
        assert f'CNOT_RUST_STREAM5_DEVICE_OK rank={rank} rounds=3'.encode() in read('stream5-nccl.log')
        assert f'CNOT_RUST_STREAM5_EXCHANGE_OK rank={rank} cases=3'.encode() in read('transport-nccl.log')
        assert f'CNOT_RADIX_NCCL_OK rank={rank} cases=3'.encode() in read('radix-nccl.log')
        assert f'CNOT_RADIX_MIXED_OPS_NCCL_OK rank={rank}'.encode() in read('radix-nccl.log')
        assert f'CNOT_NATIVE_GOAL_NCCL_OK rank={rank} cases=2'.encode() in read('goal-native-nccl.log')
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

with checked_archive(here.parent / 'cnot-native-selection-red/cnot-native-selection-red-evidence.tar.gz',
                     'ec95aff5cb49cd7d4a3c773f4e02ea244d0f4aeab77657a67f6e48095769da3b') as tar:
    prefix = 'cnot-native-selection-red-results/'
    assert b'RED_CONFIRMED\nCOMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    log = tar.extractfile(prefix + 'red.log').read()
    assert b'3 failed' in log
    assert b'legacy Python CNOT selection owner was instantiated' in log
    assert b'matching Rust-owned CNOT selection stages are required' in log

with checked_archive(here.parent / 'cnot-native-selection-smoke/cnot-native-selection-smoke-evidence.tar.gz',
                     '202a8c426df0098d707d24ebe0b07ec13c86a53996fc6a6b70b8266b827fd1fe') as tar:
    prefix = 'cnot-native-selection-smoke-results/'
    assert b'COMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'19 passed' in tar.extractfile(prefix + 'cnot-cpu.log').read()
    smoke_manifest = json.loads(tar.extractfile(prefix + 'source-manifest.json').read())
    for name, digest in manifest['overlays'].items():
        if name.startswith('rust/'):
            assert smoke_manifest['overlays'][name] == digest

with checked_archive(here.parent / 'cnot-native-selection-green/cnot-native-selection-green-evidence.tar.gz',
                     'febbd981b89f19767ef8f7a5fcde3815d6021f27e8746698b90ec66e68db565c') as tar:
    prefix = 'cnot-native-selection-green-results/'
    assert b'COMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'790 passed, 10 skipped' in tar.extractfile(prefix + 'python-tests.log').read()
    preliminary_manifest = json.loads(tar.extractfile(prefix + 'source-manifest.json').read())
    for name, digest in manifest['overlays'].items():
        if name.startswith('rust/'):
            assert preliminary_manifest['overlays'][name] == digest

with checked_archive(here.parent / 'cnot-native-selection-smoke2/cnot-native-selection-smoke2-evidence.tar.gz',
                     '20c77b56a315d7f190f08cbccac7e951df53f6951cc5768b2cc5e342c1e7bb32') as tar:
    prefix = 'cnot-native-selection-smoke2-results/'
    assert b'COMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'22 passed' in tar.extractfile(prefix + 'cnot-cpu.log').read()
    smoke_manifest = json.loads(tar.extractfile(prefix + 'source-manifest.json').read())
    for name, digest in manifest['overlays'].items():
        if name.startswith('rust/'):
            assert smoke_manifest['overlays'][name] == digest
    assert tar.extractfile(prefix + 'source-before.txt').read() == tar.extractfile(prefix + 'source-after.txt').read()

with checked_archive(here.parent / 'cnot-native-selection-memcheck2/cnot-native-selection-memcheck2-evidence.tar.gz',
                     '4e01e7b6ccf432b55d7a08a2d6695b8c646608eeb604ef13f41d5716c53743dc') as tar:
    prefix = 'cnot-native-selection-memcheck2-results/'
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
assert release['instance'] == 60980053 and release['provider_absence_verified'] is True
print('verified source overlays, Rust CNOT selection, NCCL, memcheck and lease release')
