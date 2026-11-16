"""Verify source-pinned Rust CNOT goal and physical multi-rank evidence."""
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
base = '267b84d78747e1c30dc0694f8f870ec1d0cf8111'


def archive(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return tarfile.open(path, 'r:gz')


with archive(here / 'cnot-native-goal-green-evidence.tar.gz',
             '9227d501be010c3a42b8b674d3c04fe4cb85214b663ea64f247a670d5ae58369') as tar:
    def read(name):
        return tar.extractfile('cnot-native-goal-green-results/' + name).read()

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
    assert b'CARGO_TARGET_DIR=/workspace/target-native-goal-green' in read('run.sh')
    assert b'assert hasattr(extension(), "run_cnot_native_goal")' in read('run.sh')
    for filename, pattern in (('cnot-cpu.log', rb'\b72 passed\b'),
                              ('cnot-cuda.log', rb'\b62 passed\b'),
                              ('python-tests.log', rb'\b812 passed, 10 skipped\b')):
        assert re.search(pattern, read(filename)), filename
    for rank in range(2):
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

with archive(here.parent / 'cnot-native-goal-red/cnot-native-goal-red-evidence.tar.gz',
             '93fff15745bded35dd39e5d0802a38afefc5d8355f3b2d9892b59399839a0d49') as tar:
    prefix = 'cnot-native-goal-red-results/'
    assert b'RED_CONFIRMED\nCOMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    log = tar.extractfile(prefix + 'red.log').read()
    assert b'3 failed' in log and b'legacy Python goal coordinator was called' in log
    assert b'matching Rust CNOT goal coordinator is required' in log

with archive(here.parent / 'cnot-native-goal-smoke/cnot-native-goal-smoke-evidence.tar.gz',
             '2e3152008807e6557acdc2c2c806d4502381bdc74235ce914e6ff2664ac33a18') as tar:
    prefix = 'cnot-native-goal-smoke-results/'
    assert b'NATIVE_GOAL_INTERFACE\nEXIT_CODE=1\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'target-native-selection-red' in tar.extractfile(prefix + 'run.sh').read()

with archive(here.parent / 'cnot-native-goal-smoke2/cnot-native-goal-smoke2-evidence.tar.gz',
             '7aa09f95686ea1ab5706614bb019bddc1f830e4e44d5e04672d3fb9e2352c7c9') as tar:
    prefix = 'cnot-native-goal-smoke2-results/'
    assert b'SMOKE_TEST\nEXIT_CODE=1\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b"'list' object cannot be converted to 'PyTuple'" in tar.extractfile(prefix + 'smoke.log').read()

with archive(here.parent / 'cnot-native-goal-smoke3/cnot-native-goal-smoke3-evidence.tar.gz',
             '3783895ebb65c57665f3288ce3e5754fbaa102398bd59deff6f283eb5c29bbe1') as tar:
    prefix = 'cnot-native-goal-smoke3-results/'
    assert b'COMPLETE\nEXIT_CODE=0\n' in tar.extractfile(prefix + 'status.txt').read()
    assert b'29 passed' in tar.extractfile(prefix + 'smoke.log').read()
    smoke_manifest = json.loads(tar.extractfile(prefix + 'source-manifest.json').read())
    for name, digest in manifest['overlays'].items():
        if name.startswith('rust/'):
            assert smoke_manifest['overlays'][name] == digest

with archive(here.parent / 'cnot-native-goal-memcheck/cnot-native-goal-memcheck-evidence.tar.gz',
             '5d218488f9bee374421be27a88ef1ac4b685f526573235741a92dcf27987a45a') as tar:
    prefix = 'cnot-native-goal-memcheck-results/'
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
assert release['instance'] == 60984269 and release['provider_absence_verified'] is True
print('verified source overlays, Rust CNOT goal, NCCL, memcheck and lease release')
