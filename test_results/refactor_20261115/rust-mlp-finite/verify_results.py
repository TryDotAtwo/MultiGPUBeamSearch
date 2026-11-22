"""Verify the exploratory pre-flag source archive and its CPU/GPU results."""
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '678d3d69cac42398f849b7722b776b2b8562ef43'
BASE_SHA256 = '0531a2e1c5154d3cac914a9c243bd5a66070555ea1d77394d6b3dd9e49244519'
OVERLAY_SHA256 = '2dfc66116b7c0bcda698f31f76d8b75394b1a73c3651349575878a4cdb8a1e6d'
ARCHIVE_SHA256 = '1d5096342fd01462a3c8e2f08ad1a72fcd0a3e475a7104d75ebdb3f503557fa1'
sha = lambda data: hashlib.sha256(data).hexdigest()


def replay(case, path):
    center = [i // 16 for i in range(96)]
    count = {'first': 3, 'repeat': 4, 'unsolved': 5, 'model-change': 3, 'beam-change': 3}[case]
    state = center.copy()
    for move in range(count):
        state[move], state[move + 24] = state[move + 24], state[move]
    for move in path:
        assert type(move) is int and 0 <= move < 24
        state[move], state[move + 24] = state[move + 24], state[move]
    assert state == center, (case, path)


data = (HERE / 'evidence.tar.gz').read_bytes()
assert sha(data) == ARCHIVE_SHA256
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    entries = archive.getmembers()
    assert len(entries) == len(set(m.name for m in entries))
    assert all((m.isdir() or m.isfile()) and m.name.startswith('finite-results') and
               '..' not in Path(m.name).parts for m in entries)
    get = lambda name: archive.extractfile('finite-results/' + name).read()
    status = get('status.txt').decode().splitlines()
    assert status == ['SOURCE_OK', 'RUST_TESTS_START', 'RUST_TESTS_OK', 'RUST_BUILD_OK',
        'EXHAUSTIVE_OK', 'PYTHON_OK', 'SNAPSHOT_GPU_START', 'SNAPSHOT_GPU_OK',
        'PAIR_START', 'PAIR_OK', 'COMPLETE', 'EXIT_CODE=0'], status
    assert get('source-before.txt') == get('source-after.txt')
    assert b'test result: ok.' in get('rust-tests.log')
    assert re.search(rb'\d+ passed, \d+ skipped', get('python-tests.log'))
    assert get('gpus.csv').count(b'RTX 3060') == 2
    exhaustive = json.loads(get('exhaustive.json'))
    assert exhaustive['counts'] == {'fp16': {'tested': 65536, 'finite': 63488, 'nonfinite': 2048},
                                     'bf16': {'tested': 65536, 'finite': 65280, 'nonfinite': 256}}
    manifest = json.loads(get('source-manifest.json'))
    assert manifest['base_commit'] == BASE and len(manifest['baseline']) == 534
    source_base = get('source/finite-base.tar.gz')
    source_overlay = get('source/finite-overlay.tar.gz')
    assert sha(source_base) == BASE_SHA256 and sha(source_overlay) == OVERLAY_SHA256
    with tarfile.open(fileobj=io.BytesIO(source_base), mode='r:gz') as tar:
        base_files = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
    assert {name: sha(raw) for name, raw in base_files.items()} == manifest['baseline']
    requests = ''.join(BASE + ':' + name + '\n' for name in base_files).encode()
    proc = subprocess.run(['git', 'cat-file', '--batch'], input=requests,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT, check=True)
    output = io.BytesIO(proc.stdout)
    for name, raw in base_files.items():
        header = output.readline().split()
        assert len(header) == 3 and header[1] == b'blob', (name, header)
        committed = output.read(int(header[2]))
        assert committed.replace(b'\r\n', b'\n') == raw.replace(b'\r\n', b'\n'), name
        assert output.read(1) == b'\n'
    assert not output.read()
    with tarfile.open(fileobj=io.BytesIO(source_overlay), mode='r:gz') as tar:
        assert json.load(tar.extractfile('manifest.json')) == manifest
        for name, digest in manifest['overlays'].items():
            tested = tar.extractfile(name).read()
            assert sha(tested) == digest, name
            # The final opt-in revision intentionally changes these overlays;
            # the embedded exploratory source is pinned by its own manifest.

    assert b'PUBLIC_MODEL_SNAPSHOT_GPU_OK' in get('snapshot.log')
    snapshot = json.loads(get('snapshot/results.json'))
    records = snapshot['correctness']
    assert len(records) == 24
    replayed = 0
    for world in (1, 2):
        for case in ('first', 'repeat', 'unsolved', 'model-change', 'beam-change'):
            pair = [row for row in records if row['world'] == world and row['case'] == case]
            assert len(pair) == 2 and {r['engine'] for r in pair} == {'python', 'rust'}
            assert pair[0]['path'] == pair[1]['path']
            for row in pair:
                if row['path'] is None:
                    assert case == 'unsolved'
                else:
                    replay(case, row['path']); replayed += 1
                assert row['mlp_host_binding']['ranks'] == world
                if row['engine'] == 'rust': assert row['snapshot']['owner'] == 'rust'
    assert b'PUBLIC_RUST_MLP_FINITE_GPU_OK' in get('paired.log')
    paired = json.loads(get('paired/results.json'))
    assert paired['nonfinite_rejected_and_session_reused'] and paired['num_gpus'] == 2
    rows = paired['rows']; repeated = [r for r in rows if r['phase'] == 'repeated']
    assert len(repeated) == 10 and len({tuple(r['pids']) for r in rows}) == 1
    assert len({r['model_hash'] for r in rows}) == 1
    changes = []
    for pair in range(5):
        old = next(r for r in repeated if r['pair'] == pair and r['mode'] == 'torch')
        new = next(r for r in repeated if r['pair'] == pair and r['mode'] == 'rust')
        assert old['rust_checked_bytes'] == 0
        assert new['rust_checked_bytes'] == paired['checked_tensor_bytes']
        changes.append(new['wall_seconds'] / old['wall_seconds'] - 1)
    assert changes == paired['paired_wall_changes']
    assert statistics.median(changes) == paired['median_paired_wall_change']
    print(json.dumps({'status': 'verified', 'fp16_bf16_patterns': 131072,
        'snapshot_cases': 20, 'independently_replayed_paths': replayed,
        'paired_calls': len(repeated), 'median_paired_wall_change': statistics.median(changes),
        'median_seconds': paired['median_seconds'], 'sha256': sha(data)}, indent=2))
