"""Independently verify archived public index-sort runs against Git and replay."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
import tarfile

workspace = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path.cwd()
root = workspace / 'test_results/refactor_20261114/public-index-3060'
archive = Path(sys.argv[1])
commit = (root / 'commit.txt').read_text().strip()
assert commit == '84f8d308ea26dd6842c08f4973fe155d1919628e'
before = (root / 'inputs-before.sha256').read_text()
assert before == (root / 'inputs-after.sha256').read_text()
with tarfile.open(archive, 'r:gz') as tar:
    manifest = json.load(tar.extractfile('manifest.json'))
    builds = {}
    for name, row in manifest['records'].items():
        if name.endswith('/native-build.json'):
            builds[name] = json.load(tar.extractfile('blobs/' + row['sha256']))
assert len(builds) == 4

# Reproduce the managed builder's whole native-source digest directly from
# Git's pinned bytes; do not use the possibly CRLF-converted Windows checkout.
roots = ('src', 'cuda', 'tools', 'cmake', 'native', 'rust')
top = ('CMakeLists.txt', 'Cargo.toml', 'Cargo.lock')
extensions = {'.h', '.hpp', '.cuh', '.c', '.cpp', '.cu', '.cmake', '.py', '.in', '.rs', '.toml'}
git = subprocess.Popen(['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf',
                        'archive', '--format=tar', commit, *top, *roots],
                       cwd=workspace, stdout=subprocess.PIPE)
files = {}
with tarfile.open(fileobj=git.stdout, mode='r|') as tree:
    for member in tree:
        parts = Path(member.name).parts
        if not member.isfile() or not parts:
            continue
        if member.name not in top and (parts[0] not in roots or Path(member.name).suffix not in extensions
                                       or any(x in ('target', '__pycache__', '.pytest_cache') for x in parts[1:])):
            continue
        files[member.name] = hashlib.sha256(tree.extractfile(member).read()).digest()
assert git.wait() == 0
native_digest = hashlib.sha256()
for name in sorted(files, key=lambda item: PurePosixPath(item).parts):
    native_digest.update(name.encode())
    native_digest.update(b'\0')
    native_digest.update(files[name])
print(json.dumps({'computed_native_digest': native_digest.hexdigest(),
                  'native_source_files': len(files),
                  'recorded_digests': sorted({b['source_digest'] for b in builds.values()})}))
for name, build in builds.items():
    assert build['source_digest'] == native_digest.hexdigest(), name
    binary = name.rsplit('/', 1)[0] + '/' + build['target']
    assert manifest['records'][binary]['sha256'] == build['binary_sha256'], binary
assert {(b['inference_backend'], b.get('experimental_index_sort', False))
        for b in builds.values()} == {(backend, enabled)
            for backend in ('cutlass', 'libtorch') for enabled in (False, True)}
source_rows = {}
for line in before.splitlines():
    expected, name = line.split(maxsplit=1)
    if name.endswith(('/_beam_runtime.abi3.so', '/_bin/beam-runner')):
        row = manifest['records']['beam-work/' + name]
        actual, origin = row['sha256'], 'verified archived binary'
    else:
        blob = subprocess.run(['git', 'cat-file', 'blob', f'{commit}:{name}'],
                              cwd=workspace, stdout=subprocess.PIPE, check=True).stdout
        actual, origin = hashlib.sha256(blob).hexdigest(), 'pinned Git blob'
    assert actual == expected, name
    source_rows[name] = {'sha256': actual, 'source': origin}
(root / 'source-verification.json').write_text(json.dumps(dict(
    commit=commit, before_equals_after=True, inputs=source_rows,
    native_source_digest=native_digest.hexdigest(), native_source_files=len(files),
    build_receipts_verified=len(builds)), indent=2) + '\n')

center = tuple(i // 16 for i in range(96))
moves = []
for i in range(24):
    move = list(range(96))
    move[i], move[i + 24] = move[i + 24], move[i]
    moves.append(move)

def apply(state, move):
    return tuple(state[index] for index in moves[move])

def replay(case, path):
    count = {'first': 3, 'repeat': 4, 'unsolved': 5, 'model-change': 3,
             'index-change': 3, 'beam-change': 3, 'collect': 3}[case]
    state = center
    for move in range(count):
        state = apply(state, move)
    for move in path:
        assert type(move) is int and 0 <= move < len(moves)
        state = apply(state, move)
    return state == center

cases = ('first', 'repeat', 'unsolved', 'model-change', 'index-change', 'beam-change', 'collect')
case_counts = {}
for backend in ('cutlass', 'libtorch'):
    data = json.loads((root / f'gpu-{backend}/results.json').read_text())
    assert data['backend'] == backend
    assert data['experimental_parallel_nccl_identity'] is True
    assert data['experimental_index_sort'] is True
    rows = data['records']
    assert len(rows) == 28
    for world in (1, 2):
        for case in cases:
            pair = [row for row in rows if row['world'] == world and row['case'] == case]
            assert len(pair) == 2 and {row['engine'] for row in pair} == {'python', 'rust'}
            assert pair[0]['path'] == pair[1]['path']
            for row in pair:
                assert row['persistent']['reused'] == (case in ('repeat', 'unsolved'))
                assert len(row['persistent']['worker_pids']) == world
                assert row['width'] == (65 if case in ('beam-change', 'collect') else 64)
                assert row['experimental_index_sort'] == (case != 'index-change')
                if case == 'unsolved':
                    assert row['path'] is None and row['solutions'] == []
                else:
                    assert replay(case, row['path'])
                    assert all(replay(case, path) for path in row['solutions'])
                    if case == 'collect':
                        assert 1 <= len(row['solutions']) <= 7
        for engine in ('python', 'rust'):
            ordered = [next(row for row in rows if row['world'] == world and
                            row['engine'] == engine and row['case'] == case) for case in cases]
            for left, right in zip(ordered, ordered[1:]):
                assert (left['persistent']['worker_pids'] == right['persistent']['worker_pids']) == \
                       (right['case'] in ('repeat', 'unsolved'))
            assert ordered[3]['build_key'] != ordered[4]['build_key']
            assert ordered[3]['build_key'] == ordered[5]['build_key']
            assert ordered[3]['path'] == ordered[4]['path']
            assert ordered[3]['solutions'] == ordered[4]['solutions']
    case_counts[backend] = len(rows)

data = json.loads((root / 'paired/results.json').read_text())
assert (data['width'], data['max_steps'], data['num_gpus']) == (4096, 6, 2)
rows = data['rows']
assert len(rows) == 16
for mode in ('baseline', 'index'):
    group = [row for row in rows if row['mode'] == mode]
    assert len(group) == 8
    assert [row['pair'] for row in group if row['phase'] == 'repeat'] == list(range(5))
    assert len({tuple(row['pids']) for row in group}) == 1 and len(group[0]['pids']) == 2
    assert len({row['build_key'] for row in group}) == 1
    assert all(row['path'] is None for row in group if row['phase'] != 'solved')
    assert replay('first', next(row for row in group if row['phase'] == 'solved')['path'])
    for metric in ('wall_seconds', 'worker_seconds', 'slow_rank_reported_avg_depth_seconds'):
        median = statistics.median(row[metric] for row in group if row['phase'] == 'repeat')
        assert median == data['median_seconds'][mode][metric]
assert rows[0]['build_key'] != rows[1]['build_key']
assert [row['path'] for row in rows if row['phase'] == 'solved'][0] == \
       [row['path'] for row in rows if row['phase'] == 'solved'][1]
report = dict(case_counts=case_counts, legal_paths_checked=True, pair_rows=len(rows),
              paired_repeats_per_mode=5, median_seconds=data['median_seconds'],
              scope=data['scope'])
(root / 'result-verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
