"""Verify the immutable remote receipts and their relationship to this source tree."""
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE_SHA256 = 'e1ce4689c4a2c74d8387d3036532e14c7bc4dda47181833e6bc100016a073d7f'
BASE = '280a7a1731f946ecbcfd8c32ffe47f8abe02430a'
sha = lambda data: hashlib.sha256(data).hexdigest()
raw = (HERE / 'evidence.tar.gz').read_bytes()
assert sha(raw) == ARCHIVE_SHA256, 'archive digest'
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
    entries = archive.getmembers()
    assert all(x.isfile() and x.name.startswith('results/') and '..' not in Path(x.name).parts
               for x in entries)
    assert len(entries) == len(set(x.name for x in entries))
    get = lambda name: archive.extractfile('results/' + name).read()
    manifest = json.loads(get('evidence-files.json'))
    assert len(manifest) == 405 and len(entries) == len(manifest) + 1
    for name, entry in manifest.items():
        data = get(name)
        assert len(data) == entry['size'] and sha(data) == entry['sha256'], name

    source = json.loads(get('source-snapshots/candidate-manifest.json'))
    assert source['base_commit'] == BASE
    overlay = get('source-snapshots/candidate-overlay.tar')
    assert sha(overlay) == source['candidate_archive']
    with tarfile.open(fileobj=io.BytesIO(overlay)) as changes:
        assert {x.name for x in changes if x.isfile()} == set(source['overlay_files'])
        for name, digest in source['overlay_files'].items():
            tested = changes.extractfile(name).read()
            assert sha(tested) == digest, name
            assert (ROOT / name).read_bytes().replace(b'\r\n', b'\n') == tested.replace(b'\r\n', b'\n'), name

    base_files = {}
    for name, digest in source['base_archives'].items():
        data = get('source-snapshots/' + name)
        assert sha(data) == digest, name
        with tarfile.open(fileobj=io.BytesIO(data)) as base_archive:
            for entry in base_archive:
                if entry.isfile():
                    assert entry.name not in base_files, entry.name
                    base_files[entry.name] = base_archive.extractfile(entry).read()
    requests = ''.join(BASE + ':' + name + '\n' for name in base_files).encode()
    proc = subprocess.run(['git', 'cat-file', '--batch'], input=requests,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT, check=True)
    output = io.BytesIO(proc.stdout)
    for name, data in base_files.items():
        header = output.readline().split()
        assert len(header) == 3 and header[1] == b'blob', (name, header)
        committed = output.read(int(header[2]))
        assert output.read(1) == b'\n'
        assert data.replace(b'\r\n', b'\n') == committed.replace(b'\r\n', b'\n'), name
    assert not output.read()

    architecture = json.loads(get('arch-compile.json'))
    for name in ('stream4.cu', 'static_memory.cu'):
        assert architecture[name]['architectures'] == [75, 80, 90, 120]
        assert architecture[name]['source_sha256'] == source['overlay_files']['cuda/' + name]
    binary = json.loads(get('binary-comparison.json'))
    assert binary['normalized_bytes_identical'] and len(binary['byte_difference_offsets_zero_based']) == 4

    for prefix in ('final-', 'combined-', 'default-'):
        tests = ['stream4_dirty_merge_257_cuda_tests', 'stream4_bounded_graph_cuda_tests',
                 'stream4_logical_union_cuda_tests', 'static_memory_cuda_tests']
        for name in tests:
            assert get(prefix + name + '.log').strip(), (prefix, name)
    oracle = get('final-stream4_dirty_merge_cuda_tests.log').decode()
    assert 'gpu=0 dirty_merge_oracle=pass cases=35 graph_mode_mask=3' in oracle
    assert 'gpu=1 dirty_merge_oracle=pass cases=35 graph_mode_mask=3' in oracle
    assert b'ERROR SUMMARY: 0' in get('final-memcheck.log')
    for name in ('final-race-unbounded.log', 'race-merge-bounded.log', 'race-merge-unbounded.log'):
        assert b'RACECHECK SUMMARY: 0 hazards' in get(name), name
    for name in ('race-bounded.exit', 'index-control-race-bounded.exit', 'sync-1-bounded.exit'):
        assert get(name).strip() == b'99', name
    assert b'DeviceReduceByKeyKernel' in get('race-bounded.log')
    assert b'DeviceReduceByKeyKernel' in get('index-control-race-bounded.log')

    speedups = []
    for name in ('pairs/paired-results.json', 'memory-pairs/paired-results.json'):
        p = json.loads(get(name))
        assert (p['world'], p['beam'], p['precision']) == (2, 4096, 'fp16')
        assert len(p['pairs']) == 5
        row_speeds = []
        for row in p['pairs']:
            for variant in ('control', 'radix'):
                values = row[variant]['seconds']
                assert len(values) == 5 and all(v > 0 for v in values)
                assert abs(statistics.median(values) - row[variant]['median']) < 1e-12
            speed = (row['control']['median'] / row['radix']['median'] - 1) * 100
            assert abs(speed - row['speedup_percent']) < 1e-9
            row_speeds.append(speed)
        median = statistics.median(row_speeds)
        assert abs(median - p['median_speedup_percent']) < 1e-9
        speedups.append(median)
    memory = json.loads(get('memory-pairs/memory-summary.json'))
    assert memory['median_saved_net_bytes_by_gpu'] == [-2097152, -2097152]

    puzzle = json.loads(get('session/puzzle.json'))
    center = puzzle['central_state']
    moves = [puzzle['generators']['m' + str(i)] for i in range(24)]
    session = json.loads(get('session/results.json'))['results']
    assert len(session) == 12
    for world in (1, 2):
        for variant in ('reference', 'rust'):
            rows = sorted((r for r in session if r['world'] == world and r['variant'] == variant),
                          key=lambda r: r['sequence'])
            assert len(rows) == 3 and len({tuple(r['worker_pids']) for r in rows}) == 1
            assert all(r['beam_width'] == 64 and len(r['worker_pids']) == world for r in rows)
            for row in rows:
                sequence, path = row['sequence'], row['path']
                if sequence == 2:
                    assert path is None
                    continue
                state = center[:]
                for move in range(sequence + 3):
                    state = [state[i] for i in moves[move]]
                for move in path:
                    state = [state[i] for i in moves[move]]
                assert state == center and len(path) == sequence + 3
            reference = [r['path'] for r in session if r['world'] == world and r['variant'] == 'reference']
            assert [r['path'] for r in rows] == reference

print(f'PASS: {len(manifest)} evidence files, {len(base_files)} base files, '
      f'{len(source["overlay_files"])} overlays, session replay; '
      f'median speedups {speedups[0]:.4f}% and {speedups[1]:.4f}%')
