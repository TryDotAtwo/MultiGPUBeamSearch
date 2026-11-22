"""Verify the source-pinned Rust terminal transfer and replay GPU results."""
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
BASE = '4fad51d7a9dfccb5938ca1a1249f3d6e2b95f494'
ARCHIVE_SHA256 = 'fa39e23fcfebfc251e129e869b37d0f2657edd9b5741cad73374f304fc7ec64f'
BASE_SHA256 = '9474024f9a09bee464c791a57406e4d56af76c1b094c8d70d9ca675f7a586454'
OVERLAY_SHA256 = '211566b1de4b689738ba2c4cd98e5a37448be707a83ec0bf9a364bd31ce1e8b1'
sha = lambda data: hashlib.sha256(data).hexdigest()


def replay(case, path):
    center = [i // 16 for i in range(96)]
    count = {'first': 3, 'repeat': 4, 'unsolved': 5, 'model-change': 3,
             'beam-change': 3, 'collect': 3}[case]
    state = center.copy()
    for move in range(count):
        state[move], state[move + 24] = state[move + 24], state[move]
    for move in path:
        assert type(move) is int and 0 <= move < 24
        state[move], state[move + 24] = state[move + 24], state[move]
    assert state == center, (case, path)


TERMINAL_FIELDS = ('native_seconds', 'terminal_records', 'collected_solutions', 'collection_status',
    'observed_parent_batch', 'observed_parent_batch_by_rank_values',
    'observed_inference_rows', 'observed_inference_rows_by_rank_values',
    'observed_stream3_batch_candidates', 'observed_stream3_batch_candidates_by_rank_values')


def verify_log(case, log, outcome):
    paths, seconds, widths, collected, counts, status = [], [], set(), [], [], None
    batches = {key: set() for key in ('B_MICRO', 'STREAM1_ROWS_PER_JOB', 'STREAM3_BATCH_CANDIDATES')}
    for line in log.splitlines():
        widths.update(int(value) for value in re.findall(r'(?:GLOBAL_BEAM_WIDTH_EFFECTIVE|global_beam_width_effective)=(\d+)', line))
        for key, value in re.findall(r'\b(B_MICRO|STREAM1_ROWS_PER_JOB|STREAM3_BATCH_CANDIDATES)=(\d+)', line):
            batches[key].add(int(value))
        if line.startswith('solve_bucket_solution=1 '):
            row = dict(item.split('=', 1) for item in line.split())
            path = [int(token[1:]) for token in row['solution'].split('.') if token]
            replay(case, path)
            found = int(row['found_depth'])
            assert len(path) == int(row['solution_length']) and 0 <= found <= len(path)
            collected.append(dict(path=path, solved_depth=found, touch_depth=len(path)-found,
                                  owner_rank=int(row['owner_rank'])))
        if line.startswith('solve_bucket_summary=1 '):
            match = re.search(r'\brecords=(\d+)', line); assert match
            counts.append(int(match.group(1)))
        if line.startswith('collection_status='):
            status = line.split('=', 1)[1].strip()
        if 'puzzle_solved=' in line:
            match = re.search(r'puzzle_solved=([01])\s+puzzle_id=0\s+seconds=([^\s]+)\s+'
                              r'solution_length=(-?\d+).*?\s+solution=([^\s]*)\s*$', line)
            assert match, line
            found, elapsed, length, encoded = match.groups()
            path = None if found == '0' else [int(token[1:]) for token in encoded.split('.') if token]
            assert (length == '-1' and not encoded) if path is None else len(path) == int(length)
            if path is not None: replay(case, path)
            paths.append(path); seconds.append(float(elapsed))
    assert paths and all(path == paths[0] for path in paths) and widths == {outcome['effective_beam_width']}
    assert paths[0] == outcome['path']
    expected = dict(native_seconds=max(seconds), terminal_records=len(paths))
    if counts:
        assert counts == [len(collected)]
        expected.update(collected_solutions=collected, collection_status=status or 'depth_reached')
    for key, name in (('B_MICRO', 'observed_parent_batch'),
                      ('STREAM1_ROWS_PER_JOB', 'observed_inference_rows'),
                      ('STREAM3_BATCH_CANDIDATES', 'observed_stream3_batch_candidates')):
        values = batches[key]
        expected[name] = next(iter(values)) if len(values) == 1 else None
        if len(values) > 1: expected[name + '_by_rank_values'] = sorted(values)
    actual = {name: outcome['metadata'][name] for name in TERMINAL_FIELDS if name in outcome['metadata']}
    assert actual == expected, (case, actual, expected)


data = (HERE / 'evidence.tar.gz').read_bytes()
assert sha(data) == ARCHIVE_SHA256
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    entries = archive.getmembers()
    assert len(entries) == len(set(m.name for m in entries))
    assert all((m.isdir() or m.isfile()) and m.name.startswith('terminal-result-results') and
               '..' not in Path(m.name).parts for m in entries)
    get = lambda name: archive.extractfile('terminal-result-results/' + name).read()
    status = get('status.txt').decode().splitlines()
    assert status == ['SOURCE_OK', 'RUST_TESTS_START', 'RUST_TESTS_OK', 'RUST_BUILD_OK',
        'PYTHON_OK', 'GPU_cutlass_START', 'GPU_cutlass_OK', 'GPU_libtorch_START',
        'GPU_libtorch_OK', 'PAIR_START', 'PAIR_OK', 'COMPLETE', 'EXIT_CODE=0'], status
    assert get('source-before.txt') == get('source-after.txt')
    assert b'test result: ok.' in get('rust-tests.log')
    assert re.search(rb'\d+ passed, \d+ skipped', get('python-tests.log'))
    assert b'TERMINAL_PROTOCOL_V1' in get('protocol.log')
    assert b'TERMINAL_DIFFERENTIAL_OK valid=5 invalid=9 stricter=2' in get('terminal-differential.log')
    assert get('terminal_differential.py').replace(b'\r\n', b'\n') == (
        HERE / 'terminal_differential.py').read_bytes().replace(b'\r\n', b'\n')
    corpus = json.loads(get('terminal-corpus/results.json'))
    assert len(corpus) == 16
    assert all(record[0]['ok'] and not record[1]['ok']
               for name, record in corpus.items() if name in ('native-error', 'invalid-found-depth'))
    assert get('gpus.csv').count(b'RTX 3060') == 2
    manifest = json.loads(get('source-manifest.json'))
    assert manifest['base_commit'] == BASE and len(manifest['baseline']) == 533
    source_base = get('source/terminal-result-base.tar.gz')
    source_overlay = get('source/terminal-result-overlay.tar.gz')
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
            assert (ROOT / name).read_bytes().replace(b'\r\n', b'\n') == tested.replace(b'\r\n', b'\n'), name

    checked = replayed = 0
    for backend in ('cutlass', 'libtorch'):
        assert b'PUBLIC_RUST_SESSION_GPU_OK' in get(f'gpu-{backend}.log')
        result = json.loads(get(f'gpu-{backend}/results.json'))
        rows = result['records']
        assert result['backend'] == backend and len(rows) == 24
        cases = ('first', 'repeat', 'unsolved', 'model-change', 'beam-change', 'collect')
        for world in (1, 2):
            for case in cases:
                pair = [r for r in rows if r['world'] == world and r['case'] == case]
                assert len(pair) == 2 and {r['engine'] for r in pair} == {'python', 'rust'}
                assert pair[0]['path'] == pair[1]['path']
                for row in pair:
                    assert row['terminal_python_parity'] == (row['engine'] == 'rust')
                    expected_width = 65 if case in ('beam-change', 'collect') else 64
                    assert row['width'] == expected_width
                    directory = Path(row['run_dir']).name
                    outcome = json.loads(get(f'run-receipts/gpu-{backend}/{directory}/native-outcome.json'))
                    assert outcome['path'] == row['path'] and outcome['effective_beam_width'] == expected_width
                    log = get(f'run-receipts/gpu-{backend}/{directory}/native.log').decode(errors='replace')
                    verify_log(case, log, outcome)
                    if case == 'unsolved':
                        assert row['path'] is None
                    else:
                        replay(case, row['path']); replayed += 1
                    for path in row['solutions']:
                        replay(case, path); replayed += 1
                    checked += 1

    assert b'PUBLIC_RUST_TERMINAL_PAIR_GPU_OK' in get('paired.log')
    pair = json.loads(get('paired/results.json'))
    assert pair['width'] == 4096 and pair['max_steps'] == 6 and pair['num_gpus'] == 2
    rows = [r for r in pair['rows'] if r['phase'] == 'repeated']
    assert len(rows) == 10 and all(r['wall_seconds'] > 0 for r in rows)
    for i in range(5):
        assert {r['mode'] for r in rows if r['pair'] == i} == {'duplicate', 'transfer'}
    for mode in ('duplicate', 'transfer'):
        median = statistics.median(r['wall_seconds'] for r in rows if r['mode'] == mode)
        assert median == pair['median_wall_seconds'][mode]
    assert all(r['extra_parse_seconds'] == 0 for r in rows if r['mode'] == 'transfer')
    assert all(r['extra_parse_seconds'] > 0 for r in rows if r['mode'] == 'duplicate')
print(f'PASS: {len(base_files)} base files, {len(manifest["overlays"])} overlays, '
      f'{checked} public GPU cases, {replayed} path replays and five paired repeats')
