"""Independently check the recorded MLP binding run and replay its paths."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile

here = Path(__file__).resolve().parent
workspace = here.parents[2]
manifest = json.loads((here / 'tested-overlay-manifest.json').read_text())
assert manifest['base'] == '65a1952c265bd60c6d69a7ee90687c0a7d336888'
assert (here / 'status.txt').read_text().splitlines()[-1] == 'RUN_COMPLETE'
assert '627 passed, 9 skipped' in (here / 'python-tests.log').read_text()
assert '100% tests passed' in (here / 'ctest.log').read_text()

# Documentation changed after launch. Git may normalize CRLF to LF, so check
# every code file against both its tested bytes and their LF-normalized form.
for name, row in manifest['files'].items():
    if name.endswith(('.md', '.so', '/beam-runner')):
        continue
    data = (workspace / name).read_bytes()
    assert (hashlib.sha256(data).hexdigest() == row['sha256'] or
            hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == row['normalized_lf_sha256']), name

full_archive = workspace / '.local/cache/mlp-binding-20261114/evidence.tar.gz'
verification = json.loads((here / 'archive-verification.json').read_text())
if full_archive.exists():
    assert hashlib.sha256(full_archive.read_bytes()).hexdigest() == verification['archive_sha256']
archive = here / 'bootstrap-receipts.tar.gz'
receipt_summary = json.loads((here / 'bootstrap-receipts.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt_summary['archive_sha256']
assert receipt_summary['source_archive_sha256'] == verification['archive_sha256']
result = json.loads((here / 'gpu/results.json').read_text())
assert result['model_bytes'] == 18_369_835
assert result['width'] == 4096 and result['max_steps'] == 6 and result['num_gpus'] == 2
center = tuple(i // 16 for i in range(96))


def replay(count, path):
    state = list(center)
    for move in range(count):
        state[move], state[move + 24] = state[move + 24], state[move]
    for move in path:
        assert type(move) is int and 0 <= move < 24
        state[move], state[move + 24] = state[move + 24], state[move]
    return tuple(state) == center


cases = ('first', 'repeat', 'unsolved', 'model-change', 'beam-change')
rows = result['correctness']
assert len(rows) == 24
with tarfile.open(archive, 'r:gz') as tar:
    portable = json.load(tar.extractfile('manifest.json'))
    assert portable['source_archive_sha256'] == verification['archive_sha256']
    archived = portable['records']
    assert len(archived) == receipt_summary['records'] == 40

    def check_receipts(run_dir, world, artifact_hash):
        relative = PurePosixPath(run_dir).relative_to('/workspace')
        expected = 'BEAM_MLP_ARTIFACT_VERIFIED=' + artifact_hash
        for rank in range(world):
            name = (relative / 'resident-session' / f'bootstrap-{rank}.log').as_posix()
            info = archived[name]
            data = tar.extractfile('blobs/' + info['sha256']).read()
            assert hashlib.sha256(data).hexdigest() == info['sha256']
            lines = [line for line in data.decode(errors='replace').splitlines()
                     if line.startswith('BEAM_MLP_ARTIFACT_VERIFIED=')]
            assert lines == [expected], (name, lines)

    for world in (1, 2):
        for engine in ('python', 'rust'):
            group = [next(row for row in rows if (row['world'], row['engine'], row['case'])
                          == (world, engine, case)) for case in cases]
            assert [row['persistent']['reused'] for row in group] == [False, True, True, False, False]
            assert [row['snapshot']['reused'] for row in group] == [False, True, True, False, False]
            assert [row['snapshot']['copied_bytes'] == 0 for row in group] == [False, True, True, False, False]
            assert group[0]['model_hash'] == group[1]['model_hash'] == group[2]['model_hash']
            assert group[3]['model_hash'] == group[4]['model_hash'] != group[0]['model_hash']
            assert group[0]['persistent']['worker_pids'] == group[1]['persistent']['worker_pids'] == group[2]['persistent']['worker_pids']
            assert group[3]['persistent']['worker_pids'] != group[2]['persistent']['worker_pids']
            assert group[4]['persistent']['worker_pids'] != group[3]['persistent']['worker_pids']
            assert group[2]['path'] is None
            for index, count in ((0, 3), (1, 4), (3, 3), (4, 3)):
                assert replay(count, group[index]['path'])
            # Reused calls point resident-session at the original worker's
            # directory. The archive stores the real logs, not the symlinks.
            for row, owner in zip(group, (group[0], group[0], group[0], group[3], group[4])):
                assert row['mlp_host_binding'] == {'artifact_hash': row['model_hash'], 'ranks': world}
                check_receipts(owner['run_dir'], world, row['model_hash'])
            transition = next(row for row in rows if row['world'] == world and row['engine'] == engine
                              and row['case'] == 'binding-transitions')['transitions']
            assert [item['mode'] for item in transition] == ['auto', 'native', 'native', 'auto']
            assert transition[0]['pids'] != transition[1]['pids'] == transition[2]['pids']
            assert transition[2]['pids'] != transition[3]['pids']
            assert transition[0]['binding'] is None and transition[3]['binding'] is None
            for item in transition[1:3]:
                assert item['binding']['ranks'] == world
                check_receipts(transition[1]['run_dir'], world, item['binding']['artifact_hash'])
        for case in cases:
            pair = [row for row in rows if row['world'] == world and row['case'] == case]
            assert pair[0]['path'] == pair[1]['path']

timings = result['rows']
for mode in ('baseline', 'reuse'):
    group = [row for row in timings if row['mode'] == mode]
    assert all(row['pids'] == group[0]['pids'] for row in group)
    for row in group:
        assert replay(3, row['path']) if row['phase'] == 'solved' else row['path'] is None
    for key in ('wall_seconds', 'worker_seconds'):
        value = statistics.median(row[key] for row in group if row['phase'] == 'repeat')
        assert math.isclose(value, result['median_seconds'][mode][key], abs_tol=1e-12)
changes = []
for pair in range(5):
    baseline = next(row for row in timings if row['phase'] == 'repeat' and row['pair'] == pair and row['mode'] == 'baseline')
    reuse = next(row for row in timings if row['phase'] == 'repeat' and row['pair'] == pair and row['mode'] == 'reuse')
    changes.append(reuse['wall_seconds'] / baseline['wall_seconds'] - 1)
assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(changes, result['paired_wall_changes']))
assert math.isclose(statistics.median(changes), result['median_paired_wall_change'], abs_tol=1e-12)
print(json.dumps({'verification': 'PASS', 'bounded_cases': 20, 'transition_groups': 4,
                  'paired_repeats': 5, 'median_seconds': result['median_seconds'],
                  'median_paired_wall_change': statistics.median(changes)}, indent=2))
