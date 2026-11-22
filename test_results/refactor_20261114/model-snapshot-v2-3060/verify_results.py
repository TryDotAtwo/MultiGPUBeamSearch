"""Independently replay recorded paths and verify the paired measurements."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import tarfile

here = Path(__file__).resolve().parent
workspace = here.parents[2]
commit = (here / 'commit.txt').read_text().strip()
assert commit == '46c02ab50284ebe693141af8637066a7f716d054'
assert (here / 'inputs-before.sha256').read_bytes() == (here / 'inputs-after.sha256').read_bytes()
assert (here / 'status.txt').read_text().splitlines()[-1] == 'RUN_COMPLETE'
assert '607 passed, 9 skipped' in (here / 'cpu-tests.log').read_text()

archive = workspace / '.local/cache/model-snapshot-20261114/evidence-v2.tar.gz'
with tarfile.open(archive, 'r:gz') as tar:
    manifest = json.load(tar.extractfile('manifest.json'))
for line in (here / 'inputs-before.sha256').read_text().splitlines():
    expected, filename = line.split(maxsplit=1)
    if filename.endswith(('_beam_runtime.abi3.so', '_bin/beam-runner')):
        assert manifest['records']['beam-work/' + filename]['sha256'] == expected
    else:
        content = subprocess.check_output(['git', 'show', f'{commit}:{filename}'], cwd=workspace)
        assert hashlib.sha256(content).hexdigest() == expected

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
assert len(rows) == 20
for world in (1, 2):
    for engine in ('python', 'rust'):
        group = [next(row for row in rows if (row['world'], row['engine'], row['case']) == (world, engine, case))
                 for case in cases]
        assert len({row['build_key'] for row in group}) == 1
        assert [row['persistent']['reused'] for row in group] == [False, True, True, False, False]
        assert [row['snapshot']['reused'] for row in group] == [False, True, True, False, False]
        assert [row['snapshot']['copied_bytes'] == 0 for row in group] == [False, True, True, False, False]
        assert group[0]['model_hash'] == group[1]['model_hash'] == group[2]['model_hash']
        assert group[3]['model_hash'] == group[4]['model_hash'] != group[0]['model_hash']
        assert group[0]['persistent']['worker_pids'] == group[1]['persistent']['worker_pids'] == group[2]['persistent']['worker_pids']
        assert group[3]['persistent']['worker_pids'] != group[2]['persistent']['worker_pids']
        assert group[4]['persistent']['worker_pids'] != group[3]['persistent']['worker_pids']
        assert len(group[0]['persistent']['worker_pids']) == world
        assert group[2]['path'] is None
        for index, count in ((0, 3), (1, 4), (3, 3), (4, 3)):
            assert replay(count, group[index]['path'])
    for case in cases:
        paired = [row for row in rows if row['world'] == world and row['case'] == case]
        assert paired[0]['path'] == paired[1]['path']

timings = result['rows']
for mode in ('baseline', 'reuse'):
    group = [row for row in timings if row['mode'] == mode]
    assert all(row['pids'] == group[0]['pids'] for row in group)
    assert len(group[0]['pids']) == 2
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
print(json.dumps(dict(commit=commit, path_cases=len(rows), pairs=len(changes),
                      model_bytes=result['model_bytes'], median_seconds=result['median_seconds'],
                      median_paired_wall_change=statistics.median(changes), verification='PASS'), indent=2))
