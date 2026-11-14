"""Independently replay paths and compare every benchmark rank's geometry."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile

if not __debug__:
    raise SystemExit('verification requires Python assertions to be enabled')

here = Path(__file__).resolve().parent
workspace = here.parents[2]
manifest = json.loads((here / 'tested-overlay-manifest.json').read_text())
assert manifest['base'] == '30f7d71ec6787621aa7d282aee1ee8a556732d02'
base_archive = workspace / '.local/cache/rust-mlp-snapshot-20261114/base.tar.gz'
if base_archive.exists():
    assert hashlib.sha256(base_archive.read_bytes()).hexdigest() == manifest['base_archive_sha256']
assert (here / 'status.txt').read_text().splitlines()[-1] == 'RUN_COMPLETE'
assert '631 passed, 9 skipped' in (here / 'python-tests.log').read_text()
for name, row in manifest['files'].items():
    data = (workspace / name).read_bytes()
    assert (hashlib.sha256(data).hexdigest() == row['sha256'] or
            hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == row['normalized_lf_sha256']), name
archive_report = json.loads((here / 'archive-verification.json').read_text())
full_archive = workspace / '.local/cache/rust-mlp-snapshot-20261114/evidence.tar.gz'
if full_archive.exists():
    assert hashlib.sha256(full_archive.read_bytes()).hexdigest() == archive_report['archive_sha256']
archive = here / 'bootstrap-receipts.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_report['bootstrap_sha256']
result = json.loads((here / 'gpu/results.json').read_text())
assert (result['model_bytes'], result['width'], result['max_steps'], result['num_gpus']) == (18369835, 4096, 6, 2)
center = tuple(i // 16 for i in range(96))


def replay(count, path):
    state = list(center)
    for move in range(count):
        state[move], state[move + 24] = state[move + 24], state[move]
    for move in path:
        assert type(move) is int and 0 <= move < 24
        state[move], state[move + 24] = state[move + 24], state[move]
    return tuple(state) == center


keys = '''GLOBAL_BEAM_WIDTH_EFFECTIVE WORLD_SIZE B_MICRO STREAM1_ROWS_PER_JOB
STREAM1_CONCURRENCY STREAM3_RING_SLOTS RING_COUNT RING_SLOT_COUNT
STREAM3_BATCH_CANDIDATES STREAM5_BATCH_CANDIDATES SHARD_COUNT SHARD_BUFFER_COUNT
STORAGE_SHARD_COUNT STREAM4_BATCH_CANDIDATES STREAM4_TRIGGER_CANDIDATES
STREAM4_BATCH_ALIGNMENT STREAM4_ACTIVE_SORT_SLOTS SHARD_CAPACITY_CANDIDATES
SHARD_CAPACITY_SCALE_PPM GLOBAL_SPILL_CAPACITY GLOBAL_SPILL_SCALE_PPM
STREAM5_RECV_CAPACITY_SCALE_PPM FINAL_MATERIALIZE_CHUNK_CANDIDATES
FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM STREAM5_SLOT_COUNT STREAM5_SEND_SLOT_CANDIDATES
STREAM5_RECV_SLOT_CANDIDATES FINAL_MATERIALIZE_EXCHANGE_CANDIDATES N_LOCAL
LOGICAL_SHARD_SIZE GROSS_CANDIDATES_PER_DEPTH_EST STREAM3_JOBS_PER_DEPTH_EST
STREAM3_SORT_WORK_UNITS_EST STREAM4_FLOW_SCALE_PPM STREAM4_INPUT_CANDIDATES_PER_DEPTH_EST
STREAM4_JOBS_PER_SHARD_EST STREAM4_JOBS_PER_DEPTH_EST STREAM4_WAVES_PER_DEPTH_EST
STREAM4_SORT_WORK_UNITS_EST GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS BEAM_WIDTH_ALIGNMENT
SCORE_SCALE SCORE_MAX_KEY SCORE_BIN_COUNT SOLVED_RESULT_CAPACITY'''.split()
geometry = {0: [], 1: []}
admission_free = {0: [], 1: []}
cold_runs, cold_pids = set(), set()
cases = ('first', 'repeat', 'unsolved', 'model-change', 'beam-change')
with tarfile.open(archive, 'r:gz') as tar:
    portable = json.load(tar.extractfile('manifest.json'))
    assert portable['source_archive_sha256'] == archive_report['archive_sha256']
    source_manifest = tar.extractfile('source-manifest.json').read()
    source_sha = hashlib.sha256(source_manifest).hexdigest()
    assert source_sha == portable['source_manifest_sha256'] == archive_report['source_manifest_sha256']
    full_records = json.loads(source_manifest)['records']
    prefix = 'beam-results/rust-mlp-snapshot-20261114/'
    evidence = portable['evidence']
    for rel in ('gpu/results.json', 'status.txt', 'python-tests.log', 'cargo-tests.log',
                'pyo3-test.log', 'mock-tests.log', 'inputs-after.txt'):
        row = evidence[prefix + rel]
        expected = tar.extractfile('blobs/' + row['sha256']).read()
        assert len(expected) == row['bytes'] and hashlib.sha256(expected).hexdigest() == row['sha256']
        assert (here / rel).read_bytes() == expected, rel
    checked_inputs = {}
    for line in (here / 'inputs-after.txt').read_text().splitlines():
        sha, name = line.split(' ', 1)
        assert name not in checked_inputs
        checked_inputs[name] = sha
    assert checked_inputs == {name: row['sha256'] for name, row in manifest['files'].items()}
    assert all(full_records[name] == row for name, row in evidence.items())
    assert all(full_records[name] == row for name, row in portable['records'].items())
    if full_archive.exists():
        with tarfile.open(full_archive, 'r:gz') as full:
            assert full.extractfile('manifest.json').read() == source_manifest
    receipts = portable['records']
    assert len(receipts) == archive_report['bootstrap_records']

    def log(run_dir, rank):
        name = (PurePosixPath(run_dir).relative_to('/workspace') /
                'resident-session' / f'bootstrap-{rank}.log').as_posix()
        row = receipts[name]
        data = tar.extractfile('blobs/' + row['sha256']).read()
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        return data.decode(errors='replace').splitlines()

    def check_binding(run_dir, world, artifact):
        for rank in range(world):
            lines = [line for line in log(run_dir, rank) if line.startswith('BEAM_MLP_ARTIFACT_VERIFIED=')]
            assert lines == ['BEAM_MLP_ARTIFACT_VERIFIED=' + artifact]

    for world in (1, 2):
        for engine in ('python', 'rust'):
            group = [next(row for row in result['correctness'] if
                          (row['world'], row['engine'], row['case']) == (world, engine, case))
                     for case in cases]
            assert [row['persistent']['reused'] for row in group] == [False, True, True, False, False]
            assert [row['snapshot']['reused'] for row in group] == [False, True, True, False, False]
            if engine == 'rust':
                assert all(row['snapshot']['owner'] == 'rust' for row in group)
            for idx, count in ((0, 3), (1, 4), (3, 3), (4, 3)):
                assert replay(count, group[idx]['path'])
            assert group[2]['path'] is None
            for row, owner in zip(group, (group[0], group[0], group[0], group[3], group[4])):
                assert row['mlp_host_binding'] == {'artifact_hash': row['model_hash'], 'ranks': world}
                check_binding(owner['run_dir'], world, row['model_hash'])
            transition = next(row for row in result['correctness'] if
                              (row['world'], row['engine'], row['case']) ==
                              (world, engine, 'binding-transitions'))['transitions']
            assert [row['mode'] for row in transition] == ['auto', 'native', 'native', 'auto']
            assert transition[0]['pids'] != transition[1]['pids'] == transition[2]['pids']
            assert transition[2]['pids'] != transition[3]['pids']
            check_binding(transition[1]['run_dir'], world, transition[1]['binding']['artifact_hash'])

    for mode in ('baseline', 'python_owner', 'reuse'):
        for pair in range(5):
            rows = [row for row in result['rows'] if row['mode'] == mode and row['pair'] == pair]
            assert len(rows) == (4 if pair == 0 else 3)
            assert [row['phase'] for row in rows[:3]] == ['cold', 'warmup', 'repeat']
            assert all(row['pids'] == rows[0]['pids'] for row in rows)
            cold_runs.add(rows[0]['run_dir'])
            cold_pids.add(tuple(rows[0]['pids']))
            assert all(row['path'] is None for row in rows[:3])
            if pair == 0:
                assert replay(3, rows[3]['path'])
            for rank in range(2):
                lines = log(rows[0]['run_dir'], rank)
                values = dict(line.split('=', 1) for line in lines
                              if '=' in line and ' ' not in line)
                assert values['LOCAL_RANK'] == str(rank)
                assert values['CUDA_DEVICE_LOCAL_RANK'] == str(rank)
                assert all(key in values for key in keys)
                geometry[rank].append({key: values[key] for key in keys})
                admission_free[rank].append(int(values['gpu_free_before_bytes']))
                if mode != 'baseline':
                    check_binding(rows[0]['run_dir'], 2, rows[0]['model_hash'])
    assert all(all(row == group[0] for row in group) for group in geometry.values())
    assert len(cold_runs) == len(cold_pids) == 15

for mode in ('baseline', 'python_owner', 'reuse'):
    for key in ('wall_seconds', 'worker_seconds'):
        value = statistics.median(row[key] for row in result['rows']
                                  if row['mode'] == mode and row['phase'] == 'repeat')
        assert math.isclose(value, result['median_seconds'][mode][key], abs_tol=1e-12)
for base, field, median_field in [
        ('baseline', 'paired_wall_changes', 'median_paired_wall_change'),
        ('python_owner', 'paired_owner_changes', 'median_paired_owner_change')]:
    changes = []
    for pair in range(5):
        lhs = next(row for row in result['rows'] if row['phase'] == 'repeat'
                   and row['pair'] == pair and row['mode'] == base)
        rhs = next(row for row in result['rows'] if row['phase'] == 'repeat'
                   and row['pair'] == pair and row['mode'] == 'reuse')
        changes.append(rhs['wall_seconds'] / lhs['wall_seconds'] - 1)
    assert len(changes) == len(result[field]) == 5
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(changes, result[field]))
    assert math.isclose(statistics.median(changes), result[median_field], abs_tol=1e-12)
summary = {'verification': 'PASS', 'bounded_cases': 20, 'transition_groups': 4,
           'benchmark_sessions': 15, 'compared_ranks': 30, 'geometry_keys_per_rank': len(keys),
           'rank_geometry': {str(rank): group[0] for rank, group in geometry.items()},
           'admission_free_ranges': {str(rank): [min(rows), max(rows)] for rank, rows in admission_free.items()},
           'median_seconds': result['median_seconds'],
           'median_paired_wall_change': result['median_paired_wall_change'],
           'median_paired_owner_change': result['median_paired_owner_change']}
print(json.dumps(summary, indent=2))
