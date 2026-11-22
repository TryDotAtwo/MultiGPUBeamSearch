"""Verify source, paired timings, GPU rank receipts and replayable paths."""
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import tarfile

if not __debug__:
    raise SystemExit('verification requires assertions')
here = Path(__file__).resolve().parent
workspace = here.parents[2]
sha = lambda data: hashlib.sha256(data).hexdigest()
overlay = json.loads((here / 'tested-overlay-manifest.json').read_text())
assert overlay['base_commit'] == 'df696b736099330a83e5c2802a7e9bb959cdf3ff'
assert overlay['baseline_files'] == 493
source_bytes = (here / 'source-manifest.json').read_bytes()
assert sha(source_bytes) == overlay['overlay_manifest_sha256']
source = json.loads(source_bytes)
assert source['base_commit'] == overlay['base_commit'] and source['base_sha256'] == overlay['base_sha256']
assert source['cutlass_sha256'] == overlay['cutlass_sha256'] and source['files'] == overlay['files']
assert len(source['baseline']) == overlay['baseline_files']
# Independently bind every archived baseline byte stream to the pinned Git
# commit. Batch mode avoids hundreds of separate Git processes.
baseline_names = sorted(source['baseline'])
queries = ''.join(f"{source['base_commit']}:{name}\n" for name in baseline_names).encode()
reply = subprocess.run(['git', 'cat-file', '--batch'], cwd=workspace, input=queries,
                       stdout=subprocess.PIPE, check=True).stdout
stream = io.BytesIO(reply)
post_test_docs = {'ARCHITECTURE_NEED.md', 'memory/CHANGELOG.md'}
checked_checkout = 0
for name in baseline_names:
    header = stream.readline().rstrip(b'\n').split()
    assert len(header) == 3 and header[1] == b'blob', (name, header)
    data = stream.read(int(header[2]))
    assert stream.read(1) == b'\n'
    assert source['baseline'][name] in (sha(data), sha(data.replace(b'\n', b'\r\n'))), name
    if name not in source['files'] and name not in post_test_docs:
        checkout = (workspace / name).read_bytes()
        exact = sha(checkout) in (sha(data), source['baseline'][name])
        mixed_text = (b'\0' not in data and source['baseline'][name] == sha(data.replace(b'\n', b'\r\n'))
                      and sha(checkout.replace(b'\r\n', b'\n')) == sha(data))
        assert exact or mixed_text, name
        checked_checkout += 1
assert stream.read() == b''
expected_remote = dict(source['baseline'])
expected_remote.update({name: row['sha256'] for name, row in source['files'].items()})
actual_remote = {}
for line in (here / 'inputs-after.txt').read_text().splitlines():
    digest, name = line.split(' ', 1)
    assert name not in actual_remote
    actual_remote[name] = digest
assert actual_remote == expected_remote
for name, row in overlay['files'].items():
    data = (workspace / name).read_bytes()
    assert sha(data) == row['sha256'] or sha(data.replace(b'\r\n', b'\n')) == row['normalized_lf_sha256'], name
assert (here / 'status.txt').read_text().splitlines()[-1] == 'RUN_COMPLETE'
assert 'PYO3_MLP_SNAPSHOT_PASS' in (here / 'pyo3-test.log').read_text()
python_log = (here / 'python-tests.log').read_text()
assert re.search(r'\b\d+ passed\b', python_log) and ' failed' not in python_log
assert 'PUBLIC_MLP_LAUNCH_VERIFY_GPU_OK' in (here / 'gpu.log').read_text()
archive = json.loads((here / 'archive-verification.json').read_text())
portable = here / 'bootstrap-receipts.tar.gz'
assert sha(portable.read_bytes()) == archive['bootstrap_sha256']
result = json.loads((here / 'gpu/results.json').read_text())
assert (result['model_bytes'], result['width'], result['max_steps'], result['num_gpus']) == (18369835, 4096, 6, 2)
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
center = tuple(i // 16 for i in range(96))
def replay(path):
    state = list(center)
    for move in range(3):
        state[move], state[move + 24] = state[move + 24], state[move]
    for move in path:
        assert type(move) is int and 0 <= move < 24
        state[move], state[move + 24] = state[move + 24], state[move]
    return tuple(state) == center

geometry, admission, runs, pids = {0: [], 1: []}, {0: [], 1: []}, set(), set()
with tarfile.open(portable, 'r:gz') as tar:
    receipts = json.load(tar.extractfile('manifest.json'))
    assert receipts['source_archive_sha256'] == archive['archive_sha256']
    evidence_manifest_bytes = tar.extractfile('source-manifest.json').read()
    assert sha(evidence_manifest_bytes) == receipts['source_manifest_sha256'] == archive['source_manifest_sha256']
    evidence_manifest = json.loads(evidence_manifest_bytes)
    assert len(evidence_manifest['records']) == archive['records']
    assert len(evidence_manifest['blob_bytes']) == archive['blobs']
    assert len(evidence_manifest['omitted']) == archive['omitted']
    assert evidence_manifest['records']['verify-overlay.tar.gz']['sha256'] == overlay['overlay_sha256']
    assert sha((here / 'remote-run.sh').read_bytes()) == evidence_manifest['records']['verify-run.sh']['sha256']
    for name, row in receipts['evidence'].items():
        assert evidence_manifest['records'][name] == row
    assert len(receipts['records']) == archive['bootstrap_records']
    for name, row in receipts['records'].items():
        assert evidence_manifest['records'][name] == row
    prefix = 'beam-results/rust-mlp-verify-20261114/'
    for name in ('status.txt', 'inputs-after.txt', 'cargo-tests.log', 'pyo3-test.log',
                 'python-tests.log', 'gpu.log', 'gpu/results.json', 'gpus.csv', 'software.txt'):
        row = receipts['evidence'][prefix + name]
        data = (here / name).read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256'], name
    for mode in ('python_verify', 'rust_verify'):
        for pair in range(5):
            rows = [r for r in result['rows'] if r['mode'] == mode and r['pair'] == pair]
            assert [r['phase'] for r in rows] == (['cold', 'warmup', 'repeat', 'solved'] if pair == 0
                                                   else ['cold', 'warmup', 'repeat'])
            assert all(r['pids'] == rows[0]['pids'] for r in rows)
            assert all(r['path'] is None for r in rows[:3])
            if pair == 0:
                assert replay(rows[3]['path'])
            runs.add(rows[0]['run_dir'])
            pids.add(tuple(rows[0]['pids']))
            assert rows[-1]['checks']['launch_checks'] == len(rows)
            if mode == 'rust_verify':
                assert rows[-1]['checks']['rust_checks'] == len(rows)
                assert rows[-1]['checks']['python_checks'] == 0
            else:
                assert rows[-1]['checks']['rust_checks'] == 0
                assert rows[-1]['checks']['python_checks'] >= len(rows)
            assert rows[0]['checks']['backend_checks'] >= 1
            assert all(r['checks']['backend_checks'] == rows[0]['checks']['backend_checks'] for r in rows)
            for rank in range(2):
                name = (PurePosixPath(rows[0]['run_dir']).relative_to('/workspace') /
                        'resident-session' / f'bootstrap-{rank}.log').as_posix()
                receipt = receipts['records'][name]
                data = tar.extractfile('blobs/' + receipt['sha256']).read()
                assert sha(data) == receipt['sha256'] and len(data) == receipt['bytes']
                lines = data.decode(errors='replace').splitlines()
                values = dict(line.split('=', 1) for line in lines if '=' in line and ' ' not in line)
                assert values['LOCAL_RANK'] == values['CUDA_DEVICE_LOCAL_RANK'] == str(rank)
                assert all(key in values for key in keys)
                assert [line for line in lines if line.startswith('BEAM_MLP_ARTIFACT_VERIFIED=')] == [
                    'BEAM_MLP_ARTIFACT_VERIFIED=' + rows[0]['model_hash']]
                geometry[rank].append({key: values[key] for key in keys})
                admission[rank].append(int(values['gpu_free_before_bytes']))
assert len(runs) == len(pids) == 10
assert all(all(row == group[0] for row in group) for group in geometry.values())
assert len({r['model_hash'] for r in result['rows']}) == 1
changes = []
for pair in range(5):
    old = next(r['wall_seconds'] for r in result['rows'] if (r['pair'], r['mode'], r['phase']) ==
               (pair, 'python_verify', 'repeat'))
    new = next(r['wall_seconds'] for r in result['rows'] if (r['pair'], r['mode'], r['phase']) ==
               (pair, 'rust_verify', 'repeat'))
    changes.append(new / old - 1)
assert len(changes) == len(result['paired_changes']) == 5
assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(changes, result['paired_changes']))
assert math.isclose(statistics.median(changes), result['median_paired_change'], abs_tol=1e-12)
for mode in ('python_verify', 'rust_verify'):
    median = statistics.median(r['wall_seconds'] for r in result['rows'] if r['mode'] == mode and r['phase'] == 'repeat')
    assert math.isclose(median, result['median_seconds'][mode], abs_tol=1e-12)
summary = {'verification': 'PASS', 'git_baseline_files': len(baseline_names),
           'unchanged_checkout_files': checked_checkout, 'post_test_docs_excluded': sorted(post_test_docs),
           'sessions': 10, 'ranks': 20, 'geometry_keys_per_rank': len(keys),
           'admission_free_ranges': {str(rank): [min(rows), max(rows)] for rank, rows in admission.items()},
           'median_seconds': result['median_seconds'], 'paired_changes': changes,
           'median_paired_change': statistics.median(changes)}
print(json.dumps(summary, indent=2))
