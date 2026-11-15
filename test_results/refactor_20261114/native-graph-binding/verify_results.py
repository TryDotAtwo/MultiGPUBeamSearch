"""Verify the pinned source, compact receipts, negative cases, and GPU replays."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

here = Path(__file__).resolve().parent
root = here.parents[2]
cache = root / '.local/cache/native-graph-binding-20261115'
sha = lambda value: hashlib.sha256(value).hexdigest()
package = json.loads((here / 'package.json').read_text())
for name, key in [('remote-run.sh', 'graph-run.sh'), ('archive.py', 'graph-archive.py')]:
    assert sha((here / name).read_bytes()) == package['inputs'][key]
base, overlay = cache / 'base.tar.gz', cache / 'overlay.tar.gz'
assert sha(base.read_bytes()) == package['inputs']['graph-base.tar.gz']
assert sha(overlay.read_bytes()) == package['inputs']['graph-overlay.tar.gz']
with tarfile.open(overlay, 'r:gz') as tar:
    manifest = json.load(tar.extractfile('manifest.json'))
    for name, record in manifest['files'].items():
        assert sha(tar.extractfile(name).read()) == record['sha256']
        assert sha((root / name).read_bytes().replace(b'\r\n', b'\n')) == record['normalized_lf_sha256'], name
assert manifest['base_commit'] == package['base_commit']
names = sorted(manifest['baseline'])
queries = ''.join(f"{manifest['base_commit']}:{name}\n" for name in names).encode()
stream = subprocess.run(['git', 'cat-file', '--batch'], cwd=root, input=queries,
                        capture_output=True, check=True).stdout
position = 0; blobs = {}
for name in names:
    end = stream.index(b'\n', position)
    header = stream[position:end].split()
    assert len(header) == 3 and header[1] == b'blob'
    size = int(header[2]); position = end + 1
    blobs[name] = stream[position:position + size]
    assert len(blobs[name]) == size and stream[position + size:position + size + 1] == b'\n'
    position += size + 1
assert position == len(stream)
with tarfile.open(base, 'r:gz') as tar:
    found = {}
    for member in tar:
        if not member.isfile(): continue
        name = member.name.removeprefix('./')
        data = tar.extractfile(member).read()
        assert sha(data) == manifest['baseline'][name]
        assert data == blobs[name] or data.replace(b'\r\n', b'\n') == blobs[name].replace(b'\r\n', b'\n')
        found[name] = sha(data)
    assert found == manifest['baseline']
report = json.loads((here / 'archive-verification.json').read_text())
trace = here / 'trace-receipts.tar.gz'
assert sha(trace.read_bytes()) == report['trace_sha256']
traces = {}
with tarfile.open(trace, 'r:gz') as tar:
    receipt = json.load(tar.extractfile('manifest.json'))
    source_data = tar.extractfile('source-manifest.json').read()
    assert sha(source_data) == receipt['source_manifest_sha256'] == report['source_manifest_sha256']
    assert receipt['source_archive_sha256'] == report['archive_sha256']
    source = json.loads(source_data)
    for name, record in receipt['records'].items():
        assert source['records'][name] == record
        data = tar.extractfile('blobs/' + record['sha256']).read()
        assert sha(data) == record['sha256'] and len(data) == record['bytes']
        traces[name] = data
        prefix = 'beam-results/native-generator/'
        extracted = here / 'evidence' / name.removeprefix(prefix)
        if name.startswith(prefix) and extracted.is_file():
            assert extracted.read_bytes() == data
prefix = 'beam-results/native-generator/'
read = lambda name: traces[prefix + name].decode()
assert read('status.txt').splitlines() == ['SOURCE_VERIFY_OK', 'CPU_TEST_START', 'CPU_TEST_OK',
    'CARGO_TEST_START', 'CARGO_TEST_OK', 'PYTHON_START', 'PYTHON_OK', 'GPU_PUBLIC_START',
    'GPU_PUBLIC_OK', 'GPU_MISMATCH_START', 'GPU_MISMATCH_OK', 'COMPLETE']
assert read('hash-tests.log').count('NATIVE_LOADED_GRAPH_HASH_OK') == 2
assert 'NATIVE_LOADED_GRAPH_HASH_OK' in read('hash-asan.log')
assert 'ERROR: AddressSanitizer' not in read('hash-asan.log')
assert 'runtime error:' not in read('hash-asan.log')
assert 'test result: FAILED' not in read('cargo-tests.log')
assert '2 passed; 0 failed' in read('ffi-tests.log')
assert 'PYO3_SEARCH_SESSION_CANCEL_TRIVIAL_OK' in read('pyo3-session.log')
assert 'PYO3_SEARCH_SIGINT_CLEANUP_OK' in read('pyo3-interrupt.log')
assert re.search(r'\d+ passed, \d+ skipped', read('python-tests.log'))
assert '100% tests passed' in read('ctest.log')
assert json.loads(read('protocol-legacy.json')) == {'protocol':'resident-search-v1'}
assert json.loads(read('protocol-binding.json')) == {'protocol':'resident-search-v1','loaded_graph_binding':1}
rows = json.loads(read('gpu/results.json'))
assert [(row['world'], row['reused']) for row in rows] == [(1,False),(1,True),(2,False),(2,True)]
center = tuple(i // 16 for i in range(96)); moves = []
for i in range(24):
    move = list(range(96)); move[i], move[i + 24] = move[i + 24], move[i]; moves.append(move)
digest = sha(json.dumps(dict(central_state=center, generators=moves, generators_type='PERMUTATION'),
                        sort_keys=True).encode())
for row in rows:
    assert row['graph_hash'] == digest and len(row['pids']) == row['world']
    state = tuple(row['start'])
    for move in row['path']:
        assert type(move) is int and 0 <= move < len(moves)
        state = tuple(state[i] for i in moves[move])
    assert state == center
assert rows[0]['pids'] == rows[1]['pids'] and rows[2]['pids'] == rows[3]['pids']
for row in (rows[0], rows[2]):
    run = row['run_dir'].removeprefix('/workspace/')
    for rank in range(row['world']):
        log = traces[run + f'/resident-session/bootstrap-{rank}.log'].decode()
        assert f"communicator_rank={rank} world_size={row['world']} " in log
        assert 'GLOBAL_BEAM_WIDTH_EFFECTIVE=64' in log
cases = json.loads(read('mismatch/results.json'))
assert [case['case'] for case in cases] == ['moves','center','target','repair','rank1-center']
for case in cases:
    name = case['case']; rank = int(name == 'rank1-center')
    assert case['world'] == (2 if rank else 1) and len(case['pids']) == case['world']
    assert 'exited' in case['error']
    log = read(f'mismatch/{name}/resident-session/bootstrap-{rank}.log')
    expected = {'target':'bound session target differs from central state',
                'repair':'bound session cannot use repair mode'}.get(name, 'native loaded graph hash mismatch')
    assert expected in log
print(f"PASS: {len(names)} base files, {len(manifest['files'])} overlays, "
      f"two CPU golden shapes, {len(rows)} replayed GPU paths, {len(cases)} rejected cases")
