"""Verify source-pinned CNOT snapshot tests and independently replay their paths."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '3592a50f10db30ac64cb1832ba95459209937972'
ARCHIVE_SHA256 = '10496e4c7f7dcf4daf88886a4f6a4d34cf25818a6bacbfbf4c67ce3140be5b0b'
BASE_SHA256 = 'de325add79b65aad3a49ba2c6e16e6b1fb050372c5e9e4acea1ecc74fb7e8dfc'
OVERLAY_SHA256 = 'a8f0257c8f0f5640ab16a3290a11cda54979bc413e2b676493988eac5adb6e54'
sha = lambda data: hashlib.sha256(data).hexdigest()

def replay(n, gates):
    rows = [1 << i for i in range(n)]
    for gate in gates:
        assert 0 <= gate < n * (n - 1)
        control, target = divmod(gate, n - 1)
        target += target >= control
        rows[target] ^= rows[control]
    return rows

data = (HERE / 'evidence.tar.gz').read_bytes()
assert sha(data) == ARCHIVE_SHA256
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    entries = archive.getmembers()
    assert len(entries) == len(set(m.name for m in entries))
    assert all((m.isdir() or m.isfile()) and m.name.startswith('cnot-snapshot-results') and
               '..' not in Path(m.name).parts for m in entries)
    get = lambda name: archive.extractfile('cnot-snapshot-results/' + name).read()
    status = get('status.txt').decode().splitlines()
    assert status == ['SOURCE_OK', 'RUST_START', 'RUST_OK', 'PYTHON_START', 'PYTHON_OK',
                      'PUBLIC_CPU_START', 'PUBLIC_CPU_OK', 'PUBLIC_GPU_START', 'PUBLIC_GPU_OK',
                      'COMPLETE', 'EXIT_CODE=0'], status
    assert get('source-before.txt') == get('source-after.txt')
    assert b'651 passed, 11 skipped' in get('python-tests.log')
    assert b'PYO3_CNOT_INDEPENDENT_REPLAY_OK' in get('pyo3-contract.log')
    assert get('gpus.csv').count(b'RTX 3060') == 2
    manifest = json.loads(get('source-manifest.json'))
    assert manifest['base_commit'] == BASE and len(manifest['baseline']) == 533
    source_base = get('source/cnot-snapshot-base.tar.gz')
    source_overlay = get('source/cnot-snapshot-overlay.tar.gz')
    assert sha(source_base) == BASE_SHA256 and sha(source_overlay) == OVERLAY_SHA256
    with tarfile.open(fileobj=io.BytesIO(source_base), mode='r:gz') as tar:
        base_files = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
    assert set(base_files) == set(manifest['baseline'])
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

    paths = snapshots = 0
    for kind, world, device in [('cpu-1', 1, 'cpu'), ('cpu-2', 2, 'cpu'),
                                ('gpu-1', 1, 'cuda'), ('gpu-2', 2, 'cuda')]:
        for rank in range(world):
            receipt = json.loads(get(f'receipts/{kind}/rank-{rank}.json'))
            assert (receipt['rank'], receipt['world'], receipt['device']) == (rank, world, device)
            records = receipt['records']
            assert len(records) == (10 if world == 2 else 8)
            for row in records[:4]:
                assert row['reference'] == row['rust']
                gates = row['rust']['gates']
                if gates is not None:
                    assert replay(len(row['rows']), gates) == row['rows']
                    paths += 1
            for row in records[4:8]:
                assert row['strided_input'] and row['snapshot_validation'] in ('python', 'rust')
                gates = row['gates']
                expected = [0] if row['original_gate'] else []
                assert gates == expected
                assert replay(4, gates) == replay(4, expected)
                snapshots += 1
            if world == 2:
                assert {r['rejected'] for r in records[8:]} == {'validator_mismatch', 'rank_1_singular'}
print(f'PASS: {len(base_files)} base files, {len(manifest["overlays"])} overlays, '
      f'{paths} path replays, {snapshots} owned-snapshot cases across CPU/GPU ranks')
