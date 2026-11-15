"""Verify the source-bound CNOT admission evidence and replay solved paths."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
info = json.loads((out / 'archive-verification.json').read_text(encoding='utf-8'))
data = (out / 'remote-evidence.tar.gz').read_bytes()
assert len(data) == info['bytes'] and sha(data) == info['sha256']
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
    files = {}
    for member in tar:
        if member.isdir():
            continue
        assert member.isfile() and member.name.startswith('cnot-admission-results/')
        name = member.name.removeprefix('cnot-admission-results/')
        assert name not in files and '..' not in Path(name).parts
        contents = tar.extractfile(member).read()
        assert len(contents) == member.size
        files[name] = contents

red = json.loads(files['red-manifest.json'])
green = json.loads(files['green-manifest.json'])
assert red['base_commit'] == green['base_commit'] == 'cfba6e472093735dad0acfcb6fcf26bbb135be6b'
base_archive = subprocess.check_output(['git', 'archive', '--format=tar', red['base_commit'],
                                        '--', *sorted(red['baseline'])], cwd=root)
with tarfile.open(fileobj=io.BytesIO(base_archive)) as tar:
    actual_base = {entry.name: sha(tar.extractfile(entry).read()) for entry in tar if entry.isfile()}
assert actual_base == red['baseline']
with tarfile.open(fileobj=io.BytesIO(files['source/base.tar.gz']), mode='r:gz') as tar:
    supplied_base = {entry.name: sha(tar.extractfile(entry).read()) for entry in tar if entry.isfile()}
assert supplied_base == red['baseline']
for name, digest in green['overlays'].items():
    assert sha((root / name).read_bytes().replace(b'\r\n', b'\n')) == digest, name
with tarfile.open(fileobj=io.BytesIO(files['source/green-input.tar.gz']), mode='r:gz') as tar:
    for name, digest in green['overlays'].items():
        assert sha(tar.extractfile(name).read()) == digest
assert files['source-before.txt'] == files['source-after.txt']
assert files['red-status.txt'].decode().splitlines() == ['RED_EXIT=1', 'RED_FINISHED']
red_log = files['red.log'].decode()
assert '4 failed' in red_log
assert 'test_rank_local_metadata_failure_is_collective_and_group_is_reusable' in red_log
assert 'metadata unavailable' in red_log and 'not JSON serializable' in red_log
assert files['status.txt'].decode().splitlines()[-2:] == ['COMPLETE', 'EXIT_CODE=0']
assert b'674 passed, 11 skipped' in files['python-tests.log']
assert b'PYO3_CNOT_INDEPENDENT_REPLAY_OK' in files['pyo3-contract.log']

paths = 0
failures = 0
for kind, world, device in [('cpu-1', 1, 'cpu'), ('cpu-2', 2, 'cpu'),
                            ('gpu-1', 1, 'cuda'), ('gpu-2', 2, 'cuda')]:
    ranks = []
    for rank in range(world):
        receipt = json.loads(files[f'receipts/{kind}/rank-{rank}.json'])
        assert (receipt['rank'], receipt['world'], receipt['device']) == (rank, world, device)
        records = receipt['records']
        assert len(records) == (14 if world == 2 else 8)
        for record in records[:4]:
            assert record['reference'] == record['rust']
            result, target = record['rust'], record['rows']
            if result['gates'] is None:
                assert result['status'] in ('depth_limit', 'exhausted')
                continue
            assert result['status'] == 'solved' and len(result['gates']) <= record['options']['max_depth']
            n = len(target)
            rows = [1 << i for i in range(n)]
            for move in result['gates']:
                assert 0 <= move < n * (n - 1)
                control, offset = divmod(move, n - 1)
                rows[offset + (offset >= control)] ^= rows[control]
            assert rows == target
            paths += 1
        for record in records[4:8]:
            assert record['strided_input'] and record['gates'] == ([0] if record['original_gate'] else [])
        if world == 2:
            assert records[8]['rejected'] == 'validator_mismatch' and 'differ' in records[8]['error']
            assert records[9]['rejected'] == 'rank_1_singular' and 'rank 1' in records[9]['error']
            expected = [('python', 0, 'property'), ('python', 1, 'json'),
                        ('rust', 0, 'property'), ('rust', 1, 'json')]
            for record, (validation, fault_rank, failure) in zip(records[10:], expected):
                assert record['validation'] == validation and record['fault_rank'] == fault_rank
                assert record['rejected'] == 'metadata_' + failure and record['reused'] is True
                assert f'rank {fault_rank}:' in record['error']
                assert ('metadata unavailable' if failure == 'property' else 'not JSON serializable') in record['error']
                failures += 1
        ranks.append(receipt)
    if world == 2:
        assert [r['rust'] for r in ranks[0]['records'][:4]] == [r['rust'] for r in ranks[1]['records'][:4]]
        routes = ranks[0]['records'][1]['rust']['path_routes']
        assert any(edge['move'] > 255 and len({edge['source'], edge['hash_owner'], edge['target']}) > 1
                   for edge in routes)
assert paths >= 12 and failures == 16
print(f'PASS: {len(actual_base)} base files, {len(green["overlays"])} overlays, '
      f'{paths} independently replayed paths, {failures} coordinated rank-error receipts')
