"""Verify source-bound CNOT receipts and replay paths with Python integers."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
evidence = out / 'evidence'
manifest = json.loads((evidence / 'source-manifest.json').read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
archive_info = json.loads((out / 'archive-verification.json').read_text())
archive = out / 'remote-evidence.tar.gz'
archive_data = archive.read_bytes()
assert sha(archive_data) == archive_info['sha256'] and len(archive_data) == archive_info['bytes']
archive_only = '--archive-only' in sys.argv[1:]
members = set()
archived_files = {}
with tarfile.open(fileobj=io.BytesIO(archive_data), mode='r:gz') as tar:
    for member in tar:
        if member.isdir():
            continue
        assert member.isfile() and member.name.startswith('cnot-rust-validation-final/')
        relative = member.name.removeprefix('cnot-rust-validation-final/')
        target = (evidence / relative).resolve()
        assert target.is_relative_to(evidence.resolve()) and relative not in members
        data = tar.extractfile(member).read()
        assert len(data) == member.size
        if not archive_only and target.is_file():
            assert target.read_bytes() == data
        archived_files[relative] = data
        members.add(relative)
assert members == set(archive_info['members']) and len(members) == archive_info['files']
names = sorted(manifest['baseline'])
archived = subprocess.check_output(['git', 'archive', '--format=tar', manifest['base_commit'], '--', *names], cwd=root)
with tarfile.open(fileobj=io.BytesIO(archived)) as tar:
    files = {member.name.removeprefix('./'): sha(tar.extractfile(member).read())
             for member in tar if member.isfile()}
assert files == manifest['baseline']
for name, hashes in manifest['overlays'].items():
    actual = (root / name).read_bytes().replace(b'\r\n', b'\n')
    assert sha(actual) == hashes['normalized_lf_sha256'], name
status = archived_files['status.txt'].decode().splitlines()
assert status[-2:] == ['COMPLETE', 'EXIT_CODE=0'], status
assert archived_files['source-before.txt'] == archived_files['source-after.txt']
assert b'PYO3_CNOT_INDEPENDENT_REPLAY_OK' in archived_files['pyo3-contract.log']

paths = 0
for kind, world, device in [('cpu-1', 1, 'cpu'), ('cpu-2', 2, 'cpu'),
                            ('gpu-1', 1, 'cuda'), ('gpu-2', 2, 'cuda')]:
    ranks = []
    for rank in range(world):
        data = json.loads(archived_files[f'receipts/{kind}/rank-{rank}.json'])
        assert (data['rank'], data['world'], data['device']) == (rank, world, device)
        assert len(data['records']) == (6 if world == 2 else 4)
        for record in data['records'][:4]:
            assert record['reference'] == record['rust']
            result, target = record['rust'], record['rows']
            if result['gates'] is None:
                assert result['status'] in ('depth_limit', 'exhausted')
                continue
            assert result['status'] == 'solved'
            assert len(result['gates']) <= record['options']['max_depth']
            n = len(target)
            rows = [1 << i for i in range(n)]
            for move in result['gates']:
                assert 0 <= move < n * (n - 1)
                control, slot = divmod(move, n - 1)
                target_row = slot + (slot >= control)
                rows[target_row] ^= rows[control]
            assert rows == target
            paths += 1
        if world == 2:
            assert [entry['rejected'] for entry in data['records'][4:]] == ['validator_mismatch', 'rank_1_singular']
            assert 'differ' in data['records'][4]['error']
            assert 'rank 1' in data['records'][5]['error'] and 'singular' in data['records'][5]['error']
        ranks.append(data)
    if world == 2:
        assert [r['rust'] for r in ranks[0]['records'][:4]] == [r['rust'] for r in ranks[1]['records'][:4]]
        routes = ranks[0]['records'][1]['rust']['path_routes']
        assert any(edge['move'] > 255 and len({edge['source'], edge['hash_owner'], edge['target']}) > 1
                   for edge in routes)
assert paths >= 12
print(f"PASS: {len(files)} base files, {len(manifest['overlays'])} overlays, {paths} independently replayed rank paths, CPU/GPU rank errors")
