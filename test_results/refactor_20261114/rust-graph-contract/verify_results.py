"""Independently verify the tested source, portable receipts and GPU paths.

Requires Git, the recorded base commit and the local cached source archives.
It does not claim sanitizer, performance, other-hardware or CNOT coverage.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=ROOT/'.local/cache/rust-graph-contract-20261115'
sha=lambda data:hashlib.sha256(data).hexdigest()
package=json.loads((HERE/'package.json').read_text())
for local,key in [('remote-run.sh','graph-run.sh'),('archive.py','graph-archive.py')]:
    assert sha((HERE/local).read_bytes())==package['inputs'][key],local
base=CACHE/'base.tar.gz'
overlay=CACHE/'overlay.tar.gz'
assert sha(base.read_bytes())==package['inputs']['graph-base.tar.gz']
assert sha(overlay.read_bytes())==package['inputs']['graph-overlay.tar.gz']
with tarfile.open(overlay,'r:gz') as tar:
    manifest=json.load(tar.extractfile('manifest.json'))
    for name,row in manifest['files'].items():
        data=tar.extractfile(name).read()
        assert sha(data)==row['sha256']
        current=(ROOT/name).read_bytes()
        assert sha(current.replace(b'\r\n',b'\n'))==row['normalized_lf_sha256'],name
assert manifest['base_commit']==package['base_commit']
assert manifest['base_sha256']==sha(base.read_bytes())
names=sorted(manifest['baseline'])
queries=''.join(f"{manifest['base_commit']}:{name}\n" for name in names).encode()
proc=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=queries,capture_output=True,check=True)
stream=proc.stdout;position=0;blobs={}
for name in names:
    end=stream.index(b'\n',position)
    header=stream[position:end].split()
    assert len(header)==3 and header[1]==b'blob',(name,header)
    size=int(header[2]);position=end+1
    blobs[name]=stream[position:position+size]
    assert len(blobs[name])==size and stream[position+size:position+size+1]==b'\n'
    position+=size+1
assert position==len(stream)
with tarfile.open(base,'r:gz') as tar:
    found={}
    for member in tar:
        if not member.isfile():continue
        name=member.name.removeprefix('./')
        data=tar.extractfile(member).read()
        assert name in blobs and sha(data)==manifest['baseline'][name],name
        assert data==blobs[name] or data.replace(b'\r\n',b'\n')==blobs[name].replace(b'\r\n',b'\n'),name
        found[name]=sha(data)
    assert found==manifest['baseline']
archive_report=json.loads((HERE/'archive-verification.json').read_text())
portable=HERE/'bootstrap-receipts.tar.gz'
assert sha(portable.read_bytes())==archive_report['bootstrap_sha256']
bootstraps={}
with tarfile.open(portable,'r:gz') as tar:
    receipt=json.load(tar.extractfile('manifest.json'))
    source_bytes=tar.extractfile('source-manifest.json').read()
    assert sha(source_bytes)==receipt['source_manifest_sha256']==archive_report['source_manifest_sha256']
    assert receipt['source_archive_sha256']==archive_report['archive_sha256']
    source=json.loads(source_bytes)
    for name,row in receipt['evidence'].items():
        assert source['records'][name]==row
        data=(HERE/name.removeprefix('beam-results/rust-graph-contract/')).read_bytes()
        assert sha(data)==row['sha256'] and len(data)==row['bytes'],name
    for name,row in receipt['records'].items():
        assert source['records'][name]==row
        data=tar.extractfile('blobs/'+row['sha256']).read()
        assert sha(data)==row['sha256'] and len(data)==row['bytes'],name
        bootstraps[name]=data.decode('utf-8')
status=(HERE/'status.txt').read_text().splitlines()
assert status==['SOURCE_VERIFY_START','SOURCE_VERIFY_OK','CARGO_TEST_START','CARGO_TEST_OK',
    'BUILD_START','BUILD_OK','PYO3_OK','PYTHON_START','PYTHON_OK','GPU_START','GPU_OK','COMPLETE']
cargo=(HERE/'cargo-tests.log').read_text()
assert 'test result: FAILED' not in cargo and 'graph_identity' in cargo
assert 'PYO3_SEARCH_SESSION_CANCEL_TRIVIAL_OK' in (HERE/'pyo3-session.log').read_text()
assert 'PYO3_SEARCH_SIGINT_CLEANUP_OK' in (HERE/'pyo3-interrupt.log').read_text()
assert re.search(r'643 passed, 9 skipped', (HERE/'python-tests.log').read_text())
rows=json.loads((HERE/'gpu/results.json').read_text())
assert [(r['world'],r['reused']) for r in rows]==[(1,False),(1,True),(2,False),(2,True)]
center=tuple(i//16 for i in range(96))
moves=[]
for i in range(24):
    move=list(range(96));move[i],move[i+24]=move[i+24],move[i];moves.append(move)
graph=dict(central_state=center,generators=moves,generators_type='PERMUTATION')
graph_hash=sha(json.dumps(graph,sort_keys=True).encode())
for row in rows:
    assert row['graph_hash']==graph_hash and len(row['pids'])==row['world']
    state=tuple(row['start'])
    for move in row['path']:
        assert type(move) is int and 0<=move<len(moves)
        state=tuple(state[i] for i in moves[move])
    assert state==center
assert rows[0]['pids']==rows[1]['pids'] and rows[2]['pids']==rows[3]['pids']
assert len(receipt['records'])==3
for row in (rows[0],rows[2]):
    prefix=row['run_dir'].removeprefix('/workspace/')+'/resident-session/bootstrap-'
    for rank in range(row['world']):
        log=bootstraps[prefix+str(rank)+'.log']
        assert f"communicator_rank={rank} world_size={row['world']} " in log
        assert 'GLOBAL_BEAM_WIDTH_EFFECTIVE=64' in log
print(f"PASS: {len(names)} base files, {len(manifest['files'])} overlays, "
      f"{len(receipt['records'])} bootstrap receipts and {len(rows)} independently replayed GPU paths")
