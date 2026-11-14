"""Independently verify the tested source, portable receipts and GPU paths.

Requires Git, the recorded base commit and the local cached source archives.
It does not claim sanitizer, other-hardware or CNOT coverage.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=ROOT/'.local/cache/native-generator-20261115'
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
portable=HERE/'trace-receipts.tar.gz'
assert sha(portable.read_bytes())==archive_report['trace_sha256']
traces={}
with tarfile.open(portable,'r:gz') as tar:
    receipt=json.load(tar.extractfile('manifest.json'))
    source_bytes=tar.extractfile('source-manifest.json').read()
    assert sha(source_bytes)==receipt['source_manifest_sha256']==archive_report['source_manifest_sha256']
    assert receipt['source_archive_sha256']==archive_report['archive_sha256']
    source=json.loads(source_bytes)
    for name,row in receipt['evidence'].items():
        assert source['records'][name]==row
        data=(HERE/name.removeprefix('beam-results/native-generator/')).read_bytes()
        assert sha(data)==row['sha256'] and len(data)==row['bytes'],name
    for name,row in receipt['records'].items():
        assert source['records'][name]==row
        data=tar.extractfile('blobs/'+row['sha256']).read()
        assert sha(data)==row['sha256'] and len(data)==row['bytes'],name
        traces[name]=data

status=(HERE/'status.txt').read_text().splitlines()
assert status==['SOURCE_VERIFY_START','SOURCE_VERIFY_OK','CPU_BUILD_START','CPU_TEST_OK',
    'CARGO_TEST_START','CARGO_TEST_OK','PYTHON_START','PYTHON_OK','GPU_PUBLIC_START',
    'GPU_PUBLIC_OK','GPU_PAIR_BUILD_START','GPU_PAIR_START','GPU_PAIR_OK','COMPLETE']
assert (HERE/'fifo.log').read_text().count('NATIVE_GENERATOR_ONE_READ_FIFO_OK')==3
assert 'TimeoutExpired' in (HERE/'fifo-old-red.log').read_text()
assert 'native_input checks=45 failures=0' in (HERE/'native-input-tests.log').read_text()
assert 'NATIVE_INPUT_ASAN_UBSAN_OK' in (HERE/'asan.log').read_text()
assert 'NATIVE_GENERATOR_FORMATS_OK valid=9 rejected=1' in (HERE/'formats.log').read_text()
assert 'NATIVE_GENERATOR_FORMATS_OK valid=9 rejected=1' in (HERE/'formats-baseline.log').read_text()
assert (HERE/'baseline-exact.log').read_text().strip()=='native_generator_snapshot PASS'
assert sha((HERE/'format-check.py').read_bytes().replace(b'\r\n',b'\n'))==sha(
    (ROOT/'tests/native_generator_formats.py').read_bytes().replace(b'\r\n',b'\n'))
assert 'test result: FAILED' not in (HERE/'cargo-tests.log').read_text()
assert re.search(r'643 passed, 9 skipped', (HERE/'python-tests.log').read_text())

import statistics
for shape in ('96-112-24','120-128-18','150-160-255'):
    result=json.loads((HERE/f'cpu-{shape}/results.json').read_text())
    logical,physical,count=map(int,shape.split('-'))
    assert (result['state_len'],result['moves'])==(logical,count)
    generators={f'm{move}':[(i+move)%logical for i in range(logical)] for move in range(count)}
    assert result['fixture_sha256']==sha((json.dumps(dict(generators=generators))+'\n').encode())
    assert len(result['pairs'])==5
    for pair in result['pairs']:
        assert pair['baseline']['checksum']==pair['candidate']['checksum']
        assert pair['baseline']['repeats']==pair['candidate']['repeats']==20
    for name in ('baseline','candidate'):
        for key,value in result['medians'][name].items():
            assert value==statistics.median(pair[name][key] for pair in result['pairs'])

assert 'NATIVE_GENERATOR_UNINSTRUMENTED_TIMING_OK' in (HERE/'timing.log').read_text()
for name in ('native_generator_parse_timing.cpp','native_generator_parse_timing.py'):
    archived=(HERE/'timing-source'/name).read_bytes().replace(b'\r\n',b'\n')
    assert sha(archived)==sha((ROOT/'tests'/name).read_bytes().replace(b'\r\n',b'\n'))
for shape in ('96-112-24','150-160-255'):
    measured=json.loads((HERE/f'timing/{shape}.json').read_text())
    allocation=json.loads((HERE/f'cpu-{shape}/results.json').read_text())
    assert measured['fixture_sha256']==allocation['fixture_sha256']
    assert len(measured['pairs'])==5
    for name,variant in (('baseline','old'),('candidate','new')):
        key=f'beam-results/native-generator/timing/{variant}-{shape}'
        assert measured['binary_sha256'][name]==source['records'][key]['sha256']
        assert measured['median_seconds'][name]==statistics.median(p[name]['seconds'] for p in measured['pairs'])
    for p in measured['pairs']:
        assert p['baseline']['checksum']==p['candidate']['checksum']
        assert p['baseline']['repeats']==p['candidate']['repeats']==100
        assert abs(p['change']-(p['candidate']['seconds']/p['baseline']['seconds']-1))<1e-10
    assert measured['median_paired_change']==statistics.median(p['change'] for p in measured['pairs'])

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
for row in (rows[0],rows[2]):
    prefix=row['run_dir'].removeprefix('/workspace/')+'/resident-session/bootstrap-'
    for rank in range(row['world']):
        log=traces[prefix+str(rank)+'.log'].decode('utf-8')
        assert f"communicator_rank={rank} world_size={row['world']} " in log
        assert 'GLOBAL_BEAM_WIDTH_EFFECTIVE=64' in log

pair=json.loads((HERE/'pair/paired-results.json').read_text())
assert pair['world']==2 and pair['beam']==4096 and len(pair['pairs'])==5
geometry=('GLOBAL_BEAM_WIDTH_EFFECTIVE','BEAM_WIDTH_ALIGNMENT','WORLD_SIZE','LOCAL_RANK',
    'B_MICRO','RING_COUNT','SHARD_COUNT','SHARD_CAPACITY_CANDIDATES',
    'frontier_state_capacity','STREAM3_RING_SLOTS','STREAM4_ACTIVE_SORT_SLOTS',
    'STREAM4_BATCH_ALIGNMENT','STREAM4_BATCH_CANDIDATES')
for old,new in zip(pair['plans']['control'],pair['plans']['radix']):
    assert all(old[key]==new[key] for key in geometry)
    assert old['GLOBAL_BEAM_WIDTH_EFFECTIVE']==4096 and old['WORLD_SIZE']==2
for label in ('control','radix'):
    runner=pair['runners'][label]
    record=source['records'][runner['path'].removeprefix('/workspace/')]
    assert runner['sha256']==record['sha256']
fixture='beam-results/native-generator/fixture/'
assert pair['input_sha256']==sha(traces[fixture+'puzzle.json'])
for name,digest in pair['model_files_sha256'].items():
    assert digest==sha(traces[fixture+'weights/'+name])
for row in pair['pairs']:
    for label in ('control','radix'):
        assert len(row[label]['seconds'])==5
        assert row[label]['median']==statistics.median(row[label]['seconds'])
        run=row[label]['run'].removeprefix('/workspace/')+'/'
        telemetry=json.loads(traces[run+'gpu-telemetry.json'])
        assert not telemetry['errors'] and telemetry['throttled'] is False
        assert len(telemetry['samples'])>=2
        for sample in telemetry['samples']:
            devices={device['index']:device for device in sample}
            assert len(devices)==len(sample) and {0,1}<=devices.keys()
            assert all(devices[index]['throttled'] is False for index in (0,1))
        by_rank=[]
        for rank in (0,1):
            log=traces[run+f'rank-{rank}.log'].decode('utf-8')
            lines=[line for line in log.splitlines() if line.startswith('calibration_depth=')]
            assert len(lines)==6
            parsed=[dict(part.split('=',1) for part in line.split()) for line in lines]
            assert [int(item['calibration_depth']) for item in parsed]==list(range(6))
            assert all(int(item['rank'])==rank and int(item['parents'])==2048 for item in parsed)
            by_rank.append([float(item['depth_sec']) for item in parsed[1:]])
            assert 'stream1_executor=native_cuda_graph' in log
        assert row[label]['seconds']==[max(values) for values in zip(*by_rank)]
    assert abs(row['speedup_percent']-100*(row['control']['median']/row['radix']['median']-1))<1e-10
assert pair['median_speedup_percent']==statistics.median(row['speedup_percent'] for row in pair['pairs'])
print(f"PASS: {len(names)} base files, {len(manifest['files'])} overlays, "
      f"three CPU shapes, {len(rows)} replayed GPU paths, and five paired full-step receipts")
