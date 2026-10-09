import pathlib,json,tarfile,hashlib,subprocess,os
root=pathlib.Path('/tmp/fp16-results');full=root/'full';full.mkdir(exist_ok=True)
assert json.loads((root/'profile.json').read_text())['status']=='PASS'
assert json.loads((root/'readout_acceptance.json').read_text())['status']=='PASS'
archive=pathlib.Path('/tmp/prior-speed.tar.gz')
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='5479d3ce894964333382deb12186d0752f3e08329da18ed886fa06ba75fe1f67'
prior=pathlib.Path('/tmp/prior-speed');prior.mkdir(exist_ok=True)
with tarfile.open(archive) as bundle:
 for entry in bundle:
  if entry.name.startswith('speed-results/packed/') and entry.isfile():bundle.extract(entry,prior,filter='data')
from decode_fixture import decode
fixture=full/'frontier';fixture.mkdir(exist_ok=True)
manifest=json.loads((prior/'speed-results/packed/manifest.json').read_text())
for entry in manifest['entries']:
 packed=prior/'speed-results/packed'/entry['packed_file']
 with packed.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==entry['packed_sha256']
 decode(packed,fixture/f"rank{entry['rank']}.bin",entry['parents'],entry['sha256'])
(fixture/'manifest.json').write_text(json.dumps(manifest,indent=2))
(full/'calibration.json').write_text(json.dumps(dict(reserve_bytes=1342177280,scope='Same conservative reserve and beam as prior FP32 pair; not a new FP16 maximum search')))
source=pathlib.Path('/tmp/blend-production');m=json.loads(pathlib.Path('/tmp/source_manifest.json').read_text())
for name,digest in m.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
binary=source/'build/production_runner_libtorch_stream1'
(root/'provenance.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256(pathlib.Path('/tmp/source_manifest.json').read_bytes()).hexdigest(),source_files=len(m),precision='FP16 weights/activations; FP32 reductions/readout accumulator/blend',build_info=json.loads(subprocess.check_output([str(binary),'--build-info']))),indent=2))
monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu','--format=csv,noheader','-l','5'],stdout=(root/'gpu.csv').open('w'))
(root/'monitor_pid.json').write_text(json.dumps(dict(pid=monitor.pid)))
from run_core import run
pilot=full/'pilot';pilot.mkdir(exist_ok=True)
for rank in (0,1):
 with (fixture/f'rank{rank}.bin').open('rb') as f:(pilot/f'rank{rank}.bin').write_bytes(f.read(131072*112))
for only in (True,False):
 good,v,logs=run(262144,'pilot_transformer' if only else 'pilot_blend',only=only,fixture=pilot,timeout=600)
 assert good,v
pair=[]
for only in (True,False):
 good,v,logs=run(34930688,'full_transformer' if only else 'full_blend',only=only,fixture=fixture,timeout=5400)
 assert good,v
 pair.append(v);(full/'progress.json').write_text(json.dumps(pair,indent=2))
a=max(pair[0]['depth_seconds']);b=max(pair[1]['depth_seconds'])
result=dict(status='PASS',global_beam=34930688,precision='FP16',transformer_seconds=a,blend_seconds=b,extra_seconds=b-a,slowdown_percent=(b/a-1)*100,repeats_per_mode=1,order='AB',same_frontier_as_fp32=True)
(full/'paired_result.json').write_text(json.dumps(result,indent=2));monitor.terminate();print(json.dumps(result),flush=True)
