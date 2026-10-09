"""Archive remote evidence and the exact measured frozen executable."""
import hashlib,json,pathlib,tarfile
root=pathlib.Path('/workspace');out=root/'results'
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
build=json.loads((out/'build_receipt.json').read_text())
assert digest(root/'runner-frozen')==build['binary_sha256']
sweep=json.loads((out/'sweep.json').read_text());assert len(sweep)==4
assert all(x['timing']['exit_codes']==[0]*8 for x in sweep if x['status']=='PASS')
assert all(x['status'] in ('PASS','STOPPED_THERMAL_THROTTLING','NOT_RUN_HOST_REJECTED') for x in sweep)
assert json.loads((out/'path_replay.json').read_text())['status']=='PASS'
for x in sweep:
 if x['status']!='PASS':continue
 for rank in range(8):
  log=(out/f"depth_{x['beam']}"/f'rank{rank}.log').read_text()
  assert 'blend_execution_precision=fp16' in log
  assert 'blend_launch_order=transformer_first' in log
  assert 'benchmark_full_frontier=1' in log
manifest={}
archive=root/'cube444-8x3090-evidence.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as bundle:
 files=[p for p in out.rglob('*') if p.is_file() and p.suffix!='.bin']
 files += [root/'runner-frozen',root/'bootstrap.py',root/'sweep.py',root/'archive.py',root/'inputs/bundle/blend.json',root/'inputs/bundle/tensors.tsv']
 for path in files:
  name=str(path.relative_to(root));manifest[name]={'sha256':digest(path),'bytes':path.stat().st_size}
  bundle.add(path,arcname=name,recursive=False)
# Independently read each archived byte stream, including the measured executable.
with tarfile.open(archive) as bundle:
 for member in bundle:
  if member.isfile():
   stream=bundle.extractfile(member);assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest[member.name]['sha256']
receipt={'status':'PASS','binary_sha256':build['binary_sha256'],'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,'files':manifest,
 'source_commit':build['github_commit'],'fixture_scope':'Generated legal 80-move states from index1000; exact per-rank seeds/fixture hashes in results; raw frontiers are reproducible, not included.',
 'weights_scope':'Kaggle input hashes and export metadata are included; weights remain at their original provider.'}
(root/'archive_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)

