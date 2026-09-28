import json,time,numpy as np,psutil
from pathlib import Path
root=Path('test_results/cube555_20260928')
a=json.loads((root/'assets/puzzle_info.json').read_text());g=np.array(list(a['generators'].values()),dtype=np.int64)
# Exact full-state deduplication: no probabilistic hashes.
seen=np.arange(150,dtype=np.uint8).reshape(1,150).view('V150').ravel();front=seen.copy();rows=[]
for depth in range(1,6):
 start=time.perf_counter();count=len(front)*len(g)+len(seen);need=count*150*3
 if depth==5:
  partition=root/'touch_partitions';partition.mkdir(exist_ok=True)
  files=[(partition/f'p{i}.bin').open('wb') for i in range(32)]
  def emit(records):
   matrix=records.view(np.uint8).reshape(-1,150)
   hashes=np.zeros(len(records),dtype=np.uint32)
   for column in range(150): hashes=(hashes*33)^matrix[:,column]
   buckets=hashes%32
   for i,f in enumerate(files): f.write(records[buckets==i].tobytes())
  emit(seen)
  states=front.view(np.uint8).reshape(-1,150)
  for perm in g: emit(np.ascontiguousarray(states[:,perm]).view('V150').ravel())
  for f in files:f.close()
  total=0
  for i in range(32):
   path=partition/f'p{i}.bin';chunk=np.fromfile(path,dtype='V150');total+=len(np.unique(chunk));del chunk;path.unlink()
  rows.append(dict(radius=5,new_states=total-len(seen),total_states=total,cpu_seconds=time.perf_counter()-start,method='exact full-state partition sort'))
  print(rows[-1],flush=True);(root/'touch_radius_exact_counts.json').write_text(json.dumps(rows,indent=2));break
 if need>psutil.virtual_memory().available*.8: raise RuntimeError(f'RAM guard: {need} bytes estimated temporary storage')
 candidates=np.empty(count,dtype='V150');candidates[:len(seen)]=seen
 states=front.view(np.uint8).reshape(-1,150);offset=len(seen)
 for perm in g:
  candidates[offset:offset+len(front)]=np.ascontiguousarray(states[:,perm]).view('V150').ravel();offset+=len(front)
 union=np.unique(candidates);del candidates
 front=np.setdiff1d(union,seen,assume_unique=True);seen=union
 row=dict(radius=depth,new_states=len(front),total_states=len(seen),cpu_seconds=time.perf_counter()-start)
 rows.append(row);print(row,flush=True)
 (root/'touch_radius_exact_counts.json').write_text(json.dumps(rows,indent=2))
