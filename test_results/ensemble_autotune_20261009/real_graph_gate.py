"""Real graph states and full native-backbone/epilogue acceptance on both GPUs."""
import json,pathlib,random,subprocess,numpy as np
root=pathlib.Path('/workspace')
graph=json.loads((root/'inputs/assets/p002.json').read_text())
moves=graph['moves']
state=np.load(root/'inputs/assets/solved_state.npy').reshape(-1).astype(int).tolist()
assert len(state)==96 and len(moves)==24
rng=random.Random(20261009);samples=[]
for i in range(128):
 for j in range(16):state=[state[k] for k in moves[rng.randrange(24)]]
 samples.append(state[:])
manifest=json.loads((root/'ensemble/ensemble.json').read_text())
manifest.update(calibration_states=samples,generators=moves)
path=root/'ensemble-real';path.mkdir(exist_ok=True)
(path/'ensemble.json').write_text(json.dumps(manifest))
out=root/'results/ensemble-real';out.mkdir(exist_ok=True)
processes=[]
for device in (0,1):
 log=(out/f'gpu{device}.log').open('w')
 proc=subprocess.Popen([str(root/'build-ensemble/stream1_ensemble_benchmark'),str(path),'1024','8192',str(device)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 processes.append((device,proc,log))
for device,proc,log in processes:
 code=proc.wait(timeout=180);log.close()
 print(json.dumps({'device':device,'exit':code,'output':(out/f'gpu{device}.log').read_text()}),flush=True)
 if code:raise SystemExit(code)
