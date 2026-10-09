import json,pathlib,subprocess,shutil,hashlib
from multigpubeamsearch.build import source_digest,file_sha256,build_key
root=pathlib.Path('/workspace/fast-results')
prepared=json.loads((root/'prepared.json').read_text());bin=pathlib.Path(prepared['options']['runner_path']).parent
meta=json.loads((bin/'native-build.json').read_text());build=root/'cache/builds'/meta['build_key']
with (root/'incremental-helper-build.log').open('wb') as log:
 subprocess.run(['cmake','--build',str(build),'--target','stream1_ensemble_benchmark','--parallel','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
helper=build/'stream1_ensemble_benchmark';shutil.copy2(helper,bin/helper.name)
meta['source_digest']=source_digest(pathlib.Path('/workspace/fast-source'))
meta['calibration_binary_sha256']=file_sha256(helper)
spec={k:v for k,v in meta.items() if k not in ('build_key','binary_sha256','calibration_binary_sha256')}
meta['build_key']=build_key(spec)
for p in (bin/'native-build.json',build/'native-build.json'):p.write_text(json.dumps(meta,indent=2))
(root/'incremental-helper-receipt.json').write_text(json.dumps(dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/workspace/fast-source',text=True).strip(),metadata=meta,only_native_file_changed='tools/stream1_ensemble_benchmark.cpp'),indent=2))
print(json.dumps({'ready':True,'helper_sha256':meta['calibration_binary_sha256']}),flush=True)
