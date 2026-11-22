import json,time,traceback
from pathlib import Path
from dataclasses import asdict
from cayleypy import CayleyGraph,PermutationGroups
from multigpubeamsearch import NativeOptions,NativeEnsemble,prepare_native
root=Path('/workspace/fast-results')
try:
 graph=CayleyGraph(PermutationGroups.lrx(13),device='cuda',random_seed=20261009)
 options=NativeOptions(source_dir='/workspace/fast-source',cache_dir=root/'cache',num_gpus=2,build_jobs=2,build_timeout_seconds=2400,calibration_max_batch=2048)
 prepared=prepare_native(graph,NativeEnsemble((None,None,None),(.5,.25,.25)),native_options=options)
 receipt=asdict(prepared)
 (root/'prepared.json').write_text(json.dumps(receipt,indent=2,default=str))
 print(json.dumps(dict(ready=True,runner=str(prepared.options.runner_path),seconds=prepared.preparation_seconds)),flush=True)
except BaseException:
 traceback.print_exc()
 raise
