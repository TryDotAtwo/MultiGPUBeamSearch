"""Clean installed wheel: bundled source, automatic CUTLASS setup, public API."""
import hashlib
import json
from pathlib import Path
import multigpubeamsearch
from multigpubeamsearch import NativeEnsemble,NativeOptions,prepare_native,beam_search
from cayleypy import CayleyGraph,PermutationGroups
from multigpubeamsearch.build import source_digest

root=Path('/workspace/results/installed-wheel-api-v2');root.mkdir(exist_ok=False)
assert str(Path(multigpubeamsearch.__file__).resolve()).startswith('/workspace/compact-installed-v2/')
graph=CayleyGraph(PermutationGroups.lrx(8),device='cuda',random_seed=20261009)
options=NativeOptions(cache_dir=Path('/workspace/results/installed-wheel-api/cache'),num_gpus=2,build_jobs=2,
    build_timeout_seconds=1800,timeout_seconds=900,touch_bfs_radius=2,
    calibration_seconds=120,calibration_max_batch=256,
    calibration_pipeline_seconds=600,calibration_frontier_max_states=8192)
prepared=prepare_native(graph,NativeEnsemble((None,None,None),(.5,.25,.25)),native_options=options)
bundle=Path(multigpubeamsearch.__file__).parent/'_native_source'
provenance=json.loads((prepared.preparation_dir/'native-preparation.json').read_text())
assert source_digest(bundle)==provenance['build']['source_digest']
assert prepared.options.source_dir is None and prepared.options.cutlass_dir is None
assert prepared.options.runner_path.is_file()
original=Path('/workspace/results/installed-wheel-api/cache/prepared/bd96d826d7be43da86b75404868f6fe6')
original_provenance=json.loads((original/'native-preparation.json').read_text())
assert original_provenance['build']['source_digest']==provenance['build']['source_digest']
assert original_provenance['build']['cutlass_digest']==provenance['build']['cutlass_digest']
(root/'prepared.json').write_text(json.dumps(dict(preparation_dir=str(prepared.preparation_dir),
    runner_sha256=prepared.runner_sha256,bundled_source=str(bundle),
    source_digest=provenance['build']['source_digest'],cutlass_digest=provenance['build']['cutlass_digest'],
    original_automatic_preparation=str(original)),indent=2))
start=graph.apply_path(graph.central_state,[0,2,0,2]).reshape(-1).tolist()
result=beam_search(graph,start_state=start,predictor=prepared.model,
    native_options=prepared.options,backend='native',beam_width=8192,max_steps=8,return_path=True)
assert result.path_found and graph.apply_path(start,result.path).reshape(-1).tolist()==graph.central_state.tolist()
metadata=result.native_metadata
assert metadata['profile']['inference_autotuned'] and metadata['profile']['pipeline_autotuned']
assert metadata['build']['backend']=='ensemble'
assert metadata['build']['inference_backend']=='cutlass'
wheel=next(Path('/workspace/compact-dist-v2').glob('*.whl'))
receipt=dict(path_found=True,replay_valid=True,path_length=result.path_length,
    package_path=str(multigpubeamsearch.__file__),
    wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    wheel_bytes=wheel.stat().st_size,source_commit='5d2ff3b5f216c50e2dde0f280c9d08fea7476512',
    automatic_source=True,automatic_cutlass=True,metadata=metadata)
(root/'acceptance.json').write_text(json.dumps(receipt,indent=2,default=str))
print(json.dumps(dict(path_found=True,replay_valid=True,path_length=result.path_length,
    wheel_sha256=receipt['wheel_sha256'],inference_micro=metadata['profile']['inference_calibration']['parent_batch'])),flush=True)
