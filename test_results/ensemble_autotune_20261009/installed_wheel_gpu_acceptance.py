"""Clean installed wheel: bundled source, automatic CUTLASS setup, public API."""
import hashlib
import json
from pathlib import Path
import multigpubeamsearch
from multigpubeamsearch import NativeEnsemble,NativeOptions,prepare_native,beam_search
from cayleypy import CayleyGraph,PermutationGroups

root=Path('/workspace/results/installed-wheel-api');root.mkdir(exist_ok=False)
assert str(Path(multigpubeamsearch.__file__).resolve()).startswith('/workspace/compact-installed/')
graph=CayleyGraph(PermutationGroups.lrx(8),device='cuda',random_seed=20261009)
options=NativeOptions(cache_dir=root/'cache',num_gpus=2,build_jobs=2,
    build_timeout_seconds=1800,timeout_seconds=900,touch_bfs_radius=2,
    calibration_seconds=120,calibration_max_batch=256,
    calibration_pipeline_seconds=600,calibration_frontier_max_states=8192)
prepared=prepare_native(graph,NativeEnsemble((None,None,None),(.5,.25,.25)),native_options=options)
assert '/compact-installed/multigpubeamsearch/_native_source' in str(prepared.options.source_dir)
assert prepared.options.cutlass_dir.is_dir()
(root/'prepared.json').write_text(json.dumps(dict(preparation_dir=str(prepared.preparation_dir),
    runner_sha256=prepared.runner_sha256,source_dir=str(prepared.options.source_dir),
    cutlass_dir=str(prepared.options.cutlass_dir)),indent=2))
start=graph.apply_path(graph.central_state,[0,2,0,2]).reshape(-1).tolist()
result=beam_search(graph,start_state=start,predictor=prepared.model,
    native_options=prepared.options,backend='native',beam_width=8192,max_steps=8,return_path=True)
assert result.path_found and graph.apply_path(start,result.path).reshape(-1).tolist()==graph.central_state.tolist()
metadata=result.native_metadata
assert metadata['profile']['inference_autotuned'] and metadata['profile']['pipeline_autotuned']
assert metadata['build']['backend']=='ensemble'
assert metadata['build']['inference_backend']=='cutlass'
wheel=next(Path('/workspace/compact-dist').glob('*.whl'))
receipt=dict(path_found=True,replay_valid=True,path_length=result.path_length,
    package_path=str(multigpubeamsearch.__file__),
    wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    wheel_bytes=wheel.stat().st_size,source_commit='5403dfb1dfc91741e601e61f2483c94e71e6bf64',
    automatic_source=True,automatic_cutlass=True,metadata=metadata)
(root/'acceptance.json').write_text(json.dumps(receipt,indent=2,default=str))
print(json.dumps(dict(path_found=True,replay_valid=True,path_length=result.path_length,
    wheel_sha256=receipt['wheel_sha256'],inference_micro=metadata['profile']['inference_calibration']['parent_batch'])),flush=True)
