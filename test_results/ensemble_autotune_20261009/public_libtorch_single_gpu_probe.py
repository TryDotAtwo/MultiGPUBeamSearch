"""Forced LibTorch path on 2x3060; this is not T4 hardware acceptance."""
import json
from pathlib import Path
from cayleypy import CayleyGraph, PermutationGroups
from multigpubeamsearch import NativeOptions, prepare_native, beam_search

root=Path('/workspace/results/public-libtorch-single-api');root.mkdir(exist_ok=False)
graph=CayleyGraph(PermutationGroups.lrx(8),device='cuda',random_seed=20261009)
options=NativeOptions(source_dir='/workspace/source',cutlass_dir='/workspace/cutlass',
    inference_backend='libtorch',cache_dir='/workspace/results/public-ensemble-api/cache',num_gpus=2,
    build_jobs=2,build_timeout_seconds=1800,timeout_seconds=900,
    calibration_seconds=120,calibration_max_batch=256,
    calibration_pipeline_seconds=600,calibration_frontier_max_states=8192,
    touch_bfs_radius=2)
prepared=prepare_native(graph,native_options=options)
(root/'prepared.json').write_text(json.dumps(dict(
    runner_sha256=prepared.runner_sha256,preparation_dir=str(prepared.preparation_dir),
    preparation_seconds=prepared.preparation_seconds),indent=2))
start=graph.apply_path(graph.central_state,[0,2,0,2]).reshape(-1).tolist()
result=beam_search(graph,start_state=start,predictor=prepared.model,
    native_options=prepared.options,backend='native',beam_width=8192,
    max_steps=8,return_path=True)
assert result.path_found
assert graph.apply_path(start,result.path).reshape(-1).tolist()==graph.central_state.tolist()
metadata=result.native_metadata
assert metadata['profile']['inference_backend']=='libtorch'
assert metadata['profile']['inference_autotuned'] is True
assert metadata['profile']['pipeline_autotuned'] is True
assert metadata['build']['target']=='production_runner_libtorch_stream1'
assert metadata['build']['calibration_binary_name']=='stream1_libtorch_mlp_benchmark'
micro=metadata['profile']['inference_calibration']['parent_batch']
env=metadata['profile']['pipeline_calibration']['environment']
assert int(env['BEAM_B_MICRO'])==micro*3
assert metadata['microbatch']['derived_parent_batch']==micro
receipt=dict(path_found=True,replay_valid=True,path_length=result.path_length,
    backend=result.backend,metadata=metadata,hardware_scope='2xRTX3060 forced LibTorch; not T4')
(root/'acceptance.json').write_text(json.dumps(receipt,indent=2,default=str))
print(json.dumps(dict(path_found=True,replay_valid=True,path_length=result.path_length,
    inference_micro=micro,downstream=env)),flush=True)

