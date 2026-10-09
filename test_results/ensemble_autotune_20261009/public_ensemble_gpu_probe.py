"""Public API acceptance, built and executed only on the owned remote host."""
import json
import os
from pathlib import Path
import time

from cayleypy import CayleyGraph, PermutationGroups
from multigpubeamsearch import NativeEnsemble, NativeOptions, prepare_native, beam_search


root = Path('/workspace/results/public-ensemble-api')
root.mkdir(exist_ok=False)
graph = CayleyGraph(PermutationGroups.lrx(8), device='cuda', random_seed=20261009)
options = NativeOptions(source_dir='/workspace/source', cutlass_dir='/workspace/cutlass',
    cache_dir=root/'cache', num_gpus=2, build_jobs=2, build_timeout_seconds=1800,
    timeout_seconds=900, calibration_seconds=120, calibration_max_batch=256,
    calibration_pipeline_seconds=600, calibration_frontier_max_states=8192,
    touch_bfs_radius=2)
ensemble = NativeEnsemble((None, None, None), (.5, .25, .25))
prepared = prepare_native(graph, ensemble, native_options=options)
(root/'prepared.json').write_text(json.dumps(dict(
    runner_sha256=prepared.runner_sha256,
    preparation_seconds=prepared.preparation_seconds,
    preparation_dir=str(prepared.preparation_dir)), indent=2))
# Do not interfere with the already-running fixed-work benchmark.
deadline = time.monotonic()+900
while True:
    try:
        os.kill(4287, 0)
    except ProcessLookupError:
        break
    if time.monotonic()>=deadline:
        raise RuntimeError('existing sweep still live; acceptance did not start')
    time.sleep(5)
start = graph.apply_path(graph.central_state, [0, 2, 0, 2]).reshape(-1).tolist()
result = beam_search(graph, start_state=start, predictor=prepared.model,
    native_options=prepared.options, backend='native', beam_width=8192,
    max_steps=8, return_path=True)
assert result.path_found, 'public ensemble search did not find path'
assert graph.apply_path(start, result.path).reshape(-1).tolist()==graph.central_state.tolist()
metadata = result.native_metadata
assert metadata['profile']['inference_autotuned'] is True
assert metadata['profile']['pipeline_autotuned'] is True
assert metadata['profile']['pipeline_calibration']['phase']=='pipeline_measured'
receipt = dict(path_found=True, replay_valid=True, path_length=result.path_length,
    backend=result.backend, metadata=metadata)
(root/'acceptance.json').write_text(json.dumps(receipt, indent=2, default=str))
print(json.dumps(receipt, default=str), flush=True)

