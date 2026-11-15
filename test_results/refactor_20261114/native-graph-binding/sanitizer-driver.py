"""One bounded public Rust session for an unfiltered Compute Sanitizer run."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from multigpubeamsearch import NativeModel, NativeOptions, NativeSession
from multigpubeamsearch.hamming import write_hamming_artifact
from public_model_snapshot_gpu import graph_fixture, start_state

parser = argparse.ArgumentParser()
for name in ('source', 'cutlass', 'cache', 'out'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
root = args.out.resolve(); root.mkdir(parents=True, exist_ok=False)
graph, contract, moves = graph_fixture()
start = start_state(contract.center, moves, 3)
model = NativeModel.for_graph(graph, write_hamming_artifact(contract, root))
options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass, cache_dir=args.cache,
    num_gpus=1, inference_backend='cutlass', autotune=False, report_calibration=False,
    build_jobs=2, build_timeout_seconds=1800, timeout_seconds=300)
with NativeSession(engine='rust') as session:
    result = session.beam_search(graph, native_options=options, start_state=start,
        predictor=model, beam_width=64, max_steps=4, return_path=True)
    meta = result.native_metadata
    assert result.path_found and replace(contract, start=start).replay(result.path)
    assert meta['effective_beam_width'] == 64 and meta['graph_hash'] == contract.graph_hash
    pids = meta['persistent_session']['worker_pids']
    assert len(pids) == 1
assert all(not Path(f'/proc/{pid}').exists() for pid in pids)
(root / 'result.json').write_text(json.dumps(dict(world=1, beam=64, max_steps=4,
    graph_hash=contract.graph_hash, start=start, path=result.path, pids=pids,
    run_dir=meta['run_dir']), indent=2) + '\n')
print('GRAPH_BINDING_SANITIZER_SEARCH_OK', flush=True)
