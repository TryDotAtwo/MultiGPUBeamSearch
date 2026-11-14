"""Bounded public Rust-session check of CayleyPy-compatible graph identity."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from multigpubeamsearch import NativeModel, NativeOptions, NativeSession
from multigpubeamsearch.hamming import write_hamming_artifact
from public_model_snapshot_gpu import graph_fixture, start_state


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "cutlass", "cache", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    graph, contract, moves = graph_fixture()
    model = NativeModel.for_graph(graph, write_hamming_artifact(contract, root))
    rows = []
    for world in (1, 2):
        options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass,
            cache_dir=args.cache, num_gpus=world, inference_backend="cutlass",
            autotune=False, report_calibration=False, build_jobs=2,
            build_timeout_seconds=1800, timeout_seconds=180)
        pids = None
        with NativeSession(engine="rust") as session:
            for count in (3, 4):
                state = start_state(contract.center, moves, count)
                result = session.beam_search(graph, native_options=options,
                    start_state=state, predictor=model, beam_width=64,
                    max_steps=4, return_path=True)
                meta = result.native_metadata
                resident = meta["persistent_session"]
                assert meta["graph_hash"] == contract.graph_hash
                assert meta["effective_beam_width"] == 64
                assert result.path_found and replace(contract, start=state).replay(result.path)
                assert resident["reused"] == (pids is not None)
                assert len(resident["worker_pids"]) == world
                if pids is not None:
                    assert resident["worker_pids"] == pids
                pids = resident["worker_pids"]
                rows.append(dict(world=world, start=state, path=result.path,
                    graph_hash=contract.graph_hash, pids=pids,
                    reused=resident["reused"], run_dir=meta["run_dir"]))
                print(f"RUST_GRAPH_GPU world={world} moves={count} replay=ok", flush=True)
        assert all(not Path(f"/proc/{pid}").exists() for pid in pids)
    (root / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("RUST_GRAPH_GPU_OK", flush=True)


if __name__ == "__main__":
    main()
