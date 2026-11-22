"""Compare supervisors with the real two-GPU native search (no timing claims)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from multigpubeamsearch import NativeOptions
from multigpubeamsearch.backend import (microbatch_environment, parse_terminal, prepare_runtime,
                                        snapshot_runtime_nccl)
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.models import prepare_model


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "cutlass", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    center = tuple(i // 16 for i in range(96))
    moves = []
    for i in range(24):
        move = list(range(96))
        move[i], move[i + 24] = move[i + 24], move[i]
        moves.append(tuple(move))
    definition = SimpleNamespace(state_size=96, central_state=center,
        generators_permutations=moves, generator_names=[f"m{i}" for i in range(24)],
        is_permutation_group=lambda: True)
    graph = SimpleNamespace(definition=definition, device="cpu")
    start = center
    for move in moves[:3]:
        start = tuple(start[index] for index in move)
    contract = GraphContract.from_graph(graph, start)
    prepared = root / "prepared"
    prepared.mkdir()
    options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass,
        cache_dir=root / "cache", num_gpus=2, inference_backend="cutlass", autotune=False,
        report_calibration=False, build_jobs=4, build_timeout_seconds=1200, timeout_seconds=120)
    model = prepare_model(None, contract, options, prepared)
    runtime = prepare_runtime(contract, model, options, prepared, (0, 1), touch_bfs_radius=0)
    puzzle = prepared / "puzzle_info.json"
    puzzle.write_text(json.dumps(contract.to_puzzle_info()) + "\n")
    inputs = prepared / "test.csv"
    inputs.write_text('initial_state_id,initial_state\n0,"' + ",".join(map(str, start)) + '"\n')
    nccl_library = snapshot_runtime_nccl(runtime, prepared)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1",
        BEAM_GENERATOR_PATH=str(puzzle), BEAM_PUZZLE_INFO_JSON=str(puzzle),
        BEAM_TEST_CSV=str(inputs), BEAM_WEIGHT_DIR=str(model.weights_dir),
        BEAM_RUNTIME_CONFIG_MODE="auto", BEAM_SOLVED_NEIGHBORHOOD_RADIUS="0",
        BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES=str(options.touch_bfs_max_entries),
        BEAM_STREAM2_SUFFIX_RADIUS="0", BEAM_SOLVE_BUCKET_MODE="0")
    env["LD_LIBRARY_PATH"] = str(nccl_library.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env.update(microbatch_environment(contract, model, 64, 2)[0])
    records = []
    for engine in ("python", "rust"):
        directory = root / engine
        command = ["torchrun", "--standalone", "--nproc-per-node=2", "-m",
                   "multigpubeamsearch.cluster_worker", "--shared-dir", str(directory),
                   "--timeout", "120", "--engine", engine, "--", str(runtime.runner), "0", "4", "64"]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        (root / f"{engine}-launcher.log").write_text(result.stdout + "\n" + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{engine} launcher failed: {result.returncode}")
        logs = sorted(directory.glob("*/attempt-0/rank-*/native.log"))
        assert len(logs) == 2, logs
        combined = root / f"{engine}-combined.log"
        combined.write_text("\n".join(path.read_text(errors="replace") for path in logs))
        path, width, metadata = parse_terminal(combined, contract, max_steps=4)
        assert path is not None and contract.replay(path) and width == 64, (path, width)
        records.append(dict(engine=engine, path=path, global_beam=width,
                            terminal_records=metadata["terminal_records"]))
    assert records[0]["path"] == records[1]["path"], records
    (root / "result.json").write_text(json.dumps({"records": records,
        "runner_sha256": runtime.build_metadata["binary_sha256"],
        "graph_hash": contract.graph_hash, "model_hash": model.artifact_hash}, indent=2) + "\n")
    print("CLUSTER_NATIVE_SEARCH_OK", flush=True)


if __name__ == "__main__":
    main()
