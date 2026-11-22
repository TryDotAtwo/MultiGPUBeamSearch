"""Five matched full public Rust-session calls with NCCL identity A/B."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

from multigpubeamsearch import NativeModel, NativeOptions, NativeSession
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.hamming import write_hamming_artifact


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "cutlass", "cache", "out"):
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
    graph = SimpleNamespace(device="cpu", definition=SimpleNamespace(
        state_size=96, central_state=center, generators_permutations=moves,
        generator_names=[f"m{i}" for i in range(24)], is_permutation_group=lambda: True))
    contract = GraphContract.from_graph(graph, center)
    model = NativeModel.for_graph(graph, write_hamming_artifact(contract, root))
    start = center
    for move in moves[:12]:
        start = tuple(start[i] for i in move)
    solvable = center
    for move in moves[:3]:
        solvable = tuple(solvable[i] for i in move)
    options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass,
        cache_dir=args.cache, num_gpus=2, inference_backend="cutlass", autotune=False,
        report_calibration=False, build_jobs=2, build_timeout_seconds=1200, timeout_seconds=180)
    sessions = {
        "baseline": NativeSession(engine="rust"),
        "parallel": NativeSession(engine="rust", experimental_parallel_nccl_identity=True),
    }
    rows = []

    def run(mode, phase, pair, state):
        before = time.monotonic()
        result = sessions[mode].beam_search(graph, native_options=options, start_state=state,
            predictor=model, beam_width=4096, max_steps=6, return_path=True)
        wall = time.monotonic() - before
        meta = result.native_metadata
        assert meta["effective_beam_width"] == 4096
        assert meta["persistent_session"]["reused"] == (phase != "cold")
        if phase == "solved":
            assert result.path_found and replace(contract, start=state).replay(result.path)
        else:
            assert not result.path_found and result.path is None
        row = dict(mode=mode, phase=phase, pair=pair, wall_seconds=wall,
            worker_seconds=meta["elapsed_seconds"], run_dir=meta["run_dir"], path=result.path,
            pids=meta["persistent_session"]["worker_pids"])
        rows.append(row)
        print(f"NCCL_PAIR {phase} pair={pair} mode={mode} wall={wall:.6f}", flush=True)
        return result.path

    try:
        for mode in ("baseline", "parallel"):
            run(mode, "cold", None, start)
        for mode in ("baseline", "parallel"):
            run(mode, "warmup", None, start)
        for pair in range(5):
            order = ("baseline", "parallel") if pair % 2 == 0 else ("parallel", "baseline")
            for mode in order:
                run(mode, "repeat", pair, start)
        paths = [run(mode, "solved", None, solvable) for mode in ("baseline", "parallel")]
        assert paths[0] == paths[1]
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    finally:
        for session in sessions.values():
            session.close()
    for mode in sessions:
        pids = [row["pids"] for row in rows if row["mode"] == mode]
        assert all(ids == pids[0] for ids in pids)
        assert all(not Path(f"/proc/{pid}").exists() for pid in pids[0])
    medians = {mode: statistics.median(row["wall_seconds"] for row in rows
               if row["mode"] == mode and row["phase"] == "repeat") for mode in sessions}
    (root / "results.json").write_text(json.dumps(dict(
        scope="five alternating pairs of complete six-depth two-GPU public calls; cold calls share a build cache",
        width=4096, max_steps=6, num_gpus=2, model="synthetic Hamming FP16",
        median_wall_seconds=medians, gpu_memory_mib_with_both_sessions=memory, rows=rows), indent=2))
    print("NCCL_IDENTITY_PAIR_GPU_OK", flush=True)


if __name__ == "__main__":
    main()
