"""Solve a Cube555 instance through CayleyPy with a native two-GPU blend."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle-id", default="1020")
    parser.add_argument("--beam-width", type=int, default=1_048_576)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--transformer-weight", type=float, default=0.8)
    parser.add_argument("--bfs-radius", type=int, default=5)
    parser.add_argument("--collect", action="store_true", help="Collect replayable alternatives")
    parser.add_argument("--work-dir", type=Path, default=Path("beam_run"))
    parser.add_argument("--assets", type=Path, default=Path(__file__).resolve().parent / "assets")
    parser.add_argument("--check", action="store_true", help="Check local inputs without querying CUDA")
    args = parser.parse_args(argv)
    if args.beam_width < 1 or args.max_steps < 1:
        parser.error("--beam-width and --max-steps must be positive")
    if not 0.0 <= args.transformer_weight <= 1.0:
        parser.error("--transformer-weight must be between 0 and 1")
    if not 0 <= args.bfs_radius <= 5:
        parser.error("--bfs-radius must be between 0 and 5")

    assets = args.assets.resolve()
    puzzle_file = assets / "puzzle_info.json"
    starts_file = assets / "test.csv"
    cutlass_dir = assets / "cutlass"
    transformer_dir = assets / "models" / "cube555_transformer"
    mlp_dir = assets / "models" / "cube555_mlp"
    for path in (puzzle_file, starts_file, cutlass_dir / "include/cutlass/cutlass.h",
                 transformer_dir / "manifest.json", mlp_dir / "manifest.json"):
        if not path.is_file():
            parser.error(f"Bundle file missing: {path}")

    from cayleypy import CayleyGraph, CayleyGraphDef
    from multigpubeamsearch import (
        NativeEnsemble, NativeModel, NativeOptions, enable_native, read_states, save_result,
    )
    import torch

    problem = json.loads(puzzle_file.read_text(encoding="utf-8"))
    generators = problem["generators"]
    graph = CayleyGraph(
        CayleyGraphDef.create(
            list(generators.values()),
            generator_names=list(generators),
            central_state=problem["central_state"],
        ),
        device="cpu",
    )
    starts = read_states(starts_file, id_column="initial_state_id", state_column="initial_state")
    if args.puzzle_id not in starts:
        parser.error(f"Unknown puzzle ID: {args.puzzle_id}")
    start = starts[args.puzzle_id]
    if args.check:
        print(f"Local Cube555 inputs ready. Puzzle {args.puzzle_id}, {len(start)} stickers.")
        return

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise SystemExit(f"Two visible CUDA GPUs are required (2); PyTorch sees {gpu_count}.")
    print("CUDA devices:", ", ".join(torch.cuda.get_device_name(i) for i in range(2)), flush=True)

    transformer = NativeModel.for_graph(graph, transformer_dir, backend="cube555_transformer")
    mlp = NativeModel.for_graph(graph, mlp_dir, backend="cube555_mlp")
    w = args.transformer_weight
    predictor = NativeEnsemble((transformer, mlp), (w, 1 - w))

    work_dir = args.work_dir.resolve()
    enable_native(NativeOptions(num_gpus=2, cutlass_dir=cutlass_dir,
                                cache_dir=work_dir / "cache", autotune=True))
    result = graph.beam_search(
        start_state=start,
        predictor=predictor,
        backend="native",
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        use_child_scores=True,
        return_path=True,
        solution_mode="collect" if args.collect else "first",
        bfs_radius=args.bfs_radius,
        bfs_max_entries=30_000_000,
    )
    saved = save_result(result, graph, start, work_dir / "results",
                        metadata={"puzzle_id": args.puzzle_id, "blend": [w, 1 - w]})
    print(f"Puzzle {args.puzzle_id}  found={result.path_found}  length={result.path_length}")
    print(f"Result and profile: {saved}")


if __name__ == "__main__":
    main()
