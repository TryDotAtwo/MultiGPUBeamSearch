"""Solve a Cube555 instance through CayleyPy with a native two-GPU blend."""

import argparse
import json
import tempfile
from pathlib import Path


def beam_width(value):
    if value == "max":
        return value
    try:
        width = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use max or a positive integer") from error
    if width < 1:
        raise argparse.ArgumentTypeError("use max or a positive integer")
    return width


def profile_record(result):
    native = result.native_metadata
    profile = native["profile"]
    inference = profile["inference_calibration"]
    pipeline = profile["pipeline_calibration"]
    if not pipeline.get("selection") or not pipeline.get("environment"):
        raise RuntimeError("Native calibration did not select a pipeline profile")
    return {
        "requested_beam": native["requested_beam_width"],
        "effective_beam": native["effective_beam_width"],
        "inference_parent_batch": inference["parent_batch"],
        "pipeline_selection": pipeline["selection"],
        "environment": pipeline["environment"],
        "inference_cache_hit": inference.get("cache_hit", False),
        "pipeline_cache_hit": pipeline.get("cache_hit", False),
        "measurement_scope": pipeline.get("measurement_scope", pipeline["phase"]),
        "run_dir": native["run_dir"],
    }


def prepare_profile_grid(graph, max_result, search, cache_dir):
    native = max_result.native_metadata
    effective_max = native["effective_beam_width"]
    anchors = [1 << exponent for exponent in range(16, effective_max.bit_length())
               if 1 << exponent < effective_max]
    # Native cache admission still runs whenever an anchor is used for search.
    signature = dict(native["profile"]["inference_calibration"]["signature"])
    signature.pop("beam_width", None)
    identity = {"signature": signature, "bfs_radius": search["bfs_radius"], "anchors": anchors}
    index = cache_dir / "profile-grid.json"
    try:
        saved = json.loads(index.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        saved = {}
    if not isinstance(saved, dict) or saved.get("identity") != identity:
        saved = {"identity": identity, "profiles": {}}
    profiles = saved["profiles"]
    reused = set(profiles) == set(map(str, anchors))
    saved["max"] = profile_record(max_result)
    saved["grid_reused"] = reused

    # Write each completed point so an interrupted calibration can resume.
    cache_dir.mkdir(parents=True, exist_ok=True)
    for width in anchors:
        if str(width) in profiles:
            continue
        print(f"Calibrating beam {width:,}", flush=True)
        probe = graph.beam_search(
            **search,
            beam_width=width,
            max_steps=search["bfs_radius"] + 2,
            solution_mode="first",
        )
        profiles[str(width)] = profile_record(probe)
        save_grid(index, saved)
    save_grid(index, saved)
    print(f"{'Reused' if reused else 'Prepared'} profile grid ({len(anchors) + 1} widths): {index}")


def save_grid(index, data):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=index.parent, delete=False,
    ) as output:
        temporary = Path(output.name)
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")
    temporary.replace(index)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle-id", default="1020")
    parser.add_argument("--beam-width", type=beam_width, default="max")
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--transformer-weight", type=float, default=0.8)
    parser.add_argument("--bfs-radius", type=int, default=5)
    parser.add_argument("--collect", action="store_true", help="Collect replayable alternatives")
    parser.add_argument("--work-dir", type=Path, default=Path("beam_run"))
    parser.add_argument("--assets", type=Path, default=Path(__file__).resolve().parent / "assets")
    parser.add_argument("--check", action="store_true", help="Check local inputs without querying CUDA")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
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
    cache_dir = work_dir / "cache"
    enable_native(NativeOptions(num_gpus=2, cutlass_dir=cutlass_dir,
                                cache_dir=cache_dir, autotune=True,
                                calibration_pipeline_seconds=3600))
    search = dict(
        start_state=start, predictor=predictor, backend="native",
        use_child_scores=True, return_path=True,
        bfs_radius=min(args.bfs_radius, args.max_steps - 1),
        bfs_max_entries=30_000_000,
    )
    result = graph.beam_search(
        **search, beam_width=args.beam_width, max_steps=args.max_steps,
        solution_mode="collect" if args.collect else "first",
    )
    saved = save_result(result, graph, start, work_dir / "results",
                        metadata={"puzzle_id": args.puzzle_id, "blend": [w, 1 - w]})
    print(f"Puzzle {args.puzzle_id}  found={result.path_found}  length={result.path_length}")
    print(f"Result and profile: {saved}")
    if args.beam_width == "max" and result.native_metadata.get("effective_beam_width"):
        # Calibration depends on graph/model, not on the chosen start.
        prepare_profile_grid(graph, result, dict(search, start_state=starts["1020"]), cache_dir)


if __name__ == "__main__":
    main()
