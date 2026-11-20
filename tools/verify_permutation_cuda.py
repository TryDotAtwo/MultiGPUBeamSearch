#!/usr/bin/env python3
"""Exercise the installed Rust/CUDA permutation API on one and two physical GPUs."""

from __future__ import annotations

import argparse
from collections import deque
import itertools
import json
from pathlib import Path

import cayleypy as cp
import torch
from multigpubeamsearch import NativeModel, NativeOptions, NativeSession, beam_search
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.results import resolve_move_path


def move(state, generator):
    return tuple(state[source] for source in generator)


def distance(start, target, generators):
    queue = deque([(tuple(start), 0)])
    seen = {tuple(start)}
    while queue:
        state, depth = queue.popleft()
        if state == tuple(target):
            return depth
        for generator in generators:
            child = move(state, generator)
            if child not in seen:
                seen.add(child)
                queue.append((child, depth + 1))
    return None


def fixture(contract):
    # Select a deterministic reachable state whose exact distance is at least two.
    for length in range(2, 6):
        for sequence in itertools.product(range(contract.move_count), repeat=length):
            state = tuple(contract.center)
            for index in sequence:
                state = move(state, contract.generators[index])
            exact = distance(state, contract.center, contract.generators)
            if exact is not None and 2 <= exact <= 4:
                return state, exact, list(sequence)
    raise AssertionError("the LRX(4) fixture did not provide a distance-2-to-4 state")


def write_zero_mlp(root, contract):
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "state_len": contract.state_len,
        "num_classes": contract.num_classes,
        "hd1": 16,
        "hd2": 8,
        "nrd": 1,
        "output_dim": 1,
        "dtype": "fp16",
        "normalization": "batchnorm_folded",
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    sizes = {
        "input_weight_hxk": 4 * contract.num_classes * 16,
        "input_bias": 16,
        "hidden_weight_hxk": 16 * 8,
        "hidden_bias": 8,
        "output_weight_hxk": 8,
        "output_bias": 1,
        "residual0_fc1_weight_hxk": 8 * 8,
        "residual0_fc1_bias": 8,
        "residual0_fc2_weight_hxk": 8 * 8,
        "residual0_fc2_bias": 8,
    }
    for name, count in sizes.items():
        (root / f"{name}.fp16").write_bytes(b"\x00\x00" * count)


def run_layout(graph, contract, model, source_dir, cache_dir, devices, exact_distance, scramble):
    options = NativeOptions(
        source_dir=source_dir,
        cache_dir=cache_dir,
        devices=devices,
    )
    with NativeSession(engine="rust") as session:
        if session._engine != "rust":
            raise AssertionError("NativeSession did not select the Rust owner")
        results = []
        worker_pids = None
        for call_index in range(2):
            result = beam_search(
                graph,
                backend="native",
                native_options=options,
                predictor=model,
                start_state=list(contract.start),
                beam_width=64,
                max_steps=exact_distance,
                return_path=True,
            )
            if result.path_found is not True:
                raise AssertionError(f"Rust permutation search did not solve on devices {devices}")
            path = resolve_move_path(graph, result.path)
            replayed = tuple(contract.start)
            for move_index in path:
                replayed = move(replayed, contract.generators[move_index])
            if replayed != tuple(contract.center):
                raise AssertionError("public result path failed independent permutation replay")
            if len(path) != exact_distance:
                raise AssertionError(
                    f"expected exact distance {exact_distance}, received {len(path)}"
                )
            metadata_devices = tuple(result.native_metadata.get("devices", ()))
            if metadata_devices != tuple(devices):
                raise AssertionError(
                    f"result reports devices {metadata_devices}, requested {tuple(devices)}"
                )
            workers = getattr(session, "_workers", None)
            pids = tuple(workers.pids) if workers is not None else ()
            if len(pids) != len(devices) or len(set(pids)) != len(pids):
                raise AssertionError(
                    f"expected one distinct Rust rank worker per device: "
                    f"devices={tuple(devices)}, pids={pids}"
                )
            if worker_pids is None:
                worker_pids = pids
            elif worker_pids != pids:
                raise AssertionError("repeated search replaced the Rust rank workers")
            results.append(
                {
                    "call": call_index + 1,
                    "path": list(path),
                    "length": len(path),
                    "replay_valid": True,
                    "backend": result.native_metadata.get("backend"),
                    "inference_backend": result.native_metadata.get("inference_backend"),
                    "devices": list(metadata_devices),
                    "elapsed_seconds": result.native_metadata.get("elapsed_seconds"),
                }
            )
        return {
            "world_size": len(worker_pids),
            "requested_devices": list(devices),
            "devices": list(devices),
            "gpu_names": [torch.cuda.get_device_name(device) for device in devices],
            "engine": "rust",
            "exact_distance": exact_distance,
            "scramble": scramble,
            "worker_pids": list(worker_pids),
            "worker_pids_reused": True,
            "searches": results,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("the acceptance fixture requires the existing two-GPU CUDA instance")

    graph = cp.CayleyGraph(cp.PermutationGroups.lrx(4), device="cpu")
    base = GraphContract.from_graph(graph, graph.definition.central_state)
    start, exact, scramble = fixture(base)
    contract = GraphContract.from_graph(graph, start)

    model_dir = args.cache_dir.resolve() / "zero-mlp"
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    write_zero_mlp(model_dir, contract)
    model = NativeModel(model_dir, contract.graph_hash)

    reports = [
        run_layout(graph, contract, model, args.source_dir.resolve(), args.cache_dir.resolve(), (0,), exact, scramble),
        run_layout(graph, contract, model, args.source_dir.resolve(), args.cache_dir.resolve(), (0, 1), exact, scramble),
    ]
    report = {
        "fixture": "cayleypy.LRX(4)",
        "state_len": contract.state_len,
        "move_count": contract.move_count,
        "model": "known finite FP16 MLP artifact",
        "python": __import__("sys").version.split()[0],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
