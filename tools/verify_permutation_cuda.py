#!/usr/bin/env python3
"""Exercise the installed Rust/CUDA permutation API on one and two physical GPUs."""

from __future__ import annotations

import argparse
from collections import deque
import itertools
import json
import os
import re
from pathlib import Path

import cayleypy as cp
import torch
from multigpubeamsearch import NativeModel, NativeOptions, NativeSession, beam_search
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.results import resolve_move_path
from multigpubeamsearch.build import shape_contract
from permutation_acceptance import (
    build_saturated_reference_fixture,
    decode_frontier_snapshot,
    hash_state,
    validate_depth_flow_trace,
    permutation_report_sha256,
)


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


def run_layout(graph, contract, model, source_dir, cache_dir, cutlass_dir, devices, exact_distance, scramble):
    options = NativeOptions(
        source_dir=source_dir,
        cache_dir=cache_dir,
        cutlass_dir=cutlass_dir,
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


def run_saturated_layout(graph, contract, model, source_dir, cache_dir, cutlass_dir, devices, fixture, *, source_sha, wheel_sha256, run_id):
    """Run the B=3 cutoff twice in one Rust session and validate rank-local CUDA evidence."""
    world_size = len(devices)
    case_id = f"permutation-saturated-b3-world-{world_size}"
    identity = {
        "MGBS_ACCEPTANCE_RANK_TRACE": "1",
        "MGBS_ACCEPTANCE_RUN_ID": run_id,
        "MGBS_ACCEPTANCE_CASE_ID": case_id,
        "MGBS_ACCEPTANCE_SOURCE_SHA": source_sha,
        "MGBS_ACCEPTANCE_WHEEL_SHA256": wheel_sha256,
    }
    old_environment = {name: os.environ.get(name) for name in identity}
    os.environ.update(identity)
    try:
        options = NativeOptions(source_dir=source_dir, cache_dir=cache_dir, cutlass_dir=cutlass_dir, devices=devices)
        expected = fixture.worlds[world_size].steps[1]
        if expected.keep_count != 3 or expected.eligible_distinct <= 3:
            raise AssertionError("CPU oracle did not produce a saturated B=3 global cutoff")
        with NativeSession(engine="rust") as session:
            if session._engine != "rust":
                raise AssertionError("saturated search did not select the Rust session owner")
            reports = []
            worker_pids = None
            rank_identities = None
            for call_id in (1, 2):
                result = beam_search(
                    graph, backend="native", native_options=options, predictor=model,
                    start_state=list(fixture.start), beam_width=3, max_steps=2, return_path=True,
                )
                if result.path_found is True or result.path is not None:
                    raise AssertionError("distance-at-least-three fixture unexpectedly solved within two steps")
                metadata = result.native_metadata
                build = metadata.get("build")
                if not isinstance(build, dict) or build.get("acceptance_rank_trace") is not True:
                    raise AssertionError("Rust result lacks acceptance-instrumented runner provenance")
                build_key = build.get("build_key")
                binary_sha256 = build.get("binary_sha256")
                log_value = metadata.get("log_path")
                if not isinstance(log_value, str) or not log_value:
                    run_dir_value = metadata.get("run_dir")
                    if isinstance(run_dir_value, str) and run_dir_value:
                        log_value = str(Path(run_dir_value) / "native.log")
                log_path = Path(log_value or "")
                if not log_path.is_file():
                    raise AssertionError(f"per-rank native log is missing: {log_path}")
                rows = validate_depth_flow_trace(
                    log_path.read_text(encoding="utf-8", errors="replace"),
                    run_id=run_id, case_id=case_id, call_id=call_id,
                    world_size=world_size, depth=1, beam_width=3,
                    source_sha=source_sha, wheel_sha256=wheel_sha256,
                    build_key=build_key, binary_sha256=binary_sha256,
                    require_frontier_snapshot=True,
                )
                if any(int(row["shard_count"]) != 64 for row in rows):
                    raise AssertionError("the accepted search did not use the requested 64 shards")
                if any(int(row["final_global_less"]) != 0 for row in rows):
                    raise AssertionError("equal-score fixture unexpectedly has candidates below its cutoff")
                if any(int(row["final_global_equal"]) != expected.eligible_distinct for row in rows):
                    raise AssertionError("global equal-score count differs from the independent distinct oracle")
                if any(int(row["final_total_available"]) != expected.eligible_distinct for row in rows):
                    raise AssertionError("global candidate total differs from the independent distinct oracle")
                if any(int(row["final_global_keep"]) != 3 for row in rows):
                    raise AssertionError("the global winner count differs from the requested B=3")
                if [int(row["next_frontier_size"]) for row in rows] != [len(frontier) for frontier in expected.frontier_by_rank]:
                    raise AssertionError("rank-local winner counts differ from the CPU partition")
                if sum(int(row["final_candidate_count"]) for row in rows) != expected.eligible_distinct:
                    raise AssertionError("rank candidate counts differ from the independent CPU dedup oracle")
                if sum(int(row["final_request_count"]) for row in rows) != 3:
                    raise AssertionError("rank final requests do not sum to the exact B=3 global cutoff")
                if [int(row["frontier_size"]) for row in rows] != list(expected.input_counts):
                    raise AssertionError("rank input counts differ from the CPU frontier partition")
                if [int(row["generated"]) for row in rows] != [count * contract.move_count for count in expected.input_counts]:
                    raise AssertionError("rank expansion counts differ from frontier size times move count")
                observed_frontier = []
                for rank, row in enumerate(rows):
                    raw = Path(str(row["frontier_snapshot"])).read_bytes()
                    states = decode_frontier_snapshot(raw, state_len=contract.state_len,
                                                     storage_len=shape_contract(contract)["storage_len"])
                    expected_states = tuple(candidate.state for candidate in expected.frontier_by_rank[rank])
                    if states != expected_states:
                        raise AssertionError(f"rank {rank} frontier differs from the independent CPU winner oracle")
                    observed_frontier.extend(states)
                expected_order = tuple(candidate.state for candidate in expected.ordered_survivors)
                if tuple(observed_frontier) != expected_order:
                    raise AssertionError("observed rank frontiers do not preserve the global survivor order")
                for candidate in expected.ordered_survivors:
                    replayed = tuple(fixture.start)
                    for move_index in candidate.path:
                        replayed = move(replayed, contract.generators[move_index])
                    if replayed != candidate.state:
                        raise AssertionError("independent replay failed for a retained CPU-oracle candidate")
                workers = getattr(session, "_workers", None)
                pids = tuple(workers.pids) if workers is not None else ()
                if len(pids) != world_size or len(set(pids)) != world_size:
                    raise AssertionError(f"expected one Rust rank worker per device; got {pids}")
                trace_pids = tuple(int(row["pid"]) for row in rows)
                current_rank_identities = tuple((int(row["pid"]), str(row["gpu_uuid"])) for row in rows)
                if trace_pids != pids:
                    raise AssertionError("rank trace PIDs do not match the Rust session worker processes")
                if worker_pids is None:
                    worker_pids = pids
                    rank_identities = current_rank_identities
                elif worker_pids != pids:
                    raise AssertionError("Rust rank workers changed between repeated session calls")
                elif current_rank_identities != rank_identities:
                    raise AssertionError("repeated Rust session calls changed rank-to-device identity")
                reused = getattr(workers, "_last_reused", None) if workers is not None else None
                if call_id == 2 and reused is not True:
                    raise AssertionError("second saturated search did not reuse the existing Rust workers")
                reports.append({
                    "call_id": call_id, "session_reused": bool(reused),
                    "worker_pids": list(pids), "build_key": build_key,
                    "binary_sha256": binary_sha256, "log_path": str(log_path),
                    "eligible_distinct": expected.eligible_distinct, "beam_width": 3,
                    "cross_source_duplicates": expected.cross_source_duplicate_count,
                    "rank_evidence": [
                        {key: row[key] for key in ("rank", "pid", "gpu_uuid", "frontier_size", "generated", "final_global_less", "final_global_equal", "final_total_available", "final_global_keep", "final_candidate_count", "final_request_count", "next_frontier_size", "frontier_snapshot")}
                        for row in rows
                    ],
                    "ordered_survivor_hashes": [
                        {"lo": f"{candidate.hash128[0]:016x}", "hi": f"{candidate.hash128[1]:016x}"}
                        for candidate in expected.ordered_survivors
                    ],
                })
            return {
                "case_id": case_id, "world_size": world_size,
                "devices": list(devices), "worker_pids": list(worker_pids or ()),
                "gpu_names": [torch.cuda.get_device_name(device) for device in devices],
                "fixture_start": list(fixture.start), "scramble": list(fixture.scramble),
                "exact_distance": fixture.distance, "input_counts": list(expected.input_counts),
                "eligible_distinct": expected.eligible_distinct, "keep_count": expected.keep_count,
                "cross_source_duplicates": expected.cross_source_duplicate_count,
                "ordered_survivor_hashes": reports[0]["ordered_survivor_hashes"],
                "calls": reports, "independent_replay": True,
            }
    finally:
        for name, old_value in old_environment.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cutlass-dir", type=Path, required=True,
                        help="verified, pinned CUTLASS source prepared before the GPU smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--acceptance-run-id", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is None:
        raise SystemExit("--source-sha must be a lowercase 40-digit Git SHA")
    if re.fullmatch(r"[0-9a-f]{64}", args.wheel_sha256) is None:
        raise SystemExit("--wheel-sha256 must be a lowercase SHA256")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", args.acceptance_run_id) is None:
        raise SystemExit("--acceptance-run-id must be a portable nonempty label")

    if not (args.cutlass_dir / "include/cutlass/cutlass.h").is_file():
        raise SystemExit(f"verified CUTLASS headers are missing: {args.cutlass_dir}")

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
    saturated_graph = cp.CayleyGraph(cp.PermutationGroups.lrx(5), device="cpu")
    saturated_base = GraphContract.from_graph(
        saturated_graph, saturated_graph.definition.central_state
    )
    saturated = build_saturated_reference_fixture(
        saturated_base.center, saturated_base.generators, beam_width=3, shard_count=64
    )
    saturated_contract = GraphContract.from_graph(saturated_graph, saturated.start)
    saturated_model_dir = args.cache_dir.resolve() / "zero-mlp-saturated-lrx5"
    write_zero_mlp(saturated_model_dir, saturated_contract)
    saturated_model = NativeModel(saturated_model_dir, saturated_contract.graph_hash)

    reports = [
        run_layout(graph, contract, model, args.source_dir.resolve(), args.cache_dir.resolve(),
                   args.cutlass_dir.resolve(), (0,), exact, scramble),
        run_layout(graph, contract, model, args.source_dir.resolve(), args.cache_dir.resolve(),
                   args.cutlass_dir.resolve(), (0, 1), exact, scramble),
    ]
    saturated_reports = [
        run_saturated_layout(saturated_graph, saturated_contract, saturated_model, args.source_dir.resolve(),
                             args.cache_dir.resolve(), args.cutlass_dir.resolve(), (0,), saturated,
                             source_sha=args.source_sha, wheel_sha256=args.wheel_sha256,
                             run_id=args.acceptance_run_id),
        run_saturated_layout(saturated_graph, saturated_contract, saturated_model, args.source_dir.resolve(),
                             args.cache_dir.resolve(), args.cutlass_dir.resolve(), (0, 1), saturated,
                             source_sha=args.source_sha, wheel_sha256=args.wheel_sha256,
                             run_id=args.acceptance_run_id),
    ]
    if saturated_reports[0]["ordered_survivor_hashes"] != saturated_reports[1]["ordered_survivor_hashes"]:
        raise AssertionError("one- and two-GPU saturated searches selected different ordered survivor hashes")
    report = {
        "fixture": "cayleypy.LRX(4)",
        "saturated_fixture": "cayleypy.LRX(5)",
        "state_len": contract.state_len,
        "move_count": contract.move_count,
        "saturated_state_len": saturated_contract.state_len,
        "saturated_move_count": saturated_contract.move_count,
        "model": "known finite FP16 MLP artifact",
        "source_sha": args.source_sha,
        "wheel_sha256": args.wheel_sha256,
        "acceptance_run_id": args.acceptance_run_id,
        "python": __import__("sys").version.split()[0],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "runs": reports,
        "saturated_b3": saturated_reports,
    }
    report["report_sha256"] = permutation_report_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
