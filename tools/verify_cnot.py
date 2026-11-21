"""Small, exhaustive reference for issue #6 receipts. Run directly or under torchrun."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import struct

import torch
import torch.distributed as dist

from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
from multigpubeamsearch.cnot_acceptance import (
    EVENT_SCHEMA,
    EVENT_VERSION,
    acceptance_identity,
    emit_event,
    gpu_uuid_for_rank,
    report_sha256,
)
from multigpubeamsearch.cnot_cli import initialize_world
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_scorer import TQPScorer


def rows_of(lanes, n):
    words = (n + 63) // 64
    return tuple(sum((lanes[i + j] % (1 << 64)) << (64 * j) for j in range(words))
                 for i in range(0, len(lanes), words))


def gate_pairs(n):
    return [(source, target) for source in range(n) for target in range(n) if target != source]


def apply(rows, pair):
    c, t = pair
    result = list(rows)
    result[t] ^= result[c]
    return tuple(result)


def float_value(bits):
    return struct.unpack("=d", struct.pack("=q", bits))[0]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reference_cutoff(eligible, reported_scores, reference_scores, width):
    """Order the CPU child oracle and classify uncertainty at the global cutoff."""
    ordered = sorted(eligible, key=lambda rows: (reference_scores[rows], eligible[rows][1]))
    selected = ordered[:width]
    if len(ordered) <= width:
        return selected, "underfilled", None

    kept, excluded = ordered[width - 1], ordered[width]
    kept_score, excluded_score = reference_scores[kept], reference_scores[excluded]
    gap = excluded_score - kept_score
    if gap == 0:
        tied = [rows for rows in ordered if reference_scores[rows] == kept_score]
        exact = all(reported_scores[rows] == reference_scores[rows] for rows in tied)
        status = "exact_tie_full_word" if exact else "ambiguous_within_score_tolerance"
        return selected, status, gap

    # Match assert_close(rtol=1e-6, atol=1e-7) conservatively for each side.
    kept_tolerance = 1e-7 + 1e-6 * max(abs(kept_score), abs(reported_scores[kept]))
    excluded_tolerance = 1e-7 + 1e-6 * max(abs(excluded_score), abs(reported_scores[excluded]))
    status = ("reference_separated" if gap > kept_tolerance + excluded_tolerance
              else "ambiguous_within_score_tolerance")
    return selected, status, gap


def small_exact_distance(target, pairs, bound):
    start = tuple(1 << i for i in range(len(target)))
    frontier, seen = {start}, {start}
    for depth in range(bound + 1):
        if target in frontier:
            return depth
        if depth < bound:
            frontier = {apply(rows, pair) for rows in frontier for pair in pairs} - seen
            seen.update(frontier)
    raise AssertionError("target is outside the exact BFS ball")


def verify_layers(root, scorer, mode, width, k, layers, device):
    """Enumerate on Python integers. Compare the entire eligible union, path representatives,
    measured values, and every ordered survivor against the independent single-beam result."""
    n, closed = len(root), {root}
    pairs = gate_pairs(n)
    parents = [(root, ())]
    reports = []
    for layer in layers:
        if k < len(pairs):
            batch = torch.cat([pack_rows(rows, device=device) for rows, word in parents])
            head = scorer.parent_scores(batch, mode).tolist()
        eligible = {}
        for i, (rows, word) in enumerate(parents):
            actions = range(len(pairs))
            if k < len(pairs):
                actions = sorted(actions, key=lambda a: head[i][a], reverse=mode == "proposal")[:k]
            for action in actions:
                child = apply(rows, pairs[action])
                if child not in closed:
                    eligible.setdefault(child, (i, word + (action,)))
        measured = {}
        for bits, parent, _lex, _generation, route, *lanes in layer["candidates"]:
            child = rows_of(lanes, n)
            require(child not in measured and child in eligible, "duplicate or extraneous child")
            i, word = eligible[child]
            require(parent == i and (route & 65535) == word[-1], "wrong stable representative")
            measured[child] = float_value(bits)
        require(set(measured) == set(eligible), "candidate missing from the global union")
        reference_scores = {}
        if measured:
            children = torch.cat([pack_rows(rows, device=device) for rows in measured])
            actual = scorer.child_values(children).double().cpu()
            torch.testing.assert_close(actual, torch.tensor(list(measured.values()), dtype=torch.float64),
                                       rtol=1e-6, atol=1e-7)
            reference_scores = dict(zip(measured, (float(value) for value in actual.tolist())))
        expected = sorted(eligible, key=lambda rows: (measured[rows], eligible[rows][1]))[:width]
        expected_reference, reference_status, boundary_gap = reference_cutoff(
            eligible, measured, reference_scores, width
        )
        if reference_status in ("reference_separated", "exact_tie_full_word"):
            require(expected == expected_reference,
                    "reported-score order differs from the independent reference cutoff")
        winners = [rows_of(item[5:], n) for item in layer["winners"]]
        require(winners == expected, "global top B differs")
        digest = hashlib.sha256(json.dumps(winners, separators=(",", ":")).encode()).hexdigest()
        reference_digest = hashlib.sha256(
            json.dumps(expected_reference, separators=(",", ":")).encode()
        ).hexdigest()
        reports.append(dict(
            depth=layer["depth"], distinct=len(eligible), retained=len(winners),
            winners_sha256=digest, by_rank=layer["by_rank"],
            selection_oracle="reported_scores",
            reference_order_status=reference_status,
            reference_boundary_gap=boundary_gap,
            reference_ordered_winners_sha256=reference_digest,
        ))
        parents = [(rows, eligible[rows][1]) for rows in expected]
        closed.update(winners)
    return reports


class WireFixture:
    """Diagnostic scores forcing one wide routed move; never used as the project model."""
    def __init__(self, undo):
        self.undo = undo

    def parent_scores(self, packed, mode):
        alphabet = packed.size(1) * (packed.size(1) - 1)
        q = torch.arange(alphabet, device=packed.device, dtype=torch.float64)
        q[self.undo] = -1
        return (q if mode == "qhat" else -q).expand(len(packed), -1)

    def child_values(self, children):
        # Popcount distance to the identity, using an independent byte lookup.
        n = children.size(1)
        eye = pack_rows([1 << j for j in range(n)], device=children.device)
        diff = (children ^ eye).contiguous().view(torch.uint8).reshape(len(children), -1).long()
        lut = torch.tensor([i.bit_count() for i in range(256)], device=children.device)
        return lut[diff].sum(1).double()


def wide_case():
    n = 144
    first, second = 130 * 143 + 2, 2 * 143 + 142
    return replay_cnot(n, (first, second)), WireFixture(second)


def validate_case(rows, scorer, mode, width, k, depth, device, require_cross_rank=False, engine="auto"):
    result = search_cnot(pack_rows(rows, device=device), scorer, width=width, shortlist=k,
                         max_depth=depth, mode=mode, audit=True, engine=engine)
    with World(device) as world:
        def inspect():
            if world.rank:
                return None
            layers = verify_layers(rows, scorer, mode, width, k, result.layers, device)
            require(result.status == "solved", f"expected a solution, got {result.status}")
            pairs = gate_pairs(len(rows))
            replay = tuple(1 << i for i in range(len(rows)))
            for action in result.gates:
                replay = apply(replay, pairs[action])
            require(replay == rows, "independent construction does not match the input")
            if require_cross_rank and world.size > 1:
                require(any(edge["move"] > 255 and len({edge["source"], edge["hash_owner"], edge["target"]}) > 1
                            for edge in result.path_routes), "large physical move never crossed a rank")
            require(any(layer["distinct"] > width and layer["retained"] == width and
                        (world.size == 1 or sum(rank[3] > 0 for rank in layer["by_rank"]) >= 2)
                        for layer in layers), "missing a saturated layer with participating owners")
            report = dict(n=len(rows), width=width, shortlist=k, mode=mode, world=world.size,
                          scorer=type(scorer).__name__,
                          scorer_runtime=f"{type(scorer).__module__}.{type(scorer).__qualname__}",
                          root_sha256=hashlib.sha256(
                              json.dumps(rows, separators=(",", ":")).encode("utf-8")
                          ).hexdigest(),
                          precision_contract={"scorer_output_dtype": getattr(
                              getattr(scorer, "metadata", {}), "get", lambda _key, _default=None: None
                          )("scorer_output_dtype", "unspecified"), "selection_dtype": "float64"},
            engine=engine, model=getattr(scorer, "metadata", None),
            layers=layers, status=result.status,
            physical_ids=list(result.gates), directed_gates=[pairs[i] for i in result.gates],
                          length=result.length, bitwise_replay=True, routes=result.path_routes)
            if len(rows) == 4 and result.length <= 3:
                report["exact_distance_bfs"] = small_exact_distance(rows, pairs, result.length)
            return report
        return world.checked("independent validation", inspect)


def validate_distribution(device, engine="auto"):
    """Exercise every rank as a parent source, value owner and frontier recipient."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    n = 6
    rows = tuple(1 << i for i in reversed(range(n)))
    width, k = 17, 25
    if world_size > width:
        raise ValueError("this small all-rank fixture covers up to 17 workers")
    scorer = WireFixture(29)
    result = search_cnot(pack_rows(rows, device=device), scorer, width=width, shortlist=k,
                         max_depth=3, audit=True, engine=engine)
    with World(device) as world:
        def inspect():
            if world.rank:
                return None
            layers = verify_layers(rows, scorer, "qhat", width, k, result.layers, device)
            require(len(layers) == 2, "expected two complete expansions before the depth bound")
            per_rank = layers[1]["by_rank"]
            require(all(row[0] and row[3] and row[4] for row in per_rank),
                    "a rank did not expand parents, score distinct candidates or retain winners")
            require(layers[1]["retained"] == width and layers[1]["distinct"] > width, "unsaturated frontier")
            return dict(n=n, width=width, shortlist=k, mode="qhat", world=world.size,
                        scorer=type(scorer).__name__, engine=engine, status=result.status, layers=layers)
        return world.checked("all-rank validation", inspect)


def main(argv=None, *, scorer_factory=None, manage_process_group=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("trained", "trained-deep", "wide", "topology"), required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--mode", choices=("qhat", "proposal"), default="qhat")
    parser.add_argument("--width", type=int)
    parser.add_argument("--library")
    parser.add_argument("--manifest")
    parser.add_argument("--engine", choices=("auto", "python", "rust"), default="auto",
                        help="select the search controller; rust fails closed when unavailable")
    parser.add_argument("--require-rust-scorer", action="store_true",
                        help="require the Rust-owned checkpoint scorer and Rust controller")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.require_rust_scorer:
        if args.case not in ("trained", "trained-deep"):
            parser.error("--require-rust-scorer requires a trained checkpoint case")
        if args.engine == "python":
            parser.error("--require-rust-scorer cannot be combined with --engine python")
        args.engine = "rust"
    if manage_process_group:
        if dist.is_initialized():
            parser.error("verify_cnot.main owns process-group initialization when manage_process_group is true")
        device = initialize_world(args.backend)
    else:
        if not dist.is_initialized():
            parser.error("verify_cnot.main requires an initialized process group when manage_process_group is false")
        initialized_backend = str(dist.get_backend()).lower()
        if args.backend not in initialized_backend:
            parser.error(
                f"requested backend {args.backend!r} differs from initialized process group {initialized_backend!r}"
            )
        device = torch.device("cuda", torch.cuda.current_device()) if "nccl" in initialized_backend else torch.device("cpu")
    try:
        with World(device) as world:
            identity = world.checked("prepare CNOT acceptance identity", acceptance_identity)
            identity_contract = None if identity is None else {
                "run_id": identity["run_id"],
                "case_id": identity["case_id"],
                "source_sha": identity["source_sha"],
                "wheel_sha256": identity["wheel_sha256"],
                "gpu_uuids": identity["gpu_uuids"],
                "evidence_dir": str(identity["evidence_dir"]),
            }
            identity_digest = hashlib.sha256(json.dumps(
                identity_contract, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).digest()
            world.require_engine_consensus(list(identity_digest))
            if identity is not None:
                world.checked(
                    "observe CNOT physical GPU UUID",
                    lambda: gpu_uuid_for_rank(
                        world.rank, world.size, device=device, torch_module=torch
                    ),
                )
            if args.case in ("trained", "trained-deep"):
                rows, k, default_width = ((1, 7, 4, 10), 3, 2) if args.case == "trained-deep" else ((1, 2, 7, 8), 11, 8)
                width = default_width if args.width is None else args.width
                construct_scorer = TQPScorer if scorer_factory is None else scorer_factory
                context = world.checked("load trained scorer", lambda:
                    construct_scorer(args.library, args.manifest, n=len(rows), device=device, inference_batch=4))
            elif args.case == "wide":
                rows, scorer = wide_case()
                context, k, width = nullcontext(scorer), 3, 1 if args.width is None else args.width
            else:
                context = nullcontext()
            with context as scorer:
                if args.require_rust_scorer:
                    def require_native_scorer():
                        from multigpubeamsearch.cnot_scorer import _rust_tqp_scorer
                        native_type = _rust_tqp_scorer()
                        require(native_type is not None and type(scorer) is native_type,
                                "Rust-owned checkpoint scorer is required")
                    world.checked("verify Rust-owned checkpoint scorer", require_native_scorer)
                output = (validate_distribution(device, args.engine) if args.case == "topology" else
                          validate_case(rows, scorer, args.mode, width, k, 3, device,
                                        args.case == "wide", engine=args.engine))
                if args.case == "trained-deep" and world.size == 2:
                    def check_participation():
                        if world.rank == 0:
                            require(any(all(rank[0] and rank[3] and rank[4] for rank in layer["by_rank"])
                                        and layer["distinct"] > layer["retained"] == width for layer in output["layers"]),
                                    "both GPUs must expand, score and retain in the same saturated layer")
                    world.checked("trained layer participation", check_participation)
                if identity is not None:

                    def prepare_hardware_identity():
                        return dict(
                            rank=world.rank,
                            pid=os.getpid(),
                            host=socket.gethostname(),
                            device=str(device),
                            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                            gpu_uuid=gpu_uuid_for_rank(world.rank, world.size, device=device),
                        )

                    hardware = world.checked("prepare CNOT rank/device receipt", prepare_hardware_identity)
                else:
                    hardware = dict(
                        rank=world.rank,
                        pid=os.getpid(),
                        host=socket.gethostname(),
                        device=str(device),
                        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        gpu_uuid=None,
                    )
                devices = [None] * world.size
                if world.size > 1:
                    dist.all_gather_object(devices, hardware, group=world._control_group)
                else:
                    devices[0] = hardware

                if identity is not None:
                    def hash_result():
                        if world.rank == 0:
                            if not isinstance(output, dict):
                                raise RuntimeError("rank zero has no CNOT result report to bind")
                            summary = dict(output, devices=devices, torch=str(torch.__version__), backend=args.backend)
                            return report_sha256(summary), str(output.get("status", "completed"))
                        return None

                    summary = world.checked("hash CNOT acceptance result", hash_result)
                    summaries = [summary if world.rank == 0 else None]
                    if world.size > 1:
                        dist.broadcast_object_list(summaries, src=0, group=world._control_group)
                    report_digest, terminal_status = summaries[0]
                    if not isinstance(report_digest, str) or len(report_digest) != 64:
                        raise RuntimeError("CNOT acceptance result hash was not broadcast to every rank")
                    if world.rank == 0:
                        output.update(devices=devices, torch=str(torch.__version__), backend=args.backend)
                        output["acceptance"] = {
                            "schema": EVENT_SCHEMA,
                            "version": EVENT_VERSION,
                            "run_id": identity["run_id"],
                            "case_id": identity["case_id"],
                            "source_sha": identity["source_sha"],
                            "wheel_sha256": identity["wheel_sha256"],
                            "world": world.size,
                            "transport": "single_rank_no_exchange" if world.size == 1 else "rust_stream5_nccl",
                            "report_sha256": report_digest,
                        }
                    world.checked("write CNOT terminal acceptance receipts", lambda: emit_event(
                        "search_terminal", world,
                        {"status": terminal_status,
                         "report_sha256": report_digest},
                    ))
                def write_result():
                    if world.rank == 0:
                        output.update(devices=devices, torch=str(torch.__version__), backend=args.backend)
                        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
                        print(json.dumps(output), flush=True)

                world.checked("write CNOT result JSON", write_result)
    finally:
        if manage_process_group and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
