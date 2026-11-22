"""Small, exhaustive reference for issue #6 receipts. Run directly or under torchrun."""
import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys

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


def _load_handoff_resolver():
    root = Path(__file__).resolve().parents[1]
    path = root / "integrations" / "cnot_scorer" / "handoff.py"
    spec = importlib.util.spec_from_file_location("mgbs_cnot_handoff_resolver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("CNOT handoff resolver is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _adapter_provenance():
    root = Path(__file__).resolve().parents[1]
    build_path = root / "integrations" / "cnot_scorer" / "build.py"
    bridge_path = root / "integrations" / "cnot_scorer" / "bridge_panel_v2.rs"
    match = re.search(
        r'^PANEL_SEARCH_V2_SOURCE_COMMIT = "([0-9a-f]{40})"$',
        build_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("CNOT panel adapter source pin is malformed")
    return {
        "native_adapter_source_commit": match.group(1),
        "adapter_source_sha256": hashlib.sha256(bridge_path.read_bytes()).hexdigest(),
        "endpoint_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv=None, *, scorer_factory=None, manage_process_group=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("trained", "trained-deep", "wide", "topology", "panel"), required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--mode", choices=("qhat", "proposal"), default="qhat")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=4)
    parser.add_argument("--width", type=int)
    parser.add_argument("--library")
    parser.add_argument("--manifest", help="direct checkpoint manifest for isolated smoke cases only")
    parser.add_argument("--handoff-receipt", help="private producer learning-curve receipt for paired panel acceptance")
    parser.add_argument("--run-root", help="private run root containing the pinned dataset and manifests")
    parser.add_argument("--engine", choices=("auto", "python", "rust"), default="auto",
                        help="select the search controller; rust fails closed when unavailable")
    parser.add_argument("--require-rust-scorer", action="store_true",
                        help="require the Rust-owned checkpoint scorer and Rust controller")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    resolved_handoff = None
    if args.case == "panel":
        if args.handoff_receipt is None or args.run_root is None:
            parser.error("--case panel requires --handoff-receipt and --run-root")
        if args.manifest is not None:
            parser.error("--case panel accepts the producer receipt, not a direct manifest")
        if args.library is None:
            parser.error("--case panel requires the compiled CNOT scorer library")
        if args.mode != "qhat" or args.audit or args.inference_batch != 4 or args.width is not None:
            parser.error("paired panel acceptance uses the pinned Qhat search and inference batch")
        if args.engine == "python":
            parser.error("paired panel acceptance requires the compiled Rust search engine")
        args.engine = "rust"
        args.require_rust_scorer = True
        try:
            resolver = _load_handoff_resolver()
            resolved_handoff = resolver.resolve_panel_v2_handoff(
                args.run_root, args.handoff_receipt
            )
            output_root = Path(args.run_root).resolve(strict=True)
            args.output = args.output.resolve()
            args.output.relative_to(output_root)
            if args.output.exists():
                raise ValueError("private panel result path already exists")
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    elif args.handoff_receipt is not None or args.run_root is not None:
        parser.error("--handoff-receipt and --run-root are valid only for --case panel")
    elif args.case in ("trained", "trained-deep") and (args.manifest is None or args.library is None):
        parser.error("direct smoke cases require --manifest and --library")
    if args.require_rust_scorer:
        if args.case not in ("trained", "trained-deep", "panel"):
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
            if args.case == "panel":
                context = nullcontext(None)
            elif args.case in ("trained", "trained-deep"):
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
                if args.require_rust_scorer and args.case != "panel":
                    def require_native_scorer():
                        from multigpubeamsearch.cnot_scorer import _rust_tqp_scorer
                        native_type = _rust_tqp_scorer()
                        require(native_type is not None and type(scorer) is native_type,
                                "Rust-owned checkpoint scorer is required")
                    world.checked("verify Rust-owned checkpoint scorer", require_native_scorer)
                if args.case == "panel":
                    output = _search_handoff_panel(
                        args, resolved_handoff, device, world, scorer_factory=scorer_factory
                    )
                else:
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
                        if args.case == "panel":
                            with args.output.open("x", encoding="utf-8") as receipt_file:
                                receipt_file.write(json.dumps(output, indent=2) + "\n")
                            print("cnot_handoff=completed", flush=True)
                        else:
                            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
                            print(json.dumps(output), flush=True)

                world.checked("write CNOT result JSON", write_result)
    finally:
        if manage_process_group and dist.is_initialized():
            dist.destroy_process_group()

def _rows_from_panel_root(panel_root):
    n = panel_root["dimension"]
    words = (n + 63) // 64
    lanes = panel_root["start_packed_rows"]
    if len(lanes) != n * words:
        raise ValueError("sealed root lane count differs from its dimension")
    return tuple(
        sum(lanes[row * words + word] << (64 * word) for word in range(words))
        for row in range(n)
    )


def _control_target(move, n):
    if type(move) is not int or not 0 <= move < n * (n - 1):
        raise ValueError("CNOT move ID is outside the sealed alphabet")
    control, slot = divmod(move, n - 1)
    return control, slot + (slot >= control)


def _panel_result_row(panel_root, result):
    n = panel_root["dimension"]
    start = _rows_from_panel_root(panel_root)
    status = result.status
    if status not in {"solved", "depth_limit", "resource_limit", "exhausted"}:
        raise AssertionError("CNOT runtime returned an unknown terminal status")
    if status != "solved":
        if result.gates is not None:
            raise AssertionError("unsolved panel search returned a construction word")
        word, end = None, None
    else:
        if result.gates is None:
            raise AssertionError("solved panel search omitted its construction word")
        construction_moves = tuple(result.gates)
        if replay_cnot(n, construction_moves) != start:
            raise AssertionError("independent construction replay differs from the sealed root")
        reduced = list(start)
        reduction_moves = tuple(reversed(construction_moves))
        for move in reduction_moves:
            control, target = _control_target(move, n)
            reduced[target] ^= reduced[control]
        identity = tuple(1 << row for row in range(n))
        if tuple(reduced) != identity:
            raise AssertionError("independent reduction replay did not reach identity")
        word = {
            "control_target": [list(_control_target(move, n)) for move in reduction_moves],
            "physical_cost": len(reduction_moves),
            "replayed": True,
        }
        end = [1 << row for row in range(n)]
    return {
        "panel_index": panel_root["panel_index"],
        "case_id": panel_root["case_id"],
        "dimension": n,
        "alphabet_id": panel_root["alphabet_id"],
        "orientation": panel_root["orientation"],
        "start_packed_rows": panel_root["start_packed_rows"],
        "status": status,
        "word": word,
        "end_packed_rows": end,
        "terminal_reason": status,
        "expanded_layers": len(result.layers),
    }


def _search_handoff_panel(args, resolved, device, world, *, scorer_factory):
    resolver = _load_handoff_resolver()
    expansion_bound = resolver.validate_search_contract(resolved.search)
    provenance = _adapter_provenance()
    identity = {
        "producer_source_revision": resolved.producer_source_revision,
        "producer_emitter_sha256": resolved.producer_emitter_sha256,
        "accepted_updates": resolved.accepted_updates,
        "checkpoint_sha256": resolved.checkpoint_sha256,
        "dataset_sha256": resolved.dataset_sha256,
        "search_manifest_sha256": resolved.search_manifest_sha256,
        "panel_digest": resolved.panel_digest,
        "panel_indices": list(resolved.panel_indices),
        "panel_ids": list(resolved.panel_ids),
        "search": resolved.search,
        "head": "qhat",
        "adapter": provenance,
    }
    consensus = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    world.require_engine_consensus(list(consensus))

    construct_scorer = TQPScorer if scorer_factory is None else scorer_factory
    result_rows, scorers = [], {}
    with ExitStack() as stack:
        for panel_root in resolved.panel_roots:
            n = panel_root["dimension"]
            if n not in scorers:
                context = world.checked(
                    "load handoff-pinned CNOT scorer",
                    lambda dimension=n: construct_scorer(
                        str(args.library), str(resolved.search_manifest),
                        n=dimension, device=device, inference_batch=4,
                    ),
                )
                scorer = stack.enter_context(context)
                if args.require_rust_scorer and scorer_factory is None:
                    def verify_native_type():
                        from multigpubeamsearch.cnot_scorer import _rust_tqp_scorer
                        native_type = _rust_tqp_scorer()
                        require(native_type is not None and type(scorer) is native_type,
                                "Rust-owned checkpoint scorer is required")
                    world.checked("verify Rust-owned handoff scorer", verify_native_type)
                scorers[n] = scorer
            root = pack_rows(_rows_from_panel_root(panel_root), device=device)
            result = search_cnot(
                root, scorers[n],
                width=resolved.search["global_width"],
                shortlist=resolved.search["shortlist"],
                max_depth=resolved.search["max_depth"],
                mode="qhat", audit=False, validation="rust", engine="rust",
            )
            row = world.checked(
                "independently replay handoff panel result",
                lambda: _panel_result_row(panel_root, result) if world.rank == 0 else None,
            )
            if world.rank == 0:
                result_rows.append(row)

    return {
        "schema": "cnot103_mgbs_panel_search_result_v1",
        "producer_source_revision": resolved.producer_source_revision,
        "producer_emitter_sha256": resolved.producer_emitter_sha256,
        "accepted_updates": resolved.accepted_updates,
        "checkpoint_sha256": resolved.checkpoint_sha256,
        "dataset_sha256": resolved.dataset_sha256,
        "search_manifest_sha256": resolved.search_manifest_sha256,
        "training_manifest_sha256": resolved.training_manifest_sha256,
        "training_result_sha256": resolved.training_result_sha256,
        "panel": "test",
        "panel_indices": list(resolved.panel_indices),
        "panel_ids": list(resolved.panel_ids),
        "panel_digest": resolved.panel_digest,
        "panel_roots": [dict(root) for root in resolved.panel_roots],
        "search": dict(resolved.search),
        "head": "qhat",
        "runtime_contract": "native_device_search_v13",
        "expansion_upper_bound": expansion_bound,
        "orientation": resolved.orientation,
        "adapter": provenance,
        "execution_backend": "cpu_gloo" if device.type == "cpu" else "cuda_distributed",
        "world_size": world.size,
        "rows": result_rows if world.rank == 0 else None,
    }

if __name__ == "__main__":
    main()
