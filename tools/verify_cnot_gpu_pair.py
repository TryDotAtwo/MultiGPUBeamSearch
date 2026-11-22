"""Check public result parity and acceptance receipts for one bounded GPU campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from cnot_acceptance_events import report_sha256
from permutation_acceptance import permutation_report_sha256
from vast_instance_contract import validate_instance_id


SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
UUID32 = re.compile(r"(?:GPU-)?[0-9a-fA-F]{32}\Z")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report_integrity(report, expected_world, report_name):
    world = report.get("world")
    require(type(world) is int and world == expected_world,
            f"{report_name}: report world does not match its expected topology")
    acceptance = report.get("acceptance")
    require(isinstance(acceptance, dict) and acceptance.get("world") == world,
            f"{report_name}: acceptance and report world sizes differ")
    digest = acceptance.get("report_sha256")
    require(isinstance(digest, str) and SHA64.fullmatch(digest),
            f"{report_name}: report SHA-256 is missing or malformed")
    require(digest == report_sha256(report),
            f"{report_name}: report bytes differ from the accepted report digest")
    root_sha = report.get("root_sha256")
    require(isinstance(root_sha, str) and SHA64.fullmatch(root_sha),
            f"{report_name}: root state SHA-256 is missing or malformed")
    layers = report.get("layers")
    require(isinstance(layers, list) and layers,
            f"{report_name}: ordered survivor layers are missing")
    for index, layer in enumerate(layers):
        winner_sha = layer.get("winners_sha256") if isinstance(layer, dict) else None
        require(isinstance(winner_sha, str) and SHA64.fullmatch(winner_sha),
                f"{report_name}: layer {index} survivor SHA-256 is missing or malformed")
    return digest




def _is_int(value):
    return type(value) is int


def _validate_hash128_rows(rows, name):
    require(isinstance(rows, list) and len(rows) == 3,
            f"{name}: expected three ordered Hash128 survivors")
    normalized = []
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"{name}: survivor {index} is malformed")
        lo, hi = row.get("lo"), row.get("hi")
        require(
            isinstance(lo, str) and re.fullmatch(r"[0-9a-f]{16}", lo)
            and isinstance(hi, str) and re.fullmatch(r"[0-9a-f]{16}", hi),
            f"{name}: survivor {index} does not contain both Hash128 halves",
        )
        normalized.append((lo, hi))
    require(len(set(normalized)) == 3, f"{name}: survivor hashes are not distinct")
    return tuple(normalized)


def validate_permutation_report(report, source_sha, wheel_sha, run_id):
    """Validate the exact public permutation smoke and saturated B=3 pair."""
    require(isinstance(report, dict), "permutation acceptance report is not an object")
    digest = report.get("report_sha256")
    require(isinstance(digest, str) and SHA64.fullmatch(digest),
            "permutation acceptance report SHA-256 is missing or malformed")
    require(
        digest == permutation_report_sha256(report),
        "permutation acceptance report changed after its canonical digest",
    )
    require(
        report.get("source_sha") == source_sha
        and report.get("wheel_sha256") == wheel_sha
        and report.get("acceptance_run_id") == run_id,
        "permutation report belongs to another source, wheel or acceptance run",
    )
    require(
        report.get("fixture") == "cayleypy.LRX(4)"
        and report.get("saturated_fixture") == "cayleypy.LRX(5)",
        "permutation report is missing the public smoke or saturated LRX fixture",
    )
    require(
        _is_int(report.get("state_len")) and report["state_len"] > 0
        and _is_int(report.get("move_count")) and report["move_count"] > 0
        and _is_int(report.get("saturated_state_len")) and report["saturated_state_len"] > 0
        and _is_int(report.get("saturated_move_count")) and report["saturated_move_count"] > 0,
        "permutation report lacks valid logical geometry",
    )

    smoke_runs = report.get("runs")
    require(isinstance(smoke_runs, list) and len(smoke_runs) == 2,
            "permutation one- and two-GPU public smoke results are missing")
    smoke_distance = None
    for expected_world, run in enumerate(smoke_runs, start=1):
        name = f"permutation LRX(4) world={expected_world}"
        require(isinstance(run, dict), f"{name}: run record is malformed")
        require(
            _is_int(run.get("world_size")) and run["world_size"] == expected_world,
            f"{name}: expected world size {expected_world}",
        )
        devices = run.get("devices")
        pids = run.get("worker_pids")
        require(
            isinstance(devices, list) and devices == list(range(expected_world))
            and run.get("requested_devices") == devices,
            f"{name}: requested and reported device layouts differ",
        )
        require(
            isinstance(pids, list) and len(pids) == expected_world
            and all(_is_int(pid) and pid > 0 for pid in pids)
            and len(set(pids)) == expected_world,
            f"{name}: unique Rust worker PIDs are missing",
        )
        require(run.get("engine") == "rust" and run.get("worker_pids_reused") is True,
                f"{name}: Rust session reuse evidence is missing")
        distance = run.get("exact_distance")
        require(_is_int(distance) and distance > 0,
                f"{name}: exact public-smoke distance is missing")
        if smoke_distance is None:
            smoke_distance = distance
        require(distance == smoke_distance, "one- and two-GPU permutation smoke fixtures differ")
        searches = run.get("searches")
        require(isinstance(searches, list) and len(searches) == 2,
                f"{name}: two calls on the same session are required")
        paths = []
        for expected_call, search in enumerate(searches, start=1):
            require(isinstance(search, dict), f"{name}: search result is malformed")
            path = search.get("path")
            length = search.get("length")
            require(
                search.get("call") == expected_call
                and search.get("replay_valid") is True
                and _is_int(length) and length == distance
                and isinstance(path, list) and len(path) == length
                and all(_is_int(move) and move >= 0 for move in path)
                and search.get("devices") == devices,
                f"{name}: exact path replay or device binding is missing",
            )
            paths.append(tuple(path))
        require(paths[0] == paths[1], f"{name}: repeated session calls changed the replay path")

    saturated = report.get("saturated_b3")
    require(isinstance(saturated, list) and len(saturated) == 2,
            "saturated B=3 one- and two-GPU reports are missing")
    normalized_winners = None
    common_fixture = None
    common_eligible = None
    case_summaries = []
    for expected_world, case in enumerate(saturated, start=1):
        name = f"permutation-saturated-b3-world-{expected_world}"
        require(isinstance(case, dict), f"{name}: case record is malformed")
        require(
            case.get("case_id") == name
            and _is_int(case.get("world_size")) and case["world_size"] == expected_world,
            f"{name}: case identity or world size is invalid",
        )
        devices = case.get("devices")
        worker_pids = case.get("worker_pids")
        require(
            isinstance(devices, list) and devices == list(range(expected_world)),
            f"{name}: requested CUDA ordinal layout is invalid",
        )
        require(
            isinstance(worker_pids, list) and len(worker_pids) == expected_world
            and all(_is_int(pid) and pid > 0 for pid in worker_pids)
            and len(set(worker_pids)) == expected_world,
            f"{name}: unique Rust worker PIDs are missing",
        )
        fixture_start, scramble = case.get("fixture_start"), case.get("scramble")
        distance = case.get("exact_distance")
        require(
            isinstance(fixture_start, list) and len(fixture_start) == report["saturated_state_len"]
            and all(_is_int(value) and value >= 0 for value in fixture_start)
            and isinstance(scramble, list) and scramble
            and all(_is_int(move) and move >= 0 for move in scramble)
            and _is_int(distance) and distance >= 3,
            f"{name}: deterministic saturated fixture definition is incomplete",
        )
        fixture_identity = (tuple(fixture_start), tuple(scramble), distance)
        if common_fixture is None:
            common_fixture = fixture_identity
        require(fixture_identity == common_fixture,
                "one- and two-GPU saturated permutation fixtures differ")
        expected_inputs = [3] if expected_world == 1 else [2, 1]
        input_counts = case.get("input_counts")
        require(
            isinstance(input_counts, list) and input_counts == expected_inputs,
            f"{name}: CPU oracle frontier partition is missing or changed",
        )
        eligible = case.get("eligible_distinct")
        require(
            _is_int(eligible) and eligible > 3
            and _is_int(case.get("keep_count")) and case["keep_count"] == 3,
            f"{name}: B=3 is not an exact saturated global cutoff",
        )
        if common_eligible is None:
            common_eligible = eligible
        require(eligible == common_eligible,
                "one- and two-GPU permutation oracle candidate totals differ")
        duplicates = case.get("cross_source_duplicates")
        require(_is_int(duplicates) and duplicates >= 0,
                f"{name}: independent cross-source dedup count is missing")
        if expected_world == 2:
            require(duplicates > 0,
                    f"{name}: fixture does not cover a cross-rank duplicate")
        case_winners = _validate_hash128_rows(case.get("ordered_survivor_hashes"), name)
        if normalized_winners is None:
            normalized_winners = case_winners
        require(case_winners == normalized_winners,
                "ordered Hash128 survivors differ between one and two GPUs")
        require(case.get("independent_replay") is True,
                f"{name}: independent retained-state replay is missing")
        gpu_names = case.get("gpu_names")
        require(isinstance(gpu_names, list) and len(gpu_names) == expected_world
                and all(isinstance(value, str) and value for value in gpu_names),
                f"{name}: rank GPU descriptions are missing")

        calls = case.get("calls")
        require(isinstance(calls, list) and len(calls) == 2,
                f"{name}: two calls on one persistent Rust session are required")
        prior_call_identity = None
        prior_rank_identity = None
        for expected_call, call in enumerate(calls, start=1):
            call_name = f"{name} call={expected_call}"
            require(isinstance(call, dict), f"{call_name}: call record is malformed")
            require(
                call.get("call_id") == expected_call
                and call.get("eligible_distinct") == eligible
                and call.get("beam_width") == 3
                and call.get("cross_source_duplicates") == duplicates
                and isinstance(call.get("log_path"), str) and bool(call["log_path"]),
                f"{call_name}: run-scoped trace identity is incomplete",
            )
            if expected_call == 2:
                require(call.get("session_reused") is True,
                        f"{call_name}: existing Rust workers were not reused")
            pids = call.get("worker_pids")
            require(pids == worker_pids,
                    f"{call_name}: worker PIDs differ from the saturated case")
            build_key, binary_sha = call.get("build_key"), call.get("binary_sha256")
            require(
                isinstance(build_key, str) and SHA64.fullmatch(build_key)
                and isinstance(binary_sha, str) and SHA64.fullmatch(binary_sha),
                f"{call_name}: native build or binary digest is missing",
            )
            call_winners = _validate_hash128_rows(call.get("ordered_survivor_hashes"), call_name)
            require(call_winners == case_winners,
                    f"{call_name}: ordered survivor hashes differ from its case report")
            rank_rows = call.get("rank_evidence")
            require(isinstance(rank_rows, list) and len(rank_rows) == expected_world,
                    f"{call_name}: per-rank trace evidence is incomplete")
            rank_pids, rank_uuids, rank_identity = [], [], []
            rank_candidates = rank_requests = rank_kept = 0
            for expected_rank, row in enumerate(rank_rows):
                require(isinstance(row, dict), f"{call_name}: rank record is malformed")
                require(
                    _is_int(row.get("rank")) and row["rank"] == expected_rank
                    and _is_int(row.get("pid")) and row["pid"] > 0,
                    f"{call_name}: rank number or process ID is invalid",
                )
                gpu_uuid = row.get("gpu_uuid")
                require(isinstance(gpu_uuid, str) and UUID32.fullmatch(gpu_uuid),
                        f"{call_name}: observed physical GPU UUID is missing")
                rank_pids.append(row["pid"])
                rank_uuids.append(gpu_uuid.lower().removeprefix("gpu-"))
                rank_identity.append((row["pid"], rank_uuids[-1]))
                expected_rank_values = {
                    "frontier_size": input_counts[expected_rank],
                    "generated": input_counts[expected_rank] * report["saturated_move_count"],
                    "final_global_less": 0,
                    "final_global_equal": eligible,
                    "final_total_available": eligible,
                    "final_global_keep": 3,
                }
                for field, expected_value in expected_rank_values.items():
                    require(
                        _is_int(row.get(field)) and row[field] == expected_value,
                        f"{call_name}: rank {expected_rank} disagrees on {field}",
                    )
                for field in ("final_candidate_count", "final_request_count", "next_frontier_size"):
                    require(_is_int(row.get(field)) and row[field] >= 0,
                            f"{call_name}: rank {expected_rank} has invalid {field}")
                require(row["next_frontier_size"] > 0,
                        f"{call_name}: rank {expected_rank} retained no frontier work")
                require(
                    isinstance(row.get("frontier_snapshot"), str)
                    and bool(row["frontier_snapshot"]),
                    f"{call_name}: rank {expected_rank} has no frontier snapshot evidence",
                )
                rank_candidates += row["final_candidate_count"]
                rank_requests += row["final_request_count"]
                rank_kept += row["next_frontier_size"]
            require(
                len(set(rank_pids)) == expected_world
                and rank_pids == worker_pids
                and len(set(rank_uuids)) == expected_world,
                f"{call_name}: rank records do not prove distinct processes and GPUs",
            )
            require(
                rank_candidates == eligible and rank_requests == 3 and rank_kept == 3,
                f"{call_name}: per-rank counts do not match the exact global cutoff",
            )
            if prior_call_identity is None:
                prior_call_identity = (build_key, binary_sha)
                prior_rank_identity = tuple(rank_identity)
            else:
                require(
                    prior_call_identity == (build_key, binary_sha)
                    and prior_rank_identity == tuple(rank_identity),
                    f"{call_name}: repeated session changed its binary or rank identities",
                )
        case_summaries.append(
            {
                "case_id": name,
                "world_size": expected_world,
                "worker_pids": worker_pids,
                "rank_uuids": [
                    row["gpu_uuid"] for row in calls[0]["rank_evidence"]
                ],
                "eligible_distinct": eligible,
                "keep_count": 3,
                "cross_source_duplicates": duplicates,
                "ordered_survivor_hashes": [
                    {"lo": lo, "hi": hi} for lo, hi in case_winners
                ],
                "session_reused": calls[1]["session_reused"] is True,
            }
        )
    return {
        "report_sha256": digest,
        "smoke_world_sizes": [1, 2],
        "saturated_world_sizes": [1, 2],
        "saturated_beam_width": 3,
        "eligible_distinct": common_eligible,
        "saturated_survivor_hashes_match": True,
        "independent_replay": True,
        "cases": case_summaries,
    }

def _packed_identity(n):
    words = (n + 63) // 64
    lanes = [0] * (n * words)
    for row in range(n):
        lanes[row * words + row // 64] |= 1 << (row % 64)
    return lanes


def _panel_json_digest(value):
    return __import__("hashlib").sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validate_panel_row(row, expected, root_pin, label):
    require(isinstance(row, dict), f"{label}: panel row is malformed")
    for key in ("panel_index", "case_id", "dimension", "alphabet_id"):
        require(row.get(key) == expected.get(key), f"{label}: {key} differs from the locked panel")
    require(
        row.get("orientation") == root_pin.get("orientation"),
        f"{label}: orientation differs from the sealed root",
    )
    require(row.get("terminal_reason") == row.get("status"),
            f"{label}: terminal reason differs from row status")
    require(row.get("status") in {"solved", "depth_limit", "resource_limit", "exhausted"},
            f"{label}: unknown terminal status")
    n = expected["dimension"]
    words = (n + 63) // 64
    start = row.get("start_packed_rows")
    require(
        isinstance(start, list) and len(start) == n * words
        and all(_is_int(value) and 0 <= value <= 0xFFFFFFFFFFFFFFFF for value in start)
        and _panel_json_digest(start) == root_pin["start_packed_rows_sha256"],
        f"{label}: ordered panel root does not match the sealed dataset",
    )
    require(
        type(row.get("expanded_layers")) is int and row["expanded_layers"] >= 0,
        f"{label}: expanded-layer count is invalid",
    )
    wall = row.get("wall_seconds")
    require(
        isinstance(wall, (int, float)) and not isinstance(wall, bool)
        and 0 <= wall < float("inf"),
        f"{label}: per-case wall time is invalid",
    )
    word = row.get("word")
    if row["status"] == "solved":
        require(isinstance(word, dict) and word.get("replayed") is True,
                f"{label}: solved panel result lacks replay evidence")
        require(word.get("direction") == "original_matrix_to_identity",
                f"{label}: reported word direction is not explicit")
        gates = word.get("control_target")
        cost = word.get("physical_cost")
        require(
            isinstance(gates, list) and type(cost) is int and cost == len(gates),
            f"{label}: physical word cost differs from gate count",
        )
        replay = list(start)
        for gate in gates:
            require(
                isinstance(gate, list) and len(gate) == 2
                and type(gate[0]) is int and type(gate[1]) is int
                and 0 <= gate[0] < n and 0 <= gate[1] < n and gate[0] != gate[1],
                f"{label}: physical word contains an invalid CNOT",
            )
            control, target = gate
            for lane in range(words):
                replay[target * words + lane] ^= replay[control * words + lane]
        identity = _packed_identity(n)
        require(replay == identity and row.get("end_packed_rows") == identity,
                f"{label}: returned word does not replay from the original matrix to identity")
        return cost
    require(word is None and row.get("end_packed_rows") is None,
            f"{label}: unsolved search returned a word or endpoint")
    return None


def expected_panel_adapter_provenance():
    root = Path(__file__).resolve().parents[1]
    build_path = root / "integrations" / "cnot_scorer" / "build.py"
    bridge_path = root / "integrations" / "cnot_scorer" / "bridge_panel_v2.rs"
    match = re.search(
        r'^PANEL_SEARCH_V2_SOURCE_COMMIT = "([0-9a-f]{40})"$',
        build_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    require(match is not None, "pinned native adapter source commit is missing")
    return {
        "native_adapter_source_commit": match.group(1),
        "adapter_source_sha256": hashlib.sha256(bridge_path.read_bytes()).hexdigest(),
        "endpoint_source_sha256": hashlib.sha256((root / "tools" / "verify_cnot.py").read_bytes()).hexdigest(),
    }


def validate_panel_gpu_pair(panel_result_dir, handoff_preflight_path, *,
                            source_sha, wheel_sha, scorer_library_sha, run_id):
    """Bind 1/2-rank MGBS search results to the producer's locked test receipt."""
    preflight = load_json(handoff_preflight_path)
    require(isinstance(preflight, dict)
            and preflight.get("schema") == "cnot103_mgbs_handoff_preflight_v1"
            and preflight.get("status") == "passed",
            "producer handoff preflight is missing or invalid")
    require(
        preflight.get("panel") == "test"
        and preflight.get("artifact_path_policy") == "same_run_root_mount"
        and preflight.get("orientation") == "row_target_xor_row_control",
        "producer handoff does not pin the locked test orientation and mount",
    )
    for name in (
        "checkpoint_sha256", "dataset_sha256", "training_manifest_sha256",
        "training_result_sha256", "search_manifest_sha256",
        "baseline_search_manifest_sha256", "locked_test_comparison_sha256",
        "panel_digest",
    ):
        require(isinstance(preflight.get(name), str) and SHA64.fullmatch(preflight[name]),
                f"producer handoff {name} is missing or malformed")
    require(
        isinstance(preflight.get("producer_source_revision"), str)
        and SHA40.fullmatch(preflight["producer_source_revision"])
        and isinstance(preflight.get("producer_emitter_sha256"), str)
        and SHA64.fullmatch(preflight["producer_emitter_sha256"]),
        "producer source or emitter pin is malformed",
    )
    require(
        type(preflight.get("accepted_updates")) is int and preflight["accepted_updates"] > 0,
        "producer handoff has no accepted training updates",
    )
    require(
        preflight.get("locked_test_status") in {
            "length_gain", "coverage_gain", "combined_gain", "tie", "no_practical_gain"
        },
        "producer locked-test status is unsupported",
    )
    search = preflight.get("search")
    require(
        isinstance(search, dict)
        and search == {
            "global_width": 36, "shortlist": 4, "max_depth": 8, "max_expansions": 201
        },
        "producer handoff search geometry differs from the supported preset",
    )
    require(preflight.get("search_manifest_relative_path")
            and not Path(preflight["search_manifest_relative_path"]).is_absolute()
            and ".." not in Path(preflight["search_manifest_relative_path"]).parts,
            "producer search manifest path is not run-relative")
    candidate_cases = preflight.get("candidate_cases")
    candidate_solved_count = preflight.get("candidate_solved_count")
    candidate_total_cost = preflight.get("candidate_total_physical_cost")
    require(
        type(candidate_solved_count) is int
        and type(candidate_total_cost) is int
        and isinstance(candidate_cases, list)
        and candidate_solved_count == sum(
            1 for row in candidate_cases
            if isinstance(row, dict) and row.get("status") == "solved"
        )
        and candidate_total_cost == sum(
            row["physical_cost"] for row in candidate_cases
            if isinstance(row, dict) and row.get("status") == "solved"
            and type(row.get("physical_cost")) is int
        ),
        "producer locked-test aggregate differs from its ordered candidate cases",
    )
    ids, indices = preflight.get("panel_ids"), preflight.get("panel_indices")
    expected_cases, root_pins = preflight.get("candidate_cases"), preflight.get("root_pins")
    require(
        isinstance(ids, list) and ids and len(ids) == len(set(ids))
        and isinstance(indices, list) and len(indices) == len(ids)
        and all(type(index) is int and index >= 0 for index in indices)
        and isinstance(expected_cases, list) and len(expected_cases) == len(ids)
        and isinstance(root_pins, list) and len(root_pins) == len(ids),
        "producer handoff locked cohort is empty or malformed",
    )
    expected_rows = []
    for position, (case_id, index, expected, root_pin) in enumerate(
        zip(ids, indices, expected_cases, root_pins)
    ):
        require(
            isinstance(expected, dict) and isinstance(root_pin, dict)
            and expected.get("case_id") == case_id and root_pin.get("case_id") == case_id
            and expected.get("panel_index") == index and root_pin.get("panel_index") == index
            and expected.get("dimension") == root_pin.get("dimension")
            and expected.get("alphabet_id") == root_pin.get("alphabet_id")
            and expected.get("start_packed_rows_sha256") == root_pin.get("start_packed_rows_sha256"),
            f"producer handoff ordered case {position} differs from its sealed root",
        )
        expected_rows.append(expected)

    result_dir = Path(panel_result_dir)
    paths = (
        result_dir / "cnot-panel-test-qhat-1-gpu.json",
        result_dir / "cnot-panel-test-qhat-2-gpu.json",
    )
    reports = [load_json(path) for path in paths]
    semantic_rows = []
    report_digests = []
    backend_event_receipts = {}
    for expected_world, case_id, report, path in zip((1, 2), (
        "cnot-panel-test-qhat-1-gpu", "cnot-panel-test-qhat-2-gpu"
    ), reports, paths):
        acceptance = report.get("acceptance")
        require(
            report.get("schema") == "cnot103_mgbs_panel_search_result_v1"
            and report.get("world") == expected_world
            and report.get("world_size") == expected_world
            and report.get("execution_backend") == "cuda_distributed"
            and report.get("status") == "completed"
            and report.get("engine") == "rust"
            and report.get("head") == "qhat"
            and report.get("runtime_contract") == "native_device_search_v13"
            and report.get("expansion_upper_bound") == search["max_expansions"]
            and report.get("search") == search
            and report.get("panel") == "test"
            and report.get("panel_ids") == ids
            and report.get("panel_indices") == indices
            and report.get("panel_digest") == preflight["panel_digest"]
            and report.get("orientation") == preflight["orientation"],
            f"{path.name}: panel search does not match the pinned producer contract",
        )
        for key in (
            "producer_source_revision", "producer_emitter_sha256", "accepted_updates",
            "checkpoint_sha256", "dataset_sha256", "training_manifest_sha256",
            "training_result_sha256", "search_manifest_sha256",
            "baseline_search_manifest_sha256", "locked_test_status",
            "locked_test_comparison_sha256",
        ):
            require(report.get(key) == preflight.get(key),
                    f"{path.name}: {key} differs from the producer handoff")
        require(
            report.get("adapter") == expected_panel_adapter_provenance(),
            f"{path.name}: native adapter or endpoint source SHA differs from this consumer",
        )
        require(
            report.get("scorer_library_sha256") == scorer_library_sha
            and isinstance(acceptance, dict)
            and acceptance.get("schema") == "org.trydotatwo.multigpubeamsearch.cnot.acceptance"
            and acceptance.get("run_id") == run_id
            and acceptance.get("case_id") == case_id
            and acceptance.get("source_sha") == source_sha
            and acceptance.get("wheel_sha256") == wheel_sha
            and acceptance.get("world") == expected_world
            and acceptance.get("transport") == (
                "single_rank_no_exchange" if expected_world == 1 else "rust_stream5_nccl"
            ),
            f"{path.name}: scorer or acceptance invocation identity is foreign",
        )
        digest = acceptance.get("report_sha256")
        require(
            isinstance(digest, str) and SHA64.fullmatch(digest)
            and digest == report_sha256(report),
            f"{path.name}: result body changed after its terminal receipt",
        )
        report_digests.append(digest)
        event_path = result_dir / "backend-events" / f"{case_id}.json"
        event_receipt = load_json(event_path)
        require(
            event_receipt.get("status") == "accepted"
            and event_receipt.get("run_id") == run_id
            and event_receipt.get("case_id") == case_id
            and event_receipt.get("source_sha") == source_sha
            and event_receipt.get("wheel_sha256") == wheel_sha
            and event_receipt.get("world") == expected_world
            and event_receipt.get("report_sha256") == digest
            and acceptance.get("report_sha256") == digest
            and event_receipt.get("single_rank_transport") == (expected_world == 1),
            f"{path.name}: strict backend receipt differs from the panel report",
        )
        rank_devices = event_receipt.get("rank_devices")
        report_devices = report.get("devices")
        require(
            isinstance(rank_devices, list) and len(rank_devices) == expected_world
            and isinstance(report_devices, list) and len(report_devices) == expected_world
            and sorted((row.get("rank"), row.get("pid"), row.get("device"), row.get("gpu_uuid"))
                       for row in rank_devices)
            == sorted((row.get("rank"), row.get("pid"), row.get("device"), row.get("gpu_uuid"))
                      for row in report_devices),
            f"{path.name}: strict backend receipts name different rank devices",
        )
        require(
            len({row.get("pid") for row in rank_devices}) == expected_world
            and (expected_world == 1 or len({row.get("gpu_uuid") for row in rank_devices}) == expected_world),
            f"{path.name}: strict backend receipt reuses a rank or physical GPU identity",
        )
        backend_event_receipts[case_id] = event_receipt

        report_roots = report.get("panel_roots")
        require(isinstance(report_roots, list) and len(report_roots) == len(ids),
                f"{path.name}: ordered source roots are missing")
        for root, root_pin in zip(report_roots, root_pins):
            require(
                isinstance(root, dict)
                and root.get("panel_index") == root_pin["panel_index"]
                and root.get("case_id") == root_pin["case_id"]
                and root.get("dimension") == root_pin["dimension"]
                and root.get("alphabet_id") == root_pin["alphabet_id"]
                and root.get("orientation") == root_pin["orientation"]
                and _panel_json_digest(root.get("start_packed_rows"))
                    == root_pin["start_packed_rows_sha256"],
                f"{path.name}: root differs from the sealed dataset",
            )
        rows = report.get("rows")
        require(isinstance(rows, list) and len(rows) == len(expected_rows),
                f"{path.name}: result rows differ from the frozen cohort")
        semantic = []
        for row, expected, root_pin in zip(rows, expected_rows, root_pins):
            require(
                isinstance(row, dict) and row.get("panel_index") == expected["panel_index"]
                and row.get("case_id") == expected["case_id"]
                and row.get("dimension") == expected["dimension"]
                and row.get("alphabet_id") == expected["alphabet_id"]
                and row.get("orientation") == preflight["orientation"]
                and row.get("status") == expected["status"],
                f"{path.name}: case ID, order, root or locked result differs",
            )
            cost = _validate_panel_row(row, expected, root_pin, path.name)
            require(cost == expected["physical_cost"],
                    f"{path.name}: physical cost differs from the producer's locked Qhat result")
            semantic.append({key: value for key, value in row.items() if key != "wall_seconds"})
        semantic_rows.append(semantic)
    require(semantic_rows[0] == semantic_rows[1],
            "ordered locked-test results differ between one and two GPUs")
    return {
        "status": "accepted",
        "panel": "test",
        "locked_test_status": preflight["locked_test_status"],
        "locked_test_comparison_sha256": preflight["locked_test_comparison_sha256"],
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "dataset_sha256": preflight["dataset_sha256"],
        "search_manifest_sha256": preflight["search_manifest_sha256"],
        "panel_digest": preflight["panel_digest"],
        "panel_indices": indices,
        "panel_ids": ids,
        "case_count": len(ids),
        "candidate_solved_count": preflight.get("candidate_solved_count"),
        "candidate_total_physical_cost": preflight.get("candidate_total_physical_cost"),
        "world_sizes": [1, 2],
        "report_sha256": report_digests,
        "ordered_results_match": True,
        "independent_replay": True,
        "strict_backend_receipts": backend_event_receipts,
    }
def validate(result_dir, source_sha, instance_id, manifest_sha, checkpoint_sha,
             scorer_library_sha, wheel_sha, run_id, panel_result_dir, handoff_preflight_path):
    require(validate_instance_id(str(instance_id)) == str(instance_id),
            "receipt requires the explicitly selected new positive-decimal instance ID")
    root = Path(result_dir)
    panel_verdict = validate_panel_gpu_pair(
        panel_result_dir,
        handoff_preflight_path,
        source_sha=source_sha,
        wheel_sha=wheel_sha,
        scorer_library_sha=scorer_library_sha,
        run_id=run_id,
    )
    paired_names = (
        "cnot-trained-deep-paired-1-gpu.json",
        "cnot-trained-deep-paired-2-gpu.json",
    )
    left, right = (load_json(root / name) for name in paired_names)
    paired_report_digests = [
        validate_report_integrity(left, 1, paired_names[0]),
        validate_report_integrity(right, 2, paired_names[1]),
    ]
    for report in (left, right):
        model = report.get("model") or {}
        require(
            model.get("checkpoint_sha256") == checkpoint_sha
            and model.get("search_manifest_sha256") == manifest_sha
            and model.get("scorer_library_sha256") == scorer_library_sha
            and report.get("engine") == "rust"
            and report.get("bitwise_replay"),
            "paired CNOT run lacks pinned scorer, Rust controller or independent replay",
        )
        require(
            report.get("n") == 4
            and report.get("mode") == "proposal"
            and report.get("width") == 2
            and report.get("shortlist") == 3,
            "paired CNOT runs do not share the requested logical configuration",
        )
    for key in ("root_sha256", "mode", "width", "shortlist", "n"):
        require(left.get(key) == right.get(key), f"paired 1/2 GPU CNOT inputs differ at {key}")
    require(
        left.get("precision_contract") == right.get("precision_contract")
        and left.get("precision_contract", {}).get("scorer_output_dtype") == "float32",
        "paired CNOT score or selection precision differs",
    )
    model_keys = (
        "upstream_commit", "checkpoint_sha256", "search_manifest_sha256",
        "scorer_library_sha256", "feature_registry_sha256", "training_manifest_sha256",
        "training_result_sha256", "scorer_output_dtype", "feature_ids",
    )
    require(
        {key: left["model"].get(key) for key in model_keys}
        == {key: right["model"].get(key) for key in model_keys},
        "paired CNOT model/head contract differs",
    )
    require(
        bool(left.get("layers"))
        and len(left["layers"]) == len(right.get("layers", []))
        and [item.get("winners_sha256") for item in left["layers"]]
        == [item.get("winners_sha256") for item in right["layers"]],
        "ordered global CNOT survivor hashes differ between 1 and 2 GPUs",
    )

    qhat_name = "cnot-trained-qhat-1-gpu.json"
    qhat = load_json(root / qhat_name)
    qhat_report_digest = validate_report_integrity(qhat, 1, qhat_name)
    model = qhat.get("model") or {}
    require(
        model.get("checkpoint_sha256") == checkpoint_sha
        and model.get("search_manifest_sha256") == manifest_sha
        and model.get("scorer_library_sha256") == scorer_library_sha
        and qhat.get("engine") == "rust"
        and qhat.get("world") == 1
        and qhat.get("mode") == "qhat"
        and qhat.get("bitwise_replay"),
        f"{qhat_name}: trained qhat, Rust scorer or independent replay evidence is missing",
    )
    permutation = load_json(root / "permutation-1-and-2-gpu.json")
    permutation_verdict = validate_permutation_report(
        permutation, source_sha=source_sha, wheel_sha=wheel_sha, run_id=run_id
    )

    case_expectations = {
        "cnot-trained-qhat-1-gpu": (qhat_name, 1),
        "cnot-trained-deep-paired-1-gpu": (paired_names[0], 1),
        "cnot-trained-deep-paired-2-gpu": (paired_names[1], 2),
    }
    report_digests = {
        "cnot-trained-qhat-1-gpu": qhat_report_digest,
        "cnot-trained-deep-paired-1-gpu": paired_report_digests[0],
        "cnot-trained-deep-paired-2-gpu": paired_report_digests[1],
    }
    event_receipts = {}
    for case_id, (report_name, world) in case_expectations.items():
        report = load_json(root / report_name)
        identity = report.get("acceptance") or {}
        require(
            identity.get("run_id") == run_id
            and identity.get("case_id") == case_id
            and identity.get("source_sha") == source_sha
            and identity.get("wheel_sha256") == wheel_sha
            and report.get("world") == world
            and identity.get("world") == world,
            f"{report_name}: result identity is not bound to this run, wheel and topology",
        )
        event_path = root / "backend-events" / f"{case_id}.json"
        event_receipt = load_json(event_path)
        require(
            event_receipt.get("status") == "accepted"
            and event_receipt.get("run_id") == identity["run_id"]
            and event_receipt.get("case_id") == case_id
            and event_receipt.get("source_sha") == source_sha
            and event_receipt.get("wheel_sha256") == wheel_sha
            and event_receipt.get("world") == world
            and event_receipt.get("report_sha256") == report_digests[case_id]
            and identity.get("report_sha256") == report_digests[case_id]
            and event_receipt.get("single_rank_transport") == (world == 1),
            f"{case_id}: strict backend verifier receipt does not match its result JSON",
        )
        devices = event_receipt.get("rank_devices")
        require(isinstance(devices, list) and len(devices) == world,
                f"{case_id}: per-rank process/device identities are missing")
        report_devices = report.get("devices") or []
        require(
            sorted((row.get("rank"), row.get("pid"), row.get("device"), row.get("gpu_uuid"))
                   for row in devices)
            == sorted((row.get("rank"), row.get("pid"), row.get("device"), row.get("gpu_uuid"))
                      for row in report_devices),
            f"{case_id}: backend receipts and result JSON name different rank devices",
        )
        require(len({row.get("pid") for row in devices}) == world,
                f"{case_id}: rank receipts reuse a process ID")
        if world == 2:
            require(len({row.get("gpu_uuid") for row in devices}) == world,
                    f"{case_id}: both ranks do not identify distinct physical GPUs")
        event_receipts[case_id] = event_receipt

    receipt = {
        "instance_id": instance_id,
        "source_sha": source_sha,
        "wheel_sha256": wheel_sha,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "scorer_library_sha256": scorer_library_sha,
        "upstream_commit": left["model"].get("upstream_commit"),
        "paired_cnot_survivor_hashes_match": True,
        "paired_locked_test_panel": panel_verdict,
        "paired_cnot_world_sizes": [left.get("world"), right.get("world")],
        "permutation_report_sha256": permutation_verdict["report_sha256"],
        "permutation_smoke_world_sizes": permutation_verdict["smoke_world_sizes"],
        "permutation_saturated_world_sizes": permutation_verdict["saturated_world_sizes"],
        "permutation_saturated_beam_width": permutation_verdict["saturated_beam_width"],
        "permutation_saturated_eligible_distinct": permutation_verdict["eligible_distinct"],
        "permutation_saturated_survivor_hashes_match": permutation_verdict["saturated_survivor_hashes_match"],
        "permutation_independent_replay": permutation_verdict["independent_replay"],
        "permutation_cases": permutation_verdict["cases"],
        "strict_backend_cases": event_receipts,
        "device_domain_operations": ["owners", "expansion", "full_matrix_distinct"],
        "stream5_nccl_phases": ["exchange", "all_reduce_i64", "all_gather_i64_into", "broadcast_i64"],
        "completed_cases": [
            "permutation-1-and-2-gpu.json",
            qhat_name,
            *paired_names,
            "cnot-panel-test-qhat-1-gpu.json",
            "cnot-panel-test-qhat-2-gpu.json",
        ],
        "installed_wheel": True,
        "gpu_test": "passed",
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--scorer-library-sha256", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--panel-result-dir", type=Path, required=True)
    parser.add_argument("--handoff-preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not SHA40.fullmatch(args.source_sha):
            raise ValueError("source SHA must be a 40-character lowercase hexadecimal ID")
        for name in ("manifest_sha256", "checkpoint_sha256", "scorer_library_sha256", "wheel_sha256"):
            if not SHA64.fullmatch(getattr(args, name)):
                raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
        receipt = validate(
            args.result_dir, args.source_sha, args.instance_id, args.manifest_sha256,
            args.checkpoint_sha256, args.scorer_library_sha256, args.wheel_sha256,
            args.run_id, args.panel_result_dir, args.handoff_preflight,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
