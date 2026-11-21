"""Versioned per-invocation CNOT backend trace and receipt validation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


EVENT_SCHEMA = "org.trydotatwo.multigpubeamsearch.cnot.acceptance"
EVENT_VERSION = 1
EVENT_PREFIX = "CNOT_ACCEPTANCE_EVENT "
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
SHA40_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA64_RE = re.compile(r"[0-9a-f]{64}\Z")


class ValidationError(ValueError):
    """Raised when one invocation's report, trace and rank receipts disagree."""


def serialize_event(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def make_event(*, run_id, case_id, source_sha, wheel_sha256, kind, rank, world,
               pid, device, gpu_uuid, fields):
    """Build the shared wire schema used by the Rust and Python producers."""
    return {
        "schema": EVENT_SCHEMA,
        "version": EVENT_VERSION,
        "run_id": run_id,
        "case_id": case_id,
        "source_sha": source_sha,
        "wheel_sha256": wheel_sha256,
        "kind": kind,
        "rank": rank,
        "world": world,
        "pid": pid,
        "device": device,
        "gpu_uuid": gpu_uuid,
        "fields": {key: fields[key] for key in sorted(fields)},
    }


def report_sha256(report: dict) -> str:
    """Hash the report body without its acceptance envelope."""
    payload = dict(report)
    payload.pop("acceptance", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fail(message):
    raise ValidationError(message)


def _canonical(event):
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_event(event):
    if not isinstance(event, dict):
        _fail("acceptance event must be a JSON object")
    if event.get("schema") != EVENT_SCHEMA or event.get("version") != EVENT_VERSION:
        _fail("acceptance event has an unsupported schema or version")
    for name in ("run_id", "case_id"):
        if not isinstance(event.get(name), str) or not TOKEN_RE.fullmatch(event[name]):
            _fail(f"acceptance event has an invalid {name}")
    if not isinstance(event.get("source_sha"), str) or not SHA40_RE.fullmatch(event["source_sha"]):
        _fail("acceptance event has an invalid source SHA")
    if not isinstance(event.get("wheel_sha256"), str) or not SHA64_RE.fullmatch(event["wheel_sha256"]):
        _fail("acceptance event has an invalid wheel SHA-256")
    if not isinstance(event.get("kind"), str) or not TOKEN_RE.fullmatch(event["kind"]):
        _fail("acceptance event has an invalid kind")
    world, rank, pid = event.get("world"), event.get("rank"), event.get("pid")
    if type(world) is not int or world not in (1, 2):
        _fail("acceptance event world must be 1 or 2")
    if type(rank) is not int or rank < 0 or rank >= world:
        _fail("acceptance event rank is outside its world")
    if type(pid) is not int or pid <= 0:
        _fail("acceptance event has an invalid process ID")
    for name in ("device", "gpu_uuid"):
        value = event.get(name)
        if not isinstance(value, str) or not value or any(char.isspace() for char in value):
            _fail(f"acceptance event lacks a valid {name}")
    fields = event.get("fields")
    if not isinstance(fields, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in fields.items()
    ):
        _fail("acceptance event fields must be string key/value pairs")
    return event


def parse_trace_events(trace_text: str) -> list[dict]:
    events = []
    for number, line in enumerate(trace_text.splitlines(), 1):
        position = line.find(EVENT_PREFIX)
        if position < 0:
            continue
        payload = line[position + len(EVENT_PREFIX):]
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            _fail(f"invalid acceptance event JSON on trace line {number}: {error}")
        events.append(_validate_event(event))
    return events


def _read_receipts(receipt_root: Path, run_id: str, case_id: str) -> list[dict]:
    directory = receipt_root / run_id / case_id
    if not directory.is_dir():
        _fail(f"per-rank receipt directory is missing for {run_id}/{case_id}")
    paths = sorted(directory.glob("rank-*-pid-*.jsonl"))
    if not paths:
        _fail(f"per-rank JSONL receipts are missing for {run_id}/{case_id}")
    events = []
    for path in paths:
        match = re.fullmatch(r"rank-(\d+)-pid-(\d+)\.jsonl", path.name)
        if not match:
            _fail(f"unexpected per-rank receipt filename: {path.name}")
        expected_rank, expected_pid = map(int, match.groups())
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            _fail(f"cannot read per-rank receipt {path.name}: {error}")
        if not lines:
            _fail(f"per-rank receipt is empty: {path.name}")
        for number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                _fail(f"invalid JSON in {path.name}:{number}: {error}")
            _validate_event(event)
            if event["rank"] != expected_rank or event["pid"] != expected_pid:
                _fail(f"receipt filename identity differs from event in {path.name}")
            events.append(event)
    return events


def _expect_one(by_rank, rank, kind):
    matching = by_rank[rank][kind]
    if len(matching) != 1:
        _fail(f"rank {rank} must have exactly one {kind} event; found {len(matching)}")
    return matching[0]


def _positive_rows(event):
    try:
        return int(event["fields"].get("rows", "0")) > 0
    except (TypeError, ValueError):
        return False


def validate_case(trace_text: str, receipt_root, report: dict, *, expected_source_sha: str,
                  expected_wheel_sha256: str, expected_checkpoint_sha256: str,
                  expected_manifest_sha256: str, expected_scorer_library_sha256: str) -> dict:
    """Validate exactly one report invocation against its log and per-rank JSONL files."""
    acceptance = report.get("acceptance") if isinstance(report, dict) else None
    if not isinstance(acceptance, dict):
        _fail("CNOT result JSON has no acceptance identity")
    if acceptance.get("schema") != EVENT_SCHEMA or acceptance.get("version") != EVENT_VERSION:
        _fail("CNOT result JSON has an unsupported acceptance schema")
    run_id, case_id = acceptance.get("run_id"), acceptance.get("case_id")
    if not isinstance(run_id, str) or not TOKEN_RE.fullmatch(run_id):
        _fail("CNOT result JSON has an invalid run ID")
    if not isinstance(case_id, str) or not TOKEN_RE.fullmatch(case_id):
        _fail("CNOT result JSON has an invalid case ID")
    if expected_source_sha != acceptance.get("source_sha") or not SHA40_RE.fullmatch(expected_source_sha):
        _fail("result source SHA differs from the source under test")
    if expected_wheel_sha256 != acceptance.get("wheel_sha256") or not SHA64_RE.fullmatch(expected_wheel_sha256):
        _fail("result wheel SHA differs from the installed wheel")
    world = report.get("world")
    if type(world) is not int or world not in (1, 2) or acceptance.get("world") != world:
        _fail("CNOT result JSON has an invalid or inconsistent world size")
    if acceptance.get("report_sha256") != report_sha256(report):
        _fail("result JSON hash differs from its terminal acceptance event")
    transport = "single_rank_no_exchange" if world == 1 else "rust_stream5_nccl"
    if acceptance.get("transport") != transport:
        _fail("result JSON does not state the expected transport for this world size")
    for name, expected in (
        ("checkpoint_sha256", expected_checkpoint_sha256),
        ("search_manifest_sha256", expected_manifest_sha256),
        ("scorer_library_sha256", expected_scorer_library_sha256),
    ):
        if not isinstance(expected, str) or not SHA64_RE.fullmatch(expected):
            _fail(f"expected {name} is not a SHA-256")
        if (report.get("model") or {}).get(name) != expected:
            _fail(f"result JSON has a different {name}")

    trace_events = parse_trace_events(trace_text)
    matching_identity = [event for event in trace_events
                         if event["run_id"] == run_id and event["case_id"] == case_id]
    for event in matching_identity:
        if event["source_sha"] != expected_source_sha or event["wheel_sha256"] != expected_wheel_sha256:
            _fail("trace contains the same case/run under a different source or wheel")
    selected_trace = [event for event in matching_identity
                      if event["source_sha"] == expected_source_sha
                      and event["wheel_sha256"] == expected_wheel_sha256]
    receipt_events = _read_receipts(Path(receipt_root), run_id, case_id)
    if any(event["run_id"] != run_id or event["case_id"] != case_id
           or event["source_sha"] != expected_source_sha
           or event["wheel_sha256"] != expected_wheel_sha256 for event in receipt_events):
        _fail("per-rank receipt contains a foreign run, case, source or wheel")
    if Counter(map(_canonical, selected_trace)) != Counter(map(_canonical, receipt_events)):
        _fail("trace events and per-rank JSONL receipts do not describe the same invocation")
    if not receipt_events:
        _fail("no backend acceptance events were recorded")

    if not isinstance(report.get("devices"), list) or len(report["devices"]) != world:
        _fail("result JSON device list does not match the actual world size")
    report_devices = {}
    for item in report["devices"]:
        if not isinstance(item, dict) or type(item.get("rank")) is not int:
            _fail("result JSON has an invalid rank/device record")
        rank = item["rank"]
        if rank in report_devices or rank < 0 or rank >= world:
            _fail("result JSON has duplicate or out-of-range ranks")
        report_devices[rank] = item
    if set(report_devices) != set(range(world)):
        _fail("result JSON omits a rank")
    if world == 2 and len({item.get("gpu_uuid") for item in report_devices.values()}) != 2:
        _fail("two-rank result JSON maps both ranks to the same GPU UUID")

    by_rank = {rank: defaultdict(list) for rank in range(world)}
    rank_identity = {}
    for event in receipt_events:
        rank = event["rank"]
        if event["world"] != world:
            _fail("receipt event has a world size different from its result JSON")
        device = report_devices[rank]
        identity = (event["pid"], event["device"], event["gpu_uuid"])
        if rank in rank_identity and rank_identity[rank] != identity:
            _fail(f"rank {rank} changed process or device identity within one invocation")
        rank_identity[rank] = identity
        if (device.get("pid"), device.get("device"), device.get("gpu_uuid")) != identity:
            _fail(f"result JSON device identity does not match rank {rank} receipts")
        by_rank[rank][event["kind"]].append(event)
    if set(rank_identity) != set(range(world)):
        _fail("per-rank event receipts do not cover the actual world")
    if world == 2 and len({identity[2] for identity in rank_identity.values()}) != 2:
        _fail("two ranks recorded the same physical GPU UUID")

    admissions = {}
    admission_contracts = []
    stream5_contracts = []
    for rank in range(world):
        admission = _expect_one(by_rank, rank, "backend_admission")
        controller = _expect_one(by_rank, rank, "rust_controller")
        terminal = _expect_one(by_rank, rank, "search_terminal")
        af = admission["fields"]
        if (af.get("strict") != "1" or af.get("domain_abi") != "1"
                or af.get("device_abi") != "1" or af.get("scorer_abi") != "1"
                or af.get("scorer") != "rust-tqp"
                or af.get("capabilities") != "owners,expansion,full_matrix_distinct"):
            _fail(f"rank {rank} backend admission lacks the required ABI or capabilities")
        if not SHA64_RE.fullmatch(af.get("domain_sha256", "")):
            _fail(f"rank {rank} backend admission lacks a valid DeviceDomain hash")
        if (af.get("checkpoint_sha256") != expected_checkpoint_sha256
                or af.get("search_manifest_sha256") != expected_manifest_sha256
                or af.get("scorer_library_sha256") != expected_scorer_library_sha256):
            _fail(f"rank {rank} backend admission differs from the pinned trained scorer")
        admission_contracts.append(tuple(af.get(key) for key in (
            "domain_abi", "device_abi", "capabilities", "domain_sha256",
            "scorer", "scorer_abi", "checkpoint_sha256",
            "search_manifest_sha256", "scorer_library_sha256",
        )))
        if (controller["fields"].get("implementation") != "RustNativeStages"
                or controller["fields"].get("abi") != "1"
                or controller["fields"].get("active") != "1"):
            _fail(f"rank {rank} did not report the Rust CNOT controller ABI")
        if terminal["fields"].get("report_sha256") != acceptance.get("report_sha256"):
            _fail(f"rank {rank} terminal event does not identify this result JSON")
        if terminal["fields"].get("status") != report.get("status"):
            _fail(f"rank {rank} terminal status differs from the result JSON")
        admissions[rank] = af

        for operation in ("owners", "expansion", "full_matrix_distinct"):
            matching = [event for event in by_rank[rank]["device_domain_dispatch"]
                        if event["fields"].get("operation") == operation]
            if not matching:
                _fail(f"rank {rank} has no DeviceDomain {operation} dispatch")
            for event in matching:
                fields = event["fields"]
                if (fields.get("implementation") != "DeviceDomain" or fields.get("abi") != "1"
                        or fields.get("device_abi") != "1"
                        or fields.get("library_sha256") != af["domain_sha256"]):
                    _fail(f"rank {rank} DeviceDomain {operation} dispatch differs from admission")

        for operation in ("parent_scores", "child_values"):
            matching = [event for event in by_rank[rank]["tqp_scorer_dispatch"]
                        if event["fields"].get("operation") == operation]
            for event in matching:
                fields = event["fields"]
                if (fields.get("implementation") != "RustTqpScorer" or fields.get("abi") != "1"
                        or fields.get("scorer_abi") != "1"
                        or fields.get("checkpoint_sha256") != af["checkpoint_sha256"]
                        or fields.get("search_manifest_sha256") != af["search_manifest_sha256"]
                        or fields.get("scorer_library_sha256") != af["scorer_library_sha256"]):
                    _fail(f"rank {rank} Rust TQP {operation} dispatch differs from admission")

        if world == 1:
            absent = _expect_one(by_rank, rank, "stream5_absent")
            if (absent["fields"].get("reason") != "single_rank"
                    or absent["fields"].get("exchange") != "not_applicable"):
                _fail("single-rank result does not explicitly declare Stream 5 exchange absent")
            if any(by_rank[rank][kind] for kind in
                   ("stream5_admission", "stream5_active", "stream5_dispatch")):
                _fail("single-rank result unexpectedly reports a Stream 5 collective")
        else:
            stream_admission = _expect_one(by_rank, rank, "stream5_admission")["fields"]
            stream_active = _expect_one(by_rank, rank, "stream5_active")["fields"]
            if any(stream_admission.get(key) != "1" for key in
                   ("abi", "owned_abi", "exchange_abi", "collectives_abi")):
                _fail(f"rank {rank} Stream 5 admission lacks a required ABI")
            stream_hash = stream_admission.get("library_sha256")
            if not SHA64_RE.fullmatch(stream_hash or ""):
                _fail(f"rank {rank} Stream 5 admission lacks a library hash")
            if (stream_active.get("implementation") != "RustStream5/NCCL"
                    or stream_active.get("abi") != "1"
                    or stream_active.get("library_sha256") != stream_hash):
                _fail(f"rank {rank} Stream 5 active backend differs from admission")
            stream5_contracts.append((
                stream_admission.get("abi"), stream_admission.get("owned_abi"),
                stream_admission.get("exchange_abi"), stream_admission.get("collectives_abi"),
                stream_hash,
            ))
            for operation in ("exchange", "all_reduce_i64", "all_gather_i64_into", "broadcast_i64"):
                matching = [event for event in by_rank[rank]["stream5_dispatch"]
                            if event["fields"].get("operation", "").split(":", 1)[0] == operation]
                if not matching:
                    _fail(f"rank {rank} has no actual Rust Stream 5 {operation} dispatch")
                for event in matching:
                    fields = event["fields"]
                    if (fields.get("implementation") != "RustStream5/NCCL"
                            or fields.get("abi") != "1"
                            or fields.get("library_sha256") != stream_hash):
                        _fail(f"rank {rank} Stream 5 {operation} dispatch differs from admission")

    if len(set(admission_contracts)) != 1:
        _fail("ranks admitted different DeviceDomain or TQP scorer contracts")
    if world == 2 and len(set(stream5_contracts)) != 1:
        _fail("ranks admitted different Rust Stream 5 ABI or library hashes")

    for operation in ("owners", "expansion", "full_matrix_distinct"):
        matching = [event for event in receipt_events
                    if event["kind"] == "device_domain_dispatch"
                    and event["fields"].get("operation") == operation]
        if not any(_positive_rows(event) for event in matching):
            _fail(f"DeviceDomain {operation} had no non-empty batch in this invocation")
    for operation in ("parent_scores", "child_values"):
        matching = [event for event in receipt_events
                    if event["kind"] == "tqp_scorer_dispatch"
                    and event["fields"].get("operation") == operation]
        if not matching or not any(_positive_rows(event) for event in matching):
            _fail(f"Rust TQP {operation} had no non-empty dispatch in this invocation")

    return {
        "run_id": run_id,
        "case_id": case_id,
        "source_sha": expected_source_sha,
        "wheel_sha256": expected_wheel_sha256,
        "world": world,
        "event_count": len(receipt_events),
        "device_domain_operations": ["owners", "expansion", "full_matrix_distinct"],
        "stream5_operations": [] if world == 1 else
            ["exchange", "all_reduce_i64", "all_gather_i64_into", "broadcast_i64"],
        "single_rank_transport": world == 1,
        "status": "accepted",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate one result JSON and its rank receipts")
    validate.add_argument("--trace", type=Path, required=True)
    validate.add_argument("--receipts", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--expected-source-sha", required=True)
    validate.add_argument("--expected-wheel-sha256", required=True)
    validate.add_argument("--expected-checkpoint-sha256", required=True)
    validate.add_argument("--expected-manifest-sha256", required=True)
    validate.add_argument("--expected-scorer-library-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_case(
            args.trace.read_text(encoding="utf-8"),
            args.receipts,
            json.loads(args.report.read_text(encoding="utf-8")),
            expected_source_sha=args.expected_source_sha,
            expected_wheel_sha256=args.expected_wheel_sha256,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_scorer_library_sha256=args.expected_scorer_library_sha256,
        )
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
