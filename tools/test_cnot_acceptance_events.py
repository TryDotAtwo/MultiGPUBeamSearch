"""Hosted CPU contract tests for run-scoped CNOT acceptance receipts."""
import json
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import cnot_acceptance_events as protocol


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "cnot_acceptance_event_v1.jsonl"
PYTHON_PRODUCER_PATH = ROOT.parent / "integrations" / "cayleypy_native" / "multigpubeamsearch" / "cnot_acceptance.py"
sys.path.insert(0, str(PYTHON_PRODUCER_PATH.parent))
PYTHON_PRODUCER_SPEC = importlib.util.spec_from_file_location("cnot_acceptance_python_producer", PYTHON_PRODUCER_PATH)
PYTHON_PRODUCER = importlib.util.module_from_spec(PYTHON_PRODUCER_SPEC)
PYTHON_PRODUCER_SPEC.loader.exec_module(PYTHON_PRODUCER)
SOURCE_SHA = "a" * 40
WHEEL_SHA = "b" * 64
CHECKPOINT_SHA = "c" * 64
MANIFEST_SHA = "d" * 64
SCORER_SHA = "e" * 64
DOMAIN_SHA = "f" * 64
STREAM_SHA = "1" * 64


def make_report(world, identity="attempt-17", case="trained-paired-2-gpu"):
    devices = [
        {"rank": rank, "pid": 4000 + rank, "device": f"cuda:{rank}",
         "gpu_uuid": f"GPU-{rank + 1:04d}"}
        for rank in range(world)
    ]
    report = {
        "world": world,
        "devices": devices,
        "status": "solved",
        "model": {
            "checkpoint_sha256": CHECKPOINT_SHA,
            "search_manifest_sha256": MANIFEST_SHA,
            "scorer_library_sha256": SCORER_SHA,
        },
        "layers": [{"winners_sha256": "2" * 64}],
    }
    acceptance = {
        "schema": protocol.EVENT_SCHEMA,
        "version": protocol.EVENT_VERSION,
        "run_id": identity,
        "case_id": case,
        "source_sha": SOURCE_SHA,
        "wheel_sha256": WHEEL_SHA,
        "world": world,
        "transport": "single_rank_no_exchange" if world == 1 else "rust_stream5_nccl",
    }
    acceptance["report_sha256"] = protocol.report_sha256(report)
    report["acceptance"] = acceptance
    return report


def build_bundle(root, world=2, run_id="attempt-17", case_id="trained-paired-2-gpu"):
    report = make_report(world, run_id, case_id)
    events = []

    def add(kind, rank, fields):
        device = report["devices"][rank]
        event = protocol.make_event(
            run_id=run_id,
            case_id=case_id,
            source_sha=SOURCE_SHA,
            wheel_sha256=WHEEL_SHA,
            kind=kind,
            rank=rank,
            world=world,
            pid=device["pid"],
            device=device["device"],
            gpu_uuid=device["gpu_uuid"],
            fields=fields,
        )
        events.append(event)

    for rank in range(world):
        add("backend_admission", rank, {
            "strict": "1", "domain_abi": "1", "device_abi": "1",
            "scorer": "rust-tqp", "scorer_abi": "1",
            "capabilities": "owners,expansion,full_matrix_distinct",
            "domain_sha256": DOMAIN_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
            "search_manifest_sha256": MANIFEST_SHA, "scorer_library_sha256": SCORER_SHA,
        })
        add("rust_controller", rank, {"implementation": "RustNativeStages", "abi": "1", "active": "1"})
        for operation in ("owners", "expansion", "full_matrix_distinct"):
            add("device_domain_dispatch", rank, {
                "implementation": "DeviceDomain", "operation": operation, "rows": "1",
                "abi": "1", "device_abi": "1", "library_sha256": DOMAIN_SHA,
            })
        for operation in ("parent_scores", "child_values"):
            add("tqp_scorer_dispatch", rank, {
                "implementation": "RustTqpScorer", "operation": operation, "rows": "1",
                "abi": "1", "scorer_abi": "1", "checkpoint_sha256": CHECKPOINT_SHA,
                "search_manifest_sha256": MANIFEST_SHA, "scorer_library_sha256": SCORER_SHA,
            })
        if world == 1:
            add("stream5_absent", rank, {"reason": "single_rank", "exchange": "not_applicable"})
        else:
            add("stream5_admission", rank, {
                "abi": "1", "owned_abi": "1", "exchange_abi": "1", "collectives_abi": "1",
                "library_sha256": STREAM_SHA,
            })
            add("stream5_active", rank, {
                "implementation": "RustStream5/NCCL", "abi": "1", "library_sha256": STREAM_SHA,
            })
            for operation in ("exchange", "all_reduce_i64", "all_gather_i64_into", "broadcast_i64"):
                add("stream5_dispatch", rank, {
                    "implementation": "RustStream5/NCCL", "operation": operation, "rows": "1",
                    "abi": "1", "owned_abi": "1", "exchange_abi": "1", "library_sha256": STREAM_SHA,
                })

    for rank in range(world):
        add("search_terminal", rank, {
            "status": "solved", "report_sha256": report["acceptance"]["report_sha256"],
        })

    receipt_dir = Path(root) / run_id / case_id
    receipt_dir.mkdir(parents=True)
    trace_lines = []
    per_rank = {}
    for event in events:
        serialized = protocol.serialize_event(event)
        trace_lines.append(protocol.EVENT_PREFIX + serialized)
        per_rank.setdefault((event["rank"], event["pid"]), []).append(serialized)
    for (rank, pid), records in per_rank.items():
        path = receipt_dir / f"rank-{rank}-pid-{pid}.jsonl"
        path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return "\n".join(trace_lines) + "\n", report, events


def rewrite_bundle(root, run_id, case_id, events):
    """Rewrite both evidence channels from the same mutated producer records."""
    receipt_dir = Path(root) / run_id / case_id
    for path in receipt_dir.glob("rank-*-pid-*.jsonl"):
        path.unlink()
    per_rank = {}
    trace_lines = []
    for event in events:
        serialized = protocol.serialize_event(event)
        trace_lines.append(protocol.EVENT_PREFIX + serialized)
        per_rank.setdefault((event["rank"], event["pid"]), []).append(serialized)
    for (rank, pid), records in per_rank.items():
        (receipt_dir / f"rank-{rank}-pid-{pid}.jsonl").write_text(
            "\n".join(records) + "\n", encoding="utf-8"
        )
    return "\n".join(trace_lines) + "\n"


class AcceptanceEventProtocolTest(unittest.TestCase):
    def test_rust_serde_event_fixture_has_the_versioned_schema(self):
        parsed = protocol.parse_trace_events(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["schema"], protocol.EVENT_SCHEMA)
        self.assertEqual(parsed[0]["version"], protocol.EVENT_VERSION)
        self.assertEqual(parsed[0]["fields"]["scorer"], "rust-tqp")
        self.assertEqual(parsed[0]["fields"]["domain_abi"], "1")
        self.assertEqual(parsed[0]["fields"]["device_abi"], "1")
        self.assertEqual(parsed[0]["fields"]["scorer_abi"], "1")

    def test_python_producer_matches_the_rust_jsonl_wire_record(self):
        environment = {
            "MGBS_CNOT_ACCEPTANCE_RUN_ID": "attempt-17",
            "MGBS_CNOT_ACCEPTANCE_CASE": "trained-paired-1-gpu",
            "MGBS_CNOT_ACCEPTANCE_SOURCE_SHA": SOURCE_SHA,
            "MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256": WHEEL_SHA,
            "MGBS_CNOT_ACCEPTANCE_GPU_UUIDS": "GPU-0001",
            "CNOT_RUST_EVIDENCE": str(ROOT / "unused-receipts"),
        }
        world = type("WorldFixture", (), {"rank": 0, "size": 1, "device": "cuda:0"})()
        fields = {
            "strict": "1", "scorer": "rust-tqp", "domain_abi": "1",
            "device_abi": "1", "scorer_abi": "1",
            "capabilities": "owners,expansion,full_matrix_distinct",
            "domain_sha256": DOMAIN_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
            "search_manifest_sha256": MANIFEST_SHA, "scorer_library_sha256": SCORER_SHA,
        }
        PYTHON_PRODUCER._PINNED_GPU_OBSERVATIONS.clear()
        with mock.patch.dict(os.environ, environment), \
                mock.patch.object(PYTHON_PRODUCER.os, "getpid", return_value=4000), \
                mock.patch.object(PYTHON_PRODUCER, "observe_current_cuda_context", return_value=SimpleNamespace(device_ordinal=0, gpu_uuid="GPU-0001", context_handle=0xCAFE, thread_id=42)):
            identity = PYTHON_PRODUCER.acceptance_identity()
            event = PYTHON_PRODUCER._event(identity, "backend_admission", world, fields)
            self.assertEqual(event["gpu_uuid"], "GPU-0001")
        serialized = protocol.serialize_event(event)
        expected = FIXTURE.read_text(encoding="utf-8").strip().split(protocol.EVENT_PREFIX, 1)[1]
        self.assertEqual(serialized, expected)

    def test_python_producer_rejects_a_uuid_that_differs_from_the_selected_gpu(self):
        environment = {
            "MGBS_CNOT_ACCEPTANCE_RUN_ID": "attempt-17",
            "MGBS_CNOT_ACCEPTANCE_CASE": "trained-paired-1-gpu",
            "MGBS_CNOT_ACCEPTANCE_SOURCE_SHA": SOURCE_SHA,
            "MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256": WHEEL_SHA,
            "MGBS_CNOT_ACCEPTANCE_GPU_UUIDS": "GPU-expected",
            "CNOT_RUST_EVIDENCE": str(ROOT / "unused-receipts"),
        }
        PYTHON_PRODUCER._PINNED_GPU_OBSERVATIONS.clear()
        with mock.patch.dict(os.environ, environment), \
                mock.patch.object(PYTHON_PRODUCER, "observe_current_cuda_context", return_value=SimpleNamespace(device_ordinal=0, gpu_uuid="GPU-other", context_handle=0xCAFE, thread_id=42)):
            identity = PYTHON_PRODUCER.acceptance_identity()
            with self.assertRaisesRegex(RuntimeError, "observed GPU GPU-other, expected GPU-expected"):
                PYTHON_PRODUCER.gpu_uuid_for_rank(0, 1, device="cuda:0")

    def test_acceptance_rejects_duplicate_expected_physical_uuids(self):
        environment = {
            "MGBS_CNOT_ACCEPTANCE_RUN_ID": "attempt-17",
            "MGBS_CNOT_ACCEPTANCE_CASE": "trained-paired-2-gpu",
            "MGBS_CNOT_ACCEPTANCE_SOURCE_SHA": SOURCE_SHA,
            "MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256": WHEEL_SHA,
            "MGBS_CNOT_ACCEPTANCE_GPU_UUIDS": "GPU-same,GPU-same",
            "CNOT_RUST_EVIDENCE": str(ROOT / "unused-receipts"),
        }
        with mock.patch.dict(os.environ, environment):
            with self.assertRaisesRegex(RuntimeError, "unique physical GPU UUIDs"):
                PYTHON_PRODUCER.acceptance_identity()

    def test_world_one_and_two_receipts_validate_without_cross_run_merging(self):
        for world in (1, 2):
            with self.subTest(world=world), tempfile.TemporaryDirectory() as temporary:
                trace, report, _ = build_bundle(temporary, world)
                receipt = protocol.validate_case(
                    trace, temporary, report,
                    expected_source_sha=SOURCE_SHA,
                    expected_wheel_sha256=WHEEL_SHA,
                    expected_checkpoint_sha256=CHECKPOINT_SHA,
                    expected_manifest_sha256=MANIFEST_SHA,
                    expected_scorer_library_sha256=SCORER_SHA,
                )
                self.assertEqual(receipt["world"], world)
                self.assertEqual(receipt["single_rank_transport"], world == 1)

    def test_repeated_operation_dispatch_is_valid_and_not_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace, report, events = build_bundle(temporary, 2)
            original = next(event for event in events if event["kind"] == "stream5_dispatch"
                            and event["rank"] == 0 and event["fields"]["operation"] == "exchange")
            repeated = dict(original)
            repeated["fields"] = dict(original["fields"], rows="2")
            path = Path(temporary) / report["acceptance"]["run_id"] / report["acceptance"]["case_id"] / "rank-0-pid-4000.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(protocol.serialize_event(repeated) + "\n")
            trace += protocol.EVENT_PREFIX + protocol.serialize_event(repeated) + "\n"
            receipt = protocol.validate_case(
                trace, temporary, report, expected_source_sha=SOURCE_SHA,
                expected_wheel_sha256=WHEEL_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                expected_manifest_sha256=MANIFEST_SHA,
                expected_scorer_library_sha256=SCORER_SHA,
            )
            self.assertEqual(receipt["event_count"], len(events) + 1)

    def test_rejects_missing_rank_or_required_operation(self):
        cases = (
            (lambda item: item["rank"] == 1, "missing rank"),
            (lambda item: item["rank"] == 0 and item["kind"] == "device_domain_dispatch"
             and item["fields"]["operation"] == "full_matrix_distinct", "missing required operation"),
        )
        for predicate, description in cases:
            with self.subTest(mutation=description), tempfile.TemporaryDirectory() as temporary:
                _trace, report, events = build_bundle(temporary, 2)
                remaining = [item for item in events if not predicate(item)]
                trace = rewrite_bundle(
                    temporary, report["acceptance"]["run_id"],
                    report["acceptance"]["case_id"], remaining,
                )
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_case(trace, temporary, report, expected_source_sha=SOURCE_SHA,
                        expected_wheel_sha256=WHEEL_SHA,
                        expected_checkpoint_sha256=CHECKPOINT_SHA,
                        expected_manifest_sha256=MANIFEST_SHA,
                        expected_scorer_library_sha256=SCORER_SHA,
                    )

    def test_rejects_duplicate_admission_and_terminal_records(self):
        for kind in ("backend_admission", "search_terminal"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                trace, report, events = build_bundle(temporary, 2)
                duplicate = next(item for item in events if item["kind"] == kind and item["rank"] == 0)
                receipt_path = Path(temporary) / report["acceptance"]["run_id"] / report["acceptance"]["case_id"] / "rank-0-pid-4000.jsonl"
                with receipt_path.open("a", encoding="utf-8") as stream:
                    stream.write(protocol.serialize_event(duplicate) + "\n")
                trace += protocol.EVENT_PREFIX + protocol.serialize_event(duplicate) + "\n"
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_case(
                        trace, temporary, report, expected_source_sha=SOURCE_SHA,
                        expected_wheel_sha256=WHEEL_SHA,
                        expected_checkpoint_sha256=CHECKPOINT_SHA,
                        expected_manifest_sha256=MANIFEST_SHA,
                        expected_scorer_library_sha256=SCORER_SHA,
                    )

    def test_rejects_foreign_run_or_case_and_different_trace_invocation(self):
        for key, value in (("run_id", "other-run"), ("case_id", "other-case")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                trace, report, events = build_bundle(temporary, 2)
                events[0][key] = value
                trace = protocol.EVENT_PREFIX + protocol.serialize_event(events[0]) + "\n"
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_case(
                        trace, temporary, report, expected_source_sha=SOURCE_SHA,
                        expected_wheel_sha256=WHEEL_SHA,
                        expected_checkpoint_sha256=CHECKPOINT_SHA,
                        expected_manifest_sha256=MANIFEST_SHA,
                        expected_scorer_library_sha256=SCORER_SHA,
                    )

    def test_rejects_a_trace_from_a_different_complete_invocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace_a, report_a, _ = build_bundle(temporary, 2, run_id="attempt-a")
            trace_b, _report_b, _ = build_bundle(temporary, 2, run_id="attempt-b")
            with self.assertRaises(protocol.ValidationError):
                protocol.validate_case(
                    trace_b, temporary, report_a,
                    expected_source_sha=SOURCE_SHA, expected_wheel_sha256=WHEEL_SHA,
                    expected_checkpoint_sha256=CHECKPOINT_SHA,
                    expected_manifest_sha256=MANIFEST_SHA,
                    expected_scorer_library_sha256=SCORER_SHA,
                )

    def test_rejects_wrong_abi_or_backend_hash(self):
        for field, value in (("domain_abi", "9"), ("domain_sha256", "9" * 64)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                trace, report, events = build_bundle(temporary, 2)
                changed = next(item for item in events if item["kind"] == "backend_admission" and item["rank"] == 0)
                changed["fields"][field] = value
                receipt_path = Path(temporary) / report["acceptance"]["run_id"] / report["acceptance"]["case_id"] / "rank-0-pid-4000.jsonl"
                lines = receipt_path.read_text(encoding="utf-8").splitlines()
                parsed = [json.loads(line) for line in lines]
                parsed[0] = changed
                receipt_path.write_text("\n".join(protocol.serialize_event(item) for item in parsed) + "\n", encoding="utf-8")
                # Keep the trace and receipt internally consistent so the ABI/hash check is exercised.
                records = protocol.parse_trace_events(trace)
                records[0] = changed
                trace = "\n".join(protocol.EVENT_PREFIX + protocol.serialize_event(item) for item in records) + "\n"
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_case(
                        trace, temporary, report, expected_source_sha=SOURCE_SHA,
                        expected_wheel_sha256=WHEEL_SHA,
                        expected_checkpoint_sha256=CHECKPOINT_SHA,
                        expected_manifest_sha256=MANIFEST_SHA,
                        expected_scorer_library_sha256=SCORER_SHA,
                    )

    def test_rejects_invalid_world_duplicate_gpu_and_report_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace, report, _ = build_bundle(temporary, 2)
            report["world"] = 3
            with self.assertRaises(protocol.ValidationError):
                protocol.validate_case(
                    trace, temporary, report, expected_source_sha=SOURCE_SHA,
                    expected_wheel_sha256=WHEEL_SHA,
                    expected_checkpoint_sha256=CHECKPOINT_SHA,
                    expected_manifest_sha256=MANIFEST_SHA,
                    expected_scorer_library_sha256=SCORER_SHA,
                )

        with tempfile.TemporaryDirectory() as temporary:
            trace, report, events = build_bundle(temporary, 2)
            report["devices"][1]["gpu_uuid"] = report["devices"][0]["gpu_uuid"]
            report["acceptance"]["report_sha256"] = protocol.report_sha256(report)
            terminal_events = [item for item in events if item["kind"] == "search_terminal"]
            for item in terminal_events:
                item["fields"]["report_sha256"] = report["acceptance"]["report_sha256"]
            for rank in range(2):
                receipt_path = Path(temporary) / report["acceptance"]["run_id"] / report["acceptance"]["case_id"] / f"rank-{rank}-pid-{4000 + rank}.jsonl"
                parsed = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
                for item in parsed:
                    if item["kind"] == "search_terminal":
                        item["fields"]["report_sha256"] = report["acceptance"]["report_sha256"]
                receipt_path.write_text("\n".join(protocol.serialize_event(item) for item in parsed) + "\n", encoding="utf-8")
            trace = "\n".join(protocol.EVENT_PREFIX + protocol.serialize_event(item) for item in events) + "\n"
            with self.assertRaises(protocol.ValidationError):
                protocol.validate_case(
                    trace, temporary, report, expected_source_sha=SOURCE_SHA,
                    expected_wheel_sha256=WHEEL_SHA,
                    expected_checkpoint_sha256=CHECKPOINT_SHA,
                    expected_manifest_sha256=MANIFEST_SHA,
                    expected_scorer_library_sha256=SCORER_SHA,
                )

        with tempfile.TemporaryDirectory() as temporary:
            trace, report, _ = build_bundle(temporary, 1)
            report["acceptance"]["report_sha256"] = "9" * 64
            with self.assertRaises(protocol.ValidationError):
                protocol.validate_case(
                    trace, temporary, report, expected_source_sha=SOURCE_SHA,
                    expected_wheel_sha256=WHEEL_SHA,
                    expected_checkpoint_sha256=CHECKPOINT_SHA,
                    expected_manifest_sha256=MANIFEST_SHA,
                    expected_scorer_library_sha256=SCORER_SHA,
                )


if __name__ == "__main__":
    unittest.main()
