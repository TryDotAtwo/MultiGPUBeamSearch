"""Hosted CPU tests for the final CNOT 1/2-GPU acceptance verifier."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import cnot_acceptance_events as protocol
import test_cnot_acceptance_events as event_fixture
import verify_cnot_gpu_pair
from permutation_acceptance import permutation_report_sha256


SOURCE_SHA = "a" * 40
WHEEL_SHA = "b" * 64
CHECKPOINT_SHA = event_fixture.CHECKPOINT_SHA
MANIFEST_SHA = event_fixture.MANIFEST_SHA
SCORER_SHA = event_fixture.SCORER_SHA
RUN_ID = "hosted-attempt-17"
MODEL = {
    "upstream_commit": "c" * 40,
    "checkpoint_sha256": CHECKPOINT_SHA,
    "search_manifest_sha256": MANIFEST_SHA,
    "scorer_library_sha256": SCORER_SHA,
    "feature_registry_sha256": "1" * 64,
    "training_manifest_sha256": "2" * 64,
    "training_result_sha256": "3" * 64,
    "scorer_output_dtype": "float32",
    "feature_ids": [f"feature-{index}" for index in range(6)],
}


def _build_case(root: Path, case_id: str, world: int, mode: str, width: int, shortlist: int):
    _trace, report, events = event_fixture.build_bundle(
        root, world=world, run_id=RUN_ID, case_id=case_id
    )
    report.update(
        n=4,
        mode=mode,
        width=width,
        shortlist=shortlist,
        root_sha256="4" * 64,
        engine="rust",
        bitwise_replay=True,
        precision_contract={"scorer_output_dtype": "float32", "selection_dtype": "float64"},
        model=dict(MODEL),
        layers=[
            {"winners_sha256": "5" * 64},
            {"winners_sha256": "6" * 64},
        ],
    )
    report["acceptance"]["report_sha256"] = protocol.report_sha256(report)
    for event in events:
        if event["kind"] == "search_terminal":
            event["fields"]["report_sha256"] = report["acceptance"]["report_sha256"]
    trace = event_fixture.rewrite_bundle(root, RUN_ID, case_id, events)
    backend_receipt = protocol.validate_case(
        trace,
        root,
        report,
        expected_source_sha=SOURCE_SHA,
        expected_wheel_sha256=WHEEL_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_manifest_sha256=MANIFEST_SHA,
        expected_scorer_library_sha256=SCORER_SHA,
    )
    result_name = {
        "cnot-trained-qhat-1-gpu": "cnot-trained-qhat-1-gpu.json",
        "cnot-trained-deep-paired-1-gpu": "cnot-trained-deep-paired-1-gpu.json",
        "cnot-trained-deep-paired-2-gpu": "cnot-trained-deep-paired-2-gpu.json",
    }[case_id]
    (root / result_name).write_text(json.dumps(report), encoding="utf-8")
    backend_dir = root / "backend-events"
    backend_dir.mkdir(exist_ok=True)
    (backend_dir / f"{case_id}.json").write_text(json.dumps(backend_receipt), encoding="utf-8")
    return report



def _permutation_report_sha256(report):
    return permutation_report_sha256(report)


def _build_permutation_report():
    winners = [
        {"lo": "0000000000000001", "hi": "1000000000000001"},
        {"lo": "0000000000000002", "hi": "1000000000000002"},
        {"lo": "0000000000000003", "hi": "1000000000000003"},
    ]

    def smoke(world):
        devices = list(range(world))
        pids = [5100 + world * 10 + rank for rank in range(world)]
        return {
            "world_size": world,
            "requested_devices": devices,
            "devices": devices,
            "gpu_names": ["NVIDIA RTX 3060"] * world,
            "engine": "rust",
            "exact_distance": 5,
            "scramble": [0, 1, 2, 3, 4],
            "worker_pids": pids,
            "worker_pids_reused": True,
            "searches": [
                {
                    "call": call,
                    "path": [0, 1, 2, 3, 4],
                    "length": 5,
                    "replay_valid": True,
                    "backend": "native",
                    "inference_backend": "mlp",
                    "devices": devices,
                    "elapsed_seconds": 0.25,
                }
                for call in (1, 2)
            ],
        }

    def saturated(world):
        input_counts = [3] if world == 1 else [2, 1]
        pids = [5200 + world * 10 + rank for rank in range(world)]
        uuids = [f"{rank + 1:032x}" for rank in range(world)]
        next_counts = [3] if world == 1 else [2, 1]
        candidate_counts = [12] if world == 1 else [7, 5]
        request_counts = [3] if world == 1 else [2, 1]
        ranks = [
            {
                "rank": rank,
                "pid": pids[rank],
                "gpu_uuid": uuids[rank],
                "frontier_size": input_counts[rank],
                "generated": input_counts[rank] * 10,
                "final_global_less": 0,
                "final_global_equal": 12,
                "final_total_available": 12,
                "final_global_keep": 3,
                "final_candidate_count": candidate_counts[rank],
                "final_request_count": request_counts[rank],
                "next_frontier_size": next_counts[rank],
                "frontier_snapshot": f"frontier-rank-{rank}.bin",
            }
            for rank in range(world)
        ]
        build_key = ("7" if world == 1 else "8") * 64
        binary_sha = ("9" if world == 1 else "a") * 64
        calls = [
            {
                "call_id": call,
                "session_reused": call == 2,
                "worker_pids": pids,
                "build_key": build_key,
                "binary_sha256": binary_sha,
                "log_path": f"/remote/test/{world}-{call}.log",
                "eligible_distinct": 12,
                "beam_width": 3,
                "cross_source_duplicates": 1,
                "rank_evidence": [dict(row) for row in ranks],
                "ordered_survivor_hashes": list(winners),
            }
            for call in (1, 2)
        ]
        return {
            "case_id": f"permutation-saturated-b3-world-{world}",
            "world_size": world,
            "devices": list(range(world)),
            "worker_pids": pids,
            "gpu_names": ["NVIDIA RTX 3060"] * world,
            "fixture_start": [0, 1, 2, 3, 4],
            "scramble": [0, 1, 2],
            "exact_distance": 3,
            "input_counts": input_counts,
            "eligible_distinct": 12,
            "keep_count": 3,
            "cross_source_duplicates": 1,
            "ordered_survivor_hashes": list(winners),
            "calls": calls,
            "independent_replay": True,
        }

    report = {
        "fixture": "cayleypy.LRX(4)",
        "saturated_fixture": "cayleypy.LRX(5)",
        "state_len": 24,
        "move_count": 10,
        "saturated_state_len": 5,
        "saturated_move_count": 10,
        "model": "known finite FP16 MLP artifact",
        "source_sha": SOURCE_SHA,
        "wheel_sha256": WHEEL_SHA,
        "acceptance_run_id": RUN_ID,
        "python": "3.11.0",
        "torch": "2.9.0+cpu",
        "cuda": None,
        "runs": [smoke(1), smoke(2)],
        "saturated_b3": [saturated(1), saturated(2)],
    }
    report["report_sha256"] = _permutation_report_sha256(report)
    return report


def _write_permutation_report(root, report):
    (root / "permutation-1-and-2-gpu.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def _build_campaign(root: Path):
    _build_case(root, "cnot-trained-qhat-1-gpu", 1, "qhat", 8, 11)
    left = _build_case(root, "cnot-trained-deep-paired-1-gpu", 1, "proposal", 2, 3)
    right = _build_case(root, "cnot-trained-deep-paired-2-gpu", 2, "proposal", 2, 3)
    _write_permutation_report(root, _build_permutation_report())
    return left, right


def _validate(root: Path, instance_id="987654321"):
    return verify_cnot_gpu_pair.validate(
        root,
        SOURCE_SHA,
        instance_id,
        MANIFEST_SHA,
        CHECKPOINT_SHA,
        SCORER_SHA,
        WHEEL_SHA,
        RUN_ID,
    )


class VerifyCnotGpuPairTests(unittest.TestCase):
    def test_serialized_per_case_receipts_join_to_one_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_campaign(root)
            result = _validate(root)
            self.assertTrue(result["paired_cnot_survivor_hashes_match"])
            self.assertEqual(result["paired_cnot_world_sizes"], [1, 2])
            self.assertEqual(result["permutation_report_sha256"], _build_permutation_report()["report_sha256"])
            self.assertEqual(result["permutation_saturated_world_sizes"], [1, 2])
            self.assertTrue(result["permutation_saturated_survivor_hashes_match"])
            self.assertEqual(
                result["strict_backend_cases"]["cnot-trained-deep-paired-2-gpu"]["rank_devices"],
                [
                    {"rank": 0, "pid": 4000, "device": "cuda:0", "gpu_uuid": "GPU-0001"},
                    {"rank": 1, "pid": 4001, "device": "cuda:1", "gpu_uuid": "GPU-0002"},
                ],
            )
            self.assertEqual(json.loads((root / "receipt.json").read_text())["gpu_test"], "passed")


    def test_receipt_binds_saturated_permutation_report_and_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_campaign(root)
            result = _validate(root)
            report = json.loads((root / "permutation-1-and-2-gpu.json").read_text())
            self.assertEqual(result["permutation_report_sha256"], report["report_sha256"])
            self.assertEqual(result["permutation_saturated_world_sizes"], [1, 2])
            self.assertTrue(result["permutation_saturated_survivor_hashes_match"])

    def test_rejects_world_size_only_permutation_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_campaign(root)
            _write_permutation_report(root, {"runs": [{"world_size": 1}, {"world_size": 2}]})
            with self.assertRaises(ValueError):
                _validate(root)

    def test_rejects_foreign_or_incomplete_saturated_permutation_evidence(self):
        mutations = (
            "source",
            "wheel",
            "run",
            "no_digest",
            "unsaturated",
            "missing_rank",
            "duplicate_pid",
            "duplicate_uuid",
            "rank_did_no_work",
            "session_not_reused",
            "winner_mismatch",
            "smoke_not_reused",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _build_campaign(root)
                report_path = root / "permutation-1-and-2-gpu.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                case = report["saturated_b3"][1]
                rank = case["calls"][0]["rank_evidence"][0]
                resign = True
                if mutation == "source":
                    report["source_sha"] = "f" * 40
                elif mutation == "wheel":
                    report["wheel_sha256"] = "e" * 64
                elif mutation == "run":
                    report["acceptance_run_id"] = "foreign-run"
                elif mutation == "no_digest":
                    report.pop("report_sha256", None)
                    resign = False
                elif mutation == "unsaturated":
                    for saturated_case in report["saturated_b3"]:
                        saturated_case["eligible_distinct"] = 3
                        for call in saturated_case["calls"]:
                            call["eligible_distinct"] = 3
                            for evidence in call["rank_evidence"]:
                                evidence["final_global_equal"] = 3
                                evidence["final_total_available"] = 3
                elif mutation == "missing_rank":
                    case["calls"][0]["rank_evidence"].pop()
                elif mutation == "duplicate_pid":
                    case["calls"][0]["rank_evidence"][1]["pid"] = case["calls"][0]["rank_evidence"][0]["pid"]
                elif mutation == "duplicate_uuid":
                    case["calls"][0]["rank_evidence"][1]["gpu_uuid"] = case["calls"][0]["rank_evidence"][0]["gpu_uuid"]
                elif mutation == "rank_did_no_work":
                    rank["generated"] = 0
                elif mutation == "session_not_reused":
                    case["calls"][1]["session_reused"] = False
                elif mutation == "winner_mismatch":
                    case["ordered_survivor_hashes"][0]["hi"] = "f" * 16
                elif mutation == "smoke_not_reused":
                    report["runs"][1]["worker_pids_reused"] = False
                if resign:
                    report["report_sha256"] = _permutation_report_sha256(report)
                _write_permutation_report(root, report)
                with self.assertRaises(ValueError):
                    _validate(root)

    def test_rejects_permutation_report_body_changed_after_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_campaign(root)
            report_path = root / "permutation-1-and-2-gpu.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["saturated_b3"][0]["fixture_start"][0] ^= 1
            _write_permutation_report(root, report)
            with self.assertRaises(ValueError):
                _validate(root)

    def test_rejects_different_pair_survivors_or_foreign_case_identity(self):
        for mutation in ("survivors", "case"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                left, right = _build_campaign(root)
                if mutation == "survivors":
                    right["layers"][0]["winners_sha256"] = "9" * 64
                    (root / "cnot-trained-deep-paired-2-gpu.json").write_text(
                        json.dumps(right), encoding="utf-8"
                    )
                else:
                    left["acceptance"]["case_id"] = "foreign-case"
                    (root / "cnot-trained-deep-paired-1-gpu.json").write_text(
                        json.dumps(left), encoding="utf-8"
                    )
                with self.assertRaises(ValueError):
                    _validate(root)

    def test_rejects_missing_rank_device_evidence_and_duplicate_gpu(self):
        for mutation in ("missing", "duplicate_gpu"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _build_campaign(root)
                path = root / "backend-events" / "cnot-trained-deep-paired-2-gpu.json"
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if mutation == "missing":
                    receipt["rank_devices"].pop()
                else:
                    receipt["rank_devices"][1]["gpu_uuid"] = receipt["rank_devices"][0]["gpu_uuid"]
                path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _validate(root)

    def test_rejects_report_world_mutated_after_case_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_campaign(root)
            path = root / "cnot-trained-deep-paired-2-gpu.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["world"] = 1
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate(root)

    def test_rejects_missing_report_digest_root_or_survivor_hashes(self):
        mutations = ("report_sha256", "root_sha256", "winners_sha256")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _build_campaign(root)
                names = (
                    "cnot-trained-qhat-1-gpu.json",
                    "cnot-trained-deep-paired-1-gpu.json",
                    "cnot-trained-deep-paired-2-gpu.json",
                )
                for name in names:
                    path = root / name
                    report = json.loads(path.read_text(encoding="utf-8"))
                    if mutation == "report_sha256":
                        report["acceptance"].pop("report_sha256", None)
                    elif mutation == "root_sha256":
                        report.pop("root_sha256", None)
                    else:
                        for layer in report["layers"]:
                            layer.pop("winners_sha256", None)
                    if mutation != "report_sha256":
                        report["acceptance"]["report_sha256"] = protocol.report_sha256(report)
                    path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _validate(root)

    def test_backend_verdict_carries_and_matches_report_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_campaign(root)
            report_path = root / "cnot-trained-deep-paired-2-gpu.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_digest = protocol.report_sha256(report)
            receipt_path = root / "backend-events" / "cnot-trained-deep-paired-2-gpu.json"
            backend_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(backend_receipt.get("report_sha256"), report_digest)
            backend_receipt["report_sha256"] = "9" * 64
            receipt_path.write_text(json.dumps(backend_receipt), encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate(root)

    def test_rejects_legacy_or_non_decimal_instance_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_campaign(root)
            for instance_id in ("61080396", "not-an-id", "0"):
                with self.subTest(instance_id=instance_id), self.assertRaises(ValueError):
                    _validate(root, instance_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
