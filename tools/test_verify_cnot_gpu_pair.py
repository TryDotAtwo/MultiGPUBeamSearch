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


def _build_campaign(root: Path):
    _build_case(root, "cnot-trained-qhat-1-gpu", 1, "qhat", 8, 11)
    left = _build_case(root, "cnot-trained-deep-paired-1-gpu", 1, "proposal", 2, 3)
    right = _build_case(root, "cnot-trained-deep-paired-2-gpu", 2, "proposal", 2, 3)
    (root / "permutation-1-and-2-gpu.json").write_text(
        json.dumps({"runs": [{"world_size": 1}, {"world_size": 2}]}), encoding="utf-8"
    )
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
            self.assertEqual(
                result["strict_backend_cases"]["cnot-trained-deep-paired-2-gpu"]["rank_devices"],
                [
                    {"rank": 0, "pid": 4000, "device": "cuda:0", "gpu_uuid": "GPU-0001"},
                    {"rank": 1, "pid": 4001, "device": "cuda:1", "gpu_uuid": "GPU-0002"},
                ],
            )
            self.assertEqual(json.loads((root / "receipt.json").read_text())["gpu_test"], "passed")

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
