"""Synthetic tests for resolving the producer's private panel-v2 handoff."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from handoff import HandoffError, resolve_panel_v2_handoff


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return digest_bytes(raw)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


class PanelV2HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.manifests = self.root / "manifests"
        self.reports = self.root / "reports"
        self.manifests.mkdir()
        self.reports.mkdir()

        self.accepted_updates = 64
        self.checkpoint_sha = "c" * 64
        self.dataset_sha = "d" * 64
        self.indices = [3, 9]
        self.case_ids = ["locked-case-a", "locked-case-b"]
        self.panel_digest = canonical_digest({
            "panel": "test",
            "indices": self.indices,
            "ids": self.case_ids,
        })
        self.search = {
            "global_width": 36,
            "shortlist": 4,
            "max_depth": 8,
            "max_expansions": 201,
        }
        self.training_manifest = write_json(
            self.manifests / "training.json",
            {"schema": "cnot103_training_run_v1", "fixture": "synthetic"},
        )
        self.training_result = write_json(
            self.reports / "training-result.json",
            {
                "schema": "cnot103_training_result_v1",
                "end_progress": {"accepted": self.accepted_updates},
                "checkpoint": {"sha256": self.checkpoint_sha},
                "model_parameters": 136421807,
                "feature_model_inputs": 102,
                "dataset_sha256": self.dataset_sha,
            },
        )
        self.search_manifest = self.manifests / "locked-test-search.json"
        self.comparison = {
            "status": "length_gain",
            "case_count": 2,
            "baseline_solved": 2,
            "candidate_solved": 2,
            "paired_cost_delta": -1,
        }
        self.handoff = {
            "schema": "cnot103_multigpu_beam_handoff_v1",
            "consumer": "MultiGPUBeamSearch",
            "producer_source_revision": "e" * 40,
            "accepted_updates": self.accepted_updates,
            "checkpoint_sha256": self.checkpoint_sha,
            "model": {
                "structural_blocks": 8,
                "parameters": 136421807,
                "feature_schema": "runtime_packet_layout_v25",
                "input_slots": 102,
            },
            "dataset_sha256": self.dataset_sha,
            "training_manifest": {
                "path": "manifests/training.json",
                "sha256": digest_bytes(self.training_manifest.read_bytes()),
            },
            "training_result": {
                "path": "reports/training-result.json",
                "sha256": digest_bytes(self.training_result.read_bytes()),
            },
            "search_manifest": {},
            "panel": "test",
            "panel_indices": list(self.indices),
            "panel_ids": list(self.case_ids),
            "panel_digest": self.panel_digest,
            "search": dict(self.search),
            "head": "qhat",
            "locked_test_status": self.comparison["status"],
            "locked_test_comparison_sha256": canonical_digest(self.comparison),
        }
        write_json(
            self.search_manifest,
            {
                "schema": "cnot103_panel_search_v2",
                "training_manifest": {
                    "path": str(self.training_manifest),
                    "sha256": self.handoff["training_manifest"]["sha256"],
                },
                "training_result": {
                    "path": str(self.training_result),
                    "sha256": self.handoff["training_result"]["sha256"],
                },
                "panel": "test",
                "panel_indices": list(self.indices),
                "panel_ids": list(self.case_ids),
                "panel_digest": self.panel_digest,
                "search": dict(self.search),
                "head": "qhat",
            },
        )
        self.handoff["search_manifest"] = {
            "path": "manifests/locked-test-search.json",
            "sha256": digest_bytes(self.search_manifest.read_bytes()),
        }
        self.receipt = self.reports / "learning-curve.private.json"
        self.write_receipt()

    def tearDown(self):
        self.temp.cleanup()

    def write_receipt(self):
        write_json(
            self.receipt,
            {
                "schema": "cnot103_learning_curve_campaign_v1",
                "multi_gpu_beam_handoff": copy.deepcopy(self.handoff),
                "locked_test": {
                    "status": "evaluated",
                    "accepted_updates": self.accepted_updates,
                    "checkpoint_sha256": self.checkpoint_sha,
                    "comparison": copy.deepcopy(self.comparison),
                    "consumer_handoff": copy.deepcopy(self.handoff),
                },
            },
        )

    def test_resolves_a_fully_pinned_locked_test_handoff(self):
        resolved = resolve_panel_v2_handoff(self.root, self.receipt)
        self.assertEqual(resolved.search_manifest, self.search_manifest.resolve())
        self.assertEqual(resolved.training_manifest, self.training_manifest.resolve())
        self.assertEqual(resolved.training_result, self.training_result.resolve())
        self.assertEqual(resolved.checkpoint_sha256, self.checkpoint_sha)
        self.assertEqual(resolved.dataset_sha256, self.dataset_sha)
        self.assertEqual(resolved.accepted_updates, self.accepted_updates)
        self.assertEqual(resolved.panel_indices, tuple(self.indices))
        self.assertEqual(resolved.panel_ids, tuple(self.case_ids))
        self.assertEqual(resolved.panel_digest, self.panel_digest)
        self.assertEqual(resolved.locked_test_status, "length_gain")

    def test_rejects_a_changed_training_result_even_when_json_still_parses(self):
        self.training_result.write_text('{"schema":"changed"}', encoding="utf-8")
        with self.assertRaises(HandoffError):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_a_checkpoint_swap(self):
        self.handoff["checkpoint_sha256"] = "f" * 64
        self.write_receipt()
        with self.assertRaises(HandoffError):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_a_changed_locked_test_comparison(self):
        self.comparison["paired_cost_delta"] = 1
        self.write_receipt()
        with self.assertRaises(HandoffError):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_search_manifest_panel_order_change_with_resealed_file_hash(self):
        manifest = json.loads(self.search_manifest.read_text(encoding="utf-8"))
        manifest["panel_ids"] = list(reversed(self.case_ids))
        write_json(self.search_manifest, manifest)
        self.handoff["search_manifest"]["sha256"] = digest_bytes(
            self.search_manifest.read_bytes()
        )
        self.write_receipt()
        with self.assertRaises(HandoffError):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_artifact_paths_that_escape_the_private_run_root(self):
        self.handoff["search_manifest"]["path"] = "../outside-search.json"
        self.write_receipt()
        with self.assertRaises(HandoffError):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_a_model_contract_mismatch(self):
        self.handoff["model"]["input_slots"] = 101
        self.write_receipt()
        with self.assertRaises(HandoffError):
            resolve_panel_v2_handoff(self.root, self.receipt)


if __name__ == "__main__":
    unittest.main()
