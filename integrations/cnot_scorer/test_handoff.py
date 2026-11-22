"""Synthetic tests for resolving the producer's private panel-v2 handoff."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from handoff import HandoffError, resolve_panel_v2_handoff, validate_search_contract


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
        records = [
            {
                "lineage_id": "locked-case-a" if index == 3 else "locked-case-b" if index == 9 else f"other-{index}",
                "state": {"n": 2, "alphabet_id": "all_to_all_unit_cnot:2", "packed_rows": [1, 2]},
            }
            for index in range(10)
        ]
        dataset_value = {"schema": "synthetic-sealed-dataset", "panels": {"test": records}}
        dataset_bytes = (json.dumps(dataset_value, sort_keys=True) + "\n").encode("utf-8")
        self.dataset_path = self.root / "data" / "frozen.json"
        self.dataset_path.parent.mkdir(parents=True)
        self.dataset_path.write_bytes(dataset_bytes)
        self.dataset_sha = digest_bytes(dataset_bytes)
        self.training_manifest = write_json(
            self.manifests / "training.json",
            {
                "schema": "cnot103_training_run_v1",
                "dataset": {"path": str(self.dataset_path), "sha256": self.dataset_sha},
            },
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
        self.comparison = self._comparison_fixture()
        self.handoff["locked_test_status"] = self.comparison["status"]
        self.handoff["locked_test_comparison_sha256"] = canonical_digest(self.comparison)
        self.receipt_source_revision = self.handoff["producer_source_revision"]
        self.top_level_comparison = copy.deepcopy(self.comparison)
        self.selected_checkpoint_sha = self.checkpoint_sha
        self.receipt = self.reports / "learning-curve.private.json"
        self.write_receipt()

    def _comparison_fixture(self):
        baseline_pins = {
            "search_manifest_sha256": "a" * 64,
            "dataset_sha256": self.dataset_sha,
            "source_revision": self.handoff["producer_source_revision"],
            "runtime_contract": "native_device_search_v13",
            "training_manifest_sha256": None,
            "training_result_sha256": None,
            "checkpoint_sha256": None,
        }
        candidate_pins = {
            "search_manifest_sha256": self.handoff["search_manifest"]["sha256"],
            "dataset_sha256": self.dataset_sha,
            "training_manifest_sha256": self.handoff["training_manifest"]["sha256"],
            "training_result_sha256": self.handoff["training_result"]["sha256"],
            "checkpoint_sha256": self.checkpoint_sha,
        }
        metrics = {
            "case_count": len(self.indices),
            "panel_indices": list(self.indices),
            "panel_digest": self.panel_digest,
        }
        return {
            "status": "length_gain",
            "reason": "paired physical cost reduced with coverage preserved",
            "baseline_head": "hamming",
            "candidate_head": "qhat",
            "panel": "test",
            "case_count": len(self.indices),
            "baseline_metrics": {**metrics, "receipt_pins": baseline_pins},
            "candidate_metrics": {**metrics, "receipt_pins": candidate_pins},
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_receipt(self):
        write_json(
            self.receipt,
            {
                "schema": "cnot103_learning_curve_campaign_v1",
                "source_revision": self.receipt_source_revision,
                "dataset_sha256": self.dataset_sha,
                "search": dict(self.search),
                "locked_test_panel_digest": self.panel_digest,
                "selected": {
                    "accepted_updates": self.accepted_updates,
                    "checkpoint_sha256": self.selected_checkpoint_sha,
                },
                "locked_test_comparison": copy.deepcopy(self.top_level_comparison),
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
        self.assertEqual(resolved.dataset_path, self.dataset_path.resolve())
        self.assertEqual(resolved.orientation, "row_target_xor_row_control")
        self.assertEqual(resolved.artifact_path_policy, "same_run_root_mount")
        self.assertEqual(
            [(root["panel_index"], root["case_id"], root["dimension"], root["orientation"])
             for root in resolved.panel_roots],
            [(3, "locked-case-a", 2, "row_target_xor_row_control"),
             (9, "locked-case-b", 2, "row_target_xor_row_control")],
        )

    def test_supported_search_geometry_has_exact_v13_frontier_bound(self):
        self.assertEqual(validate_search_contract(self.search), 201)

    def test_rejects_top_level_search_mismatch_before_loading_artifacts(self):
        value = json.loads(self.receipt.read_text(encoding="utf-8"))
        value["search"]["global_width"] = 35
        write_json(self.receipt, value)
        with self.assertRaisesRegex(HandoffError, "top-level receipt"):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_altered_or_boolean_search_budget_fields(self):
        for name, value in (
            ("global_width", 35),
            ("shortlist", 3),
            ("max_depth", 7),
            ("max_expansions", 200),
            ("max_expansions", True),
        ):
            with self.subTest(name=name, value=value):
                changed = copy.deepcopy(self.search)
                changed[name] = value
                with self.assertRaises(HandoffError):
                    validate_search_contract(changed)

    def test_rejects_a_changed_training_result_even_when_json_still_parses(self):
        self.training_result.write_text('{"schema":"changed"}', encoding="utf-8")
        with self.assertRaises(HandoffError):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_a_checkpoint_swap(self):
        self.checkpoint_sha = "f" * 64
        self.handoff["checkpoint_sha256"] = self.checkpoint_sha
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

    def test_rejects_source_revision_mismatch_between_receipt_and_handoff(self):
        self.receipt_source_revision = "f" * 40
        self.write_receipt()
        with self.assertRaisesRegex(HandoffError, "source revision"):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_mismatched_top_level_locked_test_comparison(self):
        self.top_level_comparison["status"] = "coverage_gain"
        self.write_receipt()
        with self.assertRaisesRegex(HandoffError, "comparison"):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_selected_checkpoint_mismatch(self):
        self.selected_checkpoint_sha = "f" * 64
        self.write_receipt()
        with self.assertRaisesRegex(HandoffError, "selected checkpoint"):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_locked_comparison_with_wrong_panel_head_or_metric_pins(self):
        mutations = (
            lambda value: value.update(panel="dev"),
            lambda value: value.update(panel_digest="f" * 64),
            lambda value: value["baseline_metrics"].update(panel_indices=list(reversed(self.indices))),
            lambda value: value["candidate_metrics"].update(panel_digest="f" * 64),
            lambda value: value.update(candidate_head="proposal"),
            lambda value: value["candidate_metrics"]["receipt_pins"].update(checkpoint_sha256="f" * 64),
            lambda value: value["candidate_metrics"]["receipt_pins"].update(dataset_sha256="f" * 64),
            lambda value: value["candidate_metrics"]["receipt_pins"].update(training_manifest_sha256="f" * 64),
            lambda value: value["candidate_metrics"]["receipt_pins"].update(training_result_sha256="f" * 64),
            lambda value: value["candidate_metrics"]["receipt_pins"].update(search_manifest_sha256="f" * 64),
            lambda value: value["baseline_metrics"]["receipt_pins"].update(dataset_sha256="f" * 64),
            lambda value: value["baseline_metrics"]["receipt_pins"].update(source_revision="f" * 40),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.comparison = self._comparison_fixture()
                mutate(self.comparison)
                self.handoff["locked_test_status"] = self.comparison["status"]
                self.handoff["locked_test_comparison_sha256"] = canonical_digest(self.comparison)
                self.write_receipt()
                with self.assertRaises(HandoffError):
                    resolve_panel_v2_handoff(self.root, self.receipt)

    def test_absolute_search_manifest_pins_require_the_same_run_root_mount(self):
        import shutil
        relocated = Path(self.temp.name) / "relocated-run"
        shutil.copytree(self.root, relocated, ignore=shutil.ignore_patterns("relocated-run"))
        with self.assertRaisesRegex(HandoffError, "outside the private run root"):
            resolve_panel_v2_handoff(
                relocated,
                relocated / "reports" / "learning-curve.private.json",
            )


if __name__ == "__main__":
    unittest.main()
