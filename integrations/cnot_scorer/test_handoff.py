"""Synthetic tests for resolving the producer's private panel-v2 handoff."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from handoff import (
    EXPECTED_PRODUCER_EMITTER_SHA256,
    EXPECTED_PRODUCER_SOURCE_REVISION,
    HandoffError,
    resolve_panel_v2_handoff,
    validate_search_contract,
)


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
        self.producer_emitter_sha256 = EXPECTED_PRODUCER_EMITTER_SHA256
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
                "state": {
                    "n": 2,
                    "alphabet_id": "all_to_all_unit_cnot:2",
                    "packed_rows": [1, 3] if index == 3 else [3, 2] if index == 9 else [1, 2],
                },
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
            "schema": "cnot103_multigpu_beam_handoff_v3",
            "consumer": "MultiGPUBeamSearch",
            "producer_source_revision": EXPECTED_PRODUCER_SOURCE_REVISION,
            "producer_emitter_sha256": self.producer_emitter_sha256,
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
            "baseline_search_manifest": {},
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
        self.baseline_search_manifest = self.manifests / "baseline-test-hamming.json"
        dataset_value = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        baseline_roots = []
        for index, case_id in zip(self.indices, self.case_ids):
            state = dataset_value["panels"]["test"][index]["state"]
            baseline_roots.append({
                "panel_index": index,
                "case_id": case_id,
                "dimension": state["n"],
                "alphabet_id": state["alphabet_id"],
                "start_packed_rows": list(state["packed_rows"]),
            })
        write_json(
            self.baseline_search_manifest,
            {
                "schema": "cnot103_hamming_search_v1",
                "source_revision": self.handoff["producer_source_revision"],
                "dataset": {"path": str(self.dataset_path), "sha256": self.dataset_sha},
                "panel": "test",
                "panel_indices": list(self.indices),
                "panel_ids": list(self.case_ids),
                "panel_roots": baseline_roots,
                "panel_digest": self.panel_digest,
                "runtime_contract": "native_device_search_v13",
                "search": dict(self.search),
            },
        )
        self.handoff["baseline_search_manifest"] = {
            "path": "manifests/baseline-test-hamming.json",
            "sha256": digest_bytes(self.baseline_search_manifest.read_bytes()),
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
            "search_manifest_sha256": self.handoff["baseline_search_manifest"]["sha256"],
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
            "panel_ids": list(self.case_ids),
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
                "producer_emitter_sha256": self.producer_emitter_sha256,
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
        self.assertEqual(resolved.baseline_search_manifest, self.baseline_search_manifest.resolve())
        self.assertEqual(
            resolved.baseline_search_manifest_sha256,
            self.handoff["baseline_search_manifest"]["sha256"],
        )
        self.assertEqual(resolved.training_manifest, self.training_manifest.resolve())
        self.assertEqual(resolved.training_result, self.training_result.resolve())
        self.assertEqual(resolved.checkpoint_sha256, self.checkpoint_sha)
        self.assertEqual(resolved.dataset_sha256, self.dataset_sha)
        self.assertEqual(resolved.accepted_updates, self.accepted_updates)
        self.assertEqual(resolved.producer_emitter_sha256, self.producer_emitter_sha256)
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
        self.assertEqual(
            [root["start_packed_rows"] for root in resolved.panel_roots],
            [[1, 3], [3, 2]],
        )

    def test_resolver_parses_the_same_artifact_bytes_it_hashes(self):
        artifact_paths = {
            self.training_manifest.resolve(),
            self.training_result.resolve(),
            self.search_manifest.resolve(),
            self.baseline_search_manifest.resolve(),
        }
        reads = {}
        original_read_bytes = Path.read_bytes

        def counted_read_bytes(path):
            resolved = path.resolve()
            if resolved in artifact_paths:
                reads[resolved] = reads.get(resolved, 0) + 1
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", counted_read_bytes):
            resolve_panel_v2_handoff(self.root, self.receipt)

        self.assertEqual(reads, {path: 1 for path in artifact_paths})

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

    def test_rejects_an_unpinned_producer_source_revision(self):
        self.handoff["producer_source_revision"] = "f" * 40
        self.receipt_source_revision = "f" * 40
        self.write_receipt()
        with self.assertRaisesRegex(HandoffError, "producer source revision"):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_baseline_manifest_panel_swap_after_receipt_repin(self):
        manifest = json.loads(self.baseline_search_manifest.read_text(encoding="utf-8"))
        manifest["panel_ids"] = list(reversed(self.case_ids))
        write_json(self.baseline_search_manifest, manifest)
        digest = digest_bytes(self.baseline_search_manifest.read_bytes())
        self.handoff["baseline_search_manifest"]["sha256"] = digest
        self.comparison["baseline_metrics"]["receipt_pins"]["search_manifest_sha256"] = digest
        self.handoff["locked_test_comparison_sha256"] = canonical_digest(self.comparison)
        self.top_level_comparison = copy.deepcopy(self.comparison)
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
            lambda value: value["baseline_metrics"].update(panel_ids=list(reversed(self.case_ids))),
            lambda value: value["candidate_metrics"].update(panel_ids=list(reversed(self.case_ids))),
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

    def test_rejects_a_campaign_emitter_pin_mismatch(self):
        value = json.loads(self.receipt.read_text(encoding="utf-8"))
        value["producer_emitter_sha256"] = "f" * 64
        write_json(self.receipt, value)
        with self.assertRaisesRegex(HandoffError, "emitter pin"):
            resolve_panel_v2_handoff(self.root, self.receipt)

    def test_rejects_feature_schema_abi_mismatch(self):
        self.handoff["model"]["feature_schema"] = "runtime_packet_layout_v24"
        self.write_receipt()
        with self.assertRaises(HandoffError):
            resolve_panel_v2_handoff(self.root, self.receipt)

    @unittest.skipUnless(
        os.environ.get("CNOT103_PRODUCER_CHECKOUT")
        and os.environ.get("CNOT103_PRODUCER_SOURCE_SHA"),
        "current producer cross-compatibility runs in hosted CI",
    )
    def test_current_producer_emitter_builds_a_consumer_accepted_handoff(self):
        producer_root = Path(os.environ["CNOT103_PRODUCER_CHECKOUT"]).resolve()
        expected_sha = os.environ["CNOT103_PRODUCER_SOURCE_SHA"]
        observed_sha = subprocess.run(
            ["git", "-C", str(producer_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(observed_sha, expected_sha)
        scripts = producer_root / ".github" / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            source_path = scripts / "run_full_training.py"
            spec = importlib.util.spec_from_file_location(
                "cnot103_pinned_producer_run_full_training", source_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            producer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(producer)
        finally:
            sys.path.remove(str(scripts))

        comparison = copy.deepcopy(self.comparison)
        comparison["baseline_metrics"]["receipt_pins"]["source_revision"] = expected_sha
        emitted = producer.build_multigpu_beam_handoff(
            root=self.root,
            source_revision=expected_sha,
            accepted_updates=self.accepted_updates,
            checkpoint_sha256=self.checkpoint_sha,
            training_manifest_path=self.training_manifest,
            training_result_path=self.training_result,
            search_manifest_path=self.search_manifest,
            baseline_search_manifest_path=self.baseline_search_manifest,
            expected_panel="test",
            expected_panel_indices=list(self.indices),
            expected_panel_ids=list(self.case_ids),
            expected_panel_digest=self.panel_digest,
            expected_search=dict(self.search),
            locked_test_comparison=comparison,
        )
        self.assertEqual(emitted["schema"], "cnot103_multigpu_beam_handoff_v3")
        self.assertEqual(
            emitted["producer_emitter_sha256"],
            producer.multigpu_handoff_emitter_sha256(),
        )
        self.handoff = emitted
        self.comparison = comparison
        self.receipt_source_revision = expected_sha
        self.producer_emitter_sha256 = emitted["producer_emitter_sha256"]
        self.top_level_comparison = copy.deepcopy(comparison)
        self.write_receipt()
        resolved = resolve_panel_v2_handoff(self.root, self.receipt)
        self.assertEqual(resolved.producer_source_revision, expected_sha)
        self.assertEqual(
            resolved.producer_emitter_sha256,
            producer.multigpu_handoff_emitter_sha256(),
        )
        self.assertEqual(resolved.panel_ids, tuple(self.case_ids))

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
