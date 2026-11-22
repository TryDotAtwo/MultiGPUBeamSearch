from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest


from validate_cnot_handoff import build_summary, write_summary


def _metric(index, case_id, status, cost):
    return {
        "panel_index": index,
        "case_id": case_id,
        "dimension": 3,
        "alphabet_id": "all_to_all_unit_cnot:3",
        "status": status,
        "physical_cost": cost,
    }


class ValidateCnotHandoffTests(unittest.TestCase):
    def make_resolved(self, root):
        search = root / "manifests" / "search.json"
        search.parent.mkdir(parents=True)
        search.write_text("{}", encoding="utf-8")
        roots = (
            {
                "panel_index": 2,
                "case_id": "locked-a",
                "dimension": 3,
                "alphabet_id": "all_to_all_unit_cnot:3",
                "orientation": "row_target_xor_row_control",
                "start_packed_rows": [1, 3, 5],
            },
            {
                "panel_index": 8,
                "case_id": "locked-b",
                "dimension": 3,
                "alphabet_id": "all_to_all_unit_cnot:3",
                "orientation": "row_target_xor_row_control",
                "start_packed_rows": [1, 2, 4],
            },
        )
        candidate_cases = {
            "locked-a": _metric(2, "locked-a", "solved", 2),
            "locked-b": _metric(8, "locked-b", "depth_limit", None),
        }
        comparison = {
            "baseline_metrics": {"solved_count": 1, "total_physical_cost": 3},
            "candidate_metrics": {
                "solved_count": 1,
                "total_physical_cost": 2,
                "cases": candidate_cases,
            },
        }
        return SimpleNamespace(
            locked_test_comparison=comparison,
            locked_test_status="length_gain",
            locked_test_comparison_sha256="a" * 64,
            producer_source_revision="b" * 40,
            producer_emitter_sha256="c" * 64,
            accepted_updates=8,
            checkpoint_sha256="d" * 64,
            dataset_sha256="e" * 64,
            training_manifest_sha256="f" * 64,
            training_result_sha256="1" * 64,
            search_manifest_sha256="2" * 64,
            baseline_search_manifest_sha256="3" * 64,
            panel_indices=(2, 8),
            panel_ids=("locked-a", "locked-b"),
            panel_digest="4" * 64,
            search={"global_width": 36, "shortlist": 4, "max_depth": 8, "max_expansions": 201},
            expansion_upper_bound=201,
            orientation="row_target_xor_row_control",
            artifact_path_policy="same_run_root_mount",
            search_manifest=search,
            panel_roots=roots,
        )

    def test_writes_ordered_pinned_summary_without_private_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = self.make_resolved(root)
            summary = build_summary(resolved, root)
            self.assertEqual(summary["panel"], "test")
            self.assertEqual(summary["panel_ids"], ["locked-a", "locked-b"])
            self.assertEqual(summary["panel_indices"], [2, 8])
            self.assertEqual(summary["locked_test_status"], "length_gain")
            self.assertEqual(summary["expansion_upper_bound"], 201)
            self.assertEqual(summary["candidate_cases"][0]["physical_cost"], 2)
            self.assertEqual(summary["candidate_cases"][1]["physical_cost"], None)
            self.assertNotIn(str(root), str(summary))
            output = root / "preflight.json"
            write_summary(summary, output)
            self.assertEqual(__import__("json").loads(output.read_text()), summary)

    def test_rejects_candidate_case_order_or_root_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = self.make_resolved(root)
            resolved.panel_ids = ("locked-b", "locked-a")
            with self.assertRaises(ValueError):
                build_summary(resolved, root)

    def test_rejects_candidate_aggregate_that_does_not_match_ordered_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = self.make_resolved(root)
            resolved.locked_test_comparison["candidate_metrics"]["total_physical_cost"] = 3
            with self.assertRaisesRegex(ValueError, "aggregate"):
                build_summary(resolved, root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = self.make_resolved(root)
            resolved.locked_test_comparison["candidate_metrics"]["solved_count"] = 2
            with self.assertRaisesRegex(ValueError, "aggregate"):
                build_summary(resolved, root)

    def test_rejects_a_solved_candidate_without_a_physical_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = self.make_resolved(root)
            resolved.locked_test_comparison["candidate_metrics"]["cases"]["locked-a"]["physical_cost"] = None
            with self.assertRaises(ValueError):
                build_summary(resolved, root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
