"""CPU regression for one sealed-dataset resolution and small rank-local snapshots."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "cnot_scorer"))

import validate_cnot_handoff as preflight  # noqa: E402
from cnot_panel_snapshot import load_panel_v2_snapshot  # noqa: E402
from test_handoff import PanelV2HandoffTests  # noqa: E402


class PanelSnapshotReadCountTests(unittest.TestCase):
    def test_one_full_dataset_read_feeds_one_and_two_rank_runtime_snapshots(self):
        fixture = PanelV2HandoffTests()
        fixture.setUp()
        try:
            dataset_path = fixture.dataset_path.resolve()
            original_read_bytes = Path.read_bytes
            dataset_reads = 0

            def count_dataset_read(path):
                nonlocal dataset_reads
                if Path(path).resolve() == dataset_path:
                    dataset_reads += 1
                return original_read_bytes(path)

            with tempfile.TemporaryDirectory() as temporary:
                snapshot_path = Path(temporary) / "resolved-panel.json"
                with patch.object(Path, "read_bytes", count_dataset_read):
                    summary = preflight.resolve_handoff(fixture.root, ROOT)
                    preflight.write_summary(summary, snapshot_path)
                    raw = original_read_bytes(snapshot_path)
                    snapshot_sha = hashlib.sha256(raw).hexdigest()

                    # One local process for world size 1, then both local ranks for world size 2.
                    snapshots = [
                        load_panel_v2_snapshot(
                            snapshot_path,
                            fixture.root,
                            snapshot_sha,
                            expected_checkpoint_sha256=summary["checkpoint_sha256"],
                            expected_search_manifest_sha256=summary["search_manifest_sha256"],
                        )
                        for _ in range(3)
                    ]

                    self.assertEqual(dataset_reads, 1)
                    self.assertEqual(
                        [root["case_id"] for root in snapshots[0].panel_roots],
                        fixture.case_ids,
                    )
                    self.assertEqual(
                        [root["start_packed_rows"] for root in snapshots[0].panel_roots],
                        fixture.root_rows,
                    )
                    self.assertTrue(all(
                        snapshot.search == summary["search"]
                        and snapshot.expansion_upper_bound == summary["expansion_upper_bound"]
                        and snapshot.search_manifest_sha256 == summary["search_manifest_sha256"]
                        for snapshot in snapshots
                    ))

                    with self.assertRaisesRegex(ValueError, "changed after preflight"):
                        load_panel_v2_snapshot(
                            snapshot_path,
                            fixture.root,
                            "0" * 64,
                            expected_checkpoint_sha256=summary["checkpoint_sha256"],
                            expected_search_manifest_sha256=summary["search_manifest_sha256"],
                        )
                    self.assertEqual(dataset_reads, 1)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
