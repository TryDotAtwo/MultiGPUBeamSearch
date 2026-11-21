from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "tools" / "vast_public_status.py"


class VastPublicStatusTests(unittest.TestCase):
    def test_status_contains_only_timestamp_and_allowlisted_outcomes(self):
        if not WRITER.is_file():
            self.fail("redacted public status writer is missing")
        spec = importlib.util.spec_from_file_location("vast_public_status", WRITER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        report = module.make_public_status(
            checked_at_utc="2026-11-22T10:00:00+00:00",
            inventory_query="success",
            target_preflight="passed",
            gpu_acceptance="failed",
            cleanup="unverified",
            watchdog="not_run",
        )
        self.assertEqual(
            report,
            {
                "checked_at_utc": "2026-11-22T10:00:00+00:00",
                "inventory_query": "success",
                "target_preflight": "passed",
                "gpu_acceptance": "failed",
                "cleanup": "unverified",
                "watchdog": "not_run",
            },
        )
        self.assertNotIn("instance_id", json.dumps(report))
        self.assertNotIn("checkpoint", json.dumps(report))
        self.assertNotIn("private", json.dumps(report))
        with self.assertRaisesRegex(ValueError, "not allowlisted"):
            module.make_public_status(
                checked_at_utc="2026-11-22T10:00:00+00:00",
                inventory_query="success",
                target_preflight="passed",
                gpu_acceptance="failed",
                cleanup="unverified",
                watchdog="private label and ID",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
