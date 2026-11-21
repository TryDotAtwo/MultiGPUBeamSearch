from __future__ import annotations

import unittest
from decimal import Decimal

from vast_campaign_guard import (
    campaign_label,
    find_campaign_target,
    parse_campaign_label,
    stop_reason,
)


def instance(**changes):
    row = {
        "id": 987654321,
        "actual_status": "running",
        "gpu_name": "RTX 3060",
        "num_gpus": 2,
        "label": campaign_label(12345, 1, "10"),
        "dph_total": 0.48,
        "storage_cost": 0.02,
        "inet_down_cost": 0.05,
        "inet_up_cost": 0.06,
        "start_date": 1_800_000_000.0,
        "public_ipaddr": "192.0.2.40",
        "ports": {"22/tcp": [{"HostPort": 22440}]},
    }
    row.update(changes)
    return row


class VastCampaignGuardTests(unittest.TestCase):
    def test_campaign_marker_is_reproducible_and_bounded(self):
        label = campaign_label(12345, 2, "9.50")
        self.assertEqual(label, "mgbs-cnot-acceptance:12345:2:9.5")
        self.assertEqual(parse_campaign_label(label), {
            "run_id": 12345, "attempt": 2, "spend_cap_usd": Decimal("9.5")
        })
        for values in ((0, 1, "10"), (1, 0, "10"), (1, 1, "10.01"), (1, 1, "NaN")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                campaign_label(*values)

    def test_selects_only_the_exact_account_owned_campaign_target(self):
        row = instance()
        self.assertIs(find_campaign_target({"instances": [row]}, row["label"]), row)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_campaign_target({"instances": []}, row["label"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_campaign_target({"instances": [row, dict(row)]}, row["label"])

    def test_rejects_legacy_or_wrong_hardware_even_with_campaign_label(self):
        for changes in (
            {"id": 61080396},
            {"gpu_name": "RTX 3060 Ti"},
            {"num_gpus": 1},
        ):
            row = instance(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                find_campaign_target({"instances": [row]}, row["label"])

    def test_external_watchdog_stops_completed_or_deadline_expired_campaign(self):
        self.assertEqual(stop_reason("completed", 5000, 1200, 1000), "workflow_completed")
        self.assertEqual(stop_reason("in_progress", 5000, 1200, 3800), "billing_deadline")
        self.assertIsNone(stop_reason("in_progress", 5000, 1200, 3799))

    def test_watchdog_fails_closed_for_unknown_workflow_status(self):
        with self.assertRaisesRegex(ValueError, "workflow status"):
            stop_reason("unknown", 5000, 1200, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
