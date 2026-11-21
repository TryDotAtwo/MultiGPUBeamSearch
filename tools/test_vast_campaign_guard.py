from __future__ import annotations

import unittest
from decimal import Decimal

from vast_campaign_guard import (
    campaign_deadline,
    campaign_label,
    find_campaign_target,
    parse_campaign_label,
    stop_reason,
    verify_restored_label,
)


def instance(**changes):
    row = {
        "id": 987654321,
        "actual_status": "running",
        "gpu_name": "RTX 3060",
        "num_gpus": 2,
        "label": campaign_label(987654321, 12345, 1, "10", "old label"),
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
    def test_campaign_marker_is_bound_to_instance_attempt_cap_and_prior_label(self):
        label = campaign_label(987654321, 12345, 2, "9.50", "old label")
        self.assertEqual(
            label,
            "mgbs-cnot-acceptance:987654321:12345:2:9.5:b2xkIGxhYmVs",
        )
        self.assertEqual(parse_campaign_label(label), {
            "instance_id": "987654321",
            "run_id": 12345,
            "attempt": 2,
            "spend_cap_usd": Decimal("9.5"),
            "original_label": "old label",
        })
        for values in (
            ("61080396", 12345, 1, "10", ""),
            (987654321, 0, 1, "10", ""),
            (987654321, 12345, 0, "10", ""),
            (987654321, 12345, 1, "10.01", ""),
            (987654321, 12345, 1, "NaN", ""),
            (987654321, 12345, 1, "10", "x" * 1100),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                campaign_label(*values)

    def test_selects_only_the_exact_account_owned_campaign_target(self):
        row = instance()
        self.assertIs(find_campaign_target({"instances": [row]}, row["label"]), row)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_campaign_target({"instances": [], "label": row["label"]}, row["label"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_campaign_target({"instances": [row, dict(row)]}, row["label"])

    def test_rejects_legacy_wrong_hardware_and_moved_labels(self):
        row = instance(id=61080396)
        with self.assertRaises(ValueError):
            find_campaign_target({"instances": [row]}, row["label"])
        row = instance(id=987654322)
        with self.assertRaisesRegex(ValueError, "different instance ID"):
            find_campaign_target({"instances": [row]}, row["label"])
        for changes in ({"gpu_name": "RTX 3060 Ti"}, {"num_gpus": 1}):
            target = instance(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                find_campaign_target({"instances": [target]}, target["label"])

    def test_campaign_deadline_does_not_depend_on_ssh_endpoint(self):
        row = instance(ssh_host=None, ssh_port=None, public_ipaddr=None, ports={})
        deadline = campaign_deadline(row, row["label"], now_epoch=1_800_001_800)
        self.assertEqual(deadline["budget_deadline_epoch"], 1_800_070_514.0)

    def test_external_watchdog_stops_completed_or_deadline_expired_campaign(self):
        self.assertEqual(stop_reason("completed", 5000, 1200, 1000), "workflow_completed")
        self.assertEqual(stop_reason("in_progress", 5000, 1200, 3800), "billing_deadline")
        self.assertIsNone(stop_reason("in_progress", 5000, 1200, 3799))
        self.assertEqual(stop_reason("unknown", 5000, 1200, 3800), "billing_deadline")

    def test_label_restoration_requires_fresh_exact_id_readback(self):
        campaign = instance()
        identity = parse_campaign_label(campaign["label"])
        restored = dict(campaign, label=identity["original_label"])
        self.assertIs(verify_restored_label({"instances": [restored]}, campaign["label"]), restored)
        moved = dict(restored, id=987654322)
        with self.assertRaisesRegex(ValueError, "exactly one campaign instance"):
            verify_restored_label({"instances": [moved]}, campaign["label"])
        not_restored = dict(campaign, label="unexpected")
        with self.assertRaisesRegex(ValueError, "did not confirm restoration"):
            verify_restored_label({"instances": [not_restored]}, campaign["label"])

    def test_watchdog_fails_closed_for_unknown_workflow_status_before_deadline(self):
        with self.assertRaisesRegex(ValueError, "workflow status"):
            stop_reason("unknown", 5000, 1200, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
