#!/usr/bin/env python3
"""CPU-only Vast target and lifetime-budget contract tests."""

import unittest
from decimal import Decimal

from vast_instance_contract import (
    OLD_INSTANCE_ID,
    acceptance_plan,
    find_owned_target,
    parse_cap,
    validate_instance_id,
)


def instance(**changes):
    row = {
        "id": 987654321,
        "actual_status": "running",
        "gpu_name": "RTX 3060",
        "num_gpus": 2,
        "dph_total": 0.48,
        "start_date": 1_800_000_000.0,
        "public_ipaddr": "192.0.2.40",
        "ports": {"22/tcp": [{"HostPort": 22440}]},
    }
    row.update(changes)
    return row


class VastInstanceContractTests(unittest.TestCase):
    def test_only_positive_decimal_new_instance_id_is_accepted(self):
        self.assertEqual(validate_instance_id("987654321"), "987654321")
        for value in ("", "0", "-1", "1.5", "123x", OLD_INSTANCE_ID):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_instance_id(value)

    def test_spend_cap_is_bounded_by_remaining_ampere_budget(self):
        self.assertEqual(parse_cap("10.00"), Decimal("10.00"))
        for value in ("0", "-1", "10.01", "NaN", "Infinity", "oops"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cap(value)

    def test_target_must_be_owned_exactly_two_desktop_rtx_3060(self):
        payload = {"instances": [instance()]}
        self.assertIs(find_owned_target(payload, "987654321"), payload["instances"][0])
        with self.assertRaisesRegex(ValueError, "not uniquely present"):
            find_owned_target({"instances": []}, "987654321")
        for changes in (
            {"num_gpus": 1},
            {"num_gpus": 4},
            {"gpu_name": "RTX 3060 Ti"},
            {"gpu_name": "RTX 3060 Laptop GPU"},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "2x desktop RTX 3060"):
                find_owned_target({"instances": [instance(**changes)]}, "987654321")

    def test_runtime_uses_instance_age_rate_and_shutdown_reserve(self):
        row = instance()
        plan = acceptance_plan(row, Decimal("10"), now_epoch=1_800_001_800)
        # 30 minutes have already elapsed at $0.48/hour; reserve another
        # 20 minutes of the $10 lifetime allowance for setup and stopping.
        self.assertEqual(plan["elapsed_before_dispatch_seconds"], 1800)
        self.assertEqual(plan["allowed_remote_runtime_seconds"], 5400)
        self.assertEqual(plan["ssh_host"], "192.0.2.40")
        self.assertEqual(plan["ssh_port"], 22440)

    def test_stale_instance_and_missing_ssh_or_rate_fail_closed(self):
        for changes, message in (
            ({"actual_status": "stopped"}, "must already be running"),
            ({"public_ipaddr": None, "ssh_host": None}, "SSH host"),
            ({"ports": {}}, "SSH port"),
            ({"dph_total": 0}, "hourly rate"),
            ({"start_date": 1_799_900_000}, "less than 20 minutes"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                acceptance_plan(instance(**changes), Decimal("1"), now_epoch=1_800_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
