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
        "dph_base": 0.46,
        "storage_cost": 0.02,
        "inet_down_cost": 0.05,
        "inet_up_cost": 0.06,
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
        # Thirty minutes have elapsed at $0.48/hour. The remaining lifetime
        # deadline includes staging, transfer, retrieval and stop reserves.
        self.assertEqual(plan["elapsed_before_dispatch_seconds"], 1800)
        self.assertEqual(plan["allowed_remote_runtime_seconds"], 5400)
        self.assertEqual(plan["budget_deadline_epoch"], 1_800_075_000.0)
        self.assertEqual(plan["stop_reserve_seconds"], 1200)
        self.assertEqual(plan["retrieval_reserve_seconds"], 300)
        self.assertEqual(plan["setup_transfer_reserve_seconds"], 1260)
        self.assertEqual(plan["ssh_host"], "192.0.2.40")
        self.assertEqual(plan["ssh_port"], 22440)

    def test_spend_plan_reserves_bounded_network_transfer_and_stopped_storage(self):
        plan = acceptance_plan(instance(), Decimal("10"), now_epoch=1_800_001_800)
        self.assertEqual(plan["network_transfer_reserve_usd"], "0.11811160064")
        self.assertEqual(plan["storage_retention_reserve_hours"], 24)
        self.assertEqual(plan["storage_retention_reserve_usd"], "0.48")
        self.assertEqual(plan["max_source_archive_bytes"], 1_073_741_824)
        self.assertEqual(plan["max_result_archive_bytes"], 1_073_741_824)
        self.assertEqual(plan["budget_deadline_epoch"], 1_800_070_514.0)

    def test_missing_network_or_storage_rates_fail_closed(self):
        for field in ("storage_cost", "inet_down_cost", "inet_up_cost"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "storage and ingress/egress bandwidth rates"):
                values = instance()
                values.pop(field)
                acceptance_plan(values, Decimal("10"), now_epoch=1_800_000_000)

    def test_ssh_host_and_port_are_selected_from_one_complete_api_pair(self):
        direct = acceptance_plan(
            instance(ssh_host="proxy.vast.example", ssh_port=30222),
            Decimal("10"),
            now_epoch=1_800_000_000,
        )
        self.assertEqual((direct["ssh_host"], direct["ssh_port"]), ("proxy.vast.example", 30222))
        mapped = acceptance_plan(
            instance(ssh_host=None, ssh_port=None),
            Decimal("10"),
            now_epoch=1_800_000_000,
        )
        self.assertEqual((mapped["ssh_host"], mapped["ssh_port"]), ("192.0.2.40", 22440))

    def test_incomplete_ssh_route_is_not_mixed_across_api_fields(self):
        with self.assertRaisesRegex(ValueError, "SSH port"):
            acceptance_plan(
                instance(ssh_host="proxy.vast.example", ssh_port=None, public_ipaddr="192.0.2.40", ports={}),
                Decimal("10"),
                now_epoch=1_800_000_000,
            )
        with self.assertRaisesRegex(ValueError, "SSH port"):
            acceptance_plan(
                instance(ssh_host=None, ssh_port=None, public_ipaddr="192.0.2.40", ports={"22/tcp": [{"HostPort": 22.5}]}),
                Decimal("10"),
                now_epoch=1_800_000_000,
            )

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
