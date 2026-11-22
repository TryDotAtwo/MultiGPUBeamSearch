from __future__ import annotations

import unittest
from unittest.mock import patch

import vast_campaign_watchdog as watchdog
from vast_campaign_guard import campaign_label


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
    return {"instances": [row]}


def stopped(payload):
    row = dict(payload["instances"][0])
    row["actual_status"] = "stopped"
    return {"instances": [row]}


def restored(payload):
    row = dict(payload["instances"][0])
    row["actual_status"] = "stopped"
    row["label"] = "old label"
    return {"instances": [row]}


class VastCampaignWatchdogTests(unittest.TestCase):
    def test_completed_workflow_stops_exact_labelled_id_and_confirms_api_readback(self):
        running = instance()
        with (
            patch.object(watchdog, "_instances", side_effect=[running, running, stopped(running), restored(running)]),
            patch.object(watchdog, "_run_status", return_value="completed"),
            patch.object(watchdog, "_request_json", return_value={"success": True}) as request,
            patch.object(watchdog.time, "sleep"),
        ):
            report = watchdog.execute_watchdog(
                vast_token="vast-test",
                github_token="github-test",
                repository="TryDotAtwo/MultiGPUBeamSearch",
                now_epoch=1_800_001_000,
            )
        item = report["campaigns"][0]
        self.assertEqual(item["reason"], "workflow_completed")
        self.assertEqual(item["instance_id"], "987654321")
        self.assertTrue(item["confirmed"])
        self.assertTrue(report["storage_billing_continues_after_stop"])
        stop_call = request.call_args_list[0]
        restore_call = request.call_args_list[1]
        self.assertEqual(stop_call.kwargs["method"], "PUT")
        self.assertEqual(stop_call.args[0], "https://console.vast.ai/api/v0/instances/987654321/")
        self.assertEqual(restore_call.kwargs["payload"], {"label": "old label"})
        self.assertTrue(item["prior_label_restored"])
        self.assertEqual(request.call_count, 2)
        self.assertNotIn("DELETE", [call.kwargs.get("method") for call in request.call_args_list])

    def test_budget_deadline_stops_even_when_github_readback_is_unavailable(self):
        running = instance()
        with (
            patch.object(watchdog, "_instances", side_effect=[running, running, stopped(running), restored(running)]),
            patch.object(watchdog, "_run_status", side_effect=watchdog.WatchdogError("runner API unavailable")),
            patch.object(watchdog, "_request_json", return_value={"success": True}),
            patch.object(watchdog.time, "sleep"),
        ):
            report = watchdog.execute_watchdog(
                vast_token="vast-test",
                github_token="github-test",
                repository="TryDotAtwo/MultiGPUBeamSearch",
                now_epoch=1_800_070_514 - 1200,
            )
        self.assertEqual(report["campaigns"][0]["reason"], "billing_deadline")
        self.assertTrue(report["campaigns"][0]["confirmed"])
        self.assertTrue(report["campaigns"][0]["prior_label_restored"])
        self.assertFalse(report["errors"])

    def test_active_workflow_before_deadline_is_not_stopped(self):
        running = instance()
        with (
            patch.object(watchdog, "_instances", return_value=running),
            patch.object(watchdog, "_run_status", return_value="in_progress"),
            patch.object(watchdog, "_request_json") as request,
        ):
            report = watchdog.execute_watchdog(
                vast_token="vast-test",
                github_token="github-test",
                repository="TryDotAtwo/MultiGPUBeamSearch",
                now_epoch=1_800_001_000,
            )
        self.assertEqual(report["campaigns"][0]["reason"], "active_before_deadline")
        request.assert_not_called()

    def test_label_bound_to_another_instance_never_stops(self):
        payload = instance()
        payload["instances"][0]["id"] = 987654322
        with patch.object(watchdog, "_instances", return_value=payload):
            report = watchdog.execute_watchdog(
                vast_token="vast-test",
                github_token="github-test",
                repository="TryDotAtwo/MultiGPUBeamSearch",
                now_epoch=1_800_001_000,
            )
        self.assertIn("different instance ID", report["campaigns"][0]["error"])

    def test_stop_readback_is_bounded_and_failure_is_reported(self):
        running = instance()
        with (
            patch.object(watchdog, "_instances", side_effect=[running, running] + [running] * 6) as fetch,
            patch.object(watchdog, "_run_status", return_value="completed"),
            patch.object(watchdog, "_request_json", return_value={"success": True}),
            patch.object(watchdog.time, "sleep"),
        ):
            report = watchdog.execute_watchdog(
                vast_token="vast-test",
                github_token="github-test",
                repository="TryDotAtwo/MultiGPUBeamSearch",
                now_epoch=1_800_001_000,
            )
        self.assertEqual(fetch.call_count, 8)
        self.assertIn("stop readback was not confirmed", report["campaigns"][0]["error"])
        self.assertEqual(len(report["errors"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
