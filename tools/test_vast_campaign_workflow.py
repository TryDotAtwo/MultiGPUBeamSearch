from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VastCampaignWorkflowTests(unittest.TestCase):
    def test_campaign_marker_and_cleanup_are_bound_to_one_instance(self):
        workflow = (ROOT / ".github/workflows/vast-cnot-control-collectives.yml").read_text(encoding="utf-8")
        self.assertIn("run-name: mgbs-cnot-acceptance", workflow)
        contract = (ROOT / "tools/vast_instance_contract.py").read_text(encoding="utf-8")
        self.assertNotIn("safe_to_stop", contract)
        self.assertIn("group: vast-cnot-control-${{ inputs.instance_id || github.run_id }}", workflow)
        self.assertIn("--run-id \"$GITHUB_RUN_ID\" --attempt \"$GITHUB_RUN_ATTEMPT\"", workflow)

        preflight = workflow.split("id: preflight", 1)[1].split("\n      - name:", 1)[0]
        label_put = preflight.index("instances/${VAST_INSTANCE_ID}/")
        fresh_get = preflight.index('"https://console.vast.ai/api/v0/instances/"', label_put)
        verify = preflight.index("tools/vast_campaign_guard.py verify", fresh_get)
        safe_output = preflight.index('echo "safe_to_stop=true"', verify)
        self.assertLess(label_put, fresh_get)
        self.assertLess(fresh_get, verify)
        self.assertLess(verify, safe_output)

        cleanup = workflow.split("- name: Stop only the validated new instance", 1)[1].split("\n      - name:", 1)[0]
        self.assertIn("CAMPAIGN_LABEL: ${{ steps.preflight.outputs.campaign_label }}", cleanup)
        self.assertLess(cleanup.index("tools/vast_campaign_guard.py verify"), cleanup.index('{"state":"stopped"}'))
        self.assertLess(cleanup.index("tools/vast_campaign_guard.py restore-body"), cleanup.index("tools/vast_campaign_guard.py verify-restored"))

    def test_independent_watchdog_is_scheduled_and_triggered_on_completion(self):
        watchdog = (ROOT / ".github/workflows/vast-campaign-watchdog.yml").read_text(encoding="utf-8")
        self.assertIn('cron: "*/5 * * * *"', watchdog)
        self.assertIn("workflow_run:", watchdog)
        self.assertIn("types:\n      - completed", watchdog)
        self.assertIn("actions: read", watchdog)
        self.assertIn("GITHUB_TOKEN: ${{ github.token }}", watchdog)
        self.assertIn("tools/vast_campaign_watchdog.py", watchdog)


if __name__ == "__main__":
    unittest.main(verbosity=2)
