from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VastCampaignWorkflowTests(unittest.TestCase):
    def test_campaign_marker_and_cleanup_are_bound_to_one_instance(self):
        workflow = (ROOT / ".github/workflows/vast-cnot-control-collectives.yml").read_text(encoding="utf-8")
        self.assertIn("run-name: CNOT GPU acceptance", workflow)
        contract = (ROOT / "tools/vast_instance_contract.py").read_text(encoding="utf-8")
        self.assertNotIn("safe_to_stop", contract)
        self.assertIn("group: vast-cnot-control-acceptance", workflow)
        self.assertIn("--run-id \"$GITHUB_RUN_ID\" --attempt \"$GITHUB_RUN_ATTEMPT\"", workflow)

        preflight = workflow.split("id: preflight", 1)[1].split("\n      - name:", 1)[0]
        first_get = preflight.index('"https://console.vast.ai/api/v0/instances/"')
        label_put = preflight.index("instances/${VAST_INSTANCE_ID}/")
        fresh_get = preflight.index('"https://console.vast.ai/api/v0/instances/"', label_put)
        verify = preflight.index("tools/vast_campaign_guard.py verify", fresh_get)
        safe_output = preflight.index('echo "safe_to_stop=true"', verify)
        root_guard = preflight.index('[[ -n "$HANDOFF_RUN_ROOT" ]]')
        self.assertLess(root_guard, first_get)
        self.assertLess(root_guard, label_put)
        self.assertLess(root_guard, fresh_get)
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
        self.assertNotIn('cat "$report"', watchdog)
        self.assertIn('report="${RUNNER_TEMP}/vast-campaign-watchdog-private.json"', watchdog)
        self.assertIn('python3 tools/vast_campaign_watchdog.py --report "$report" >/dev/null 2>&1', watchdog)
        self.assertIn("Write redacted watchdog status", watchdog)
        self.assertIn("Upload redacted watchdog status", watchdog)
        self.assertNotIn('Marked campaigns inspected:', watchdog)
        self.assertNotIn('Confirmed stopped:', watchdog)
        self.assertNotIn('path: test_results/refactor_20261121/vast-campaign-watchdog/', watchdog)
        upload = watchdog.split("Upload redacted watchdog status", 1)[1].split("\n      - name:", 1)[0]
        self.assertIn("cnot-public-status.json", upload)
        self.assertNotIn("watchdog-${GITHUB_RUN_ID}.json", upload)
        self.assertNotIn("vast-campaign-watchdog/", upload)

    def test_public_gpu_workflow_redacts_inputs_logs_and_artifacts(self):
        workflow = (ROOT / ".github/workflows/vast-cnot-control-collectives.yml").read_text(encoding="utf-8")
        self.assertNotIn("inputs.instance_id", workflow)
        self.assertNotIn("inputs.spend_cap_usd", workflow)
        self.assertIn("secrets.VAST_CNOT_ACCEPTANCE_INSTANCE_ID", workflow)
        self.assertIn("secrets.VAST_CNOT_ACCEPTANCE_CAP_USD", workflow)
        self.assertIn("secrets.VAST_CNOT_HANDOFF_RUN_ROOT", workflow)
        self.assertNotIn("memory/CHANGELOG.md", workflow)
        self.assertNotIn('cat "$status"', workflow)
        self.assertNotIn('cat "$remote_log"', workflow)
        self.assertNotIn('cat "$lifecycle"', workflow)
        self.assertIn("Upload redacted operational status", workflow)
        self.assertIn("Write redacted source validation status", workflow)
        self.assertIn('--source-outcome "$SOURCE_OUTCOME"', workflow)
        source_upload = workflow.split("Upload redacted source validation status", 1)[1].split("\n      - name:", 1)[0]
        self.assertIn("cnot-public-status.json", source_upload)
        self.assertNotIn("test_results", source_upload)
        self.assertNotIn("GITHUB_STATUS.md", source_upload)
        upload = workflow.split("Upload redacted operational status", 1)[1].split("\n      - name:", 1)[0]
        self.assertIn("cnot-public-status.json", upload)
        for private_path in (
            "vast-private-preflight.log",
            "vast-status.json",
            "vast-stop-lifecycle.json",
            "beam-gpu-acceptance.log",
            "beam-gpu-acceptance-test-results.tar.gz",
        ):
            self.assertNotIn(private_path, upload)

    def test_private_handoff_runs_before_scorer_build_and_uses_locked_panel_pair(self):
        worker = (ROOT / "tools/vast_cnot_gpu_acceptance.sh").read_text(encoding="utf-8")
        self.assertLess(worker.index("validate_cnot_handoff.py"), worker.index("integrations/cnot_scorer/build.py"))
        self.assertIn('mgbs_vast_cnot_panel_cli_args "${handoff_run_root}"', worker)
        self.assertIn('"${panel_result_dir}/${case_id}.json" "${scorer_library}"', worker)
        panel_cli = (ROOT / "tools/vast_cnot_panel_cli_args.sh").read_text(encoding="utf-8")
        self.assertIn("--case panel", panel_cli)
        self.assertIn("--backend nccl", panel_cli)
        self.assertIn("--mode qhat", panel_cli)
        self.assertIn("--engine rust", panel_cli)
        self.assertIn("--require-rust-scorer", panel_cli)
        self.assertIn('--run-root "$run_root"', panel_cli)
        self.assertIn('--output "$output"', panel_cli)
        self.assertIn('--handoff-preflight "$handoff_preflight"', panel_cli)
        self.assertIn('--handoff-preflight-sha256 "$handoff_preflight_sha256"', panel_cli)
        self.assertIn('run_panel_case cnot-panel-test-qhat-1-gpu "${physical_gpu_uuids[0]}" 1', worker)
        self.assertIn('run_panel_case cnot-panel-test-qhat-2-gpu "${physical_gpu_uuids[0]},${physical_gpu_uuids[1]}" 2', worker)
        self.assertIn('--panel-result-dir "$panel_result_dir" --handoff-preflight "$handoff_preflight"', worker)
        self.assertIn('--manifest "$manifest"', worker, "GL4 direct-manifest runs remain isolated smoke cases")

        workflow = (ROOT / ".github/workflows/vast-cnot-control-collectives.yml").read_text(encoding="utf-8")
        self.assertIn("REMOTE_CNOT_HANDOFF_ROOT_PREFLIGHT_OK status=ready", workflow)
        self.assertIn('VAST_REMOTE_HANDOFF_RUN_ROOT="$HANDOFF_RUN_ROOT"', workflow)
        self.assertIn('VAST_REMOTE_RESULTS_RELATIVE="private-cnot-acceptance-v1"', workflow)
        preflight_parser = (ROOT / "tools/vast_private_preflight.py").read_text(encoding="utf-8")
        self.assertIn("HANDOFF_ROOT_RECEIPT_PATTERN", preflight_parser)
        self.assertIn("handoff_root_preflight", preflight_parser)

    def test_installed_wheel_ci_publishes_only_redacted_status(self):
        workflow = (ROOT / ".github/workflows/installed-wheel-validation.yml").read_text(encoding="utf-8")
        self.assertIn("Write redacted hosted validation status", workflow)
        self.assertIn("Upload redacted hosted validation status", workflow)
        self.assertIn('--source-outcome "$SOURCE_OUTCOME"', workflow)
        self.assertNotIn("| tee ", workflow)
        upload = workflow.split("Upload redacted hosted validation status", 1)[1].split("\n      - name:", 1)[0]
        self.assertIn("cnot-public-status.json", upload)
        self.assertNotIn("test_results/", upload)
        self.assertNotIn(".log", upload)
        self.assertNotIn("hosted-run.md", upload)

    def test_result_transport_never_downloads_or_prints_private_results(self):
        transport = (ROOT / "tools/vast_result_transport.sh").read_text(encoding="utf-8")
        self.assertNotIn("VAST_SSH_RESULTS", transport)
        self.assertIn("VAST_REMOTE_RESULTS_RELATIVE", transport)
        self.assertIn("VAST_REMOTE_HANDOFF_RUN_ROOT", transport)
        self.assertIn('"private-cnot-acceptance-v1"', transport)
        self.assertNotIn("tail -n", transport)
        self.assertNotIn('cat "$VAST_SSH_LOG"', transport)
        self.assertIn("REMOTE_GPU_STATUS=passed", transport)
        self.assertIn("REMOTE_GPU_STATUS=failed", transport)


if __name__ == "__main__":
    unittest.main(verbosity=2)
