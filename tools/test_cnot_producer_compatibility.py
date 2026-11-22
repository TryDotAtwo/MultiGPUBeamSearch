"""Cheap unit checks for panel-v2 CI input selection and status carry-forward."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from cnot_producer_compatibility import (
    PANEL_V2_BUILD_INPUTS,
    STATE_SCHEMA,
    _changed_paths,
    choose_action,
    decide,
    finalize,
    input_fingerprint,
    requires_full_build,
    _write_outputs,
)


class ProducerCompatibilityGateTests(unittest.TestCase):
    def test_docs_logs_and_unrelated_components_do_not_rebuild_the_pinned_adapter(self):
        self.assertFalse(requires_full_build([
            "README.md",
            "memory/CHANGELOG.md",
            "integrations/cnot_scorer/README.md",
            "test_results/run/build.log",
            "rust/beam-runtime/src/search_session.rs",
            "integrations/cayleypy_native/multigpubeamsearch/cnot.py",
        ]))

    def test_adapter_fixture_pin_and_gate_changes_require_a_full_build(self):
        for path in (
            ".github/workflows/cnot-producer-compatibility.yml",
            "integrations/cnot_scorer/build.py",
            "integrations/cnot_scorer/bridge_panel_v2.rs",
            "integrations/cnot_scorer/handoff.py",
            "integrations/cnot_scorer/test_handoff.py",
            "tools/cnot_producer_compatibility.py",
            "tools/test_cnot_producer_compatibility.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(requires_full_build([path]))

    def test_identical_inputs_reuse_pass_but_preserve_failure_and_unknown_runs(self):
        self.assertEqual(
            choose_action(force_full=False, changed_inputs=False, previous_status="passed"),
            "reuse_pass",
        )
        self.assertEqual(
            choose_action(force_full=False, changed_inputs=False, previous_status="failed"),
            "carry_failure",
        )
        self.assertEqual(
            choose_action(force_full=False, changed_inputs=False, previous_status="unknown"),
            "run_full",
        )
        self.assertEqual(
            choose_action(force_full=True, changed_inputs=False, previous_status="passed"),
            "run_full",
        )
        self.assertEqual(
            choose_action(force_full=False, changed_inputs=True, previous_status="passed"),
            "run_full",
        )

    def test_docs_only_reuses_a_same_input_pass_and_carries_a_same_input_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            state_path.write_text(json.dumps({
                "schema": STATE_SCHEMA,
                "fingerprint": "c" * 64,
                "status": "passed",
            }), encoding="utf-8")
            self.assertEqual(
                decide(state_path, "c" * 64, force_full=False, changed_inputs=False, output=None)["action"],
                "reuse_pass",
            )
            state_path.write_text(json.dumps({
                "schema": STATE_SCHEMA,
                "fingerprint": "c" * 64,
                "status": "failed",
            }), encoding="utf-8")
            self.assertEqual(
                decide(state_path, "c" * 64, force_full=False, changed_inputs=False, output=None)["action"],
                "carry_failure",
            )


    def test_long_pr_docs_commit_ignores_old_adapter_change_but_preserves_cached_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init", "-q")
            git("config", "user.email", "ci@example.invalid")
            git("config", "user.name", "CI fixture")
            adapter = root / "integrations/cnot_scorer/build.py"
            adapter.parent.mkdir(parents=True)
            adapter.write_text("old pinned build input\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")

            adapter.write_text("changed adapter input\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "older PR adapter change")

            (root / "README.md").write_text("docs-only head commit\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "later docs-only PR commit")
            head = git("rev-parse", "HEAD")

            changed = _changed_paths(
                "pull_request",
                {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}},
                root,
            )
            self.assertEqual(changed, ["README.md"])
            self.assertFalse(requires_full_build(changed))

            adapter.write_text("new current adapter input\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "current PR adapter change")
            changed_head = git("rev-parse", "HEAD")
            changed_again = _changed_paths(
                "pull_request",
                {"pull_request": {"base": {"sha": base}, "head": {"sha": changed_head}}},
                root,
            )
            self.assertEqual(changed_again, ["integrations/cnot_scorer/build.py"])
            self.assertTrue(requires_full_build(changed_again))

            fingerprint = "d" * 64
            state_path = root / "cached-state.json"
            for previous, expected in (("passed", "reuse_pass"), ("failed", "carry_failure")):
                state_path.write_text(json.dumps({
                    "schema": STATE_SCHEMA,
                    "fingerprint": fingerprint,
                    "status": previous,
                }), encoding="utf-8")
                decision = decide(
                    state_path,
                    fingerprint,
                    force_full=False,
                    changed_inputs=requires_full_build(changed),
                    output=None,
                )
                self.assertEqual(decision["action"], expected)

            state_path.unlink()
            unknown = decide(
                state_path,
                fingerprint,
                force_full=False,
                changed_inputs=requires_full_build(changed),
                output=None,
            )
            self.assertEqual(unknown["action"], "run_full")

    def test_fingerprint_uses_adapter_inputs_and_ignores_docs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in PANEL_V2_BUILD_INPUTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"input:{relative}\n", encoding="utf-8")
            first = input_fingerprint(root)
            (root / "integrations/cnot_scorer/README.md").write_text(
                "documentation update\n", encoding="utf-8"
            )
            self.assertEqual(first, input_fingerprint(root))
            bridge = root / "integrations/cnot_scorer/bridge_panel_v2.rs"
            bridge.write_text("adapter source changed\n", encoding="utf-8")
            self.assertNotEqual(first, input_fingerprint(root))

    def test_finalizer_carries_a_same_input_failure_and_records_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            failed = finalize(
                state_path, "a" * 64, "carry_failure", "success", "skipped", "skipped"
            )
            self.assertEqual(failed["status"], "failed")
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], STATE_SCHEMA)
            self.assertEqual(saved["fingerprint"], "a" * 64)
            summary_path = Path(temporary) / "summary.md"
            passed = finalize(
                state_path, "b" * 64, "run_full", "success", "success", "success", summary_path
            )
            self.assertEqual(passed["status"], "passed")
            self.assertTrue(summary_path.read_bytes().endswith(b"\n"))
            self.assertNotIn(b"\\n", summary_path.read_bytes())

    def test_github_output_has_a_real_line_ending(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "github-output"
            _write_outputs(output, {"handoff_tests": "passed"})
            self.assertEqual(output.read_bytes(), b"handoff_tests=passed\n")
            self.assertNotIn(b"\\n", output.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
