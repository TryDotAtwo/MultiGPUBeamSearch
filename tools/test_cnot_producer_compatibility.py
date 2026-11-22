"""CPU tests for selecting real builds across PR and manual-dispatch cache scopes."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cnot_producer_compatibility import cache_refs, last_result, main


class BuildSelectionTest(unittest.TestCase):
    def test_trust_only_the_current_pr_and_its_same_repository_branch(self):
        event = {"repository": {"id": 7}, "pull_request": {"head": {"repo": {"id": 7}, "ref": "topic"}}}
        self.assertEqual(cache_refs(event, "refs/pull/10/merge"), ["refs/pull/10/merge", "refs/heads/topic"])
        event["pull_request"]["head"]["repo"]["id"] = 9
        self.assertEqual(cache_refs(event, "refs/pull/10/merge"), ["refs/pull/10/merge"])

    def test_newer_manual_failure_wins_until_a_real_retry_passes(self):
        prefix = "input-"
        pr, branch = "refs/pull/10/merge", "refs/heads/topic"
        entries = {}
        def record(ref, seq, verdict):
            entries[ref] = [{"ref": ref, "created_at": "2026-01-01T12:00:00Z",
                             "id": seq, "key": f"{prefix}{seq}-1-{verdict}"}]
        def lookup(key, ref):
            self.assertEqual(key, prefix)
            return entries.get(ref, [])
        self.assertEqual(last_result(prefix, [pr, branch], lookup), "run")
        record(pr, 1, "success")
        record(branch, 2, "failure")
        self.assertEqual(last_result(prefix, [pr, branch], lookup), "failure")
        record(pr, 3, "success")
        self.assertEqual(last_result(prefix, [pr, branch], lookup), "success")
        self.assertEqual(last_result("other-", [pr], lambda *_: entries[pr]), "run")

    def test_dispatch_and_retry_build_even_after_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            event, output = Path(tmp) / "event", Path(tmp) / "output"
            event.write_text(json.dumps({"repository": {"id": 7}}))
            env = {"GITHUB_EVENT_PATH": str(event), "GITHUB_OUTPUT": str(output),
                   "PANEL_INPUTS": "abc", "ImageOS": "ubuntu", "ImageVersion": "1",
                   "PYTHON_VERSION_USED": "3.11", "GITHUB_REF": "refs/heads/topic", "GITHUB_REF_NAME": "topic"}
            for name, attempt, expected in (("pull_request", "1", "success"),
                                            ("workflow_dispatch", "1", "run"), ("pull_request", "2", "run")):
                with self.subTest(name=name, attempt=attempt), patch.dict(os.environ, {
                    **env, "GITHUB_EVENT_NAME": name, "GITHUB_RUN_ATTEMPT": attempt
                }), patch("cnot_producer_compatibility.last_result", return_value="success"):
                    main()
                    self.assertEqual(output.read_text().splitlines()[-1], f"verdict={expected}")


if __name__ == "__main__":
    unittest.main()
