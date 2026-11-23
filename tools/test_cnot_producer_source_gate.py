from __future__ import annotations

from pathlib import Path
import unittest

from cnot_producer_source_gate import (
    PRODUCER_SOURCE_SHA,
    REQUIRED_CHECKS,
    evaluate_source_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def valid_checks():
    return {
        name: {"result": "success", "checked_sha": PRODUCER_SOURCE_SHA}
        for name in REQUIRED_CHECKS
    }


def top_level_job_body(workflow: str, name: str) -> str:
    lines = workflow.splitlines()
    marker = f"  {name}:"
    start = next((index for index, line in enumerate(lines) if line == marker), None)
    if start is None:
        raise AssertionError(f"missing top-level job: {name}")
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("  ")
            and not lines[index].startswith("    ")
            and lines[index].rstrip().endswith(":")
        ),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end])


class ProducerSourceGateTests(unittest.TestCase):
    def test_top_level_job_parser_keeps_nested_needs_and_condition(self):
        sample = (
            "jobs:\n"
            "  first:\n"
            "    if: dispatch\n"
            "    needs: [validate, gate]\n"
            "  next:\n"
            "    if: false\n"
        )
        body = top_level_job_body(sample, "first")
        self.assertIn("needs: [validate, gate]", body)
        self.assertNotIn("if: false", body)
    def test_three_successes_for_the_exact_producer_revision_pass(self):
        self.assertEqual(("passed", "none"), evaluate_source_gate(PRODUCER_SOURCE_SHA, valid_checks()))

    def test_missing_stale_skipped_failed_pending_and_unknown_checks_fail_closed(self):
        checks = valid_checks()
        checks.pop("cuda_build")
        self.assertEqual(("failed", "missing_check"), evaluate_source_gate(PRODUCER_SOURCE_SHA, checks))
        cases = (
            ("stale", "stale_source_sha", {"result": "success", "checked_sha": "0" * 40}),
            ("skipped", "source_check_not_successful", {"result": "skipped", "checked_sha": PRODUCER_SOURCE_SHA}),
            ("failure", "source_check_not_successful", {"result": "failure", "checked_sha": PRODUCER_SOURCE_SHA}),
            ("pending", "source_check_pending", {"result": "in_progress", "checked_sha": PRODUCER_SOURCE_SHA}),
            ("timeout", "source_check_not_successful", {"result": "timed_out", "checked_sha": PRODUCER_SOURCE_SHA}),
        )
        for name, reason, value in cases:
            with self.subTest(name=name):
                candidate = valid_checks()
                candidate["manifest"] = value
                self.assertEqual(("failed", reason), evaluate_source_gate(PRODUCER_SOURCE_SHA, candidate))

    def test_all_three_reusable_jobs_are_required_before_vast_mutation(self):
        workflow = (ROOT / ".github/workflows/vast-cnot-control-collectives.yml").read_text(encoding="utf-8")
        expected = {
            "producer_manifest": "check-full-training-manifest.yml",
            "producer_cpu_runtime": "check-full-training-cpu-runtime.yml",
            "producer_cuda_build": "check-full-training-cuda-build.yml",
        }
        for job, path in expected.items():
            with self.subTest(job=job):
                self.assertIn(f"{job}:", workflow)
                self.assertIn(
                    f"TryDotAtwo/cnot-neural-synth-frontier6/.github/workflows/{path}@{PRODUCER_SOURCE_SHA}",
                    workflow,
                )
                self.assertIn(f"source_sha: {PRODUCER_SOURCE_SHA}", workflow)
        gate = workflow.split("producer_source_gate:", 1)[1].split("\n  gpu-acceptance:", 1)[0]
        self.assertIn("needs: [producer_manifest, producer_cpu_runtime, producer_cuda_build]", gate)
        for name in REQUIRED_CHECKS:
            self.assertIn(f"needs.producer_{name}", gate)
        gpu = top_level_job_body(workflow, "gpu-acceptance")
        self.assertTrue(gpu.strip())
        self.assertIn("needs: [validate-source, producer_source_gate]", gpu)
        self.assertIn("needs.producer_source_gate.outputs.status == 'passed'", gpu)
        self.assertIn("github.event_name == 'workflow_dispatch'", gpu)


if __name__ == "__main__":
    unittest.main(verbosity=2)
