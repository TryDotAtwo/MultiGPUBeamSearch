"""Hosted CPU checks for the producer adapter build cache contract."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cnot-producer-compatibility.yml"


class ProducerBuildCacheWorkflowTests(unittest.TestCase):
    def test_only_successful_native_build_artifacts_are_reused_for_matching_inputs(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        # A prior negative result must never become a new PR failure by itself.
        self.assertNotIn("Carry the last failed build", workflow)
        self.assertNotIn("verdict == 'failure'", workflow)
        self.assertNotIn("actions/caches", workflow)

        # Reuse is backed by a real cache of successful adapter build outputs,
        # with a key tied to the adapter source and both pinned producer inputs.
        self.assertIn("actions/cache/restore@v4", workflow)
        self.assertIn("actions/cache/save@v4", workflow)
        self.assertIn("integrations/cnot_scorer/build.py", workflow)
        self.assertIn("integrations/cnot_scorer/bridge_panel_v2.rs", workflow)
        self.assertIn("e17953891a36e3e3ae8b038695bfac8548ab784f", workflow)
        self.assertIn("42ed72f587ab70fc74a85cf0a25001efb29cb994", workflow)
        self.assertIn("steps.build_cache.outputs.cache-hit", workflow)

        save = workflow.split("uses: actions/cache/save@v4", 1)[1]
        self.assertIn("if: success()", save)
        self.assertIn("cnot-panel-v2-target", save)
        self.assertIn("cnot-panel-v2-build", save)
        self.assertIn(".venv-cnot-panel", save)

        # cargo test builds and exercises the library; a second cargo check
        # would compile the same crate without an independent check target.
        self.assertNotIn("cargo check ", workflow)
        self.assertIn("cargo test --locked --features torch --lib", workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
