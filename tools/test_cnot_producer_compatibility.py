"""Hosted CPU checks for the producer adapter build-cache contract."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cnot-producer-compatibility.yml"


class ProducerBuildCacheWorkflowTests(unittest.TestCase):
    def test_only_successful_native_build_artifacts_are_reused_for_matching_inputs(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        # A previous negative run must never become a fresh PR failure on its own.
        self.assertNotIn("Carry the last failed build", workflow)
        self.assertNotIn("verdict == 'failure'", workflow)
        self.assertNotIn("actions/caches", workflow)
        self.assertNotIn("tools/cnot_producer_compatibility.py", workflow)

        # Reuse is an exact-input cache of real native build outputs and the
        # installed build environment, not a saved verdict.
        self.assertIn("actions/cache/restore@v4", workflow)
        self.assertIn("actions/cache/save@v4", workflow)
        self.assertIn('sha256sum "$GITHUB_WORKSPACE/integrations/cnot_scorer/build.py"', workflow)
        self.assertIn('sha256sum "$GITHUB_WORKSPACE/integrations/cnot_scorer/bridge_panel_v2.rs"', workflow)
        self.assertIn("PANEL_PRODUCER_SOURCE_SHA", workflow)
        self.assertIn("e17953891a36e3e3ae8b038695bfac8548ab784f", workflow)
        self.assertIn("PANEL_EMITTER_SOURCE_SHA", workflow)
        self.assertIn("42ed72f587ab70fc74a85cf0a25001efb29cb994", workflow)
        self.assertIn("steps.cache_validation.outputs.ready", workflow)
        self.assertEqual(workflow.count("key: ${{ steps.build_inputs.outputs.cache_key }}"), 2)

        # Save only after the native adapter's tests pass; all downstream
        # consumer fixtures still execute on cache hits.
        self.assertIn("if: success() && steps.cache_validation.outputs.ready != 'true'", workflow)
        save = workflow.split("- name: Save successful native build artifact", 1)[1].split("\n      - name:", 1)[0]
        for path in (".venv-cnot-panel", "cnot-panel-v2-build", "cnot-panel-v2-target"):
            self.assertIn(path, save)
        self.assertIn("Run the cached panel-v2 adapter tests", workflow)
        self.assertIn("Run the pinned emitter through the consumer fixture", workflow)

        # cargo test builds and exercises the library; a separate cargo check
        # adds no independent verification target.
        self.assertNotIn("cargo check ", workflow)
        self.assertIn("cargo test --locked --features torch --lib", workflow)
        self.assertEqual(workflow.count('tail -n 80 "$RUNNER_TEMP/producer-adapter-tests.log"'), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
