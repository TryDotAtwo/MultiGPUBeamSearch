"""Hosted CPU contracts for panel routing and split cache identities."""
from pathlib import Path
import tempfile
import unittest

from cnot_producer_compatibility import (
    classify_changed_paths,
    native_artifact_key,
    tree_digest,
)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cnot-producer-compatibility.yml"
PINS_PATH = ".github/cnot-producer-panel-pins.json"
BASE_PINS = {
    "schema": "cnot-producer-panel-pins-v1",
    "producer_source_sha": "e17953891a36e3e3ae8b038695bfac8548ab784f",
    "emitter_source_sha": "42ed72f587ab70fc74a85cf0a25001efb29cb994",
    "panel_manifest_sha256": "a" * 64,
}


class ProducerCompatibilityRoutingTests(unittest.TestCase):
    def test_emitter_or_panel_manifest_change_runs_only_the_fast_handoff(self):
        emitter = dict(BASE_PINS, emitter_source_sha="5" * 40)
        manifest = dict(BASE_PINS, panel_manifest_sha256="b" * 64)
        for changed in (emitter, manifest):
            with self.subTest(changed=changed):
                self.assertEqual(
                    classify_changed_paths(
                        [PINS_PATH], previous_pins=BASE_PINS, current_pins=changed
                    ),
                    {"run_handoff": True, "run_native": False},
                )

    def test_producer_source_revision_runs_native_check_without_emitter_checkout(self):
        producer = dict(BASE_PINS, producer_source_sha="6" * 40)
        self.assertEqual(
            classify_changed_paths(
                [PINS_PATH], previous_pins=BASE_PINS, current_pins=producer
            ),
            {"run_handoff": False, "run_native": True},
        )

    def test_native_adapter_paths_run_only_the_rust_cpu_gate(self):
        self.assertEqual(
            classify_changed_paths(["integrations/cnot_scorer/bridge_panel_v2.rs"]),
            {"run_handoff": False, "run_native": True},
        )
        self.assertEqual(
            classify_changed_paths(["integrations/cnot_scorer/src/bridge.rs"]),
            {"run_handoff": False, "run_native": True},
        )

    def test_handoff_contract_and_producer_manifest_changes_run_the_fast_gate(self):
        for path in (
            "integrations/cnot_scorer/handoff.py",
            "integrations/cnot_scorer/fixtures/panel-v2-manifest.json",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    classify_changed_paths([path]),
                    {"run_handoff": True, "run_native": False},
                )
        self.assertEqual(
            classify_changed_paths(
                [".github/workflows/cnot-producer-compatibility.yml"]
            ),
            {"run_handoff": False, "run_native": True},
        )

    def test_workflow_dispatch_forces_both_routes(self):
        self.assertEqual(
            classify_changed_paths([], event_name="workflow_dispatch"),
            {"run_handoff": True, "run_native": True},
        )

    def test_native_cache_key_tracks_only_prepared_inputs_abi_and_toolchains(self):
        inputs = {
            "native_inputs_sha": "1" * 64,
            "abi": "panel_search_v2",
            "toolchain": "rust-stable-cargo-1",
            "environment": "python-3.11-torch-2.9-cxxabi1",
            "runner": "Linux-X64",
        }
        before = native_artifact_key(**inputs)
        # Changing either pin changes the handoff scope but is deliberately not
        # an input to the native artifact key.
        emitter_pins = dict(BASE_PINS, emitter_source_sha="7" * 40)
        manifest_pins = dict(BASE_PINS, panel_manifest_sha256="c" * 64)
        self.assertNotEqual(emitter_pins, manifest_pins)
        self.assertEqual(before, native_artifact_key(**inputs))
        self.assertNotEqual(before, native_artifact_key(**dict(inputs, native_inputs_sha="2" * 64)))
        self.assertNotEqual(before, native_artifact_key(**dict(inputs, abi="panel_search_v3")))
        self.assertNotEqual(before, native_artifact_key(**dict(inputs, toolchain="rust-next")))
        self.assertNotEqual(before, native_artifact_key(**dict(inputs, environment="torch-3")))

    def test_native_tree_digest_changes_when_a_source_byte_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "lib.rs"
            source.parent.mkdir()
            source.write_text("pub fn score() -> u32 { 1 }\n", encoding="utf-8")
            before = tree_digest(root)
            source.write_text("pub fn score() -> u32 { 2 }\n", encoding="utf-8")
            self.assertNotEqual(before, tree_digest(root))


class ProducerBuildCacheWorkflowTests(unittest.TestCase):
    def test_prepare_precedes_torch_and_cargo_and_caches_are_split(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        prepare = workflow.index("Prepare exact panel-v2 Rust crate before Torch or Cargo")
        torch_restore = workflow.index("Restore the CPU PyTorch environment")
        torch_install = workflow.index("Install CPU PyTorch into the environment")
        cargo = workflow.index("cargo test --locked --features torch --lib")
        self.assertLess(prepare, torch_restore)
        self.assertLess(prepare, torch_install)
        self.assertLess(prepare, cargo)

        self.assertIn("actions/cache/restore@v4", workflow)
        self.assertIn("actions/cache/save@v4", workflow)
        self.assertIn("cnot-panel-python-v2-", workflow)
        self.assertIn("hashFiles('.github/cnot-panel-python-requirements.txt')", workflow)
        self.assertEqual(workflow.count("  pull_request:"), 1)
        self.assertIn('export PYTHON="$python_bin"', workflow)
        self.assertIn('export PYO3_PYTHON="$python_bin"', workflow)
        key = native_artifact_key(
            native_inputs_sha="1" * 64,
            abi="panel_search_v2",
            toolchain="rust-stable",
            environment="py311-torch290",
            runner="Linux-X64",
        )
        self.assertTrue(key.startswith("cnot-panel-native-v5-"))
        self.assertIn("--native-inputs-sha", workflow)
        native_key_step = workflow.split(
            "- name: Fingerprint prepared native inputs, ABI, and toolchain", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertNotIn("emitter_source_sha", native_key_step)
        self.assertNotIn("producer_source_sha", native_key_step)

        native_save = workflow.split("- name: Save successful native build artifact", 1)[1].split("\n      - name:", 1)[0]
        self.assertIn("if: success()", native_save)
        self.assertIn("cnot-panel-v2-target", native_save)
        self.assertNotIn(".venv-cnot-panel", native_save)
        self.assertNotIn("cnot-panel-v2-build", native_save)

    def test_emitter_route_has_no_torch_or_cargo_and_skips_native_steps(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("steps.scope.outputs.run_handoff == 'true'", workflow)
        self.assertIn("steps.scope.outputs.run_native == 'true'", workflow)
        handoff = workflow.split("- name: Run pinned emitter through the no-Torch consumer fixture", 1)[1].split("\n      - name:", 1)[0]
        self.assertIn('find_spec("torch") is None', handoff)
        self.assertIn("python -m unittest -v test_handoff", handoff)
        self.assertNotIn("cargo", handoff.lower())
        self.assertNotIn("pip install", handoff.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
