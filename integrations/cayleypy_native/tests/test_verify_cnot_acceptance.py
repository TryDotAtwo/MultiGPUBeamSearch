import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


def acceptance_env():
    env = os.environ.copy()
    package_root = str(ROOT / "integrations" / "cayleypy_native")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (package_root, env.get("PYTHONPATH", "")) if part
    )
    return env


class VerifyCnotAcceptanceCliTest(unittest.TestCase):
    def test_rust_engine_runs_full_reference_checked_cnot_search(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "wide.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "verify_cnot.py"),
                    "--case", "wide",
                    "--backend", "gloo",
                    "--engine", "rust",
                    "--output", str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=acceptance_env(),
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["engine"], "rust")
            self.assertEqual(report["n"], 144)
            self.assertTrue(report["bitwise_replay"])

    def test_help_exposes_checkpoint_scorer_requirement(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "verify_cnot.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=acceptance_env(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--require-rust-scorer", result.stdout)


if __name__ == "__main__":
    unittest.main()
