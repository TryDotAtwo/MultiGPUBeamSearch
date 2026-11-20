from pathlib import Path
import os
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]


class VerifyCnotAcceptanceCliTest(unittest.TestCase):
    def test_cli_exposes_native_search_and_native_scorer_requirements(self):
        env = os.environ.copy()
        package_root = str(ROOT / "integrations" / "cayleypy_native")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (package_root, env.get("PYTHONPATH", "")) if part
        )
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "verify_cnot.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--engine", result.stdout)
        self.assertIn("--require-rust-scorer", result.stdout)


if __name__ == "__main__":
    unittest.main()
