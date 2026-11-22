"""Exercise the exact panel CLI argv used by the physical shell runner, without launching workers."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARGUMENT_BUILDER = ROOT / "tools" / "vast_cnot_panel_cli_args.sh"
VERIFY_CNOT = ROOT / "tools" / "verify_cnot.py"


def panel_cli_arguments(run_root: str, output: str, library: str, snapshot: str, snapshot_sha: str) -> list[str]:
    command = 'source "$1"; shift; mgbs_vast_cnot_panel_cli_args "$@"'
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "panel-cli-args-test",
            str(ARGUMENT_BUILDER),
            run_root,
            output,
            library,
            snapshot,
            snapshot_sha,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    if not result.stdout.endswith(b"\0"):
        raise AssertionError("panel argument builder did not terminate its NUL-delimited argv")
    return [part.decode("utf-8") for part in result.stdout.split(b"\0")[:-1]]


class VastPanelCliArgumentsTests(unittest.TestCase):
    def test_physical_panel_argv_is_accepted_before_any_process_group(self):
        run_root = "/private/campaign with spaces/run"
        output = "/private/campaign with spaces/run/reports/panel/case-1.json"
        args = panel_cli_arguments(
            run_root,
            output,
            "/workspace/libcnot103_native.so",
            "/workspace/handoff-preflight.json",
            "a" * 64,
        )
        self.assertEqual(args[args.index("--run-root") + 1], run_root)
        self.assertEqual(args[args.index("--output") + 1], output)

        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="mgbs-panel-cli-") as cwd:
            package = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import multigpubeamsearch; print(multigpubeamsearch.__file__)",
                ],
                cwd=cwd,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(package.returncode, 0, package.stderr)
            self.assertNotIn(str(ROOT), package.stdout)

            valid = subprocess.run(
                [sys.executable, str(VERIFY_CNOT), *args, "--validate-cli-only"],
                cwd=cwd,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertIn("CNOT_PANEL_CLI_VALIDATION=passed", valid.stdout)

            for option in ("--run-root", "--output"):
                missing = list(args)
                position = missing.index(option)
                del missing[position:position + 2]
                invalid = subprocess.run(
                    [sys.executable, str(VERIFY_CNOT), *missing, "--validate-cli-only"],
                    cwd=cwd,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(invalid.returncode, 0, f"missing {option} was accepted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
