"""CPU test for the exact scorer selection environment used by the Vast launcher."""
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "vast_cnot_panel_cli_args.sh"


def scorer_environment(contract: str, dev_library: str, panel_library: str,
                       dev_sha256: str, panel_sha256: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; shift; mgbs_vast_cnot_scorer_environment "$@"',
            "vast-scorer-env-test",
            str(HELPER),
            contract,
            dev_library,
            panel_library,
            dev_sha256,
            panel_sha256,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    if not result.stdout.endswith(b"\0"):
        raise AssertionError("scorer environment must be NUL-delimited")
    parts = [item.decode("utf-8") for item in result.stdout.split(b"\0")[:-1]]
    if len(parts) % 2:
        raise AssertionError("scorer environment must contain key/value pairs")
    return dict(zip(parts[0::2], parts[1::2]))


class VastScorerEnvironmentTests(unittest.TestCase):
    def test_gl4_and_panel_choose_distinct_libraries_and_both_enable_strict_backend(self):
        dev = "/workspace/dev-v1/libcnot103_native.so"
        panel = "/workspace/panel-search-v2/libcnot103_native.so"
        dev_sha = "a" * 64
        panel_sha = "b" * 64
        dev_env = scorer_environment("dev_v1", dev, panel, dev_sha, panel_sha)
        panel_env = scorer_environment("panel_search_v2", dev, panel, dev_sha, panel_sha)

        self.assertEqual(dev_env["MGBS_TQP_LIBRARY"], dev)
        self.assertEqual(panel_env["MGBS_TQP_LIBRARY"], panel)
        self.assertEqual(dev_env["MGBS_CNOT_STRICT_BACKEND"], "1")
        self.assertEqual(panel_env["MGBS_CNOT_STRICT_BACKEND"], "1")
        self.assertEqual(dev_env["MGBS_CNOT_EXPECTED_TQP_LIBRARY_SHA256"], dev_sha)
        self.assertEqual(panel_env["MGBS_CNOT_EXPECTED_SCORER_LIBRARY_SHA256"], panel_sha)

    def test_rejects_malformed_hash(self):
        with self.assertRaises(AssertionError):
            scorer_environment("panel_search_v2", "/dev/lib.so", "/panel/lib.so",
                                "not-a-sha", "b" * 64)

    def test_rejects_one_library_for_two_contracts(self):
        library = "/workspace/libcnot103_native.so"
        with self.assertRaises(AssertionError):
            scorer_environment("panel_search_v2", library, library, "a" * 64, "b" * 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
