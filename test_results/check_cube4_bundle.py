import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "verify_cube4_bundle.py"
sys.path.insert(0, str(ROOT))
from tools.verify_cube4_bundle import required_tensor_names


class BundleChecks(unittest.TestCase):
    def test_real_archived_bundle_passes(self):
        data = ROOT / "test_results" / "hopper_runtime_v1" / "data"
        weights = ROOT.parents[1] / "test_results" / "paper_benchmarks" / "cube4_ours_t4_b4m_depth10" / "stream1_transformer_weights_fp16"
        if not data.exists() or not weights.exists():
            self.skipTest("local archived Cube4 bundle unavailable")
        result = subprocess.run([sys.executable, str(VERIFIER), str(data), str(weights), "1000"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_puzzle_or_order_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data"
            weights = base / "weights"
            data.mkdir()
            weights.mkdir()
            moves = [f"m{i}" for i in range(24)]
            (data / "puzzle_info.json").write_text(json.dumps({"central_state": [0] * 96, "generators": {m: list(range(96)) for m in moves}}))
            (data / "test.csv").write_text("initial_state_id,initial_state\n1000,\"" + ",".join(["0"] * 96) + "\"\n")
            manifest = {"backend": "piece_transformer", "state_len": 96, "move_count": 24, "output_dim": 24,
                        "seq_len": 57, "dtype": "fp16", "activation": "relu", "pooling": "cls", "move_names": moves}
            (weights / "manifest.json").write_text(json.dumps(manifest))
            for name in required_tensor_names():
                (weights / name).write_bytes(b"x")
            command = [sys.executable, str(VERIFIER), str(data), str(weights), "1000"]
            self.assertEqual(subprocess.run(command).returncode, 0)
            manifest["move_names"] = list(reversed(moves))
            (weights / "manifest.json").write_text(json.dumps(manifest))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            manifest["move_names"] = moves
            (weights / "manifest.json").write_text(json.dumps(manifest))
            (data / "puzzle_info.json").write_text(json.dumps({"central_state": [0] * 120, "generators": {m: list(range(120)) for m in moves}}))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
