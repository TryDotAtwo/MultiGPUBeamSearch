import argparse
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stream1_transformer_backends import BACKENDS, build_invocation, list_backends


class Stream1TransformerBackendRegistryTests(unittest.TestCase):
    def base_args(self, backend: str, mode: str | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            backend=backend,
            mode=mode,
            weight_dir="weights_fp16",
            build_dir="build-smoke",
            device="cuda:0",
            batches="128,256",
            warmup=1,
            iters=2,
            puzzle_id="991",
            b_micro=None,
            concurrency=None,
            report=None,
            csv=None,
        )

    def test_all_three_backend_names_are_registered(self) -> None:
        self.assertEqual(set(BACKENDS), {"pytorch", "libtorch", "native_cutlass"})
        rows = {row["backend"]: row for row in list_backends()}
        self.assertEqual(rows["pytorch"]["default_mode"], "eager")
        self.assertEqual(rows["libtorch"]["modes"], ["eager", "cuda_graph"])
        self.assertEqual(rows["native_cutlass"]["modes"], ["eager", "graph"])

    def test_pytorch_invocation_uses_python_torch_tool(self) -> None:
        invocation = build_invocation(self.base_args("pytorch"))
        self.assertEqual(invocation.backend, "pytorch")
        self.assertEqual(invocation.mode, "eager")
        command = list(invocation.command)
        self.assertTrue(command[0])
        self.assertIn("stream1_transformer_torch_benchmark.py", command[1])
        self.assertIn("--weight-dir", command)
        self.assertIn("weights_fp16", command)

    def test_libtorch_invocation_uses_opt_in_cpp_tool(self) -> None:
        invocation = build_invocation(self.base_args("libtorch", "cuda_graph"))
        self.assertEqual(invocation.backend, "libtorch")
        self.assertEqual(invocation.mode, "cuda_graph")
        command = " ".join(invocation.command)
        self.assertIn("stream1_transformer_libtorch_benchmark", command)
        self.assertIn("--cuda-graph", invocation.command)
        self.assertIn("--batches", invocation.command)

    def test_native_invocation_sets_native_environment(self) -> None:
        args = self.base_args("native_cutlass", "graph")
        args.b_micro = "256"
        args.concurrency = "2"
        invocation = build_invocation(args)
        self.assertEqual(invocation.backend, "native_cutlass")
        self.assertEqual(invocation.mode, "graph")
        binary_name = Path(invocation.command[0]).name
        self.assertIn(binary_name, {"stream_benchmark", "stream_benchmark.exe"})
        env = dict(invocation.env)
        self.assertEqual(env["BEAM_WEIGHT_DIR"], "weights_fp16")
        self.assertEqual(env["BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH"], "1")
        self.assertEqual(env["BEAM_STREAM1_TRANSFORMER_B_MICRO"], "256")
        self.assertEqual(env["BEAM_STREAM1_TRANSFORMER_CONCURRENCY"], "2")

    def test_invalid_mode_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_invocation(self.base_args("pytorch", "cuda_graph"))


if __name__ == "__main__":
    unittest.main()
