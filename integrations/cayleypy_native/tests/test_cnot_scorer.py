"""Run the weight-backed cases with the original, privately supplied GL4 fixture."""

import json
import os
from pathlib import Path
import unittest

from multigpubeamsearch.cnot_scorer import TQPScorer


LIBRARY = os.environ.get("MGBS_TQP_LIBRARY")
MANIFEST = os.environ.get("MGBS_TQP_MANIFEST")


@unittest.skipUnless(LIBRARY and MANIFEST, "requires the private GL4 model fixture and CUDA bridge")
class CheckpointScorerTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.device = torch.device(os.environ.get("MGBS_TQP_DEVICE", "cuda:0"))
        self.scorer = TQPScorer(LIBRARY, MANIFEST, n=4, device=self.device, inference_batch=4)
        self.addCleanup(self.scorer.close)

    def packed(self, rows):
        return self.torch.tensor(rows, dtype=self.torch.int64, device=self.device).reshape(-1, 4, 1)

    def test_actual_checkpoint_heads_and_live_rows(self):
        t = self.torch
        self.assertEqual(self.scorer.metadata["checkpoint_sha256"],
                         "92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4")
        self.assertEqual(self.scorer.metadata["feature_model_inputs"], 102)
        self.assertNotIn("H07", self.scorer.metadata["feature_ids"])
        rows = self.packed([[1, 2, 7, 8], [1, 2, 4, 14]])
        raw = {}
        for mode in ("qhat", "proposal"):
            values = self.scorer.parent_scores(rows, mode)
            self.assertEqual((values.shape, values.device), ((2, 12), self.device))
            t.testing.assert_close(values.flip(0), self.scorer.parent_scores(rows.flip(0), mode),
                                   rtol=2e-5, atol=2e-5)
            self.assertGreater((values[0] - values[1]).abs().max().item(), 1e-6)
            raw[mode] = values
        self.assertFalse(t.allclose(raw["qhat"], raw["proposal"]))
        vs = self.scorer.child_values(rows)
        self.assertEqual((vs.shape, vs.device), ((2,), self.device))
        t.testing.assert_close(vs.flip(0), self.scorer.child_values(rows.flip(0)), rtol=2e-5, atol=2e-5)
        t.testing.assert_close(vs[:1], self.scorer.child_values(rows[:1]), rtol=2e-5, atol=2e-5)
        self.assertEqual(self.scorer.child_values(rows[:0]).shape, (0,))

    def test_out_of_scope_feature_and_invalid_storage_fail_closed(self):
        rows = self.packed([[1, 2, 7, 8]])
        with self.assertRaisesRegex(RuntimeError, "feature F02 rejected"):
            self.scorer.child_values(self.packed([[3, 2, 6, 12]]))
        with self.assertRaises(ValueError):
            self.scorer.parent_scores(rows.cpu(), "qhat")
        with self.assertRaises(ValueError):
            self.scorer.parent_scores(rows, "unknown")
        self.scorer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.scorer.child_values(rows)

    @unittest.skipUnless(os.environ.get("MGBS_TQP_BATCH_TEST"), "requires the experimental native batch ABI")
    def test_native_batch_matches_existing_chunked_checkpoint_scorer(self):
        t = self.torch
        kwargs = dict(n=4, device=self.device, inference_batch=2)
        with TQPScorer(LIBRARY, MANIFEST, **kwargs) as reference, TQPScorer(
                LIBRARY, MANIFEST, experimental_native_batch=True, **kwargs) as actual:
            base = self.packed([[1, 2, 7, 8], [1, 2, 7, 8],
                                [1, 2, 4, 14], [1, 2, 7, 8]] * 5)
            for count in (0, 1, 2, 3, 9):
                rows = base[::2][:count]  # deliberately non-contiguous when count > 1
                for mode in ("qhat", "proposal"):
                    t.testing.assert_close(actual.parent_scores(rows, mode),
                                           reference.parent_scores(rows, mode), rtol=2e-5, atol=2e-5)
                t.testing.assert_close(actual.child_values(rows), reference.child_values(rows),
                                       rtol=2e-5, atol=2e-5)

            rejected = self.packed([[1, 2, 7, 8]] * 5 + [[3, 2, 6, 12]])
            for scorer in (reference, actual):
                with self.assertRaisesRegex(RuntimeError, "CNOT row 5: feature F02 rejected"):
                    scorer.child_values(rejected)

            # The native boundary must reject short buffers before launching any work.
            rows = self.packed([[1, 2, 7, 8], [1, 2, 4, 14]])
            output = t.empty((2,), dtype=t.float32, device=self.device)
            status = t.empty((2,), dtype=t.int64, device=self.device)
            sizes = [rows.numel() * rows.element_size(), output.numel() * output.element_size(),
                     status.numel() * status.element_size()]
            pointers = [rows.data_ptr(), output.data_ptr(), status.data_ptr()]
            for position, name in enumerate(("rows", "output", "votes")):
                available = sizes.copy()
                available[position] -= 1
                code = actual._api.mgbs_tqp_score_many(
                    actual._handle, 2, 2, 2, pointers[0], available[0],
                    pointers[1], available[1], pointers[2], available[2])
                self.assertNotEqual(code, 0)
                message = str(actual._native_error())
                self.assertIn(name, message)
                self.assertIn(f"required={sizes[position]}", message)
                self.assertIn(f"available={available[position]}", message)
            code = actual._api.mgbs_tqp_score_many(actual._handle, 2, 0, 2, None, 0, None, 0, None, 0)
            self.assertEqual(code, 0)
            for head, batch, limit in ((3, 2, 2), (2, -1, 2), (2, 2, 0),
                                       (2, 2**63 - 1, 2)):
                code = actual._api.mgbs_tqp_score_many(
                    actual._handle, head, batch, limit, pointers[0], sizes[0],
                    pointers[1], sizes[1], pointers[2], sizes[2])
                self.assertNotEqual(code, 0)

    @unittest.skipUnless(os.environ.get("MGBS_TQP_H07_REFERENCE"), "requires original H07 CLI output")
    def test_values_match_upstream_checkpoint_cli(self):
        report = json.loads(Path(os.environ["MGBS_TQP_H07_REFERENCE"]).read_text(encoding="utf-8"))
        self.assertEqual(report["checkpoint"]["sha256"], self.scorer.metadata["checkpoint_sha256"])
        rows = self.packed([[1, 2, 7, 8], [1, 2, 4, 14]])
        by_index = {r["dev_index"]: r for r in report["rows"]}
        golden = []
        for i in (3, 4):
            probes = by_index[i]["diagnostic"]["probes"]
            canonical, = [p for p in probes if p["kind"] == "canonical"]
            golden.append(canonical["prediction"])
        reference = self.torch.tensor(golden, device=self.device, dtype=self.torch.float32)
        self.torch.testing.assert_close(self.scorer.child_values(rows), reference, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
