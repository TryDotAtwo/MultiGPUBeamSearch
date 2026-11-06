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
