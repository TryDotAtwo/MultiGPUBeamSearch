"""Installed-wheel synthetic test for the private producer-handoff endpoint."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "integrations" / "cnot_scorer"))
from test_handoff import PanelV2HandoffTests  # noqa: E402

import multigpubeamsearch  # noqa: E402


def load_verify_endpoint():
    path = ROOT / "tools" / "verify_cnot.py"
    spec = importlib.util.spec_from_file_location("verify_cnot_installed_endpoint", path)
    if spec is None or spec.loader is None:
        raise AssertionError("standard CNOT verifier endpoint is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticScorer:
    metadata = {"feature_model_inputs": 102, "feature_ids": [f"f{i}" for i in range(102)]}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def parent_scores(self, packed, mode):
        alphabet = packed.size(1) * (packed.size(1) - 1)
        return torch.zeros((len(packed), alphabet), dtype=torch.float32, device=packed.device)

    def child_values(self, children):
        return torch.zeros((len(children),), dtype=torch.float32, device=children.device)


class InstalledHandoffEndpointTests(unittest.TestCase):
    def setUp(self):
        package_path = Path(multigpubeamsearch.__file__).resolve()
        self.assertNotIn(ROOT.resolve(), package_path.parents, "test imported the checkout instead of the installed wheel")
        self.fixture = PanelV2HandoffTests()
        self.fixture.setUp()
        self.owns_process_group = not dist.is_initialized()
        if self.owns_process_group:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{self.fixture.root / 'gloo-rendezvous'}",
                rank=0,
                world_size=1,
            )
        self.verify = load_verify_endpoint()
        self.scorer_creations = []
        self.search_calls = []
        self.scorer = SyntheticScorer()
        self.real_search = self.verify.search_cnot

        def factory(library, manifest, *, n, device, inference_batch=4):
            self.scorer_creations.append((library, Path(manifest).resolve(), n, str(device), inference_batch))
            return self.scorer

        def search(root, scorer, **kwargs):
            self.search_calls.append({
                "root": root.detach().cpu().tolist(),
                "same_scorer": scorer is self.scorer,
                **kwargs,
            })
            return self.real_search(root, scorer, **kwargs)

        self.verify.search_cnot = search
        self.scorer_factory = factory

    def tearDown(self):
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        self.fixture.tearDown()

    def invoke(self, *, receipt=None, output_name="consumer-panel.private.json", extra=()):
        args = [
            "--case", "panel",
            "--backend", "gloo",
            "--engine", "rust",
            "--library", "synthetic-scorer.so",
            "--run-root", str(self.fixture.root),
            "--handoff-receipt", str(receipt or self.fixture.receipt),
            "--output", str(self.fixture.reports / output_name),
            *extra,
        ]
        return self.verify.main(args, scorer_factory=self.scorer_factory, manage_process_group=False)

    def test_public_endpoint_loads_ordered_roots_and_searches_with_pinned_qhat_budget(self):
        self.invoke()
        report_path = self.fixture.reports / "consumer-panel.private.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], "cnot103_mgbs_panel_search_result_v1")
        self.assertEqual(report["producer_emitter_sha256"], self.fixture.producer_emitter_sha256)
        self.assertEqual(report["adapter"]["native_adapter_source_commit"], "e17953891a36e3e3ae8b038695bfac8548ab784f")
        bridge = ROOT / "integrations" / "cnot_scorer" / "bridge_panel_v2.rs"
        self.assertEqual(report["adapter"]["adapter_source_sha256"], hashlib.sha256(bridge.read_bytes()).hexdigest())
        self.assertEqual(report["panel"], "test")
        self.assertEqual(report["panel_indices"], [3, 9])
        self.assertEqual(report["panel_ids"], ["locked-case-a", "locked-case-b"])
        self.assertEqual(report["panel_digest"], self.fixture.panel_digest)
        self.assertEqual(report["head"], "qhat")
        self.assertEqual(report["search"], self.fixture.search)
        self.assertEqual(report["expansion_upper_bound"], 201)
        self.assertEqual(
            [(row["panel_index"], row["case_id"], row["dimension"], row["orientation"])
             for row in report["panel_roots"]],
            [
                (3, "locked-case-a", 2, "row_target_xor_row_control"),
                (9, "locked-case-b", 2, "row_target_xor_row_control"),
            ],
        )
        self.assertEqual([row["case_id"] for row in report["rows"]], self.fixture.case_ids)
        self.assertTrue(all(row["status"] == "solved" and row["word"]["replayed"] for row in report["rows"]))
        self.assertEqual(len(self.scorer_creations), 1)
        self.assertEqual(
            self.scorer_creations[0],
            (
                "synthetic-scorer.so",
                self.fixture.search_manifest.resolve(),
                2,
                "cpu",
                4,
            ),
        )
        self.assertEqual(len(self.search_calls), 2)
        self.assertTrue(all(call["same_scorer"] for call in self.search_calls))
        self.assertTrue(all(
            call["width"] == 36
            and call["shortlist"] == 4
            and call["max_depth"] == 8
            and call["mode"] == "qhat"
            and call["engine"] == "rust"
            for call in self.search_calls
        ))
        self.assertEqual([call["root"] for call in self.search_calls], [[[[1], [3]]], [[[3], [2]]]])

    def test_bad_receipt_fails_before_scorer_or_search(self):
        self.fixture.receipt_source_revision = "f" * 40
        self.fixture.write_receipt()
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.scorer_creations, [])
        self.assertEqual(self.search_calls, [])
        self.assertFalse((self.fixture.reports / "consumer-panel.private.json").exists())

    def test_missing_handoff_receipt_fails_before_scorer_or_search(self):
        missing = self.fixture.root / "reports" / "missing.json"
        with self.assertRaises(SystemExit):
            self.invoke(receipt=missing)
        self.assertEqual(self.scorer_creations, [])
        self.assertEqual(self.search_calls, [])

    def test_direct_manifest_is_not_accepted_for_paired_panel_endpoint(self):
        with self.assertRaises(SystemExit):
            self.invoke(extra=("--manifest", str(self.fixture.search_manifest)))
        self.assertEqual(self.scorer_creations, [])
        self.assertEqual(self.search_calls, [])


if __name__ == "__main__":
    unittest.main()
