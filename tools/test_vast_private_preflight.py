from __future__ import annotations

import unittest
from pathlib import Path

from vast_private_preflight import (
    EXPECTED_CHECKPOINT,
    EXPECTED_UPSTREAM,
    parse_receipt,
)


CUTLASS_REVISION = "afa1772203677c5118fcd82537a9c8fefbcc7008"
CUTLASS_SHA256 = "e62fb320c2b61e7e0c7a1163c9c5a58f6dd86025adf7df4d124933152482997f"


class VastPrivatePreflightTests(unittest.TestCase):
    def receipt(self, manifest: str = "a" * 64, upstream: str = EXPECTED_UPSTREAM,
                checkpoint: str = EXPECTED_CHECKPOINT,
                cutlass_revision: str = CUTLASS_REVISION,
                cutlass_sha256: str = CUTLASS_SHA256) -> str:
        return (
            f"REMOTE_PRIVATE_PREFLIGHT_OK upstream={upstream} "
            f"manifest_sha256={manifest} checkpoint_sha256={checkpoint}\n"
            f"REMOTE_CUTLASS_PREFLIGHT_OK revision={cutlass_revision} "
            f"archive_sha256={cutlass_sha256}\n"
        )

    def test_accepts_one_complete_pinned_receipt(self):
        result = parse_receipt("tool output\n" + self.receipt())
        self.assertEqual(result["upstream_commit"], EXPECTED_UPSTREAM)
        self.assertEqual(result["manifest_sha256"], "a" * 64)
        self.assertEqual(result["checkpoint_sha256"], EXPECTED_CHECKPOINT)
        self.assertEqual(result["cutlass_revision"], CUTLASS_REVISION)
        self.assertEqual(result["cutlass_archive_sha256"], CUTLASS_SHA256)

    def test_rejects_absent_or_duplicate_receipt(self):
        for text in ("", self.receipt() + self.receipt()):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                parse_receipt(text)

    def test_rejects_changed_upstream_or_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "upstream"):
            parse_receipt(self.receipt(upstream="0" * 40))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            parse_receipt(self.receipt(checkpoint="b" * 64))

    def test_rejects_malformed_manifest_digest(self):
        with self.assertRaises(ValueError):
            parse_receipt(self.receipt(manifest="not-a-sha"))

    def test_rejects_missing_or_duplicate_pinned_cutlass_receipt(self):
        private_only = self.receipt().split("REMOTE_CUTLASS_PREFLIGHT_OK", 1)[0]
        duplicate_cutlass = self.receipt() + self.receipt().split("REMOTE_PRIVATE_PREFLIGHT_OK", 1)[1]
        for text in (private_only, duplicate_cutlass):
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                parse_receipt(text)

    def test_rejects_wrong_cutlass_revision_or_archive_sha(self):
        with self.assertRaisesRegex(ValueError, "CUTLASS revision"):
            parse_receipt(self.receipt(cutlass_revision="0" * 40))
        with self.assertRaisesRegex(ValueError, "CUTLASS archive SHA-256"):
            parse_receipt(self.receipt(cutlass_sha256="0" * 64))

    def test_preflight_pin_matches_the_runtime_source_pin(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "integrations/cayleypy_native/multigpubeamsearch/sources.py"
        ).read_text(encoding="utf-8")
        self.assertIn(CUTLASS_REVISION, source)
        self.assertIn(CUTLASS_SHA256, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
