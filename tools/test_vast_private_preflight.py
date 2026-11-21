from __future__ import annotations

import unittest

from vast_private_preflight import (
    EXPECTED_CHECKPOINT,
    EXPECTED_UPSTREAM,
    parse_receipt,
)


class VastPrivatePreflightTests(unittest.TestCase):
    def receipt(self, manifest: str = "a" * 64, upstream: str = EXPECTED_UPSTREAM,
                checkpoint: str = EXPECTED_CHECKPOINT) -> str:
        return (f"REMOTE_PRIVATE_PREFLIGHT_OK upstream={upstream} "
                f"manifest_sha256={manifest} checkpoint_sha256={checkpoint}\n")

    def test_accepts_one_complete_pinned_receipt(self):
        result = parse_receipt("tool output\n" + self.receipt())
        self.assertEqual(result["upstream_commit"], EXPECTED_UPSTREAM)
        self.assertEqual(result["manifest_sha256"], "a" * 64)
        self.assertEqual(result["checkpoint_sha256"], EXPECTED_CHECKPOINT)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
