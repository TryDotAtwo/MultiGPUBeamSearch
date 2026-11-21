#!/usr/bin/env python3
"""CPU acceptance for the exact source archive packager used by Vast."""
from __future__ import annotations

from pathlib import Path
import tempfile
import tarfile
import unittest

from create_vast_source_archive import ARCHIVE_PATHS, REQUIRED_VENDOR_HEADERS, create_source_archive


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class VastSourceArchiveTests(unittest.TestCase):
    def test_real_archive_extracts_both_vendored_headers(self):
        with tempfile.TemporaryDirectory(prefix="mgbs-vast-source-archive-") as temporary:
            temporary_root = Path(temporary)
            archive_path = temporary_root / "source.tar.gz"
            extract_dir = temporary_root / "extracted"
            extract_dir.mkdir()

            create_source_archive(REPOSITORY_ROOT, archive_path)
            with tarfile.open(archive_path, "r:gz") as archive:
                members = set(archive.getnames())
                for relative in REQUIRED_VENDOR_HEADERS:
                    self.assertIn(relative.as_posix(), members)
                archive.extractall(extract_dir)

            for relative in REQUIRED_VENDOR_HEADERS:
                self.assertEqual(
                    (extract_dir / relative).read_bytes(),
                    (REPOSITORY_ROOT / relative).read_bytes(),
                )

    def test_packaging_rejects_a_source_tree_missing_either_vendor_header(self):
        for missing in REQUIRED_VENDOR_HEADERS:
            with self.subTest(missing=missing.as_posix()), tempfile.TemporaryDirectory(
                prefix="mgbs-vast-source-negative-"
            ) as temporary:
                source_root = Path(temporary) / "source"
                source_root.mkdir()
                for relative in ARCHIVE_PATHS:
                    path = source_root / relative
                    if relative.suffix:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("synthetic source", encoding="utf-8")
                    else:
                        path.mkdir(parents=True, exist_ok=True)
                for relative in REQUIRED_VENDOR_HEADERS:
                    if relative == missing:
                        continue
                    path = source_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("synthetic vendored header", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, missing.as_posix()):
                    create_source_archive(source_root, Path(temporary) / "source.tar.gz")


if __name__ == "__main__":
    unittest.main(verbosity=2)
