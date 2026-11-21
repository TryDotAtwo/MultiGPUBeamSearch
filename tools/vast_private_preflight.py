#!/usr/bin/env python3
"""Extract and validate the private-input receipt from a remote preflight log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


EXPECTED_UPSTREAM = "c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb"
EXPECTED_CHECKPOINT = "92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4"
EXPECTED_CUTLASS_REVISION = "afa1772203677c5118fcd82537a9c8fefbcc7008"
EXPECTED_CUTLASS_ARCHIVE_SHA256 = "e62fb320c2b61e7e0c7a1163c9c5a58f6dd86025adf7df4d124933152482997f"
RECEIPT_PATTERN = re.compile(
    r"REMOTE_PRIVATE_PREFLIGHT_OK upstream=([0-9a-f]{40}) "
    r"manifest_sha256=([0-9a-f]{64}) checkpoint_sha256=([0-9a-f]{64})"
)
CUTLASS_RECEIPT_PATTERN = re.compile(
    r"REMOTE_CUTLASS_PREFLIGHT_OK revision=([0-9a-f]{40}) "
    r"archive_sha256=([0-9a-f]{64})"
)


def parse_receipt(text: str) -> dict[str, str]:
    private_matches = list(RECEIPT_PATTERN.finditer(text))
    cutlass_matches = list(CUTLASS_RECEIPT_PATTERN.finditer(text))
    if len(private_matches) != 1:
        raise ValueError("private preflight must emit exactly one complete receipt")
    if len(cutlass_matches) != 1:
        raise ValueError("private preflight must emit exactly one pinned CUTLASS receipt")
    upstream, manifest_sha, checkpoint_sha = private_matches[0].groups()
    cutlass_revision, cutlass_sha = cutlass_matches[0].groups()
    if upstream != EXPECTED_UPSTREAM:
        raise ValueError("private preflight used an unexpected upstream commit")
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise ValueError("private preflight used an unexpected checkpoint SHA-256")
    if cutlass_revision != EXPECTED_CUTLASS_REVISION:
        raise ValueError("private preflight used an unexpected CUTLASS revision")
    if cutlass_sha != EXPECTED_CUTLASS_ARCHIVE_SHA256:
        raise ValueError("private preflight used an unexpected CUTLASS archive SHA-256")
    return {
        "upstream_commit": upstream,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": checkpoint_sha,
        "cutlass_revision": cutlass_revision,
        "cutlass_archive_sha256": cutlass_sha,
    }


def write_outputs(receipt: dict[str, str], github_output: Path, report_path: Path) -> None:
    with github_output.open("a", encoding="utf-8") as output:
        for key in ("manifest_sha256", "checkpoint_sha256", "cutlass_revision", "cutlass_archive_sha256"):
            output.write(f"{key}={receipt[key]}\n")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(private_preflight="passed", **receipt)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--github-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = parse_receipt(args.log.read_text(encoding="utf-8", errors="replace"))
        write_outputs(receipt, args.github_output, args.report)
    except Exception as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
