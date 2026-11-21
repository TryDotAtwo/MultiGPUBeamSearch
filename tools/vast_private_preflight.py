#!/usr/bin/env python3
"""Extract and validate the private-input receipt from a remote preflight log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


EXPECTED_UPSTREAM = "c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb"
EXPECTED_CHECKPOINT = "92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4"
RECEIPT_PATTERN = re.compile(
    r"REMOTE_PRIVATE_PREFLIGHT_OK upstream=([0-9a-f]{40}) "
    r"manifest_sha256=([0-9a-f]{64}) checkpoint_sha256=([0-9a-f]{64})"
)


def parse_receipt(text: str) -> dict[str, str]:
    matches = list(RECEIPT_PATTERN.finditer(text))
    if len(matches) != 1:
        raise ValueError("private preflight must emit exactly one complete receipt")
    upstream, manifest_sha, checkpoint_sha = matches[0].groups()
    if upstream != EXPECTED_UPSTREAM:
        raise ValueError("private preflight used an unexpected upstream commit")
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise ValueError("private preflight used an unexpected checkpoint SHA-256")
    return {
        "upstream_commit": upstream,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": checkpoint_sha,
    }


def write_outputs(receipt: dict[str, str], github_output: Path, report_path: Path) -> None:
    with github_output.open("a", encoding="utf-8") as output:
        for key in ("manifest_sha256", "checkpoint_sha256"):
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
