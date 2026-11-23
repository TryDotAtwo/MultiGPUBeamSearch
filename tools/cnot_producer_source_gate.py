#!/usr/bin/env python3
"""Fail-closed gate for the producer checks required by the CNOT GPU runner."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping


PRODUCER_SOURCE_SHA = "7310b1eb5ec07c76c131ba8c60b105ff4aad7a9c"
REQUIRED_CHECKS = ("manifest", "cpu_runtime", "cuda_build")
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_PENDING = {"queued", "pending", "in_progress", "requested", "waiting"}


def evaluate_source_gate(expected_sha: str, checks: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(expected_sha, str) or _SHA40.fullmatch(expected_sha) is None:
        return "failed", "invalid_expected_sha"
    if not isinstance(checks, Mapping):
        return "failed", "missing_check"
    for name in REQUIRED_CHECKS:
        check = checks.get(name)
        if not isinstance(check, Mapping):
            return "failed", "missing_check"
        result = check.get("result")
        if result in _PENDING:
            return "failed", "source_check_pending"
        if result != "success":
            return "failed", "source_check_not_successful"
        checked_sha = check.get("checked_sha")
        if not isinstance(checked_sha, str) or _SHA40.fullmatch(checked_sha) is None:
            return "failed", "missing_source_sha"
        if checked_sha != expected_sha:
            return "failed", "stale_source_sha"
    return "passed", "none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", default=PRODUCER_SOURCE_SHA)
    parser.add_argument("--output", type=Path)
    for name in REQUIRED_CHECKS:
        parser.add_argument(f"--{name.replace('_', '-')}-result", required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-sha", required=True)
    args = parser.parse_args(argv)
    checks = {
        name: {
            "result": getattr(args, f"{name}_result"),
            "checked_sha": getattr(args, f"{name}_sha"),
        }
        for name in REQUIRED_CHECKS
    }
    status, reason = evaluate_source_gate(args.expected_sha, checks)
    if args.output:
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(f"status={status}\nreason={reason}\n")
    print(f"producer_source_gate={status} reason={reason}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
