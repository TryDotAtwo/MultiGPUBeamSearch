#!/usr/bin/env python3
"""Resolve an exact transformer/GPU policy cache into explicit environment values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.stream1_transformer_autotune import (
    POLICY_FAMILIES,
    load_cached_environment,
    policy_environment_variable,
)


def policies_to_environment(policies: dict[str, str]) -> dict[str, str]:
    if set(policies) != set(POLICY_FAMILIES):
        raise ValueError("policy map must contain every policy family")
    return {policy_environment_variable(family): policies[family] for family in POLICY_FAMILIES}


def load_policy_environment(cache_path: Path, signature: dict[str, Any]) -> tuple[dict[str, str], str]:
    return load_cached_environment(cache_path, signature)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    args = parser.parse_args()
    signature = json.loads(args.signature.read_text(encoding="utf-8"))
    environment, status = load_policy_environment(args.cache, signature)
    print(json.dumps({"status": status, "environment": environment}, indent=2, sort_keys=True))
    return 0 if status == "cache_hit" else 2


if __name__ == "__main__":
    raise SystemExit(main())