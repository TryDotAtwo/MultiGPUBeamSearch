#!/usr/bin/env python3
"""Validate one explicitly selected, owned Vast 2x RTX 3060 instance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import ipaddress
import json
import re
import sys
from pathlib import Path


OLD_INSTANCE_ID = "61080396"
MAX_SPEND_CAP_USD = Decimal("10")
MAX_REMOTE_RUNTIME_SECONDS = 5400
RESERVE_SECONDS = 1200
MIN_REMOTE_RUNTIME_SECONDS = 1200
PRIVATE_PREFLIGHT_RESERVE_SECONDS = 300
SOURCE_STAGING_RESERVE_SECONDS = 60
UPLOAD_RESERVE_SECONDS = 900
RETRIEVAL_RESERVE_SECONDS = 300


def validate_instance_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError("instance_id must contain only decimal digits and be positive")
    if value == OLD_INSTANCE_ID:
        raise ValueError("the legacy instance is outside this operation")
    return value


def parse_cap(value: str) -> Decimal:
    try:
        cap = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError("spend_cap_usd must be a finite positive decimal") from None
    if not cap.is_finite() or cap <= 0 or cap > MAX_SPEND_CAP_USD:
        raise ValueError("spend_cap_usd must be greater than 0 and at most $10")
    return cap


def _rows(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("instances")
    if not isinstance(payload, list):
        raise ValueError("Vast API response does not contain an instances list")
    return [row for row in payload if isinstance(row, dict)]


def find_owned_target(payload: object, instance_id: str) -> dict:
    instance_id = validate_instance_id(instance_id)
    matches = [row for row in _rows(payload) if str(row.get("id")) == instance_id]
    if len(matches) != 1:
        raise ValueError("the requested instance is not uniquely present in this account's API listing")
    row = matches[0]
    count = row.get("num_gpus")
    try:
        exact_count = Decimal(str(count)) == Decimal(2)
    except InvalidOperation:
        exact_count = False
    name = str(row.get("gpu_name") or "").upper()
    if not exact_count or re.search(r"\bRTX\s*3060\b", name) is None or any(
        token in name for token in ("TI", "LAPTOP", "MOBILE", "SUPER")
    ):
        raise ValueError(f"instance {instance_id} is not exactly 2x desktop RTX 3060")
    return row


def _ssh_endpoint(row: dict) -> tuple[str, int]:
    direct_host, direct_port = row.get("ssh_host"), row.get("ssh_port")
    mapped_host = row.get("public_ipaddr")
    mappings = (row.get("ports") or {}).get("22/tcp") or []
    mapped_port = mappings[0].get("HostPort") if mappings and isinstance(mappings[0], dict) else None

    if direct_host is not None and direct_port is not None:
        host, port = direct_host, direct_port
    elif mapped_host is not None and mapped_port is not None:
        host, port = mapped_host, mapped_port
    elif direct_host is not None or mapped_host is not None:
        raise ValueError("Vast API did not provide an SSH port and complete endpoint pair")
    else:
        raise ValueError("Vast API did not provide an SSH host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("Vast API did not provide an SSH host")
    host = host.strip()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if re.fullmatch(r"[A-Za-z0-9.-]+", host) is None:
            raise ValueError("Vast API returned an invalid SSH host") from None
    try:
        numeric_port = Decimal(str(port))
        if not numeric_port.is_finite() or numeric_port != numeric_port.to_integral_value():
            raise ValueError
        port = int(numeric_port)
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("Vast API did not provide an SSH port") from None
    if not 1 <= port <= 65535:
        raise ValueError("Vast API returned an invalid SSH port")
    return host, port


def acceptance_plan(
    row: dict,
    spend_cap: Decimal,
    *,
    now_epoch: Decimal | int | float | None = None,
) -> dict:
    if str(row.get("actual_status") or row.get("cur_state") or "").lower() != "running":
        raise ValueError("the verified 2x RTX 3060 instance must already be running")
    try:
        rate = Decimal(str(row.get("dph_total")))
        start = Decimal(str(row.get("start_date")))
    except (InvalidOperation, TypeError):
        raise ValueError("Vast API must provide numeric dph_total and start_date") from None
    if not rate.is_finite() or rate <= 0:
        raise ValueError("Vast API did not provide a valid total hourly rate")
    if not start.is_finite() or start <= 0:
        raise ValueError("Vast API did not provide a valid instance start_date")
    now = Decimal(str(now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp()))
    if not now.is_finite() or start > now + Decimal(30):
        raise ValueError("Vast API instance start_date is in the future")
    host, port = _ssh_endpoint(row)

    lifetime_budget_seconds = int((spend_cap / rate * Decimal(3600)).to_integral_value(rounding=ROUND_FLOOR))
    elapsed_seconds = max(0, int(now - start))
    remaining_before_reserve = lifetime_budget_seconds - elapsed_seconds
    setup_transfer_reserve = (
        PRIVATE_PREFLIGHT_RESERVE_SECONDS
        + SOURCE_STAGING_RESERVE_SECONDS
        + UPLOAD_RESERVE_SECONDS
    )
    runtime_seconds = min(
        MAX_REMOTE_RUNTIME_SECONDS,
        remaining_before_reserve
        - RESERVE_SECONDS
        - RETRIEVAL_RESERVE_SECONDS
        - setup_transfer_reserve,
    )
    if runtime_seconds < MIN_REMOTE_RUNTIME_SECONDS:
        raise ValueError(
            "the $10 lifetime cap leaves less than 20 minutes for remote work after elapsed time and shutdown reserve"
        )
    return {
        "instance_id": str(row["id"]),
        "actual_status": str(row.get("actual_status") or row.get("cur_state")),
        "gpu_name": str(row.get("gpu_name")),
        "gpu_count": int(Decimal(str(row.get("num_gpus")))),
        "hourly_rate_usd": str(rate),
        "spend_cap_usd": str(spend_cap),
        "instance_start_epoch": float(start),
        "elapsed_before_dispatch_seconds": elapsed_seconds,
        "estimated_cost_before_dispatch_usd": str(rate * Decimal(elapsed_seconds) / Decimal(3600)),
        "allowed_remote_runtime_seconds": runtime_seconds,
        "budget_deadline_epoch": float(start + Decimal(lifetime_budget_seconds)),
        "stop_reserve_seconds": RESERVE_SECONDS,
        "retrieval_reserve_seconds": RETRIEVAL_RESERVE_SECONDS,
        "setup_transfer_reserve_seconds": setup_transfer_reserve,
        "ssh_host": host,
        "ssh_port": port,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--spend-cap-usd")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--verify-target-only", action="store_true")
    args = parser.parse_args()
    instance_id = None
    try:
        instance_id = validate_instance_id(args.instance_id)
        if not args.verify_target_only:
            if not args.spend_cap_usd or args.report is None or args.github_output is None:
                raise ValueError("spend cap, report path and GitHub output path are required")
            cap = parse_cap(args.spend_cap_usd)
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        row = find_owned_target(payload, instance_id)
        if args.verify_target_only:
            print(json.dumps({
                "instance_id": instance_id,
                "actual_status": str(row.get("actual_status") or row.get("cur_state") or "unknown"),
                "gpu_name": str(row.get("gpu_name")),
                "num_gpus": int(Decimal(str(row.get("num_gpus")))),
            }, sort_keys=True))
            return 0
        # Once account ownership and exact hardware are established, ensure
        # every later preflight failure can still stop this new target.
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write("safe_to_stop=true\n")
        report = acceptance_plan(row, cap)
        report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["gpu_test"] = "ready"
        with args.report.open("w", encoding="utf-8") as output:
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write("instance_running=true\n")
            output.write(f"allowed_runtime_seconds={report['allowed_remote_runtime_seconds']}\n")
            output.write(f"budget_deadline_epoch={report['budget_deadline_epoch']}\n")
            output.write(f"stop_reserve_seconds={report['stop_reserve_seconds']}\n")
            output.write(f"retrieval_reserve_seconds={report['retrieval_reserve_seconds']}\n")
            output.write(f"ssh_host={report['ssh_host']}\n")
            output.write(f"ssh_port={report['ssh_port']}\n")
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as error:
        if args.report is not None:
            try:
                current = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
            except Exception:
                current = {}
            if instance_id is not None:
                current.setdefault("instance_id", instance_id)
            current.update(preflight="failed", preflight_error=str(error))
            args.report.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Vast preflight rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
