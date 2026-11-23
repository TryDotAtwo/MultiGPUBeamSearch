#!/usr/bin/env python3
"""Campaign identity and fail-closed stop policy for the external Vast watchdog."""

from __future__ import annotations

import argparse
import base64
from decimal import Decimal, InvalidOperation
import json
import re
import sys
from pathlib import Path

from vast_instance_contract import (
    find_owned_target,
    lifetime_budget_deadline,
    parse_cap,
    validate_instance_id,
)


_LABEL_RE = re.compile(
    r"mgbs-cnot-acceptance:([1-9][0-9]*):([1-9][0-9]*):([1-9][0-9]*):([0-9]+(?:\.[0-9]+)?):([A-Za-z0-9_-]*)"
)
MAX_LABEL_LENGTH = 1024


def _positive_number(value: int | str, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive decimal integer")
    text = str(value)
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise ValueError(f"{field} must be a positive decimal integer")
    return int(text)


def _encode_original_label(value: str | None) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError("original instance label must be text")
    if value.startswith("mgbs-cnot-acceptance:"):
        raise ValueError("instance is already marked by an active campaign")
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_original_label(value: str) -> str:
    if not value:
        return ""
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("campaign label contains an invalid saved instance label") from error


def campaign_label(
    instance_id: int | str,
    run_id: int | str,
    attempt: int | str,
    spend_cap_usd: Decimal | str,
    original_label: str | None = "",
) -> str:
    target = validate_instance_id(str(instance_id))
    run = _positive_number(run_id, "run_id")
    retry = _positive_number(attempt, "attempt")
    cap = parse_cap(str(spend_cap_usd))
    encoded_label = _encode_original_label(original_label)
    label = (
        f"mgbs-cnot-acceptance:{target}:{run}:{retry}:"
        f"{format(cap.normalize(), 'f')}:{encoded_label}"
    )
    if len(label) > MAX_LABEL_LENGTH:
        raise ValueError("the prior Vast instance label is too long to preserve safely")
    return label


def parse_campaign_label(label: str) -> dict:
    if not isinstance(label, str) or len(label) > MAX_LABEL_LENGTH:
        raise ValueError("campaign label must be bounded text")
    match = _LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError("campaign label has an invalid format")
    instance_id, run_id, attempt, cap_text, original_encoded = match.groups()
    target = validate_instance_id(instance_id)
    cap = parse_cap(cap_text)
    original_label = _decode_original_label(original_encoded)
    if campaign_label(target, run_id, attempt, cap, original_label) != label:
        raise ValueError("campaign label is not canonical")
    return {
        "instance_id": target,
        "run_id": int(run_id),
        "attempt": int(attempt),
        "spend_cap_usd": cap,
        "original_label": original_label,
    }


def find_campaign_target(payload: object, label: str) -> dict:
    identity = parse_campaign_label(label)
    rows = payload.get("instances") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Vast API response does not contain an instances list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("label") == label]
    if len(matches) != 1:
        raise ValueError("campaign label must match exactly one account-owned instance")
    row = matches[0]
    if str(row.get("id")) != identity["instance_id"]:
        raise ValueError("campaign label is bound to a different instance ID")
    target = find_owned_target(payload, identity["instance_id"])
    if str(target.get("id")) != str(row.get("id")) or target.get("label") != label:
        raise ValueError("campaign label or instance identity changed during validation")
    return target


def campaign_deadline(row: dict, label: str, *, now_epoch=None) -> dict:
    identity = parse_campaign_label(label)
    if str(row.get("id")) != identity["instance_id"] or row.get("label") != label:
        raise ValueError("instance no longer matches the exact campaign identity")
    return lifetime_budget_deadline(
        row,
        identity["spend_cap_usd"],
        now_epoch=now_epoch,
    )


def verify_restored_label(payload: object, label: str) -> dict:
    identity = parse_campaign_label(label)
    rows = payload.get("instances") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Vast API response does not contain an instances list")
    matches = [row for row in rows if isinstance(row, dict) and str(row.get("id")) == identity["instance_id"]]
    if len(matches) != 1:
        raise ValueError("restored-label readback must contain exactly one campaign instance")
    row = find_owned_target(payload, identity["instance_id"])
    actual = row.get("label")
    if actual is None and identity["original_label"] == "":
        actual = ""
    if actual != identity["original_label"]:
        raise ValueError("fresh API readback did not confirm restoration of the prior instance label")
    return row


def stop_reason(workflow_status: str, deadline_epoch, stop_reserve_seconds, now_epoch):
    try:
        deadline = Decimal(str(deadline_epoch))
        now = Decimal(str(now_epoch))
        reserve_decimal = Decimal(str(stop_reserve_seconds))
    except (InvalidOperation, TypeError):
        raise ValueError("deadline, stop reserve and current time must be numeric") from None
    if (not deadline.is_finite() or not now.is_finite() or not reserve_decimal.is_finite()
            or deadline <= 0 or now <= 0 or reserve_decimal < 0
            or reserve_decimal != reserve_decimal.to_integral_value()):
        raise ValueError("deadline, stop reserve and current time are invalid")
    if now >= deadline - reserve_decimal:
        return "billing_deadline"
    if workflow_status == "completed":
        return "workflow_completed"
    if workflow_status not in {"queued", "in_progress"}:
        raise ValueError("workflow status is unknown; stop decision is withheld before the billing deadline")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    label_parser = commands.add_parser("label")
    label_parser.add_argument("--payload", required=True, type=Path)
    label_parser.add_argument("--instance-id", required=True)
    label_parser.add_argument("--run-id", required=True)
    label_parser.add_argument("--attempt", required=True)
    label_parser.add_argument("--spend-cap-usd", required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--payload", required=True, type=Path)
    verify_parser.add_argument("--instance-id", required=True)
    verify_parser.add_argument("--label", required=True)
    restore_parser = commands.add_parser("restore-body")
    restore_parser.add_argument("--label", required=True)
    restore_parser.add_argument("--output", required=True, type=Path)
    restored_parser = commands.add_parser("verify-restored")
    restored_parser.add_argument("--payload", required=True, type=Path)
    restored_parser.add_argument("--instance-id", required=True)
    restored_parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "label":
            payload = json.loads(args.payload.read_text(encoding="utf-8"))
            target = find_owned_target(payload, args.instance_id)
            original_label = target.get("label") or ""
            print(campaign_label(
                args.instance_id, args.run_id, args.attempt, args.spend_cap_usd, original_label
            ))
            return 0
        identity = parse_campaign_label(args.label)
        if args.command == "restore-body":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"label": identity["original_label"]}, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            return 0
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        instance_id = validate_instance_id(args.instance_id)
        if args.command == "verify-restored":
            target = verify_restored_label(payload, args.label)
            if str(target.get("id")) != instance_id:
                raise ValueError("restored label belongs to a different instance ID")
            print(json.dumps({
                "instance_id": instance_id,
                "restored_label": target.get("label") or "",
            }, sort_keys=True))
            return 0
        target = find_campaign_target(payload, args.label)
        if str(target.get("id")) != instance_id:
            raise ValueError("campaign label belongs to a different instance ID")
        print(json.dumps({
            "instance_id": instance_id,
            "campaign_label": args.label,
            "gpu_name": str(target.get("gpu_name")),
            "num_gpus": int(Decimal(str(target.get("num_gpus")))),
        }, sort_keys=True))
        return 0
    except Exception as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
