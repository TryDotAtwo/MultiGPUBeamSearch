"""Fail-closed identity and deadline helpers for bounded CNOT GPU campaigns."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any

_LABEL = re.compile(
    r"^mgbs-cnot-acceptance:([1-9][0-9]*):([1-9][0-9]*):"
    r"(0|[1-9][0-9]*)(?:\.([0-9]+))?$"
)
_LEGACY_INSTANCE_IDS = frozenset({61080396})
_MAX_SPEND_USD = Decimal("10")


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _spend_cap(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("spend cap must be a finite positive USD amount")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("spend cap must be a finite positive USD amount") from None
    if not amount.is_finite() or amount <= 0 or amount > _MAX_SPEND_USD:
        raise ValueError("spend cap must be positive and at most $10")
    return amount.normalize()


def campaign_label(run_id: object, attempt: object, spend_cap_usd: object) -> str:
    run = _positive_int(run_id, "run ID")
    retry = _positive_int(attempt, "attempt")
    cap = _spend_cap(spend_cap_usd)
    return f"mgbs-cnot-acceptance:{run}:{retry}:{format(cap, 'f')}"


def parse_campaign_label(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("campaign label must be text")
    match = _LABEL.fullmatch(value)
    if match is None:
        raise ValueError("campaign label is malformed")
    run_id, attempt, whole, fraction = match.groups()
    cap = _spend_cap(Decimal(whole + (("." + fraction) if fraction else "")))
    canonical = campaign_label(int(run_id), int(attempt), cap)
    if canonical != value:
        raise ValueError("campaign label is not canonical")
    return {
        "run_id": int(run_id),
        "attempt": int(attempt),
        "spend_cap_usd": cap,
    }


def find_campaign_target(payload: object, expected_label: object) -> dict[str, Any]:
    """Return one exact live campaign row, rejecting legacy or wrong hardware."""
    parsed = parse_campaign_label(expected_label)
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        raise ValueError("instance inventory is malformed")
    matches = [
        row for row in payload["instances"]
        if isinstance(row, dict) and row.get("label") == expected_label
    ]
    if len(matches) != 1:
        raise ValueError("campaign inventory must contain exactly one target")
    row = matches[0]
    ident = row.get("id")
    if type(ident) is not int or ident <= 0 or ident in _LEGACY_INSTANCE_IDS:
        raise ValueError("campaign target must be a new positive-decimal instance ID")
    if row.get("actual_status") != "running":
        raise ValueError("campaign target is not running")
    if row.get("gpu_name") != "RTX 3060" or type(row.get("num_gpus")) is not int or row["num_gpus"] != 2:
        raise ValueError("campaign target hardware is not exactly 2x RTX 3060")
    # Bind identity to the parsed campaign marker without trusting caller-provided aliases.
    if parsed["run_id"] <= 0 or parsed["attempt"] <= 0:
        raise ValueError("campaign marker identity is invalid")
    return row


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def stop_reason(
    workflow_status: object,
    billing_deadline_epoch: object,
    stop_reserve_seconds: object,
    now_epoch: object,
) -> str | None:
    """Stop completed campaigns or running ones at the billing stop margin."""
    if workflow_status == "completed":
        return "workflow_completed"
    if workflow_status != "in_progress":
        raise ValueError("workflow status is unknown")
    deadline = _finite_number(billing_deadline_epoch, "billing deadline")
    reserve = _finite_number(stop_reserve_seconds, "stop reserve")
    now = _finite_number(now_epoch, "current time")
    if deadline <= 0 or reserve < 0 or now < 0:
        raise ValueError("billing deadline inputs are outside their valid range")
    return "billing_deadline" if now >= deadline - reserve else None
