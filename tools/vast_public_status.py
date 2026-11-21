from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ALLOWED = {
    "inventory_query": {"not_run", "success", "failed"},
    "target_preflight": {"not_run", "passed", "rejected"},
    "gpu_acceptance": {"not_run", "passed", "failed"},
    "cleanup": {"not_run", "confirmed", "unverified"},
    "watchdog": {"not_run", "completed", "failed"},
}
PUBLIC_FIELDS = {"checked_at_utc", *ALLOWED}


def _validate_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("checked_at_utc must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("checked_at_utc must be a timezone-aware ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("checked_at_utc must be a timezone-aware ISO timestamp")
    return value


def make_public_status(
    *,
    checked_at_utc: str | None = None,
    inventory_query: str,
    target_preflight: str,
    gpu_acceptance: str,
    cleanup: str,
    watchdog: str,
) -> dict[str, str]:
    timestamp = checked_at_utc or datetime.now(timezone.utc).isoformat()
    report = {
        "checked_at_utc": _validate_timestamp(timestamp),
        "inventory_query": inventory_query,
        "target_preflight": target_preflight,
        "gpu_acceptance": gpu_acceptance,
        "cleanup": cleanup,
        "watchdog": watchdog,
    }
    for field, allowed in ALLOWED.items():
        if report[field] not in allowed:
            raise ValueError(f"{field} status is not allowlisted")
    return report


def _mapped_outcome(value: str, *, passed: str, failed: str) -> str:
    if value == "success":
        return passed
    if value in {"failure", "cancelled"}:
        return failed
    return "not_run"


def write_public_status(
    report: dict[str, str],
    output_path: Path,
    summary_path: Path,
) -> None:
    if set(report) != PUBLIC_FIELDS:
        raise ValueError("public status fields are not allowlisted")
    for field, allowed in ALLOWED.items():
        if report[field] not in allowed:
            raise ValueError(f"{field} status is not allowlisted")
    _validate_timestamp(report["checked_at_utc"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "## Redacted Vast operational status",
        "",
        f"- Checked at UTC: {report['checked_at_utc']}",
        f"- Inventory query: {report['inventory_query']}",
        f"- Target preflight: {report['target_preflight']}",
        f"- GPU acceptance: {report['gpu_acceptance']}",
        f"- Cleanup confirmation: {report['cleanup']}",
        f"- Watchdog: {report['watchdog']}",
        "",
    ]
    with summary_path.open("a", encoding="utf-8") as output:
        output.write("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-query", choices=sorted(ALLOWED["inventory_query"]), default="not_run")
    parser.add_argument("--preflight-outcome", choices=("success", "failure", "cancelled", "skipped"), default="skipped")
    parser.add_argument("--gpu-outcome", choices=("success", "failure", "cancelled", "skipped"), default="skipped")
    parser.add_argument("--cleanup-outcome", choices=("success", "failure", "cancelled", "skipped"), default="skipped")
    parser.add_argument("--watchdog-outcome", choices=("success", "failure", "cancelled", "skipped"), default="skipped")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    report = make_public_status(
        inventory_query=args.inventory_query,
        target_preflight=_mapped_outcome(args.preflight_outcome, passed="passed", failed="rejected"),
        gpu_acceptance=_mapped_outcome(args.gpu_outcome, passed="passed", failed="failed"),
        cleanup=_mapped_outcome(args.cleanup_outcome, passed="confirmed", failed="unverified"),
        watchdog=_mapped_outcome(args.watchdog_outcome, passed="completed", failed="failed"),
    )
    write_public_status(report, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
