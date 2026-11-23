#!/usr/bin/env python3
"""Independently stop tagged Vast acceptance instances after completion or at their spend deadline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vast_campaign_guard import campaign_deadline, find_campaign_target, parse_campaign_label, stop_reason


VAST_API = "https://console.vast.ai/api/v0/instances/"
EXPECTED_WORKFLOW_PATH = ".github/workflows/vast-cnot-control-collectives.yml"
LABEL_PREFIX = "mgbs-cnot-acceptance:"
STOP_RESERVE_SECONDS = 1200
MAX_READBACK_ATTEMPTS = 6


class WatchdogError(RuntimeError):
    pass


def _request_json(url: str, *, token: str, method: str = "GET", payload: dict | None = None,
                  timeout: int = 15, accept: str = "application/json") -> dict:
    headers = {"Authorization": f"Bearer {token}", "Accept": accept}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, ValueError) as error:
        raise WatchdogError(f"{method} {url} failed: {error}") from error
    if not isinstance(value, dict):
        raise WatchdogError(f"{method} {url} returned a non-object JSON response")
    return value


def _instances(vast_token: str) -> dict:
    value = _request_json(VAST_API, token=vast_token)
    if not isinstance(value.get("instances"), list):
        raise WatchdogError("Vast API listing did not contain instances")
    return value


def _run_status(repo: str, run_id: int, attempt: int, github_token: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None:
        raise WatchdogError("GITHUB_REPOSITORY has invalid syntax")
    url = (
        f"https://api.github.com/repos/{repo}/actions/runs/"
        f"{run_id}/attempts/{attempt}"
    )
    run = _request_json(
        url,
        token=github_token,
        accept="application/vnd.github+json",
    )
    if run.get("path") != EXPECTED_WORKFLOW_PATH:
        raise WatchdogError("campaign label does not point to the expected acceptance workflow")
    if int(run.get("run_attempt", -1)) != attempt:
        raise WatchdogError("campaign label attempt differs from the GitHub run attempt")
    status = run.get("status")
    if status not in {"queued", "in_progress", "completed"}:
        raise WatchdogError("GitHub returned an unknown workflow status")
    return status


def _row_status(row: dict) -> str:
    return str(row.get("actual_status") or row.get("cur_state") or "unknown").lower()


def _restore_original_label(instance_id: str, label: str, vast_token: str) -> None:
    identity = parse_campaign_label(label)
    if instance_id != identity["instance_id"]:
        raise WatchdogError("refusing to restore a label on an instance outside the campaign identity")
    response = _request_json(
        f"{VAST_API}{instance_id}/",
        token=vast_token,
        method="PUT",
        payload={"label": identity["original_label"]},
    )
    if response.get("success") is not True:
        raise WatchdogError(f"Vast did not restore the prior label for campaign instance {instance_id}")
    readback = _instances(vast_token)
    rows = readback.get("instances", [])
    matches = [row for row in rows if isinstance(row, dict) and str(row.get("id")) == instance_id]
    if len(matches) != 1:
        raise WatchdogError("label restoration readback did not contain exactly one campaign instance")
    actual_label = matches[0].get("label")
    if actual_label is None and identity["original_label"] == "":
        actual_label = ""
    if actual_label != identity["original_label"]:
        raise WatchdogError("fresh Vast API readback did not confirm restoration of the prior label")


def _stop_one(initial: dict, label: str, vast_token: str) -> dict:
    instance_id = str(initial.get("id"))
    if _row_status(initial) in {"stopped", "exited"}:
        _restore_original_label(instance_id, label, vast_token)
        return {
            "instance_id": instance_id,
            "label": label,
            "stop": "already_stopped",
            "readback": _row_status(initial),
            "confirmed": True,
            "prior_label_restored": True,
        }

    current_payload = _instances(vast_token)
    current = find_campaign_target(current_payload, label)
    if str(current.get("id")) != instance_id:
        raise WatchdogError("campaign target ID changed before stop request")
    state = _row_status(current)
    if state in {"stopped", "exited"}:
        _restore_original_label(instance_id, label, vast_token)
        return {
            "instance_id": instance_id,
            "label": label,
            "stop": "already_stopped",
            "readback": state,
            "confirmed": True,
            "prior_label_restored": True,
        }

    response = _request_json(
        f"{VAST_API}{instance_id}/",
        token=vast_token,
        method="PUT",
        payload={"state": "stopped"},
    )
    if response.get("success") is not True:
        raise WatchdogError(f"Vast did not accept stop request for campaign instance {instance_id}")

    last_state = "unknown"
    for attempt in range(MAX_READBACK_ATTEMPTS):
        readback = _instances(vast_token)
        current = find_campaign_target(readback, label)
        if str(current.get("id")) != instance_id:
            raise WatchdogError("campaign target ID changed during lifecycle readback")
        last_state = _row_status(current)
        if last_state in {"stopped", "exited"}:
            _restore_original_label(instance_id, label, vast_token)
            return {
                "instance_id": instance_id,
                "label": label,
                "stop": "requested",
                "readback": last_state,
                "confirmed": True,
                "prior_label_restored": True,
            }
        if attempt + 1 < MAX_READBACK_ATTEMPTS:
            time.sleep(5)
    raise WatchdogError(f"Vast stop readback was not confirmed; last state was {last_state}")


def execute_watchdog(*, vast_token: str, github_token: str, repository: str,
                     now_epoch: int | None = None) -> dict:
    if not vast_token or not github_token:
        raise WatchdogError("VAST_API_KEY and GITHUB_TOKEN are required")
    now = int(now_epoch if now_epoch is not None else time.time())
    payload = _instances(vast_token)
    rows = payload["instances"]
    labels = sorted({
        row.get("label")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("label"), str)
        and row["label"].startswith(LABEL_PREFIX)
    })
    report = {
        "schema_version": 1,
        "checked_at_epoch": now,
        "workflow_path": EXPECTED_WORKFLOW_PATH,
        "instances_scanned": len(rows),
        "campaigns": [],
        "errors": [],
        "storage_billing_continues_after_stop": True,
    }
    for label in labels:
        item = {"label": label}
        try:
            identity = parse_campaign_label(label)
            target = find_campaign_target(payload, label)
            item["instance_id"] = identity["instance_id"]
            item["run_id"] = identity["run_id"]
            item["attempt"] = identity["attempt"]
            try:
                status = _run_status(
                    repository,
                    identity["run_id"],
                    identity["attempt"],
                    github_token,
                )
            except Exception as error:
                status = "unknown"
                item["workflow_status_error"] = str(error)
            item["workflow_status"] = status
            if status == "completed":
                reason = "workflow_completed"
            else:
                item["deadline"] = campaign_deadline(target, label, now_epoch=now)
                item["storage_rate_usd_per_hour"] = item["deadline"]["storage_rate_usd_per_hour"]
                try:
                    reason = stop_reason(
                        status,
                        item["deadline"]["budget_deadline_epoch"],
                        STOP_RESERVE_SECONDS,
                        now,
                    )
                except ValueError as error:
                    item["reason"] = "awaiting_reliable_workflow_status"
                    item["warning"] = str(error)
                    report["campaigns"].append(item)
                    continue
            item["reason"] = reason or "active_before_deadline"
            if reason is not None:
                item.update(_stop_one(target, label, vast_token))
            elif _row_status(target) in {"stopped", "exited"}:
                item.update(_stop_one(target, label, vast_token))
            report["campaigns"].append(item)
        except Exception as error:
            item["error"] = str(error)
            report["errors"].append({"label": label, "error": str(error)})
            report["campaigns"].append(item)
    return report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = execute_watchdog(
            vast_token=os.environ.get("VAST_API_KEY", ""),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            repository=os.environ.get("GITHUB_REPOSITORY", ""),
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "checked_at_epoch": int(time.time()),
            "workflow_path": EXPECTED_WORKFLOW_PATH,
            "campaigns": [],
            "errors": [{"error": str(error)}],
            "storage_billing_continues_after_stop": True,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    campaigns = report.get("campaigns", [])
    outputs = Path(os.environ["GITHUB_OUTPUT"]) if os.environ.get("GITHUB_OUTPUT") else None
    if outputs is not None:
        with outputs.open("a", encoding="utf-8") as stream:
            stream.write(f"campaigns_found={'true' if campaigns else 'false'}\n")
            stream.write(
                "stop_confirmed="
                + ("true" if campaigns and all(item.get("confirmed") for item in campaigns) else "false")
                + "\n"
            )
    print(json.dumps(report, sort_keys=True))
    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
