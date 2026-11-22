#!/usr/bin/env python3
"""Stable required status and input-keyed reuse for the pinned CNOT producer build."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


STATE_SCHEMA = "cnot_panel_v2_compatibility_status_v1"
PANEL_V2_BUILD_INPUTS = (
    ".github/workflows/cnot-producer-compatibility.yml",
    "integrations/cnot_scorer/build.py",
    "integrations/cnot_scorer/bridge_panel_v2.rs",
    "integrations/cnot_scorer/handoff.py",
    "integrations/cnot_scorer/test_handoff.py",
    "tools/cnot_producer_compatibility.py",
    "tools/test_cnot_producer_compatibility.py",
)


def _normalize(path: str) -> str:
    value = path.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def requires_full_build(changed_paths: Iterable[str]) -> bool:
    relevant = set(PANEL_V2_BUILD_INPUTS)
    return any(_normalize(path) in relevant for path in changed_paths)


def input_fingerprint(workspace: Path) -> str:
    root = Path(workspace).resolve()
    digest = hashlib.sha256()
    for relative in sorted(PANEL_V2_BUILD_INPUTS):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required panel-v2 build input is missing: {relative}")
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _changed_paths(event_name: str, event: dict, workspace: Path) -> list[str] | None:
    if event_name == "workflow_dispatch":
        return []
    if event_name == "pull_request":
        pull_request = event.get("pull_request")
        if not isinstance(pull_request, dict):
            return None
        head = pull_request.get("head", {}).get("sha")
        if not isinstance(head, str):
            return None
        # A PR's base...head range keeps reporting old adapter changes on every
        # later docs-only commit. Inspect this event's head commit instead. If
        # no matching cached fingerprint exists, decide() still runs the full
        # build, so the first PR run and unknown history cannot be skipped.
        revision = f"{head}^..{head}"
    elif event_name == "push":
        before, after = event.get("before"), event.get("after")
        if not isinstance(before, str) or not isinstance(after, str) or set(before) == {"0"}:
            return None
        revision = f"{before}..{after}"
    else:
        return None
    result = subprocess.run(
        ["git", "-C", str(Path(workspace).resolve()), "diff", "--name-only", revision],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return [_normalize(line) for line in result.stdout.splitlines() if line.strip()]


def _write_outputs(path: Path | None, values: dict[str, str]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")


def prepare(event_name: str, event_path: Path, workspace: Path, output: Path | None) -> dict[str, str]:
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    fingerprint = input_fingerprint(workspace)
    changed = _changed_paths(event_name, event, workspace)
    changed_inputs = changed is None or requires_full_build(changed)
    values = {
        "fingerprint": fingerprint,
        "changed_inputs": "true" if changed_inputs else "false",
        "changed_count": "unknown" if changed is None else str(len(changed)),
        "force_full": "true" if event_name == "workflow_dispatch" else "false",
    }
    _write_outputs(output, values)
    print(json.dumps(values, sort_keys=True))
    return values


def cached_status(state_path: Path, fingerprint: str) -> str:
    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if (
        not isinstance(state, dict)
        or state.get("schema") != STATE_SCHEMA
        or state.get("fingerprint") != fingerprint
        or state.get("status") not in {"passed", "failed"}
    ):
        return "unknown"
    return state["status"]


def choose_action(*, force_full: bool, changed_inputs: bool, previous_status: str) -> str:
    if force_full or changed_inputs or previous_status == "unknown":
        return "run_full"
    if previous_status == "passed":
        return "reuse_pass"
    if previous_status == "failed":
        return "carry_failure"
    return "run_full"


def decide(
    state_path: Path,
    fingerprint: str,
    *,
    force_full: bool,
    changed_inputs: bool,
    output: Path | None,
) -> dict[str, str]:
    previous = cached_status(state_path, fingerprint)
    action = choose_action(
        force_full=force_full,
        changed_inputs=changed_inputs,
        previous_status=previous,
    )
    values = {"action": action, "previous_status": previous}
    _write_outputs(output, values)
    print(json.dumps(values, sort_keys=True))
    return values


def finalize(
    state_path: Path,
    fingerprint: str,
    action: str,
    cheap_outcome: str,
    compile_outcome: str,
    handoff_outcome: str,
    summary_path: Path | None = None,
) -> dict[str, str]:
    if cheap_outcome != "success":
        status = "failed"
        detail = "the cheap CPU contract check failed"
    elif action == "carry_failure":
        status = "failed"
        detail = "the same panel-v2 inputs have a previous failed result"
    elif action == "reuse_pass":
        status = "passed"
        detail = "reused the result for the same panel-v2 inputs after rerunning cheap CPU checks"
    elif action == "run_full" and compile_outcome == "success" and handoff_outcome == "success":
        status = "passed"
        detail = "pinned producer adapter build and handoff fixture passed"
    else:
        status = "failed"
        detail = "the required pinned producer adapter build or handoff fixture failed or did not run"

    state = {
        "schema": STATE_SCHEMA,
        "fingerprint": fingerprint,
        "status": status,
        "detail": detail,
    }
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    if summary_path is not None:
        Path(summary_path).write_text(
            f"CNOT producer panel-v2 compatibility: **{status}** — {detail}.\n",
            encoding="utf-8",
        )
    print(json.dumps(state, sort_keys=True))
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("prepare")
    plan.add_argument("--event-name", required=True)
    plan.add_argument("--event-file", type=Path, required=True)
    plan.add_argument("--workspace", type=Path, required=True)
    plan.add_argument("--output", type=Path)

    choose = commands.add_parser("decide")
    choose.add_argument("--state-file", type=Path, required=True)
    choose.add_argument("--fingerprint", required=True)
    choose.add_argument("--changed-inputs", choices=("true", "false"), required=True)
    choose.add_argument("--force-full", choices=("true", "false"), required=True)
    choose.add_argument("--output", type=Path)

    finish = commands.add_parser("finish")
    finish.add_argument("--state-file", type=Path, required=True)
    finish.add_argument("--fingerprint", required=True)
    finish.add_argument("--action", choices=("run_full", "reuse_pass", "carry_failure"), required=True)
    finish.add_argument("--cheap-outcome", required=True)
    finish.add_argument("--compile-outcome", required=True)
    finish.add_argument("--handoff-outcome", required=True)
    finish.add_argument("--summary-file", type=Path)

    check = commands.add_parser("verify-result")
    check.add_argument("--state-file", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.event_name, args.event_file, args.workspace, args.output)
        elif args.command == "decide":
            decide(
                args.state_file,
                args.fingerprint,
                force_full=args.force_full == "true",
                changed_inputs=args.changed_inputs == "true",
                output=args.output,
            )
        elif args.command == "finish":
            state = finalize(
                args.state_file,
                args.fingerprint,
                args.action,
                args.cheap_outcome,
                args.compile_outcome,
                args.handoff_outcome,
                args.summary_file,
            )
            return 0 if state["status"] == "passed" else 1
        else:
            state = json.loads(args.state_file.read_text(encoding="utf-8"))
            if state.get("schema") != STATE_SCHEMA or state.get("status") != "passed":
                print("panel-v2 compatibility result is not passed", file=sys.stderr)
                return 1
            print("panel-v2 compatibility result is passed")
    except Exception as error:
        print(f"panel-v2 compatibility gate failed: {type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
