#!/usr/bin/env python3
"""Find the latest completed panel-v2 build for the same inputs and GitHub ref."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys


def cache_refs(event, ref):
    refs = [ref]
    pr = event.get("pull_request")
    if pr and pr["head"]["repo"]["id"] == event["repository"]["id"]:
        refs.append("refs/heads/" + pr["head"]["ref"])
    return refs


def last_result(prefix, refs, list_caches):
    results = []
    for ref in refs:
        for entry in list_caches(prefix, ref):
            match = re.fullmatch(re.escape(prefix) + r"\d+-\d+-(success|failure)", entry["key"])
            if entry["ref"] == ref and match:
                results.append((datetime.fromisoformat(entry["created_at"]), entry["id"], match[1]))
    return max(results)[2] if results else "run"


def list_caches(prefix, ref):
    data = subprocess.check_output([
        "gh", "api", "--method", "GET", f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/caches",
        "-f", f"key={prefix}", "-f", f"ref={ref}", "-f", "sort=created_at",
        "-f", "direction=desc", "-f", "per_page=1",
    ], text=True)
    return json.loads(data)["actions_caches"]


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    head = event.get("pull_request", {}).get("head", {})
    identity = (
        os.environ["PANEL_INPUTS"], os.environ["ImageOS"], os.environ["ImageVersion"],
        os.environ["PYTHON_VERSION_USED"], str(event["repository"]["id"]),
        str((head.get("repo") or {}).get("id", event["repository"]["id"])),
        head.get("ref", os.environ["GITHUB_REF_NAME"]),
    )
    prefix = "cnot-panel-v3-" + hashlib.sha256("\0".join(identity).encode()).hexdigest() + "-"
    verdict = "run"
    if os.environ["GITHUB_EVENT_NAME"] != "workflow_dispatch" and os.environ["GITHUB_RUN_ATTEMPT"] == "1":
        try:
            verdict = last_result(prefix, cache_refs(event, os.environ["GITHUB_REF"]), list_caches)
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
            print(f"Compatibility cache unavailable; running full check ({type(exc).__name__})", file=sys.stderr)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        print(f"prefix={prefix}", f"verdict={verdict}", sep="\n", file=output)
    print(f"Panel-v2 compatibility: {verdict}")


if __name__ == "__main__":
    main()
