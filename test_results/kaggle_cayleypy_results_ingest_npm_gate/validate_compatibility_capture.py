from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path


KERNEL_REF = "trydotatwo/cayleypy-results-ingest-npm-gate"
CURRENT_STACK = {
    "@cloudflare/vitest-pool-workers": "0.19.0",
    "@cloudflare/workers-types": "5.20260729.1",
    "miniflare": "4.20260722.1",
    "vitest": "4.1.10",
    "workerd": "1.20260729.1",
    "wrangler": "4.115.0",
}
PROBE_STACK = {
    "@cloudflare/vitest-pool-workers": "0.18.8",
    "@cloudflare/workers-types": "5.20260727.1",
    "miniflare": "4.20260722.0",
    "vitest": "4.1.10",
    "workerd": "1.20260728.1",
    "wrangler": "4.114.0",
}
BASE_COMMANDS = {
    "node-version": 0,
    "npm-version": 0,
    "npm-view-vitest-pool": 0,
    "npm-view-wrangler": 0,
    "npm-view-workers-types": 0,
    "npm-view-vitest": 0,
    "npm-view-workerd": 0,
    "npm-install-package-lock-only": 0,
    "npm-ci": 0,
    "npm-ls-critical": 0,
}
PHASES = {
    "v26_probe": {
        "version": 26,
        "phase": "diagnostic-red",
        "all_commands_passed": False,
        "commands": {
            **BASE_COMMANDS,
            "npm-test": 1,
            "npm-test-worker": 1,
            "npm-typecheck": 2,
            "gate-assertions": 1,
        },
        "stack": PROBE_STACK,
        "registry_exact": False,
        "post_install_exact": False,
        "worker_passed": False,
        "compare_current": False,
    },
    "v28_final": {
        "version": 28,
        "phase": "typecheck-red",
        "all_commands_passed": False,
        "commands": {
            **BASE_COMMANDS,
            "npm-test": 0,
            "npm-test-worker": 0,
            "npm-typecheck": 2,
            "gate-assertions": 1,
        },
        "stack": CURRENT_STACK,
        "registry_exact": True,
        "post_install_exact": True,
        "worker_passed": True,
        "compare_current": False,
    },
    "v29_final": {
        "version": 29,
        "phase": "green",
        "all_commands_passed": True,
        "commands": {
            **BASE_COMMANDS,
            "npm-test": 0,
            "npm-test-worker": 0,
            "npm-typecheck": 0,
            "gate-assertions": 0,
        },
        "stack": CURRENT_STACK,
        "registry_exact": True,
        "post_install_exact": True,
        "worker_passed": True,
        "compare_current": True,
    },
}
CONTROL_FILES = (
    "push_receipt.txt",
    "push_observed_utc.txt",
    "status_receipt.txt",
    "status_observed_utc.txt",
    "output_receipt.txt",
)
PULL_CONTROL_FILES = (
    "pull_receipt.txt",
    "pull/kernel-metadata.json",
    "pull/cayleypy-results-ingest-npm-gate.ipynb",
)
OUTPUT_FILES = (
    "node-bootstrap.json",
    "node-version.log",
    "npm-version.log",
    "gate-assertions.log",
    "npm-gate-results.json",
    "payload-sha256.json",
    "post-install-payload-sha256.json",
    "registry-metadata.json",
    "resolved-stack.json",
    "compatibility-warning-scan.json",
    "npm-ls-critical.log",
    "npm-test.log",
    "npm-test-worker.log",
    "npm-typecheck.log",
    "npm-ci.log",
)
SECRET_PATTERNS = {
    "local_user_path": re.compile(r"(?i)(?:[A-Z]:\\Users\\|/Users/|/home/[^/\s]+/)"),
    "email": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "auth_header": re.compile(r"(?i)\bAuthorization\s*:\s*(?:Bearer|Basic)\s+[A-Za-z0-9+/=._-]{10,}"),
}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    return data.decode("utf-8")


def clean_text(path: Path) -> str:
    return ANSI.sub("", read_text(path))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def notebook_payload(path: Path) -> tuple[dict[str, str], list[str], dict[str, str]]:
    notebook = json.loads(read_text(path))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assignments: dict[str, object] = {}
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"PAYLOAD_B64", "EXPECTED_SHA256"}
        ):
            assignments[node.targets[0].id] = ast.literal_eval(node.value)
    expected = assignments["EXPECTED_SHA256"]
    encoded = assignments["PAYLOAD_B64"]
    assert isinstance(expected, dict) and isinstance(encoded, str)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        names = archive.namelist()
        embedded = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}
    return expected, names, embedded


def validate(root: Path, phase: str) -> dict[str, object]:
    expected_phase = PHASES[phase]
    evidence_root = root / "test_results/kaggle_cayleypy_results_ingest_npm_gate"
    control = evidence_root / f"control_{phase}"
    outputs = evidence_root / f"outputs_{phase}"
    retained = [control / item for item in CONTROL_FILES + PULL_CONTROL_FILES]
    retained += [outputs / item for item in OUTPUT_FILES]
    assert all(path.is_file() and path.stat().st_size > 0 for path in retained)
    assert not any(
        path.name in {"list.csv", "package-lock.json", "cayleypy-results-ingest-npm-gate.log", "npm-gate.log"}
        for path in retained
    )

    capture = json.loads(read_text(control / "capture-manifest.json"))
    assert capture["schema_version"] == 1
    assert capture["kernel_ref"] == KERNEL_REF
    assert capture["kernel_version"] == expected_phase["version"]
    assert capture["phase"] == expected_phase["phase"]
    expected_pull_provenance = "run-exact"
    if phase == "v26_probe":
        expected_pull_provenance = "stale source pull after the next push; excluded from v26 run identity"
    assert capture["pull_provenance"] == expected_pull_provenance
    assert capture["terminal_status"] == "KernelWorkerStatus.COMPLETE"
    assert capture["evidence_policy"] == "list-free; no identity-bearing kernel list capture"
    captured = {entry["path"]: entry for entry in capture["files"]}
    retained_relative = [path.relative_to(root).as_posix() for path in retained]
    assert set(captured) == set(retained_relative)
    for path, relative in zip(retained, retained_relative, strict=True):
        assert captured[relative]["bytes"] == path.stat().st_size
        assert captured[relative]["sha256"] == sha256(path)

    version = expected_phase["version"]
    assert f"Kernel version {version} successfully pushed" in read_text(control / "push_receipt.txt")
    assert "KernelWorkerStatus.COMPLETE" in read_text(control / "status_receipt.txt")
    parse_utc(read_text(control / "push_observed_utc.txt"))
    parse_utc(read_text(control / "status_observed_utc.txt"))
    manifest = json.loads(read_text(outputs / "payload-sha256.json"))
    post_manifest = json.loads(read_text(outputs / "post-install-payload-sha256.json"))
    results = json.loads(read_text(outputs / "npm-gate-results.json"))
    commands = {item["label"]: item["exit_code"] for item in results["commands"]}
    assert results["all_commands_passed"] is expected_phase["all_commands_passed"]
    assert commands == expected_phase["commands"]
    assert len(manifest) == 18 and manifest == results["payload_sha256"]
    metadata = json.loads(read_text(control / "pull/kernel-metadata.json"))
    assert metadata["id"] == KERNEL_REF
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False and metadata["enable_tpu"] is False
    assert metadata["machine_shape"] == "None" and metadata["enable_internet"] is True
    pulled_expected, names, embedded = notebook_payload(
        control / "pull/cayleypy-results-ingest-npm-gate.ipynb"
    )
    assert names == list(pulled_expected)
    assert embedded == pulled_expected
    pulled_mismatches = {
        relative: {"run": manifest.get(relative), "pulled": pulled_expected.get(relative)}
        for relative in set(manifest) | set(pulled_expected)
        if manifest.get(relative) != pulled_expected.get(relative)
    }
    assert (not pulled_mismatches) is (phase != "v26_probe")
    payload_expected = manifest
    assert post_manifest == results["post_install_payload_sha256"]
    mismatch_count = len(results["post_install_payload_mismatches"])
    assert (mismatch_count == 0) is expected_phase["post_install_exact"]
    if expected_phase["compare_current"]:
        current = {relative: sha256(root / relative) for relative in payload_expected}
        assert current == payload_expected

    bootstrap = json.loads(read_text(outputs / "node-bootstrap.json"))
    assert bootstrap == {
        "archive": "node-v22.23.1-linux-x64.tar.xz",
        "sha256": "9749e988f437343b7fa832c69ded82a312e41a03116d766797ac14f6f9eee578",
        "version": "v22.23.1",
    }
    assert read_text(outputs / "node-version.log").strip() == "v22.23.1"
    assert read_text(outputs / "npm-version.log").strip() == "10.9.8"

    registry = json.loads(read_text(outputs / "registry-metadata.json"))
    assert registry["exact"] is expected_phase["registry_exact"]
    resolved = json.loads(read_text(outputs / "resolved-stack.json"))
    assert resolved["exact"] is True
    assert resolved["expected"] == expected_phase["stack"]
    assert resolved["resolved"] == expected_phase["stack"]
    assert resolved["lockfile_workerd_paths"] == ["node_modules/workerd"]
    assert resolved["lockfile_workerd_versions"] == [expected_phase["stack"]["workerd"]]
    warning_scan = json.loads(read_text(outputs / "compatibility-warning-scan.json"))
    assert warning_scan["compatibility_date"] == "2026-07-28"
    assert warning_scan["passed"] is True and warning_scan["hits"] == []
    assert results["compatibility_warning_scan"] == warning_scan

    npm_test = clean_text(outputs / "npm-test.log")
    worker_test = clean_text(outputs / "npm-test-worker.log")
    typecheck = clean_text(outputs / "npm-typecheck.log")
    assert "Tests  12 passed (12)" in npm_test
    if expected_phase["worker_passed"]:
        assert "Tests  70 passed (70)" in npm_test
        assert "Tests  70 passed (70)" in worker_test
    else:
        marker = 'Missing "./config" specifier in "@cloudflare/vitest-pool-workers" package'
        assert marker in npm_test and marker in worker_test
    assert ("error TS" not in typecheck) is (expected_phase["commands"]["npm-typecheck"] == 0)

    semantic = capture["semantic_validation"]
    assert semantic["all_commands_passed"] is expected_phase["all_commands_passed"]
    assert semantic["schema"] == "12 passed, 12 total"
    assert semantic["typecheck_exit_code"] == expected_phase["commands"]["npm-typecheck"]
    assert semantic["payload_files"] == 18
    assert semantic["post_install_payload_mismatches"] == mismatch_count
    pulled_matches_run = not pulled_mismatches
    assert semantic["pulled_notebook_payload_matches_run"] is pulled_matches_run
    assert semantic["pulled_notebook_payload_mismatches"] == len(pulled_mismatches)
    if phase == "v26_probe":
        assert set(pulled_mismatches) == {
            "services/cayleypy-results-ingest/package.json",
            "services/cayleypy-results-ingest/tsconfig.json",
            "services/cayleypy-results-ingest/vitest.config.ts",
        }

    findings = []
    for path in retained:
        value = read_text(path)
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(value):
                findings.append({"path": path.relative_to(root).as_posix(), "pattern": label})
    assert not findings
    return {
        "phase": phase,
        "kernel_version": version,
        "terminal_status": "KernelWorkerStatus.COMPLETE",
        "payload_files": len(payload_expected),
        "post_install_payload_mismatches": mismatch_count,
        "pulled_notebook_payload_mismatches": len(pulled_mismatches),
        "retained_files": len(retained),
        "secret_findings": findings,
        "commands": commands,
        "resolved_stack": resolved["resolved"],
        "compatibility_date": warning_scan["compatibility_date"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=tuple(PHASES))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve(), args.phase), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
