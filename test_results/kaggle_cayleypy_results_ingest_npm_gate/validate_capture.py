from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


PHASES = {
    "v23_red": {
        "version": 23,
        "all_commands_passed": False,
        "commands": {
            "node-version": 0,
            "npm-version": 0,
            "npm-install-package-lock-only": 0,
            "npm-ci": 0,
            "npm-test": 1,
            "npm-test-worker": 1,
            "npm-typecheck": 0,
        },
        "schema_summary": "Tests  3 failed | 9 passed (12)",
        "worker_summary": "Tests  6 failed | 64 passed (70)",
        "compare_current": False,
    },
    "v25_green": {
        "version": 25,
        "all_commands_passed": True,
        "commands": {
            "node-version": 0,
            "npm-version": 0,
            "npm-install-package-lock-only": 0,
            "npm-ci": 0,
            "npm-test": 0,
            "npm-test-worker": 0,
            "npm-typecheck": 0,
        },
        "schema_summary": "Tests  12 passed (12)",
        "worker_summary": "Tests  70 passed (70)",
        "compare_current": True,
    },
}
CONTROL_FILES = (
    "push_receipt.txt",
    "status_latest.txt",
    "output_download.txt",
    "pull_receipt.txt",
    "pull/kernel-metadata.json",
    "pull/cayleypy-results-ingest-npm-gate.ipynb",
)
OUTPUT_FILES = (
    "node-version.log",
    "npm-version.log",
    "npm-gate-results.json",
    "payload-sha256.json",
    "npm-test.log",
    "npm-test-worker.log",
    "npm-typecheck.log",
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


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    return data.decode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if not isinstance(expected, dict):
        raise AssertionError("EXPECTED_SHA256 must be a mapping")
    encoded = assignments["PAYLOAD_B64"]
    if not isinstance(encoded, str):
        raise AssertionError("PAYLOAD_B64 must be text")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        names = archive.namelist()
        embedded = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}
    return expected, names, embedded


def validate(root: Path, phase: str) -> dict[str, object]:
    expected_phase = PHASES[phase]
    evidence_root = root / "test_results/kaggle_cayleypy_results_ingest_npm_gate"
    control = evidence_root / f"control_{phase}"
    outputs = evidence_root / f"outputs_{phase}"
    retained = [control / item for item in CONTROL_FILES] + [outputs / item for item in OUTPUT_FILES]
    assert all(path.is_file() and path.stat().st_size > 0 for path in retained)
    assert not any(path.name in {"list.csv", "package-lock.json", "cayleypy-results-ingest-npm-gate.log"} for path in retained)

    capture = json.loads(read_text(control / "capture-manifest.json"))
    captured = {entry["path"]: entry for entry in capture["files"]}
    retained_relative = [path.relative_to(root).as_posix() for path in retained]
    assert set(captured) == set(retained_relative)
    for path, relative in zip(retained, retained_relative, strict=True):
        assert captured[relative]["bytes"] == path.stat().st_size
        assert captured[relative]["sha256"] == sha256(path)

    version = expected_phase["version"]
    assert f"Kernel version {version} successfully pushed" in read_text(control / "push_receipt.txt")
    assert 'KernelWorkerStatus.COMPLETE' in read_text(control / "status_latest.txt")
    metadata = json.loads(read_text(control / "pull/kernel-metadata.json"))
    assert metadata["id"] == "trydotatwo/cayleypy-results-ingest-npm-gate"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False and metadata["enable_tpu"] is False
    assert metadata["machine_shape"] == "None" and metadata["enable_internet"] is True

    expected, names, embedded = notebook_payload(control / "pull/cayleypy-results-ingest-npm-gate.ipynb")
    manifest = json.loads(read_text(outputs / "payload-sha256.json"))
    results = json.loads(read_text(outputs / "npm-gate-results.json"))
    commands = {item["label"]: item["exit_code"] for item in results["commands"]}
    assert results["all_commands_passed"] is expected_phase["all_commands_passed"]
    assert commands == expected_phase["commands"]
    assert len(expected) == 18 and names == list(expected)
    assert embedded == expected == manifest == results["payload_sha256"]
    if expected_phase["compare_current"]:
        current = {relative: sha256(root / relative) for relative in expected}
        assert current == expected

    assert expected_phase["schema_summary"] in read_text(outputs / "npm-test.log")
    assert expected_phase["worker_summary"] in read_text(outputs / "npm-test-worker.log")
    assert "error TS" not in read_text(outputs / "npm-typecheck.log")
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
        "payload_files": len(expected),
        "payload_mismatches": 0,
        "retained_files": len(retained),
        "secret_findings": findings,
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=tuple(PHASES))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve(), args.phase), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
