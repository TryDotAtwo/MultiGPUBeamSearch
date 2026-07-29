from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent / "kernel"
FILES = (
    "configs/cayleypy_results_schema_v1.json",
    "configs/cayleypy_results_v1_golden.json",
    "services/cayleypy-results-ingest/package.json",
    "services/cayleypy-results-ingest/package-lock.json",
    "services/cayleypy-results-ingest/tsconfig.json",
    "services/cayleypy-results-ingest/wrangler.jsonc",
    "services/cayleypy-results-ingest/vitest.config.ts",
    "services/cayleypy-results-ingest/vitest.schema.config.ts",
    "services/cayleypy-results-ingest/DEPLOY_STAGING_RUNBOOK.md",
    "services/cayleypy-results-ingest/OPERATIONS.md",
    "services/cayleypy-results-ingest/scripts/invoke-staging-deployment.ps1",
    "services/cayleypy-results-ingest/src/mode.ts",
    "services/cayleypy-results-ingest/src/github-app.ts",
    "services/cayleypy-results-ingest/src/github-writer.ts",
    "services/cayleypy-results-ingest/test/github-app.test.ts",
    "services/cayleypy-results-ingest/test/github-writer.test.ts",
    "services/cayleypy-results-ingest/migrations/0001_initial.sql",
    "services/cayleypy-results-ingest/migrations/0002_ingest_rate_limits.sql",
    "services/cayleypy-results-ingest/migrations/0003_remove_legacy_status_ip_limits.sql",
    "services/cayleypy-results-ingest/src/schema.ts",
    "services/cayleypy-results-ingest/src/ids.ts",
    "services/cayleypy-results-ingest/src/db.ts",
    "services/cayleypy-results-ingest/src/storage.ts",
    "services/cayleypy-results-ingest/src/worker.ts",
    "services/cayleypy-results-ingest/src/replay.ts",
    "services/cayleypy-results-ingest/src/consumer.ts",
    "services/cayleypy-results-ingest/src/operator-replay.ts",
    "services/cayleypy-results-ingest/test/schema.test.ts",
    "services/cayleypy-results-ingest/test/apply-migrations.ts",
    "services/cayleypy-results-ingest/test/receipt.test.ts",
    "services/cayleypy-results-ingest/test/worker.test.ts",
    "services/cayleypy-results-ingest/test/consumer.test.ts",
    "services/cayleypy-results-ingest/test/replay.test.ts",
    "services/cayleypy-results-ingest/test/node-crypto.d.ts",
    "services/cayleypy-results-ingest/test/wrangler-config.test.ts",
    "services/cayleypy-results-ingest/load/k6-100-publishers.js",
    "services/cayleypy-results-ingest/test/deployment-runbook.test.ts",
    "services/cayleypy-results-ingest/test/deployment-migrations-resolver.test.ps1",
    "services/cayleypy-results-ingest/test/migration-upgrade.test.ts",
    "services/cayleypy-results-ingest/test/load-recovery-gate.test.ts",
    "services/cayleypy-results-ingest/test/recovery-audit.ts",
)


def payload() -> tuple[str, dict[str, str]]:
    archive_bytes = io.BytesIO()
    checksums: dict[str, str] = {}
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            data = (ROOT / relative).read_bytes()
            checksums[relative] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return base64.b64encode(archive_bytes.getvalue()).decode("ascii"), checksums


def notebook_source(encoded: str, checksums: dict[str, str]) -> str:
    return f'''from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")
ROOT = WORKING / "cayleypy-results-ingest-gate"
PACKAGE = ROOT / "services" / "cayleypy-results-ingest"
NPM_CACHE = Path("/tmp/cayleypy-results-ingest-npm-cache")
NODE_ROOT = Path("/tmp/cayleypy-results-ingest-node")
NODE_VERSION = "v22.23.1"
NODE_ARCHIVE_NAME = "node-" + NODE_VERSION + "-linux-x64.tar.xz"
NODE_BASE_URL = "https://nodejs.org/dist/" + NODE_VERSION
PAYLOAD_B64 = {encoded!r}
EXPECTED_SHA256 = {checksums!r}

if ROOT.exists():
    shutil.rmtree(ROOT)
if NPM_CACHE.exists():
    shutil.rmtree(NPM_CACHE)
if NODE_ROOT.exists():
    shutil.rmtree(NODE_ROOT)
ROOT.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64))) as archive:
    archive.extractall(ROOT)

shasums = urllib.request.urlopen(
    NODE_BASE_URL + "/SHASUMS256.txt", timeout=60
).read().decode("utf-8")
expected_node_sha = next(
    line.split()[0]
    for line in shasums.splitlines()
    if line.endswith("  " + NODE_ARCHIVE_NAME)
)
node_archive = urllib.request.urlopen(
    NODE_BASE_URL + "/" + NODE_ARCHIVE_NAME, timeout=180
).read()
observed_node_sha = hashlib.sha256(node_archive).hexdigest()
if observed_node_sha != expected_node_sha:
    raise RuntimeError("Node archive checksum mismatch")
NODE_ROOT.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(node_archive), mode="r:xz") as archive:
    archive.extractall(NODE_ROOT)
NODE_BIN = NODE_ROOT / NODE_ARCHIVE_NAME.removesuffix(".tar.xz") / "bin"
os.environ["PATH"] = str(NODE_BIN) + os.pathsep + os.environ["PATH"]
(WORKING / "node-bootstrap.json").write_text(
    json.dumps(
        {{"version": NODE_VERSION, "archive": NODE_ARCHIVE_NAME, "sha256": observed_node_sha}},
        indent=2,
        sort_keys=True,
    ) + "\\n",
    encoding="utf-8",
)

observed = {{}}
for relative in EXPECTED_SHA256:
    observed[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
if observed != EXPECTED_SHA256:
    raise RuntimeError("embedded payload checksum mismatch")
(WORKING / "payload-sha256.json").write_text(
    json.dumps(observed, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)

combined_log = WORKING / "npm-gate.log"
combined_log.write_text("", encoding="utf-8")
results = {{"payload_sha256": observed, "commands": []}}
command_outputs = {{}}
command_stdout = {{}}
RUN_ENV = {{
    **os.environ,
    "NPM_CONFIG_CACHE": str(NPM_CACHE),
    "NPM_CONFIG_REGISTRY": "https://registry.npmjs.org/",
    "NPM_CONFIG_FETCH_RETRIES": "3",
    "NPM_CONFIG_UPDATE_NOTIFIER": "false",
    "NO_COLOR": "1",
    "FORCE_COLOR": "0",
}}


def run(label: str, argv: list[str]) -> int:
    completed = subprocess.run(
        argv,
        cwd=PACKAGE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=RUN_ENV,
    )
    command_stdout[label] = completed.stdout
    output = completed.stdout + completed.stderr
    command_outputs[label] = output
    section = f"\\n===== {{label}} (exit={{completed.returncode}}) =====\\n{{output}}"
    print(section, flush=True)
    with combined_log.open("a", encoding="utf-8") as handle:
        handle.write(section)
    (WORKING / f"{{label}}.log").write_text(output, encoding="utf-8")
    results["commands"].append(
        {{"label": label, "argv": argv, "exit_code": completed.returncode}}
    )
    return completed.returncode

REGISTRY_QUERIES = {{
    "npm-view-vitest-pool": [
        "npm", "view", "@cloudflare/vitest-pool-workers@0.19.0",
        "version", "dist-tags", "dependencies", "peerDependencies", "engines", "dist.integrity", "--json",
    ],
    "npm-view-wrangler": [
        "npm", "view", "wrangler@4.115.0",
        "version", "dist-tags", "dependencies", "engines", "dist.integrity", "--json",
    ],
    "npm-view-workers-types": [
        "npm", "view", "@cloudflare/workers-types@5.20260729.1",
        "version", "dist-tags", "dependencies", "engines", "dist.integrity", "--json",
    ],
    "npm-view-vitest": [
        "npm", "view", "vitest@4.1.10",
        "version", "dist-tags", "dependencies", "peerDependencies", "engines", "dist.integrity", "--json",
    ],
    "npm-view-workerd": [
        "npm", "view", "workerd@1.20260729.1",
        "version", "dist-tags", "dependencies", "engines", "dist.integrity", "--json",
    ],
}}
REGISTRY_EXPECTED = {{
    "npm-view-vitest-pool": "0.19.0",
    "npm-view-wrangler": "4.115.0",
    "npm-view-workers-types": "5.20260729.1",
    "npm-view-vitest": "4.1.10",
    "npm-view-workerd": "1.20260729.1",
}}

run("node-version", ["node", "--version"])
run("npm-version", ["npm", "--version"])
for label, argv in REGISTRY_QUERIES.items():
    run(label, argv)
run("npm-install-package-lock-only", ["npm", "install", "--package-lock-only", "--no-audit", "--no-fund"])
run("npm-ci", ["npm", "ci", "--no-audit", "--no-fund"])
run(
    "npm-ls-critical",
    [
        "npm", "ls", "@cloudflare/vitest-pool-workers", "@cloudflare/workers-types",
        "vitest", "wrangler", "miniflare", "workerd", "--json",
    ],
)
run("npm-test", ["npm", "test"])
run("npm-test-worker", ["npm", "exec", "--", "vitest", "run", "--config", "vitest.config.ts"])
run("npm-typecheck", ["npm", "run", "typecheck"])

registry_metadata = {{}}
registry_parse_errors = []
for label, expected_version in REGISTRY_EXPECTED.items():
    try:
        metadata = json.loads(command_stdout[label])
    except (KeyError, json.JSONDecodeError) as exc:
        registry_parse_errors.append({{"label": label, "error": type(exc).__name__}})
        continue
    registry_metadata[label] = metadata
registry_exact = not registry_parse_errors and all(
    registry_metadata[label].get("version") == expected
    for label, expected in REGISTRY_EXPECTED.items()
)
(WORKING / "registry-metadata.json").write_text(
    json.dumps(
        {{
            "expected_versions": REGISTRY_EXPECTED,
            "exact": registry_exact,
            "parse_errors": registry_parse_errors,
            "registry": registry_metadata,
        }},
        indent=2,
        sort_keys=True,
    ) + "\\n",
    encoding="utf-8",
)

lockfile = PACKAGE / "package-lock.json"
lock = json.loads(lockfile.read_text(encoding="utf-8")) if lockfile.is_file() else {{"packages": {{}}}}
lock_packages = lock.get("packages", {{}})
workerd_paths = sorted(
    path for path in lock_packages if path == "node_modules/workerd" or path.endswith("/node_modules/workerd")
)
workerd_versions = sorted({{
    lock_packages[path].get("version") for path in workerd_paths if lock_packages[path].get("version")
}})

def installed_version(relative: str) -> str | None:
    path = PACKAGE / "node_modules" / relative / "package.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["version"]

expected_stack = {{
    "@cloudflare/vitest-pool-workers": "0.19.0",
    "@cloudflare/workers-types": "5.20260729.1",
    "vitest": "4.1.10",
    "wrangler": "4.115.0",
    "miniflare": "4.20260722.1",
    "workerd": "1.20260729.1",
}}
resolved_stack = {{name: installed_version(name) for name in expected_stack}}
resolved_stack_exact = (
    resolved_stack == expected_stack
    and workerd_versions == ["1.20260729.1"]
    and bool(workerd_paths)
)
resolved_report = {{
    "expected": expected_stack,
    "resolved": resolved_stack,
    "exact": resolved_stack_exact,
    "lockfile_workerd_paths": workerd_paths,
    "lockfile_workerd_versions": workerd_versions,
}}
(WORKING / "resolved-stack.json").write_text(
    json.dumps(resolved_report, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)

warning_markers = (
    "newer than the latest supported date",
    "falling back to",
    "unsupported compatibility date",
    "does not support compatibility date",
)
warning_hits = []
for label, output in command_outputs.items():
    for line_number, line in enumerate(output.splitlines(), start=1):
        lowered = line.lower()
        if any(marker in lowered for marker in warning_markers):
            warning_hits.append({{"label": label, "line": line_number, "text": line}})
warning_report = {{
    "compatibility_date": "2026-07-28",
    "markers": list(warning_markers),
    "hits": warning_hits,
    "passed": not warning_hits,
}}
(WORKING / "compatibility-warning-scan.json").write_text(
    json.dumps(warning_report, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)

post_install_sha256 = {{
    relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    for relative in EXPECTED_SHA256
}}
post_install_mismatches = {{
    relative: {{"expected": EXPECTED_SHA256[relative], "observed": observed_sha}}
    for relative, observed_sha in post_install_sha256.items()
    if observed_sha != EXPECTED_SHA256[relative]
}}
(WORKING / "post-install-payload-sha256.json").write_text(
    json.dumps(post_install_sha256, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)

prior_commands_passed = all(item["exit_code"] == 0 for item in results["commands"])
gate_assertions = {{
    "prior_commands_passed": prior_commands_passed,
    "registry_versions_exact": registry_exact,
    "resolved_stack_exact": resolved_stack_exact,
    "workerd_override_exact": workerd_versions == ["1.20260729.1"],
    "compatibility_warning_scan_clean": not warning_hits,
    "post_install_payload_exact": not post_install_mismatches,
    "lockfile_present": lockfile.is_file(),
}}
gate_exit = 0 if all(gate_assertions.values()) else 1
gate_output = json.dumps(gate_assertions, indent=2, sort_keys=True) + "\\n"
command_outputs["gate-assertions"] = gate_output
(WORKING / "gate-assertions.log").write_text(gate_output, encoding="utf-8")
with combined_log.open("a", encoding="utf-8") as handle:
    handle.write(f"\\n===== gate-assertions (exit={{gate_exit}}) =====\\n{{gate_output}}")
results["commands"].append(
    {{"label": "gate-assertions", "argv": [], "exit_code": gate_exit}}
)

if lockfile.is_file():
    shutil.copy2(lockfile, WORKING / "package-lock.json")
results.update(
    {{
        "lockfile_present": lockfile.is_file(),
        "registry_versions_exact": registry_exact,
        "resolved_stack": resolved_report,
        "compatibility_warning_scan": warning_report,
        "post_install_payload_sha256": post_install_sha256,
        "post_install_payload_mismatches": post_install_mismatches,
        "gate_assertions": gate_assertions,
        "all_commands_passed": all(item["exit_code"] == 0 for item in results["commands"]),
    }}
)
(WORKING / "npm-gate-results.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
shutil.rmtree(ROOT)
shutil.rmtree(NPM_CACHE)
shutil.rmtree(NODE_ROOT)
print(json.dumps(results, indent=2, sort_keys=True), flush=True)
'''


def main() -> None:
    encoded, checksums = payload()
    source = notebook_source(encoded, checksums)
    ast.parse(source)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Private CayleyPy Results Ingest npm gate\n",
                    "CPU-only exact-stack Task 8 load/recovery static and unit verification. No deployment or external load.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": "trydotatwo/cayleypy-results-ingest-npm-gate",
        "title": "CayleyPy Results Ingest npm Gate",
        "code_file": "cayleypy-results-ingest-npm-gate.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / metadata["code_file"]).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (PACKAGE / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    parsed = json.loads((PACKAGE / metadata["code_file"]).read_text(encoding="utf-8"))
    for cell in parsed["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    parsed_metadata = json.loads((PACKAGE / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert parsed_metadata["is_private"] is True
    assert parsed_metadata["enable_gpu"] is False
    assert parsed_metadata["enable_internet"] is True
    print(json.dumps({"files": checksums, "metadata": parsed_metadata}, indent=2))


if __name__ == "__main__":
    main()
