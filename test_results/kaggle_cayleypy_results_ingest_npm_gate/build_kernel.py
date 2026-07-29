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
    "services/cayleypy-results-ingest/package.json",
    "services/cayleypy-results-ingest/package-lock.json",
    "services/cayleypy-results-ingest/tsconfig.json",
    "services/cayleypy-results-ingest/wrangler.jsonc",
    "services/cayleypy-results-ingest/vitest.config.ts",
    "services/cayleypy-results-ingest/vitest.schema.config.ts",
    "services/cayleypy-results-ingest/migrations/0001_initial.sql",
    "services/cayleypy-results-ingest/migrations/0002_ingest_rate_limits.sql",
    "services/cayleypy-results-ingest/src/schema.ts",
    "services/cayleypy-results-ingest/src/ids.ts",
    "services/cayleypy-results-ingest/src/db.ts",
    "services/cayleypy-results-ingest/src/storage.ts",
    "services/cayleypy-results-ingest/src/worker.ts",
    "services/cayleypy-results-ingest/test/schema.test.ts",
    "services/cayleypy-results-ingest/test/apply-migrations.ts",
    "services/cayleypy-results-ingest/test/receipt.test.ts",
    "services/cayleypy-results-ingest/test/worker.test.ts",
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
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")
ROOT = WORKING / "cayleypy-results-ingest-gate"
PACKAGE = ROOT / "services" / "cayleypy-results-ingest"
NPM_CACHE = Path("/tmp/cayleypy-results-ingest-npm-cache")
PAYLOAD_B64 = {encoded!r}
EXPECTED_SHA256 = {checksums!r}

if ROOT.exists():
    shutil.rmtree(ROOT)
if NPM_CACHE.exists():
    shutil.rmtree(NPM_CACHE)
ROOT.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64))) as archive:
    archive.extractall(ROOT)

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

def run(label: str, argv: list[str]) -> int:
    completed = subprocess.run(
        argv,
        cwd=PACKAGE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={{**os.environ, "NPM_CONFIG_CACHE": str(NPM_CACHE)}},
    )
    output = completed.stdout
    section = f"\\n===== {{label}} (exit={{completed.returncode}}) =====\\n{{output}}"
    print(section, flush=True)
    with combined_log.open("a", encoding="utf-8") as handle:
        handle.write(section)
    (WORKING / f"{{label}}.log").write_text(output, encoding="utf-8")
    results["commands"].append(
        {{"label": label, "argv": argv, "exit_code": completed.returncode}}
    )
    return completed.returncode

run("node-version", ["node", "--version"])
run("npm-version", ["npm", "--version"])
run("npm-install-package-lock-only", ["npm", "install", "--package-lock-only", "--no-audit", "--no-fund"])
run("npm-ci", ["npm", "ci", "--no-audit", "--no-fund"])
run("npm-test", ["npm", "test"])
run("npm-test-worker", ["npm", "exec", "--", "vitest", "run", "--config", "vitest.config.ts"])
run("npm-typecheck", ["npm", "run", "typecheck"])

lockfile = PACKAGE / "package-lock.json"
if lockfile.is_file():
    shutil.copy2(lockfile, WORKING / "package-lock.json")
results["lockfile_present"] = lockfile.is_file()
results["all_commands_passed"] = all(item["exit_code"] == 0 for item in results["commands"])
(WORKING / "npm-gate-results.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
shutil.rmtree(ROOT)
shutil.rmtree(NPM_CACHE)
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
                    "CPU-only dependency, test, and TypeScript verification. No deployment.\n",
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
