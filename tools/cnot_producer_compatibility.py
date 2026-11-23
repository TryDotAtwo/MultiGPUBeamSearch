"""Route producer panel checks and fingerprint prepared native inputs.

This helper uses only Python's standard library so the handoff and routing
checks can run before PyTorch or Cargo is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ".github/cnot-producer-panel-pins.json"
SCHEMA = "cnot-producer-panel-pins-v1"
HANDOFF_PATHS = {
    "integrations/cnot_scorer/handoff.py",
    "integrations/cnot_scorer/test_handoff.py",
    "integrations/cayleypy_native/tests/installed_cnot_handoff_endpoint.py",
    "tools/validate_cnot_handoff.py",
    "tools/cnot_panel_snapshot.py",
    "tools/test_cnot_panel_snapshot.py",
}
NATIVE_EXACT_PATHS = {
    "integrations/cnot_scorer/build.py",
    "integrations/cnot_scorer/bridge_panel_v2.rs",
    "integrations/cnot_scorer/panel_v2_manifest.rs",
    "integrations/cnot_scorer/Cargo.toml",
    "integrations/cnot_scorer/Cargo.lock",
    ".github/cnot-panel-python-requirements.txt",
}


def load_pins(raw: str | None) -> dict | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported panel pin schema")
    for key in ("producer_source_sha", "emitter_source_sha"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 40:
            raise ValueError(f"invalid {key}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"invalid {key}") from error
    return value


def classify_changed_paths(
    paths: list[str],
    *,
    event_name: str = "pull_request",
    previous_pins: dict | None = None,
    current_pins: dict | None = None,
) -> dict[str, bool]:
    if event_name == "workflow_dispatch":
        return {"run_handoff": True, "run_native": True}

    normalized = {path.replace("\\", "/").removeprefix("./") for path in paths}
    run_handoff = any(
        path in HANDOFF_PATHS
        or path.startswith("integrations/cnot_scorer/fixtures/")
        for path in normalized
    )
    run_native = any(
        path in NATIVE_EXACT_PATHS
        or path == ".github/workflows/cnot-producer-compatibility.yml"
        or path.startswith("integrations/cnot_scorer/src/")
        or path.startswith("integrations/cnot_scorer/tests/")
        for path in normalized
    )

    if PIN_PATH in normalized:
        old = previous_pins or {}
        new = current_pins or {}
        if previous_pins is None:
            # Initializing the pin file records the already-used producer source;
            # it only needs the small emitter/handoff verdict.
            run_handoff = True
        else:
            run_native = run_native or (
                old.get("producer_source_sha") != new.get("producer_source_sha")
            )
            changed_keys = (set(old) | set(new)) - {
                "schema", "producer_source_sha"
            }
            run_handoff = run_handoff or any(old.get(key) != new.get(key) for key in changed_keys)
            run_handoff = run_handoff or old.get("schema") != new.get("schema")

    return {"run_handoff": run_handoff, "run_native": run_native}


def read_revision_file(revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def scope_for_git_diff(event_name: str, base: str | None, head: str | None) -> tuple[dict[str, bool], dict]:
    pin_text = read_revision_file(head or "HEAD", PIN_PATH)
    pins = load_pins(pin_text) if pin_text is not None else None
    if pins is None:
        raise ValueError(f"missing or invalid current pin file: {PIN_PATH}")

    if event_name == "workflow_dispatch":
        return classify_changed_paths([], event_name=event_name), pins
    if not base or not head:
        raise ValueError("pull request base/head SHAs are required")
    diff = subprocess.run(
        ["git", "diff", "--name-only", "-z", base, head],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = [part.decode("utf-8", errors="strict") for part in diff.stdout.split(b"\0") if part]
    old_text = read_revision_file(base, PIN_PATH)
    previous = load_pins(old_text) if old_text is not None else None
    return classify_changed_paths(
        paths,
        event_name=event_name,
        previous_pins=previous,
        current_pins=pins,
    ), pins


def tree_digest(root: Path) -> str:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"native source tree is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"native source tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def native_artifact_key(*, native_inputs_sha: str, abi: str, toolchain: str,
                        environment: str, runner: str) -> str:
    values = {
        "abi": abi,
        "environment": environment,
        "native_inputs_sha256": native_inputs_sha,
        "runner": runner,
        "toolchain": toolchain,
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "cnot-panel-native-v5-" + hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", action="store_true")
    parser.add_argument("--tree-digest")
    parser.add_argument("--native-key", action="store_true")
    parser.add_argument("--native-inputs-sha")
    parser.add_argument("--abi")
    parser.add_argument("--toolchain")
    parser.add_argument("--environment")
    parser.add_argument("--runner")
    parser.add_argument("--event", default="pull_request")
    parser.add_argument("--base")
    parser.add_argument("--head")
    args = parser.parse_args(argv)

    if args.scope:
        scope, pins = scope_for_git_diff(args.event, args.base, args.head)
        for key, value in scope.items():
            print(f"{key}={str(value).lower()}")
        print(f"producer_source_sha={pins['producer_source_sha']}")
        print(f"emitter_source_sha={pins['emitter_source_sha']}")
        return 0
    if args.tree_digest:
        print(tree_digest(Path(args.tree_digest)))
        return 0
    if args.native_key:
        required = (args.native_inputs_sha, args.abi, args.toolchain, args.environment, args.runner)
        if any(value is None for value in required):
            parser.error("--native-key requires inputs SHA, ABI, toolchain, environment and runner")
        print(native_artifact_key(
            native_inputs_sha=args.native_inputs_sha,
            abi=args.abi,
            toolchain=args.toolchain,
            environment=args.environment,
            runner=args.runner,
        ))
        return 0
    parser.error("choose --scope, --tree-digest, or --native-key")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
