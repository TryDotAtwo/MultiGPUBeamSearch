"""Smoke the installed wheel without importing Python modules from the checkout."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import cayleypy_native as compatibility
import multigpubeamsearch as package
from multigpubeamsearch import NativeEnsemble, NativeModel, NativeOptions, NativeSession, beam_search
from multigpubeamsearch.rust_control import extension, worker_executable

workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()


def assert_installed(path, label):
    resolved = Path(path).resolve()
    if resolved == workspace or workspace in resolved.parents:
        raise AssertionError(f"{label} was imported from the checkout: {resolved}")
    return resolved


for name, value in (
    ("beam_search", beam_search),
    ("NativeOptions", NativeOptions),
    ("NativeModel", NativeModel),
    ("NativeEnsemble", NativeEnsemble),
    ("NativeSession", NativeSession),
):
    if getattr(compatibility, name) is not value:
        raise AssertionError(f"cayleypy_native.{name} is not the compatibility export")

runtime = extension()
if runtime.ABI_VERSION != 1:
    raise AssertionError(f"unexpected Rust ABI version: {runtime.ABI_VERSION}")
if not hasattr(runtime, "TQPScorer"):
    raise AssertionError("the installed Rust module does not expose TQPScorer")
if not hasattr(runtime, "InferenceCalibration"):
    raise AssertionError("the installed Rust module does not expose InferenceCalibration")

module = importlib.import_module("multigpubeamsearch._beam_runtime")
module_path = assert_installed(module.__file__, "Rust extension")
package_path = assert_installed(package.__file__, "multigpubeamsearch")
compatibility_path = assert_installed(compatibility.__file__, "cayleypy_native")
worker = assert_installed(worker_executable(), "beam-runner")
if not os.access(worker, os.X_OK):
    raise AssertionError(f"installed worker is not executable: {worker}")

response = subprocess.run(
    [str(worker), "--resident-protocol-info"],
    check=True,
    capture_output=True,
    text=True,
)
protocol = json.loads(response.stdout)
if protocol.get("protocol") != "resident-search-v1":
    raise AssertionError(f"unexpected resident worker protocol: {protocol}")

summary = {
    "distribution_version": importlib.metadata.version("multigpubeamsearch"),
    "python": sys.version.split()[0],
    "rust_abi": runtime.ABI_VERSION,
    "cnot_tqp_scorer": runtime.TQPScorer.__name__,
    "resident_protocol": protocol["protocol"],
    "package": str(package_path),
    "compatibility_package": str(compatibility_path),
    "extension": str(module_path),
    "worker": str(worker),
}
print(json.dumps(summary, sort_keys=True))
