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
from multigpubeamsearch.rust_control import build_preparation_identity, extension, worker_executable

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

# Exercise the real PyO3 preparation identity decoder with the public auto-max token.
# This deliberately uses the fallback wire shape, which an older Rust extension must reject
# only when it cannot advertise auto-max identity support.
graph_hash = "0dfd25a031f018028e013a6cbbf83b3588c61b5b26897bb5321d117d7d3cca8d"
center = [0, 1, 2, 3]
generators = [[1, 0, 2, 3], [0, 1, 3, 2]]
preparation_fields = {
    "graph_hash": graph_hash,
    "geometry": [4, 4, 2],
    "center": center,
    "generators": generators,
    "model_artifact_hash": "b" * 64,
    "artifact_sizes": [17],
    "model_backend": "mlp",
    "devices": [0],
    "binding_scope": False,
    "beam_width": "max",
    "max_steps": 1,
    "options_repr": "NativeOptions()",
    "cuda_visible_devices": None,
    "cuda_device_order": None,
    "runtime_build": {"binary_sha256": "c" * 64},
}
fallback_preparation = json.dumps(preparation_fields, sort_keys=True)
decode_probe = runtime.SearchSession()
assert decode_probe.begin_preparation(fallback_preparation) is False
decode_probe.remember_preparation(fallback_preparation)
assert decode_probe.preparation_matches(fallback_preparation) is True
decode_probe.close()

preparation_key = build_preparation_identity(
    graph_hash=graph_hash,
    geometry=(4, 4, 2),
    center=center,
    generators=generators,
    model_artifact_hash="b" * 64,
    artifact_sizes=(17,),
    model_backend="mlp",
    devices=(0,),
    binding_scope=False,
    beam_width="max",
    max_steps=1,
    options_repr="NativeOptions()",
    cuda_visible_devices=None,
    cuda_device_order=None,
    runtime_build={"binary_sha256": "c" * 64},
)
if not isinstance(preparation_key, str):
    raise AssertionError("Rust preparation identity builder must encode beam_width='max'")
if json.loads(preparation_key)["beam_width"] != "max":
    raise AssertionError("Rust preparation identity lost the auto-max request")

# Keep the logical session width numeric while the preparation request remains AutoMax.
session_identity = {
    "graph": {"center": center, "generators": generators},
    "graph_hash": graph_hash,
    "model_artifact_hash": "b" * 64,
    "model_backend": "mlp",
    "runtime_build": {"binary_sha256": "c" * 64},
    "devices": [0],
    "beam_width": 64,
    "max_steps": 1,
    "options": {"solution_mode": "first"},
    "cuda_visible_devices": None,
    "cuda_device_order": None,
}
session_identity_json = json.dumps(session_identity, sort_keys=True)
with tempfile.TemporaryDirectory(prefix="beam-installed-wheel-resident-") as temporary:
    root = Path(temporary)
    fake_worker = root / "fake_worker.py"
    fake_worker.write_text(
        r"""import json, os, sys, time
from pathlib import Path
if '--resident-protocol-info' in sys.argv:
    print(json.dumps(dict(protocol='resident-search-v1', loaded_graph_binding=1)), flush=True)
    raise SystemExit(0)
directory = Path(os.environ['BEAM_SESSION_DIR'])
rank = int(os.environ['RANK'])
sequence = 0
while not (directory / 'stop').exists():
    request_path = directory / f'request-{sequence}.json'
    if not request_path.exists():
        time.sleep(.002)
        continue
    request = json.loads(request_path.read_text())
    output = Path(request['run_dir'])
    text = 'GLOBAL_BEAM_WIDTH_EFFECTIVE=' + os.environ['FIXTURE_BEAM'] + '\n'
    text += 'B_MICRO=64\n'
    text += 'puzzle_solved=1 puzzle_id=0 seconds=.1 solution_length=1 solution=m0\n'
    (output / f'resident-rank-{rank}.log').write_text(text)
    done = directory / f'done-{sequence}-{rank}.json'
    temp = done.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(sequence=sequence, rank=rank)))
    temp.replace(done)
    sequence += 1
""",
        encoding="utf-8",
    )
    command = [sys.executable, str(fake_worker)]
    launch = {
        "command": command,
        "protocol_command": command + ["--resident-protocol-info"],
        "environment": dict(os.environ, FIXTURE_BEAM="64"),
        "cwd": str(root),
    }
    owner = runtime.SearchSession()
    run_one = root / "run-one"
    run_one.mkdir()
    first = json.loads(owner.run(
        session_identity_json, json.dumps(launch), [1, 0, 2, 3], run_one, 10.0, "{}"
    ))
    assert first["path"] == [0], first
    assert first["metadata"]["persistent_session"]["reused"] is False
    assert owner.remember_preparation_result(preparation_key, json.dumps(first["metadata"])) is True
    assert owner.begin_preparation(preparation_key) is True
    assert owner.preparation_matches(preparation_key) is True

    run_two = root / "run-two"
    run_two.mkdir()
    second = json.loads(owner.run(
        session_identity_json, json.dumps(launch), [1, 0, 2, 3], run_two, 10.0, "{}"
    ))
    assert second["path"] == [0], second
    assert second["metadata"]["persistent_session"]["reused"] is True
    assert second["metadata"]["persistent_session"]["request_count"] == 2
    exact_preparation = dict(preparation_fields, beam_width=64)
    exact_key = build_preparation_identity(
        graph_hash=graph_hash,
        geometry=(4, 4, 2),
        center=center,
        generators=generators,
        model_artifact_hash="b" * 64,
        artifact_sizes=(17,),
        model_backend="mlp",
        devices=(0,),
        binding_scope=False,
        beam_width=64,
        max_steps=1,
        options_repr="NativeOptions()",
        cuda_visible_devices=None,
        cuda_device_order=None,
        runtime_build={"binary_sha256": "c" * 64},
    )
    assert json.loads(exact_key)["beam_width"] == exact_preparation["beam_width"]
    assert owner.begin_preparation(exact_key) is False
    assert owner.pids == []
    owner.close()

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
