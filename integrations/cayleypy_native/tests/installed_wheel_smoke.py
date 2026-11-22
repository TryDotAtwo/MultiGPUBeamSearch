"""Smoke the installed wheel without importing Python modules from the checkout."""
import importlib
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

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

device_resolver = getattr(runtime, "resolve_runtime_devices", None)
if not callable(device_resolver):
    raise AssertionError("the installed Rust runtime does not own CUDA device selection")
device_cases = (
    (2, None, None, [], [0, 1]),
    (2, None, None, [("cpu", None)], [0, 1]),
    (2, None, None, [("cuda", 1)], [1]),
    (2, None, None, [("cuda", None)], [0]),
    (2, None, 1, [], [0]),
    (2, [1], None, [], [1]),
)
for visible_count, explicit, requested, graph_devices, expected in device_cases:
    actual = device_resolver(visible_count, explicit, requested, graph_devices)
    if actual != expected:
        raise AssertionError(
            f"Rust device selection mismatch: {graph_devices!r}, {explicit!r}, "
            f"{requested!r} returned {actual!r}, expected {expected!r}"
        )
try:
    device_resolver(2, None, 3, [])
except ValueError as error:
    if "num_gpus exceeds" not in str(error):
        raise
else:
    raise AssertionError("Rust device selection silently shrank an oversized GPU request")

# Exercise the public Python preparation boundary with the real Rust extension.
# Python supplies only Torch/CayleyPy observations; making its fallback fail
# proves the installed route consumed the Rust decision.
import multigpubeamsearch.backend as device_backend
import torch
from types import SimpleNamespace

_original_system = device_backend.platform.system
_original_available = torch.cuda.is_available
_original_count = torch.cuda.device_count
_original_python_resolver = getattr(device_backend, "_resolve_runtime_devices_python", None)
from multigpubeamsearch import rust_control as device_rust_control
_original_rust_resolver = device_rust_control.resolve_runtime_devices
_rank_environment = {
    key: os.environ.pop(key, None)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID")
}
device_backend.platform.system = lambda: "Linux"
torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 2
device_backend._resolve_runtime_devices_python = lambda *args: (_ for _ in ()).throw(
    AssertionError("public device selection used the Python policy")
)
try:
    graph = SimpleNamespace(device_config=SimpleNamespace(devices=("cuda:1",)))
    if device_backend.runtime_devices(graph, NativeOptions()) != (1,):
        raise AssertionError("public preparation did not use the Rust-selected device")

    # An older extension without the resolver retains the same public behavior.
    device_rust_control.resolve_runtime_devices = lambda *args: None
    device_backend._resolve_runtime_devices_python = _original_python_resolver
    if device_backend.runtime_devices(graph, NativeOptions()) != (1,):
        raise AssertionError("legacy wheel device-selection fallback changed behavior")
finally:
    device_backend.platform.system = _original_system
    torch.cuda.is_available = _original_available
    torch.cuda.device_count = _original_count
    device_rust_control.resolve_runtime_devices = _original_rust_resolver
    if _original_python_resolver is not None:
        device_backend._resolve_runtime_devices_python = _original_python_resolver
    for key, value in _rank_environment.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

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
    "options": {"inference_backend": "auto"},
    "cuda_visible_devices": None,
    "cuda_device_order": None,
    "runtime_build": {"binary_sha256": "c" * 64},
}
typed_preparation = json.dumps(preparation_fields, sort_keys=True)
decode_probe = runtime.SearchSession()
assert decode_probe.begin_preparation(typed_preparation) is False
decode_probe.remember_preparation(typed_preparation)
assert decode_probe.preparation_matches(typed_preparation) is True
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
    options={"inference_backend": "auto"},
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
        options={"inference_backend": "auto"},
        cuda_visible_devices=None,
        cuda_device_order=None,
        runtime_build={"binary_sha256": "c" * 64},
    )
    assert json.loads(exact_key)["beam_width"] == exact_preparation["beam_width"]
    assert owner.begin_preparation(exact_key) is False
    assert owner.pids == []
    owner.close()


from multigpubeamsearch.shared_profiles import find_profile

native_profile_reader = getattr(runtime, "find_shared_profile", None)
if not callable(native_profile_reader):
    raise AssertionError("the installed Rust runtime does not expose shared profile lookup")

profile_identity = {"backend": "cutlass", "beam": 1_000_000, "world_size": 2}
with tempfile.TemporaryDirectory(prefix="beam-shared-profile-") as temporary:
    source_root = Path(temporary)
    profiles = source_root / "profiles"
    records = profiles / "records"
    records.mkdir(parents=True)
    profile_record = {
        "schema": 1,
        "compatibility": profile_identity,
        "inference": {"parent_batch": 512},
        "evidence": {"status": "verified", "source_commit": "a" * 40},
    }
    payload = json.dumps(profile_record, sort_keys=True).encode("utf-8")
    profile_sha = hashlib.sha256(payload).hexdigest()
    (records / f"{profile_sha}.json").write_bytes(payload)
    (profiles / "index.json").write_text(
        json.dumps(
            {"schema": 1, "profiles": [{"compatibility": profile_identity, "sha256": profile_sha}]},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rust_record = native_profile_reader(
        str(profiles), json.dumps(profile_identity, sort_keys=True)
    )
    if not isinstance(rust_record, str):
        raise AssertionError("Rust profile lookup missed a valid hash-addressed proposal")
    parsed_rust_record = json.loads(rust_record)
    if parsed_rust_record.get("registry_sha256") != profile_sha:
        raise AssertionError("Rust profile lookup did not bind the record digest")
    if find_profile(source_root, profile_identity) != parsed_rust_record:
        raise AssertionError("Python profile adapter did not use the Rust lookup result")
    if find_profile(source_root, dict(profile_identity, beam=2_000_000)) is not None:
        raise AssertionError("Rust profile lookup accepted a mismatched workload")

    from unittest.mock import patch
    from multigpubeamsearch import rust_control

    class OlderRustRuntime:
        pass

    with patch.object(rust_control, "extension", return_value=OlderRustRuntime()):
        if find_profile(source_root, profile_identity) != parsed_rust_record:
            raise AssertionError("the old-wheel Python profile fallback changed")

    (records / f"{profile_sha}.json").write_text("{}", encoding="utf-8")
    if native_profile_reader(str(profiles), json.dumps(profile_identity, sort_keys=True)) is not None:
        raise AssertionError("Rust profile lookup accepted a record with the wrong SHA")
    if find_profile(source_root, profile_identity) is not None:
        raise AssertionError("the Python adapter bypassed the Rust digest rejection")
print("INSTALLED_RUST_SHARED_PROFILE_LOOKUP_PASS", flush=True)

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
