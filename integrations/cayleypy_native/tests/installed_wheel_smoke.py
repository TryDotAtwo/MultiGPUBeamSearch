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


# Exercise the existing microbatch resource policy through the installed Rust
# runtime. A failing Python fallback proves the public helper consumed the plan.
microbatch_resolver = getattr(runtime, "resolve_microbatch_plan", None)
if not callable(microbatch_resolver):
    raise AssertionError("the installed Rust runtime does not own microbatch planning")

microbatch_cases = (
    (
        {"beam_width": 1, "world_size": 2, "move_count": 12,
         "output_dim": 1, "model_backend": "mlp", "row_limit": 8192},
        {"native_default_row_budget": 8192, "configured_row_budget": 2040,
         "derived_parent_batch": 170, "derived_candidates_per_slot": 2040,
         "reference_logical_shard": 1024, "reference_shard_storage": 2048,
         "adjusted_for_small_beam": True},
    ),
    (
        {"beam_width": 20000, "world_size": 1, "move_count": 4,
         "output_dim": 1, "model_backend": "ensemble", "row_limit": 1234},
        {"native_default_row_budget": 1234, "configured_row_budget": 1234,
         "derived_parent_batch": 1234, "derived_candidates_per_slot": 4936,
         "reference_logical_shard": 5120, "reference_shard_storage": 7168,
         "adjusted_for_small_beam": True},
    ),
    (
        {"beam_width": 100000, "world_size": 1, "move_count": 4,
         "output_dim": 4, "model_backend": "mlp", "row_limit": 8192},
        {"native_default_row_budget": 8192, "configured_row_budget": 8192,
         "derived_parent_batch": 8192, "derived_candidates_per_slot": 32768,
         "reference_logical_shard": 25600, "reference_shard_storage": 32768,
         "adjusted_for_small_beam": False},
    ),
)
microbatch_plans = []
for plan_request, expected_plan in microbatch_cases:
    actual_plan = json.loads(
        microbatch_resolver(json.dumps(plan_request, sort_keys=True))
    )
    if actual_plan != expected_plan:
        raise AssertionError(
            f"Rust microbatch plan mismatch: request={plan_request}, "
            f"expected={expected_plan}, actual={actual_plan}"
        )
    microbatch_plans.append(actual_plan)

from multigpubeamsearch import rust_control as microbatch_rust_control
from unittest.mock import patch as _patch

_python_plan = device_backend._resolve_microbatch_plan_python
device_backend._resolve_microbatch_plan_python = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("public microbatch planning used the Python policy")
)
try:
    for (plan_request, expected_plan), plan in zip(microbatch_cases, microbatch_plans):
        contract = SimpleNamespace(move_count=plan_request["move_count"])
        model = SimpleNamespace(
            backend=plan_request["model_backend"],
            manifest={"output_dim": plan_request["output_dim"]},
        )
        environment, metadata = device_backend.microbatch_environment(
            contract, model, plan_request["beam_width"], plan_request["world_size"],
            row_limit=plan_request["row_limit"],
        )
        expected_environment = (
            {} if plan["configured_row_budget"] == 8192
            else {"BEAM_B_MICRO": str(plan["configured_row_budget"])}
        )
        if environment != expected_environment or metadata != expected_plan:
            raise AssertionError(
                f"public microbatch preparation ignored the Rust plan: "
                f"expected={expected_environment, expected_plan}, actual={environment, metadata}"
            )
finally:
    device_backend._resolve_microbatch_plan_python = _python_plan

class OlderMicrobatchRuntime:
    pass

_rust_environment, _rust_metadata = device_backend.microbatch_environment(
    SimpleNamespace(move_count=12),
    SimpleNamespace(backend="mlp", manifest={"output_dim": 1}),
    1, 2, row_limit=8192,
)
with _patch.object(
    microbatch_rust_control, "extension", return_value=OlderMicrobatchRuntime()
):
    device_backend._resolve_microbatch_plan_python = _python_plan
    legacy_environment, legacy_metadata = device_backend.microbatch_environment(
        SimpleNamespace(move_count=12),
        SimpleNamespace(backend="mlp", manifest={"output_dim": 1}),
        1, 2, row_limit=8192,
    )
if legacy_environment != _rust_environment or legacy_metadata != _rust_metadata:
    raise AssertionError("old-wheel microbatch fallback changed behavior")
print("INSTALLED_RUST_MICROBATCH_PLAN_PASS", flush=True)

# Exercise the existing public multi-node rank layout through the installed Rust
# runtime. Exact per-rank placement and capacity must match the Python API.
cluster_plan_resolver = getattr(runtime, "resolve_cluster_plan", None)
if not callable(cluster_plan_resolver):
    raise AssertionError("the installed Rust runtime does not own cluster planning")

cluster_request = {
    "gpus_per_node": [2, 4, 1],
    "beam_width": 1000003,
    "shard_count": 16,
    "batch_alignment": 1024,
}
cluster_expected = {
    "ranks": [
        {"global_rank": 0, "node_rank": 0, "local_rank": 0},
        {"global_rank": 1, "node_rank": 0, "local_rank": 1},
        {"global_rank": 2, "node_rank": 1, "local_rank": 0},
        {"global_rank": 3, "node_rank": 1, "local_rank": 1},
        {"global_rank": 4, "node_rank": 1, "local_rank": 2},
        {"global_rank": 5, "node_rank": 1, "local_rank": 3},
        {"global_rank": 6, "node_rank": 2, "local_rank": 0},
    ],
    "requested_beam": 1000003,
    "effective_beam": 1000003,
    "alignment": 114688,
    "local_frontier_capacity": 142858,
    "shard_count": 16,
    "hardware_verified": False,
}
cluster_plan = json.loads(cluster_plan_resolver(json.dumps(cluster_request, sort_keys=True)))
if cluster_plan != cluster_expected:
    raise AssertionError(
        f"Rust cluster plan mismatch: expected={cluster_expected}, actual={cluster_plan}"
    )

from multigpubeamsearch import cluster as cluster_module
from multigpubeamsearch import rust_control as cluster_rust_control

observed_cluster_requests = []
_native_cluster_resolver = cluster_rust_control.resolve_cluster_plan


def _observe_cluster_resolver(request):
    observed_cluster_requests.append(request)
    return _native_cluster_resolver(request)


with _patch.object(cluster_rust_control, "resolve_cluster_plan", _observe_cluster_resolver), _patch.object(
    cluster_module, "_resolve_cluster_plan_python",
    side_effect=AssertionError("public cluster planning used the Python policy"),
):
    public_cluster_plan = cluster_module.plan_cluster((2, 4, 1), 1000003)
if not observed_cluster_requests or observed_cluster_requests[-1] != cluster_request:
    raise AssertionError("public plan_cluster did not delegate its exact request to Rust")
if (
    public_cluster_plan.requested_beam != cluster_plan["requested_beam"]
    or public_cluster_plan.effective_beam != cluster_plan["effective_beam"]
    or public_cluster_plan.alignment != cluster_plan["alignment"]
    or public_cluster_plan.local_frontier_capacity != cluster_plan["local_frontier_capacity"]
    or public_cluster_plan.shard_count != cluster_plan["shard_count"]
    or public_cluster_plan.hardware_verified is not False
    or [
        {"global_rank": rank.global_rank, "node_rank": rank.node_rank, "local_rank": rank.local_rank}
        for rank in public_cluster_plan.ranks
    ] != cluster_plan["ranks"]
):
    raise AssertionError("public ClusterPlan does not preserve the installed Rust plan")

with _patch.object(cluster_rust_control, "resolve_cluster_plan", return_value=None):
    legacy_cluster_plan = cluster_module.plan_cluster((2, 4, 1), 1000003)
if legacy_cluster_plan != public_cluster_plan:
    raise AssertionError("old-wheel cluster-plan fallback changed the public result")
print("INSTALLED_RUST_CLUSTER_PLAN_PASS", flush=True)

module = importlib.import_module("multigpubeamsearch._beam_runtime")
module_path = assert_installed(module.__file__, "Rust extension")
package_path = assert_installed(package.__file__, "multigpubeamsearch")
compatibility_path = assert_installed(compatibility.__file__, "cayleypy_native")
worker = assert_installed(worker_executable(), "beam-runner")
if not os.access(worker, os.X_OK):
    raise AssertionError(f"installed worker is not executable: {worker}")

# The public one-shot permutation path now uses the installed Rust rank
# supervisor even when there is only one local GPU.
from multigpubeamsearch.local_launcher import run_local_ranks
from multigpubeamsearch.backend import collect_worker_logs
from multigpubeamsearch import rust_control as _rank_control
from unittest.mock import patch as _rank_patch

if getattr(module, "CLUSTER_LOCAL_RANKS_MIN_WORLD_SIZE", None) != 1:
    raise AssertionError("the installed Rust runtime lacks single-rank supervision")

worker_code = """
import json, os, pathlib, sys
world, rank = sys.argv[-2:]
row = {"argv_world": world, "argv_rank": rank,
       "env_world": os.environ.get("WORLD_SIZE"),
       "env_rank": os.environ.get("RANK"),
       "env_local_rank": os.environ.get("LOCAL_RANK"),
       "run_id": os.environ.get("BEAM_NCCL_RUN_ID")}
pathlib.Path(sys.argv[1]).write_text(json.dumps(row, sort_keys=True))
print("GLOBAL_BEAM_WIDTH_EFFECTIVE=16", flush=True)
print("puzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 solution=m1", flush=True)
"""

with tempfile.TemporaryDirectory(prefix="beam-installed-single-rank-") as temporary:
    single_root = Path(temporary)
    single_worker_logs = single_root / "worker-logs"
    single_worker_logs.mkdir()
    observed_path = single_root / "observed.json"
    with _rank_patch(
        "multigpubeamsearch.local_launcher.subprocess.Popen",
        side_effect=AssertionError("single-rank launch fell back to Python Popen"),
    ):
        single_elapsed = run_local_ranks(
            [sys.executable, "-c", worker_code, str(observed_path)],
            world_size=1,
            cwd=single_root,
            env=dict(os.environ),
            timeout=10.0,
            log_path=single_root / "launcher.log",
            worker_logs=single_worker_logs,
        )
    if not isinstance(single_elapsed, float) or single_elapsed <= 0:
        raise AssertionError(f"invalid Rust single-rank runtime: {single_elapsed}")
    observed_rank = json.loads(observed_path.read_text())
    if (
        observed_rank["argv_world"] != "1"
        or observed_rank["argv_rank"] != "0"
        or observed_rank["env_world"] != "1"
        or observed_rank["env_rank"] != "0"
        or observed_rank["env_local_rank"] != "0"
        or not observed_rank["run_id"]
    ):
        raise AssertionError(f"Rust single-rank launch contract mismatch: {observed_rank}")
    single_log_metadata = collect_worker_logs(
        single_worker_logs, single_root / "launcher.log",
        single_root / "native.log", 1, strict=True,
    )
    if not single_log_metadata["worker_logs_complete"]:
        raise AssertionError("Rust single-rank launch did not collect its exact worker logs")
    merged_single_log = (single_root / "native.log").read_text()
    if "GLOBAL_BEAM_WIDTH_EFFECTIVE=16" not in merged_single_log or "puzzle_solved=1" not in merged_single_log:
        raise AssertionError("Rust single-rank worker output was not merged")
print("INSTALLED_RUST_SINGLE_RANK_SUPERVISOR_PASS", flush=True)

# Version-1 wheels predate the single-rank capability and retain the Python fallback.
legacy_rank_root = Path(tempfile.mkdtemp(prefix="beam-installed-single-rank-legacy-"))
try:
    legacy_logs = legacy_rank_root / "worker-logs"
    legacy_logs.mkdir()
    legacy_observed = legacy_rank_root / "observed.json"
    class LegacyLocalRanksRuntime:
        CLUSTER_LOCAL_RANKS_VERSION = 1

        @staticmethod
        def run_local_search_ranks(*_args):
            raise AssertionError("version-1 single-rank call must stay on the Python compatibility path")

    with _rank_patch.object(_rank_control, "extension", return_value=LegacyLocalRanksRuntime()):
        run_local_ranks(
            [sys.executable, "-c", worker_code, str(legacy_observed)],
            world_size=1,
            cwd=legacy_rank_root,
            env=dict(os.environ),
            timeout=10.0,
            log_path=legacy_rank_root / "launcher.log",
            worker_logs=legacy_logs,
        )
    legacy_rank = json.loads(legacy_observed.read_text())
    if legacy_rank["argv_world"] != "1" or legacy_rank["env_rank"] != "0":
        raise AssertionError(f"version-1 compatibility launch mismatch: {legacy_rank}")
    legacy_meta = collect_worker_logs(
        legacy_logs, legacy_rank_root / "launcher.log",
        legacy_rank_root / "native.log", 1, strict=True,
    )
    if not legacy_meta["worker_logs_complete"]:
        raise AssertionError("version-1 Python compatibility logs were incomplete")
finally:
    import shutil as _rank_shutil
    _rank_shutil.rmtree(legacy_rank_root)
print("INSTALLED_OLD_WHEEL_SINGLE_RANK_FALLBACK_PASS", flush=True)

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

# Exercise one reusable, two-rank planning session through the installed Rust owner.
from multigpubeamsearch import plan_session as plan_session_module
from multigpubeamsearch import rust_control as plan_session_rust_control
from multigpubeamsearch.plan_session import NativePlanSession
from unittest.mock import patch as _patch
import time as _time

with tempfile.TemporaryDirectory(prefix="beam-plan-session-") as temporary:
    plan_root = Path(temporary)
    fake_planner = plan_root / "fake-planner"
    fake_planner.write_text(
        """#!/usr/bin/env python3
import json
import sys
world = int(sys.argv[-2])
rank = int(sys.argv[-1])
print(json.dumps({"ready": True, "rank": rank, "component_protocol": "none"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    environment = request["environment"]
    plan = {
        "GLOBAL_BEAM_WIDTH_EFFECTIVE": request["beam"],
        "BEAM_WIDTH_ALIGNMENT": 1,
        "SHARD_COUNT": int(environment["BEAM_SHARD_COUNT"]),
        "B_MICRO": int(environment["BEAM_B_MICRO"]),
        "WORLD_SIZE": world,
        "STREAM4_BATCH_ALIGNMENT": 1024,
        "LOCAL_RANK": rank,
        "estimated_required_device_bytes": 1,
        "gpu_budget_bytes": 2,
    }
    print(json.dumps({"admitted": True, "plan": plan}), flush=True)
""",
        encoding="utf-8",
    )
    fake_planner.chmod(0o755)
    planner_environment = {
        "PATH": os.environ["PATH"],
        "BEAM_B_MICRO": "64",
        "BEAM_ENSEMBLE_INFERENCE_MICRO": "64",
    }
    planner_overrides = {
        "BEAM_B_MICRO": "64",
        "BEAM_SHARD_COUNT": "1",
        "BEAM_STREAM3_RING_SLOTS": "2",
        "BEAM_STREAM4_BATCH_CANDIDATES": "1024",
        "BEAM_STREAM4_ACTIVE_SORT_SLOTS": "1",
    }

    with _patch.object(
        plan_session_module,
        "InferenceSession",
        side_effect=AssertionError("installed planner session used Python-owned rank processes"),
    ):
        rust_owner = NativePlanSession(
            fake_planner,
            planner_environment,
            2,
            plan_root / "rust-owner",
            deadline=_time.monotonic() + 30.0,
        )
        try:
            rust_pids = tuple(rust_owner.pids)
            first_plans = rust_owner.admit(128, planner_overrides)
            second_plans = rust_owner.admit(256, planner_overrides)
            if len(rust_pids) != 2 or len(set(rust_pids)) != 2:
                raise AssertionError(f"Rust planner did not own two distinct ranks: {rust_pids}")
            if tuple(rust_owner.pids) != rust_pids:
                raise AssertionError("a repeated admission replaced the Rust-owned rank session")
            if [row["LOCAL_RANK"] for row in first_plans] != [0, 1]:
                raise AssertionError(f"Rust planner lost rank identity: {first_plans}")
            if [row["GLOBAL_BEAM_WIDTH_EFFECTIVE"] for row in first_plans] != [128, 128]:
                raise AssertionError(f"Rust planner changed the first requested width: {first_plans}")
            if [row["GLOBAL_BEAM_WIDTH_EFFECTIVE"] for row in second_plans] != [256, 256]:
                raise AssertionError(f"Rust planner changed the repeated requested width: {second_plans}")
        finally:
            rust_owner.close()

    class OlderPlanningRuntime:
        ABI_VERSION = 1
        InferenceCalibration = object

    with _patch.object(plan_session_rust_control, "extension", return_value=OlderPlanningRuntime()):
        legacy_owner = NativePlanSession(
            fake_planner,
            planner_environment,
            2,
            plan_root / "legacy-owner",
            deadline=_time.monotonic() + 30.0,
        )
        try:
            legacy_plans = legacy_owner.admit(128, planner_overrides)
            if legacy_plans != first_plans:
                raise AssertionError("older-wheel planning fallback changed the rank plans")
        finally:
            legacy_owner.close()

print("INSTALLED_RUST_PLAN_SESSION_OWNER_PASS", flush=True)

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
