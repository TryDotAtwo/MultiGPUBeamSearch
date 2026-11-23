"""CPU-only policy checks against the installed Rust/PyO3 runtime."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from multigpubeamsearch.errors import NativeBackendError, NativeUnavailable
from multigpubeamsearch.rust_control import extension

runtime = extension()
resolve_native = getattr(runtime, "resolve_inference_backend", None)
if not callable(resolve_native):
    raise AssertionError("installed Rust runtime does not expose inference backend resolution")
resolve_plan_native = getattr(runtime, "resolve_inference_plan", None)
if not callable(resolve_plan_native):
    raise AssertionError("installed Rust runtime does not expose inference plan resolution")


def request(requested, model_backend, dtype, output_dim, devices):
    return {
        "requested": requested,
        "model_backend": model_backend,
        "dtype": dtype,
        "output_dim": output_dim,
        "devices": [{"sm": sm, "name": name} for sm, name in devices],
    }


def resolve(value):
    return resolve_native(json.dumps(value, sort_keys=True))


def check(value, expected):
    actual = resolve(value)
    if actual != expected:
        raise AssertionError(f"backend selection mismatch: expected={expected}, actual={actual}, request={value}")


def resolve_plan(value):
    return json.loads(resolve_plan_native(json.dumps(value, sort_keys=True)))


def reject(value, expected_text):
    try:
        resolve(value)
    except (RuntimeError, ValueError) as error:
        if expected_text not in str(error):
            raise AssertionError(f"wrong backend-policy error: {error}") from error
    else:
        raise AssertionError(f"backend policy accepted an unsupported request: {value}")


check(request("auto", "mlp", "fp16", 8, [(86, "NVIDIA GeForce RTX 3060")]), "cutlass")
check(request("auto", "mlp", "fp16", 8, [(75, "Tesla T4")]), "libtorch")
check(request("auto", "mlp", "fp16", 7, [(75, "Tesla T4")]), "libtorch")
check(request("auto", "mlp", "fp16", 8, [(75, "NVIDIA RTX A6000"), (86, "NVIDIA GeForce RTX 3060")]), "cutlass")
check(request("auto", "mlp", "fp16", 8, [(75, "Tesla T4"), (86, "NVIDIA GeForce RTX 3060")]), "libtorch")
check(request("auto", "ensemble", "fp16", 64, [(75, "NVIDIA RTX A6000")]), "libtorch")
check(request("auto", "ensemble", "fp16", 64, [(80, "NVIDIA A100")]), "cutlass")
check(request("libtorch", "mlp", "fp16", 7, [(86, "NVIDIA A100")]), "libtorch")
check(request("cutlass", "mlp", "fp16", 1, [(75, "Tesla T4")]), "cutlass")
reject(request("cutlass", "mlp", "fp16", 7, [(86, "NVIDIA A100")]),
       "native MLP Q-head output_dim must be a multiple of 8")
reject(request("auto", "mlp", "fp16", 7, [(86, "NVIDIA A100")]),
       "native MLP Q-head output_dim must be a multiple of 8")
reject(request("auto", "mlp", "bf16", 8, [(79, "NVIDIA test GPU")]),
       "native BF16 Stream1 requires SM80+")
reject(request("auto", "mlp", "fp16", 8, [(74, "NVIDIA test GPU")]),
       "this native FP16 MLP backend uses SM75+")
reject(request("cutlass", "ensemble", "fp16", 64, [(75, "NVIDIA RTX A6000")]),
       "ensemble CUTLASS requires SM80+")
reject(request("auto", "mlp", "fp16", 8, []),
       "inference backend selection requires at least one GPU")

plan = resolve_plan(request("auto", "mlp", "fp16", 8, [(86, "NVIDIA GeForce RTX 3060")]))
if plan != {"backend": "cutlass", "policy": "cutlass-default-v1", "inference_row_limit": 8192}:
    raise AssertionError(f"Rust inference plan did not preserve the CUTLASS profile: {plan}")


from multigpubeamsearch import backend, build, rust_control
from multigpubeamsearch.options import NativeOptions
import torch

unsupported = request("auto", "mlp", "bf16", 8, [(79, "NVIDIA test GPU")])
try:
    backend._resolve_inference_backend(unsupported)
except NativeUnavailable as error:
    if "native BF16 Stream1 requires SM80+" not in str(error):
        raise AssertionError(f"Rust policy error did not reach the public capability boundary: {error}") from error
else:
    raise AssertionError("unsupported backend capabilities did not raise NativeUnavailable")


class FakeContract:
    move_count = 2


class LegacyRuntime:
    ABI_VERSION = 1
    InferenceCalibration = object()


with patch.object(rust_control, "extension", return_value=LegacyRuntime()):
    legacy_route = backend._resolve_inference_backend(
        request("auto", "mlp", "fp16", 8, [(75, "Tesla T4")])
    )
if legacy_route != "libtorch":
    raise AssertionError("the compatibility resolver changed behavior for an older wheel")


with patch.object(rust_control, "extension", return_value=LegacyRuntime()):
    legacy_plan = backend._resolve_inference_plan(
        request("auto", "mlp", "fp16", 8, [(75, "Tesla T4")])
    )
if legacy_plan != {
    "backend": "libtorch", "policy": "libtorch-conservative-v1",
    "inference_row_limit": 2048,
}:
    raise AssertionError(f"the old-wheel policy profile changed: {legacy_plan}")


class RouteOnlyRuntime:
    ABI_VERSION = 1
    InferenceCalibration = object()

    @staticmethod
    def resolve_inference_backend(_payload):
        return "libtorch"


with patch.object(rust_control, "extension", return_value=RouteOnlyRuntime()):
    route_only_plan = backend._resolve_inference_plan(
        request("auto", "mlp", "fp16", 8, [(86, "NVIDIA A100")])
    )
if route_only_plan != {
    "backend": "libtorch", "policy": "libtorch-conservative-v1",
    "inference_row_limit": 2048,
}:
    raise AssertionError(f"the route-only wheel fallback changed: {route_only_plan}")


prepared_request = request(
    "auto", "mlp", "fp16", 8, [(80, "NVIDIA A100")]
)
expected_prepared_plan = resolve_plan(prepared_request)
selected = []


def fake_ensure_runner(_contract, _model, options, _architectures, run_dir, **_kwargs):
    selected.append(options.inference_backend)
    return Path(run_dir) / "fixture-runner", {"binary_sha256": "d" * 64}


with tempfile.TemporaryDirectory(prefix="beam-inference-policy-") as temporary:
    root = Path(temporary)
    model = SimpleNamespace(
        backend="mlp",
        manifest={"dtype": "fp16", "output_dim": 8},
    )
    options = NativeOptions(cache_dir=root / "cache", inference_backend="auto")
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100"),
        patch.object(build, "ensure_runner", side_effect=fake_ensure_runner),
    ):
        prepared = backend.prepare_runtime(
            FakeContract(), model, options, root / "run", (0,)
        )
    if selected != [expected_prepared_plan["backend"]]:
        raise AssertionError(
            f"prepare_runtime did not pass the real Rust backend to runner setup: "
            f"expected={expected_prepared_plan}, selected={selected}"
        )
    if prepared.profile.get("inference_backend") != expected_prepared_plan["backend"]:
        raise AssertionError(
            f"prepared runtime backend differs from the installed Rust plan: "
            f"expected={expected_prepared_plan}, profile={prepared.profile}"
        )
    if prepared.profile.get("policy") != expected_prepared_plan["policy"]:
        raise AssertionError(
            f"prepared runtime policy differs from the installed Rust plan: "
            f"expected={expected_prepared_plan}, profile={prepared.profile}"
        )
    if prepared.profile.get("inference_row_limit") != expected_prepared_plan["inference_row_limit"]:
        raise AssertionError(
            f"prepared runtime row limit differs from the installed Rust plan: "
            f"expected={expected_prepared_plan}, profile={prepared.profile}"
        )

phase_plan_native = getattr(runtime, "resolve_search_phase_plan", None)
if not callable(phase_plan_native) or getattr(runtime, "SEARCH_PHASE_PLAN_VERSION", None) != 1:
    raise AssertionError("installed Rust runtime does not expose the versioned search phase plan")

phase_cases = [
    (
        {"autotune": True, "model_backend": "mlp",
         "calibration_binary_name": "stream1_native_mlp_benchmark",
         "maximum_requested": False, "discover_capacity": False},
        ["inference_calibration", "pipeline_calibration", "execute"],
    ),
    (
        {"autotune": True, "model_backend": "mlp",
         "calibration_binary_name": "stream1_cutlass_mlp_benchmark",
         "maximum_requested": False, "discover_capacity": False},
        ["execute"],
    ),
    (
        {"autotune": True, "model_backend": "ensemble",
         "calibration_binary_name": None,
         "maximum_requested": False, "discover_capacity": False},
        ["inference_calibration", "pipeline_calibration", "execute"],
    ),
    (
        {"autotune": True, "model_backend": "transformer",
         "calibration_binary_name": None,
         "maximum_requested": False, "discover_capacity": False},
        ["execute"],
    ),
    (
        {"autotune": True, "model_backend": "mlp",
         "calibration_binary_name": "stream1_libtorch_mlp_benchmark",
         "maximum_requested": False, "discover_capacity": True},
        ["inference_calibration", "capacity_discovery",
         "pipeline_calibration", "execute"],
    ),
    (
        {"autotune": True, "model_backend": "mlp",
         "calibration_binary_name": "stream1_libtorch_mlp_benchmark",
         "maximum_requested": True, "discover_capacity": True},
        ["inference_calibration", "maximum_frontier_search",
         "pipeline_calibration", "working_capacity_selection", "execute"],
    ),
    (
        {"autotune": True, "model_backend": "transformer",
         "calibration_binary_name": None,
         "maximum_requested": True, "discover_capacity": False},
        ["maximum_frontier_search", "execute"],
    ),
    (
        {"autotune": False, "model_backend": "mlp",
         "calibration_binary_name": "stream1_native_mlp_benchmark",
         "maximum_requested": False, "discover_capacity": False},
        ["execute"],
    ),
]

for phase_request, expected_phases in phase_cases:
    direct = json.loads(phase_plan_native(json.dumps(phase_request, sort_keys=True)))
    if direct != {"schema_version": 1, "phases": expected_phases}:
        raise AssertionError(
            f"installed Rust search phase plan mismatch: "
            f"expected={expected_phases}, actual={direct}, request={phase_request}"
        )
    adapted = rust_control.resolve_search_phase_plan(phase_request)
    if adapted != tuple(expected_phases):
        raise AssertionError(
            f"Rust phase-plan Python adapter mismatch: "
            f"expected={expected_phases}, actual={adapted}"
        )
    options = SimpleNamespace(
        autotune=phase_request["autotune"],
        calibration_discover_capacity=phase_request["discover_capacity"],
    )
    model = SimpleNamespace(backend=phase_request["model_backend"])
    prepared_runtime = SimpleNamespace(
        build_metadata={"calibration_binary_name": phase_request["calibration_binary_name"]}
    )
    with patch.object(
        backend, "_python_search_phase_plan",
        side_effect=AssertionError("installed wheel unexpectedly used Python phase selection"),
    ):
        consumed = backend._resolve_search_phase_plan(
            options, model, prepared_runtime, phase_request["maximum_requested"]
        )
    if consumed != tuple(expected_phases):
        raise AssertionError(
            f"public permutation preparation consumed a different Rust phase plan: {consumed}"
        )

with patch.object(rust_control, "extension", return_value=LegacyRuntime()):
    old_wheel_phases = backend._resolve_search_phase_plan(
        SimpleNamespace(autotune=True, calibration_discover_capacity=False),
        SimpleNamespace(backend="mlp"),
        SimpleNamespace(build_metadata={"calibration_binary_name": "stream1_native_mlp_benchmark"}),
        False,
    )
if old_wheel_phases != ("inference_calibration", "pipeline_calibration", "execute"):
    raise AssertionError(f"older-wheel Python compatibility plan changed: {old_wheel_phases}")


class BrokenPhasePlanRuntime:
    ABI_VERSION = 1
    SEARCH_PHASE_PLAN_VERSION = 1
    InferenceCalibration = object()

    @staticmethod
    def resolve_search_phase_plan(_payload):
        raise RuntimeError("injected Rust phase-plan failure")


with patch.object(rust_control, "extension", return_value=BrokenPhasePlanRuntime()):
    try:
        backend._resolve_search_phase_plan(
            SimpleNamespace(autotune=True, calibration_discover_capacity=False),
            SimpleNamespace(backend="mlp"),
            SimpleNamespace(build_metadata={"calibration_binary_name": "stream1_native_mlp_benchmark"}),
            False,
        )
    except NativeBackendError as error:
        if "phase resolution failed" not in str(error):
            raise AssertionError(f"unexpected live Rust phase-plan error: {error}") from error
    else:
        raise AssertionError("a live Rust phase-plan failure silently fell back to Python")

with patch.object(
    rust_control,
    "extension",
    side_effect=NativeUnavailable("Rust runtime is incompatible; rebuild the matching PyO3 module"),
):
    try:
        backend._resolve_search_phase_plan(
            SimpleNamespace(autotune=True, calibration_discover_capacity=False),
            SimpleNamespace(backend="mlp"),
            SimpleNamespace(build_metadata={"calibration_binary_name": "stream1_native_mlp_benchmark"}),
            False,
        )
    except NativeBackendError as error:
        if "incompatible" not in str(error):
            raise AssertionError(f"unexpected incompatible runtime error: {error}") from error
    else:
        raise AssertionError("an incompatible Rust runtime silently fell back to Python")


execution_resolver = getattr(runtime, "resolve_search_execution_plan", None)
if not callable(execution_resolver) or getattr(runtime, "SEARCH_EXECUTION_PLAN_VERSION", None) != 1:
    raise AssertionError("installed Rust runtime does not expose the typed search execution plan")

maximum_request = {
    "autotune": True,
    "model_backend": "transformer",
    "calibration_binary_name": None,
    "requested_beam_width": "max",
    "discover_capacity": False,
}
direct_execution = json.loads(execution_resolver(json.dumps(maximum_request, sort_keys=True)))
if direct_execution != {
    "schema_version": 1,
    "phases": ["maximum_frontier_search", "execute"],
    "calibration_beam_width": 1_048_576,
}:
    raise AssertionError(f"Rust maximum-width execution plan changed: {direct_execution}")
exact_request = dict(maximum_request, requested_beam_width=37, autotune=False)
direct_exact = json.loads(execution_resolver(json.dumps(exact_request, sort_keys=True)))
if direct_exact["calibration_beam_width"] != 37 or direct_exact["phases"] != ["execute"]:
    raise AssertionError(f"Rust fixed-width execution plan changed: {direct_exact}")

from multigpubeamsearch import model_snapshot

prepared_runtime = SimpleNamespace(
    build_metadata={
        "shape": "fixture-shape",
        "backend": "transformer",
        "calibration_binary_name": None,
        "binary_sha256": "c" * 64,
    },
    profile={},
)
execution_options = NativeOptions(
    cache_dir=Path(tempfile.gettempdir()) / "beam-execution-plan-cache",
    autotune=True,
)
with patch.object(
    backend, "_python_search_phase_plan",
    side_effect=AssertionError("installed wheel unexpectedly used Python execution planning"),
):
    consumed_execution = backend._resolve_search_execution_plan(
        execution_options, SimpleNamespace(backend="transformer"),
        prepared_runtime, "max"
    )
if consumed_execution != {
    "phases": ("maximum_frontier_search", "execute"),
    "calibration_beam_width": 1_048_576,
}:
    raise AssertionError(f"public preparation did not consume the Rust execution plan: {consumed_execution}")

observed_microbatch_widths = []
def record_microbatch(_contract, _model, beam_width, _world_size, *, row_limit=8192):
    observed_microbatch_widths.append(beam_width)
    return {}, {}

with tempfile.TemporaryDirectory(prefix="beam-rust-execution-plan-") as temporary:
    root = Path(temporary)
    run_dir = root / "run"
    contract = SimpleNamespace(
        move_count=2,
        graph_hash="a" * 64,
        replay=lambda _path: False,
    )
    model = SimpleNamespace(
        backend="transformer",
        manifest={"output_dim": 1, "heuristic": "fixture"},
        weights_dir=root / "weights",
    )
    prepared_runtime = SimpleNamespace(
        build_metadata={
            "shape": "fixture-shape",
            "backend": "transformer",
            "calibration_binary_name": None,
            "binary_sha256": "c" * 64,
        },
        profile={},
    )
    execution_options = NativeOptions(
        cache_dir=root / "cache",
        autotune=True,
    )
    try:
        with (
            patch.object(backend, "native_depth_budget", return_value=(1, 0)),
            patch.object(backend, "validate_touch_bfs_contract", return_value=0),
            patch.object(build, "shape_contract", return_value="fixture-shape"),
            patch.object(build, "acceptance_rank_trace_identity", return_value=None),
            patch.object(rust_control, "write_search_inputs", return_value=True),
            patch.object(backend, "child_environment", return_value={}),
            patch.object(backend, "runtime_search_environment", return_value={}),
            patch.object(model_snapshot, "requires_mlp_host_binding", return_value=False),
            patch.object(backend, "search_model_environment", return_value={}),
            patch.object(backend, "microbatch_environment", side_effect=record_microbatch),
            patch.object(backend, "snapshot_runtime_nccl", return_value=root / "libnccl.so"),
            patch.object(backend, "snapshot_runtime_runner", return_value=root / "runner"),
            patch.object(
                backend, "_python_search_phase_plan",
                side_effect=AssertionError("installed runner unexpectedly used Python execution planning"),
            ),
        ):
            backend._run_native_once(
                contract, model, execution_options, "max", 1, run_dir, (0,),
                runtime=prepared_runtime,
            )
    except NativeUnavailable as error:
        if "maximum beam requires a supported inference calibrator" not in str(error):
            raise
    else:
        raise AssertionError("maximum-width transformer unexpectedly passed the unsupported-calibrator boundary")
if observed_microbatch_widths != [1_048_576]:
    raise AssertionError(
        f"the public runner did not use Rust's calibration width: {observed_microbatch_widths}"
    )

with patch.object(rust_control, "extension", return_value=LegacyRuntime()):
    legacy_execution_plan = backend._resolve_search_execution_plan(
        SimpleNamespace(autotune=True, calibration_discover_capacity=False),
        SimpleNamespace(backend="transformer"),
        SimpleNamespace(build_metadata={"calibration_binary_name": None}),
        "max",
    )
if legacy_execution_plan != {
    "phases": ("maximum_frontier_search", "execute"),
    "calibration_beam_width": 1_048_576,
}:
    raise AssertionError(f"the old-wheel execution fallback changed: {legacy_execution_plan}")

class BrokenExecutionPlanRuntime:
    ABI_VERSION = 1
    SEARCH_EXECUTION_PLAN_VERSION = 1
    InferenceCalibration = object()

    @staticmethod
    def resolve_search_execution_plan(_payload):
        raise RuntimeError("injected Rust execution-plan failure")

with patch.object(rust_control, "extension", return_value=BrokenExecutionPlanRuntime()):
    try:
        backend._resolve_search_execution_plan(
            SimpleNamespace(autotune=True, calibration_discover_capacity=False),
            SimpleNamespace(backend="transformer"),
            SimpleNamespace(build_metadata={"calibration_binary_name": None}),
            "max",
        )
    except NativeBackendError as error:
        if "execution planning failed" not in str(error):
            raise
    else:
        raise AssertionError("a live Rust execution-plan error silently fell back to Python")

print("INSTALLED_RUST_SEARCH_EXECUTION_PLAN_PASS", flush=True)
print("INSTALLED_RUST_SEARCH_PHASE_PLAN_PASS", flush=True)
print("INSTALLED_RUST_INFERENCE_BACKEND_POLICY_PASS", flush=True)
