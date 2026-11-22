"""CPU-only policy checks against the installed Rust/PyO3 runtime."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from multigpubeamsearch.errors import NativeUnavailable
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


class PolicyProbe:
    ABI_VERSION = 1
    InferenceCalibration = object()

    def __init__(self):
        self.requests = []

    def resolve_inference_plan(self, payload):
        self.requests.append(json.loads(payload))
        return json.dumps({
            "backend": "libtorch",
            "policy": "rust-profile-probe-v1",
            "inference_row_limit": 1234,
        })


policy_probe = PolicyProbe()
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
        patch.object(rust_control, "extension", return_value=policy_probe),
        patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100"),
        patch.object(build, "ensure_runner", side_effect=fake_ensure_runner),
    ):
        prepared = backend.prepare_runtime(
            FakeContract(), model, options, root / "run", (0,)
        )
    if len(policy_probe.requests) != 1:
        raise AssertionError(f"prepare_runtime did not call the Rust policy exactly once: {policy_probe.requests}")
    if policy_probe.requests[0]["devices"] != [{"sm": 80, "name": "NVIDIA A100"}]:
        raise AssertionError(f"Rust policy received the wrong device observation: {policy_probe.requests[0]}")
    if selected != ["libtorch"]:
        raise AssertionError(f"prepare_runtime did not use the Rust policy result: {selected}")
    if prepared.profile.get("inference_backend") != "libtorch":
        raise AssertionError(f"prepared profile recorded a different backend: {prepared.profile}")
    if (prepared.profile.get("policy") != "rust-profile-probe-v1"
            or prepared.profile.get("inference_row_limit") != 1234):
        raise AssertionError(f"prepare_runtime did not use the Rust profile plan: {prepared.profile}")

print("INSTALLED_RUST_INFERENCE_BACKEND_POLICY_PASS", flush=True)
