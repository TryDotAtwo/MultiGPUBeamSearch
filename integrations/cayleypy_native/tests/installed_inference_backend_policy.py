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


# Exercise the normal preparation path with mocked CUDA observations and runner
# construction. The installed Rust resolver must select libtorch for the T4.
import torch
from multigpubeamsearch import backend, build, rust_control
from multigpubeamsearch.options import NativeOptions


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
        patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)),
        patch.object(torch.cuda, "get_device_name", return_value="Tesla T4"),
        patch.object(build, "ensure_runner", side_effect=fake_ensure_runner),
    ):
        prepared = backend.prepare_runtime(
            FakeContract(), model, options, root / "run", (0,)
        )
    if selected != ["libtorch"]:
        raise AssertionError(f"prepare_runtime did not use the Rust policy result: {selected}")
    if prepared.profile.get("inference_backend") != "libtorch":
        raise AssertionError(f"prepared profile recorded a different backend: {prepared.profile}")

print("INSTALLED_RUST_INFERENCE_BACKEND_POLICY_PASS", flush=True)
