#!/usr/bin/env python3
"""Validate and package immutable SM120 NVFP4 fused-FFN profiles.

The native encoder owns CUTLASS-specific nibble and scale-factor layouts.  This
module owns the model-level contract and deliberately fails closed before any
artifact is written.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from tools.sm120_quant_tuner import CORE_OPERATORS, validate_profile


_FFN_OPERATOR = re.compile(r"^blocks\.(\d+)\.ff\.(0|3)\.weight$")

# name, logical K, logical N, physical N.  SM120 block-scaled weights require
# N128.  The 24-logit head is therefore exported as an explicitly padded N128
# matrix; runtime must consume only the first 24 columns.
FULL_STREAM1_OPERATORS = tuple(
    (
        f"block{block}_{slug}",
        input_cols,
        output_cols,
        output_cols,
    )
    for block in range(4)
    for slug, input_cols, output_cols in (
        ("attn_qkv", 256, 768),
        ("attn_out", 256, 256),
        ("ff1", 256, 1024),
        ("ff2", 1024, 256),
    )
) + (("output_head", 256, 24, 128),)


def selected_fused_ffn_layers(profile: Mapping) -> list[int]:
    """Return eligible Cube4 blocks selected as complete NVFP4 FFN pairs."""
    validated = validate_profile(profile)
    selected = [
        name
        for name in CORE_OPERATORS
        if validated["operators"][name]["weight_dtype"] == "e2m1"
    ]
    by_layer: dict[int, set[int]] = {}
    for name in selected:
        match = _FFN_OPERATOR.fullmatch(name)
        if match is None:
            raise ValueError(
                "SM120 NVFP4 runtime currently supports only fused FF1/FF2 operators"
            )
        layer = int(match.group(1))
        if layer == 3:
            raise ValueError(
                "SM120 NVFP4 next-layer packing is not valid for the final block"
            )
        by_layer.setdefault(layer, set()).add(int(match.group(2)))

    for layer, members in sorted(by_layer.items()):
        if members != {0, 3}:
            raise ValueError(
                f"block {layer} must select a complete FF1/FF2 pair for fused NVFP4"
            )
    return sorted(by_layer)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accepted_report(path: Path, label: str) -> tuple[dict, str]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"SM120 NVFP4 {label} report must be readable JSON") from error
    if not isinstance(payload, Mapping) or payload.get("status") != "accepted":
        raise ValueError(f"SM120 NVFP4 {label} report must have status=accepted")
    return dict(payload), _sha256(path)


def _validated_build_fingerprint(value: Mapping) -> dict[str, str]:
    required = ("cuda", "cutlass_commit", "source_sha256")
    if not isinstance(value, Mapping) or any(
        not isinstance(value.get(key), str) or not value[key]
        for key in required
    ):
        raise ValueError(
            "SM120 NVFP4 build fingerprint requires cuda, cutlass_commit, and source_sha256"
        )
    if re.fullmatch(r"[0-9a-f]{40}", value["cutlass_commit"]) is None:
        raise ValueError("SM120 NVFP4 CUTLASS fingerprint must be a lowercase SHA-1")
    if re.fullmatch(r"[0-9a-f]{64}", value["source_sha256"]) is None:
        raise ValueError("SM120 NVFP4 source fingerprint must be a lowercase SHA-256")
    return {key: value[key] for key in required}


def _validate_full_stream1_benchmark(payload: Mapping) -> None:
    if payload.get("scope") != "full_stream1":
        raise ValueError("SM120 NVFP4 benchmark scope must be full_stream1")
    if payload.get("batch") != 896:
        raise ValueError("SM120 NVFP4 benchmark batch must be 896")
    numeric_limits = (
        ("median_ms", 0.333252, "median"),
        ("p95_ms", 0.350000, "p95"),
        ("stage_sum_ms", 0.300000, "stage sum"),
    )
    for key, limit, label in numeric_limits:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value > limit:
            raise ValueError(f"SM120 NVFP4 full Stream1 {label} gate was not met")
    runs = payload.get("alternating_runs")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 7:
        raise ValueError("SM120 NVFP4 full Stream1 alternating run gate was not met")
    for key, label in (
        ("oom", "OOM"), ("nan", "NaN"), ("race", "race"),
        ("barrier_errors", "barrier"),
    ):
        if payload.get(key) is not False:
            raise ValueError(f"SM120 NVFP4 full Stream1 {label} gate was not met")


def _validate_full_stream1_quality(payload: Mapping) -> None:
    if payload.get("scope") != "full_stream1":
        raise ValueError("SM120 NVFP4 quality scope must be full_stream1")
    for key in (
        "replay_valid",
        "nvfp4_solved_not_less_than_fp16",
        "nvfp4_lengths_not_greater_than_fp16",
    ):
        if payload.get(key) is not True:
            raise ValueError(f"SM120 NVFP4 quality gate requires {key}=true")
    ladder = payload.get("quality_ladder")
    expected_ladder = [(10, 2**16), (30, 2**18), (100, 2**20)]
    if not isinstance(ladder, list) or [
        (item.get("states"), item.get("beam")) if isinstance(item, Mapping) else None
        for item in ladder
    ] != expected_ladder:
        raise ValueError("SM120 NVFP4 quality ladder is incomplete")
    depth8 = payload.get("final_depth8")
    if not isinstance(depth8, Mapping) or depth8.get("beam") != 2**25 or depth8.get("completed_depth") != 8:
        raise ValueError("SM120 NVFP4 beam 2^25 depth-8 gate was not met")
    solve = payload.get("full_solve_control")
    if not isinstance(solve, Mapping) or solve.get("replay_valid") is not True or solve.get("length_not_greater_than_fp16") is not True:
        raise ValueError("SM120 NVFP4 full-solve quality gate was not met")


def export_nvfp4_full_stream1_package(
    fp32_dir: Path,
    destination: Path,
    *,
    native_exporter: Path,
    quality_report: Path,
    benchmark_report: Path,
    build_fingerprint: Mapping,
    microbatch: int = 896,
    physical_slots: int = 2,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    """Atomically package an accepted, immutable full-Stream1 NVFP4 profile.

    This is intentionally separate from the component FFN exporter.  It cannot
    produce promotion_state=accepted until the exact end-to-end performance and
    solve-quality reports satisfy the production gates.
    """
    fp32_dir = Path(fp32_dir)
    destination = Path(destination)
    native_exporter = Path(native_exporter)
    source_manifest_path = fp32_dir / "manifest.json"
    try:
        source_model = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("SM120 NVFP4 export requires a readable FP32 Cube4 ReLU manifest") from error
    expected = {
        "backend": "piece_transformer", "dtype": "fp32", "output_dim": 24,
        "d_model": 256, "num_layers": 4, "ff_dim": 1024,
        "activation": "relu", "seq_len": 57, "piece_layout": "cube4",
        "pooling": "cls",
    }
    if any(source_model.get(key) != value for key, value in expected.items()):
        raise ValueError("SM120 NVFP4 export requires the exact FP32 Cube4 ReLU model contract")
    if not isinstance(microbatch, int) or isinstance(microbatch, bool) or microbatch <= 0:
        raise ValueError("SM120 NVFP4 runtime microbatch must be a positive integer")
    if physical_slots != 2:
        raise ValueError("SM120 NVFP4 persistent runtime requires exactly two physical slots")
    quality_payload, quality_sha256 = _accepted_report(quality_report, "quality")
    benchmark_payload, benchmark_sha256 = _accepted_report(benchmark_report, "benchmark")
    _validate_full_stream1_quality(quality_payload)
    _validate_full_stream1_benchmark(benchmark_payload)
    validated_build = _validated_build_fingerprint(build_fingerprint)
    checkpoint_sha256 = source_model.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise ValueError("SM120 NVFP4 source manifest requires checkpoint_sha256")
    if destination.exists():
        raise FileExistsError(f"NVFP4 package destination already exists: {destination}")
    temporary = Path(str(destination) + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    argv = [
        str(native_exporter), "--weight-dir", str(fp32_dir),
        "--output-dir", str(temporary), "--scope", "full_stream1",
    ]
    try:
        runner(argv, check=True)
        runtime_manifest = temporary / "runtime_manifest.txt"
        text = runtime_manifest.read_text(encoding="utf-8")
        required = (
            "schema_version=2\n",
            "encoding=sm120_nvfp4_e2m1_k16_ue4m3_sfb\n",
            "weights=offline_immutable\n",
            "profile_scope=full_stream1\n",
            f"operator_count={len(FULL_STREAM1_OPERATORS)}\n",
            "operator.output_head.logical_output_cols=24\n",
            "operator.output_head.physical_output_cols=128\n",
        )
        if any(item not in text for item in required):
            raise ValueError("native NVFP4 exporter produced an incomplete full-Stream1 manifest")
        text = text.replace("schema_version=2\n", "schema_version=4\n", 1)
        text += (
            "backend=sm120a_nvfp4_persistent\n"
            "target_sm=120\n"
            "target_arch=sm120a\n"
            "promotion_state=accepted\n"
            f"microbatch={microbatch}\n"
            f"physical_slots={physical_slots}\n"
            f"checkpoint_sha256={checkpoint_sha256}\n"
            f"quality_report_sha256={quality_sha256}\n"
            f"benchmark_report_sha256={benchmark_sha256}\n"
            f"cuda={validated_build['cuda']}\n"
            f"cutlass_commit={validated_build['cutlass_commit']}\n"
            f"source_sha256={validated_build['source_sha256']}\n"
        )
        runtime_manifest.write_text(text, encoding="utf-8")
        operators = [
            {
                "name": name, "input_cols": k,
                "logical_output_cols": logical_n,
                "physical_output_cols": physical_n,
            }
            for name, k, logical_n, physical_n in FULL_STREAM1_OPERATORS
        ]
        package_manifest = {
            "schema_version": 4,
            "backend": "sm120a_nvfp4_persistent",
            "target_sm": 120,
            "target_arch": "sm120a",
            "encoding": "sm120_nvfp4_e2m1_k16_ue4m3_sfb",
            "weights": "offline_immutable",
            "profile_scope": "full_stream1",
            "promotion_state": "accepted",
            "microbatch": microbatch,
            "physical_slots": physical_slots,
            "operator_mask": [item[0] for item in FULL_STREAM1_OPERATORS],
            "operators": operators,
            "source_model": source_model,
            "source_manifest_sha256": _sha256(source_manifest_path),
            "runtime_manifest_sha256": _sha256(runtime_manifest),
            "quality_report_sha256": quality_sha256,
            "benchmark_report_sha256": benchmark_sha256,
            "quality_report": quality_payload,
            "benchmark_report": benchmark_payload,
            "build_fingerprint": validated_build,
        }
        (temporary / "package_manifest.json").write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def export_nvfp4_ffn_package(
    fp32_dir: Path,
    destination: Path,
    profile: Mapping,
    *,
    native_exporter: Path,
    quality_report: Path | None = None,
    benchmark_report: Path | None = None,
    build_fingerprint: Mapping | None = None,
    microbatch: int = 896,
    physical_slots: int = 2,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    """Atomically build one immutable SM120 NVFP4 FFN package.

    The native encoder owns the physical CUTLASS layout.  Python verifies the
    exact Cube4 graph contract and records provenance; it never quantizes a
    tensor at inference time.
    """
    fp32_dir = Path(fp32_dir)
    destination = Path(destination)
    native_exporter = Path(native_exporter)
    source_manifest_path = fp32_dir / "manifest.json"
    try:
        source_model = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("SM120 NVFP4 export requires a readable FP32 Cube4 ReLU manifest") from error
    expected = {
        "backend": "piece_transformer",
        "dtype": "fp32",
        "output_dim": 24,
        "d_model": 256,
        "num_layers": 4,
        "ff_dim": 1024,
        "activation": "relu",
        "seq_len": 57,
        "piece_layout": "cube4",
        "pooling": "cls",
    }
    if any(source_model.get(key) != value for key, value in expected.items()):
        raise ValueError("SM120 NVFP4 export requires the exact FP32 Cube4 ReLU model contract")
    if not isinstance(microbatch, int) or isinstance(microbatch, bool) or microbatch <= 0:
        raise ValueError("SM120 NVFP4 runtime microbatch must be a positive integer")
    if physical_slots != 2:
        raise ValueError("SM120 NVFP4 persistent runtime requires exactly two physical slots")
    if quality_report is None or benchmark_report is None:
        raise ValueError("SM120 NVFP4 component package requires quality and benchmark reports")
    quality_payload, quality_sha256 = _accepted_report(quality_report, "quality")
    benchmark_payload, benchmark_sha256 = _accepted_report(
        benchmark_report, "benchmark"
    )
    validated_build = _validated_build_fingerprint(build_fingerprint or {})
    validated_profile = validate_profile(profile)
    checkpoint_sha256 = source_model.get("checkpoint_sha256")
    if checkpoint_sha256 != validated_profile["fingerprints"]["checkpoint_sha256"]:
        raise ValueError("SM120 NVFP4 profile checkpoint fingerprint does not match FP32 export")
    layers = selected_fused_ffn_layers(validated_profile)
    if not layers:
        raise ValueError("SM120 NVFP4 export profile selects no fused FFN layers")
    if destination.exists():
        raise FileExistsError(f"NVFP4 package destination already exists: {destination}")
    temporary = Path(str(destination) + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    argv = [
        str(native_exporter), "--weight-dir", str(fp32_dir),
        "--output-dir", str(temporary), "--layers",
        ",".join(str(layer) for layer in layers),
    ]
    try:
        runner(argv, check=True)
        runtime_manifest = temporary / "runtime_manifest.txt"
        text = runtime_manifest.read_text(encoding="utf-8")
        required = (
            "schema_version=1\n",
            "encoding=sm120_nvfp4_e2m1_k16_ue4m3_sfb\n",
            "weights=offline_immutable\n",
        )
        if any(item not in text for item in required):
            raise ValueError("native NVFP4 exporter produced an incompatible runtime manifest")
        text = text.replace("schema_version=1\n", "schema_version=4\n", 1)
        text += (
            "backend=sm120a_nvfp4_persistent\n"
            "target_sm=120\n"
            "target_arch=sm120a\n"
            "profile_scope=ffn_only\n"
            "promotion_state=component_candidate\n"
            f"microbatch={microbatch}\n"
            f"physical_slots={physical_slots}\n"
            f"layers={','.join(str(layer) for layer in layers)}\n"
            f"checkpoint_sha256={checkpoint_sha256}\n"
            f"quality_report_sha256={quality_sha256}\n"
            f"benchmark_report_sha256={benchmark_sha256}\n"
            f"cuda={validated_build['cuda']}\n"
            f"cutlass_commit={validated_build['cutlass_commit']}\n"
            f"source_sha256={validated_build['source_sha256']}\n"
        )
        runtime_manifest.write_text(text, encoding="utf-8")
        package_manifest = {
            "schema_version": 4,
            "backend": "sm120a_nvfp4_persistent",
            "target_sm": 120,
            "target_arch": "sm120a",
            "encoding": "sm120_nvfp4_e2m1_k16_ue4m3_sfb",
            "weights": "offline_immutable",
            "profile_scope": "ffn_only",
            "promotion_state": "component_candidate",
            "microbatch": microbatch,
            "physical_slots": physical_slots,
            "layers": layers,
            "source_model": source_model,
            "source_manifest_sha256": _sha256(source_manifest_path),
            "runtime_manifest_sha256": _sha256(runtime_manifest),
            "quality_report_sha256": quality_sha256,
            "benchmark_report_sha256": benchmark_sha256,
            "quality_report": quality_payload,
            "benchmark_report": benchmark_payload,
            "build_fingerprint": validated_build,
            "precision_profile": validated_profile,
        }
        (temporary / "package_manifest.json").write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
