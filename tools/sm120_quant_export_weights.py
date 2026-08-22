#!/usr/bin/env python3
"""Create an immutable SM120 mixed-precision weight profile offline.

Weight conversion is delegated to the native SM120 CUDA encoder, so the bytes
are exactly the CUTLASS E4M3 representation consumed by production.  The
production runner never derives or mutates weights at startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tools.sm120_quant_tuner import CORE_OPERATORS, validate_profile


RUNTIME_SUPPORTED = frozenset(
    f"blocks.{layer}.{suffix}"
    for layer in range(4)
    for suffix in ("attn.in_proj_weight", "ff.0.weight")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_operators(profile: dict) -> list[str]:
    validated = validate_profile(profile)
    incompatible = [
        name for name in CORE_OPERATORS
        if validated["operators"][name]["weight_dtype"] not in ("fp16", "e4m3")
    ]
    if incompatible:
        raise ValueError(
            "E4M3 encoder cannot package operators using another low-precision layout: "
            f"{incompatible}"
        )
    selected = [
        name for name in CORE_OPERATORS
        if validated["operators"][name]["weight_dtype"] == "e4m3"
    ]
    unsupported = sorted(set(selected) - RUNTIME_SUPPORTED)
    if unsupported:
        raise ValueError(f"native runtime does not yet support offline mixed operators: {unsupported}")
    if not selected:
        raise ValueError("mixed profile must select at least one offline encoded operator")
    return selected


def _operator_files(name: str) -> tuple[str, str, str]:
    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "blocks" or not parts[1].isdigit():
        raise ValueError(f"unsupported transformed operator: {name}")
    block = int(parts[1])
    if name.endswith("attn.in_proj_weight"):
        return (
            f"block{block}_attn_qkv_weight_hxk.fp32",
            f"block{block}_ln1_gamma.fp32",
            f"block{block}_ln1_beta.fp32",
        )
    if name.endswith("ff.0.weight"):
        return (
            f"block{block}_ff1_weight_hxk.fp32",
            f"block{block}_ln2_gamma.fp32",
            f"block{block}_ln2_beta.fp32",
        )
    raise ValueError(f"offline transform is not supported for operator: {name}")


def _prepare_encoding_source(
    source_dir: Path,
    profile: dict,
    operators: list[str],
    destination: Path,
) -> dict[str, dict[str, bytes | str]]:
    """Materialize transformed FP32 weights and FP16 LN overrides offline."""
    destination.mkdir()
    shutil.copyfile(source_dir / "manifest.json", destination / "manifest.json")
    overrides: dict[str, dict[str, bytes | str]] = {}
    for name in operators:
        weight_file, gamma_file, beta_file = _operator_files(name)
        transforms = profile["operators"][name]["folded_transforms"]
        source_weight = source_dir / weight_file
        if not transforms:
            shutil.copyfile(source_weight, destination / weight_file)
            overrides[name] = {"transform": "none"}
            continue
        if len(transforms) != 1:
            raise ValueError(f"operator {name} must have exactly one offline transform")
        transform = transforms[0]
        scales = np.asarray(transform["scales"], dtype=np.float32)
        weight = np.fromfile(source_weight, dtype=np.float32)
        if weight.size % scales.size:
            raise ValueError(f"operator {name} FP32 weight shape is incompatible with transform")
        transformed_weight = (weight.reshape(scales.size, -1) / scales[:, None]).astype(np.float32)
        transformed_weight.tofile(destination / weight_file)
        gamma = np.fromfile(source_dir / gamma_file, dtype=np.float32)
        beta = np.fromfile(source_dir / beta_file, dtype=np.float32)
        if gamma.shape != scales.shape or beta.shape != scales.shape:
            raise ValueError(f"operator {name} LayerNorm vector shape is incompatible with transform")
        overrides[name] = {
            "transform": "layernorm_linear_smoothquant",
            "gamma": (gamma * scales).astype(np.float16).tobytes(),
            "beta": (beta * scales).astype(np.float16).tobytes(),
        }
    return overrides


def package_profile(
    *, encoder: Path, fp16_weight_dir: Path, fp32_weight_dir: Path, fp32_checkpoint: Path,
    profile_json: Path, output_dir: Path, weight_scale_policy: str = "mse_grid",
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"immutable profile directory already exists: {output_dir}")
    profile = validate_profile(json.loads(profile_json.read_text(encoding="utf-8")))
    if weight_scale_policy not in {"max_abs", "mse_grid"}:
        raise ValueError("weight_scale_policy must be max_abs or mse_grid")
    operators = selected_operators(profile)
    checkpoint_sha256 = sha256_file(fp32_checkpoint)
    if profile["fingerprints"]["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("profile checkpoint_sha256 does not match the original FP32 checkpoint")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    # The encoder requires a non-existent destination, while mkdtemp reserves
    # the parent name safely.  Its child is atomically promoted on success.
    encoded = temporary / "encoded"
    encoding_source = temporary / "encoding_source"
    try:
        overrides = _prepare_encoding_source(
            fp32_weight_dir, profile, operators, encoding_source
        )
        subprocess.run([
            str(encoder), "--weight-dir", str(encoding_source),
            "--output-dir", str(encoded), "--operators", ",".join(operators),
            "--weight-scale-policy", weight_scale_policy,
        ], check=True)
        runtime_manifest_path = encoded / "runtime_manifest.txt"
        runtime_manifest = runtime_manifest_path.read_text(encoding="utf-8")
        runtime_manifest += (
            "runtime_base_manifest_sha256="
            + sha256_file(fp16_weight_dir / "manifest.json") + "\n"
        )
        for name in operators:
            prefix = f"operator.{name}."
            override = overrides[name]
            runtime_manifest += prefix + "transform=" + str(override["transform"]) + "\n"
            if override["transform"] != "none":
                slug = name.replace(".", "_")
                gamma_relative = Path("weights") / f"{slug}.ln_gamma.fp16"
                beta_relative = Path("weights") / f"{slug}.ln_beta.fp16"
                (encoded / gamma_relative).write_bytes(bytes(override["gamma"]))
                (encoded / beta_relative).write_bytes(bytes(override["beta"]))
                runtime_manifest += prefix + "ln_gamma_file=" + gamma_relative.as_posix() + "\n"
                runtime_manifest += prefix + "ln_gamma_sha256=" + sha256_file(encoded / gamma_relative) + "\n"
                runtime_manifest += prefix + "ln_beta_file=" + beta_relative.as_posix() + "\n"
                runtime_manifest += prefix + "ln_beta_sha256=" + sha256_file(encoded / beta_relative) + "\n"
        runtime_manifest_path.write_text(runtime_manifest, encoding="utf-8")
        (encoded / "profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance = {
            "schema_version": 1,
            "fp32_checkpoint": {
                "filename": fp32_checkpoint.name,
                "sha256": checkpoint_sha256,
            },
            "fp16_export": {
                "manifest_sha256": sha256_file(fp16_weight_dir / "manifest.json"),
                "directory_name": fp16_weight_dir.name,
            },
            "fp32_export": {
                "manifest_sha256": sha256_file(fp32_weight_dir / "manifest.json"),
                "directory_name": fp32_weight_dir.name,
            },
            "operators": operators,
            "encoding": {
                "source_weight": "fp32",
                "weight": "cutlass_float_e4m3",
                "weight_scale": "fp32",
                "weight_scale_granularity": {"k": 128, "n": 128},
                "weight_scale_policy": weight_scale_policy,
                "activation": "dynamic_cutlass_float_e4m3",
                "activation_scale": "fp32",
                "activation_scale_granularity": {"row": 1, "k": 128},
                "accumulator": "fp32",
                "output": "fp16",
            },
        }
        (encoded / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = {
            path.relative_to(encoded).as_posix(): sha256_file(path)
            for path in sorted(encoded.rglob("*")) if path.is_file()
        }
        (encoded / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(encoded, output_dir)
    finally:
        shutil.rmtree(encoding_source, ignore_errors=True)
        try:
            temporary.rmdir()
        except OSError:
            pass
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--fp16-weight-dir", type=Path, required=True)
    parser.add_argument("--fp32-weight-dir", type=Path, required=True)
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-scale-policy", choices=("max_abs", "mse_grid"), default="mse_grid")
    args = parser.parse_args()
    result = package_profile(
        encoder=args.encoder, fp16_weight_dir=args.fp16_weight_dir,
        fp32_weight_dir=args.fp32_weight_dir,
        fp32_checkpoint=args.fp32_checkpoint, profile_json=args.profile_json,
        output_dir=args.output_dir, weight_scale_policy=args.weight_scale_policy,
    )
    print(json.dumps({"output_dir": str(result), "immutable": True}, sort_keys=True))


if __name__ == "__main__":
    main()
