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
import subprocess
import tempfile
from pathlib import Path

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


def package_profile(
    *, encoder: Path, fp16_weight_dir: Path, fp32_checkpoint: Path,
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
    try:
        subprocess.run([
            str(encoder), "--weight-dir", str(fp16_weight_dir),
            "--output-dir", str(encoded), "--operators", ",".join(operators),
            "--weight-scale-policy", weight_scale_policy,
        ], check=True)
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
            "operators": operators,
            "encoding": {
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
        try:
            temporary.rmdir()
        except OSError:
            pass
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--fp16-weight-dir", type=Path, required=True)
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-scale-policy", choices=("max_abs", "mse_grid"), default="mse_grid")
    args = parser.parse_args()
    result = package_profile(
        encoder=args.encoder, fp16_weight_dir=args.fp16_weight_dir,
        fp32_checkpoint=args.fp32_checkpoint, profile_json=args.profile_json,
        output_dir=args.output_dir, weight_scale_policy=args.weight_scale_policy,
    )
    print(json.dumps({"output_dir": str(result), "immutable": True}, sort_keys=True))


if __name__ == "__main__":
    main()
