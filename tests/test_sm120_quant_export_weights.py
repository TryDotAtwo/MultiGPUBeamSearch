from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.sm120_quant_export_weights import (
    _prepare_encoding_source,
    selected_operators,
    sha256_file,
)
from tools.sm120_quant_tuner import CORE_OPERATORS, build_profile


def _profile(checkpoint_sha256: str, selected: set[str]) -> dict:
    return build_profile(
        checkpoint_sha256=checkpoint_sha256,
        model_metadata_sha256="b" * 64,
        calibration_sha256="c" * 64,
        gpu_identity="RTX PRO 6000 Blackwell|sm120",
        cutlass_commit="d" * 40,
        operator_precision={
            name: "sm120_block_fp8" if name in selected else "fp16"
            for name in CORE_OPERATORS
        },
    )


def _int8_profile(checkpoint_sha256: str) -> dict:
    return build_profile(
        checkpoint_sha256=checkpoint_sha256,
        model_metadata_sha256="b" * 64,
        calibration_sha256="c" * 64,
        gpu_identity="RTX PRO 6000 Blackwell|sm120",
        cutlass_commit="d" * 40,
        operator_precision={name: "sm120_block_int8" for name in CORE_OPERATORS},
    )


def test_selected_operators_are_canonical_and_fail_closed() -> None:
    selected = {"blocks.0.attn.in_proj_weight", "blocks.2.ff.0.weight"}
    assert selected_operators(_profile("a" * 64, selected)) == [
        name for name in CORE_OPERATORS if name in selected
    ]
    with pytest.raises(ValueError, match="does not yet support"):
        selected_operators(_profile("a" * 64, {"blocks.0.ff.3.weight"}))
    with pytest.raises(ValueError, match="at least one"):
        selected_operators(_profile("a" * 64, set()))
    with pytest.raises(ValueError, match="another low-precision layout"):
        selected_operators(_int8_profile("a" * 64))


def test_sha256_file_matches_known_vector(tmp_path: Path) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_prepare_encoding_source_folds_equalization_offline(tmp_path: Path) -> None:
    source = tmp_path / "fp32"
    source.mkdir()
    (source / "manifest.json").write_text('{"dtype":"fp32"}\n')
    name = "blocks.0.attn.in_proj_weight"
    weight = np.arange(256 * 768, dtype=np.float32).reshape(256, 768) / 1024.0
    gamma = np.linspace(0.5, 1.5, 256, dtype=np.float32)
    beta = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    weight.tofile(source / "block0_attn_qkv_weight_hxk.fp32")
    gamma.tofile(source / "block0_ln1_gamma.fp32")
    beta.tofile(source / "block0_ln1_beta.fp32")
    transforms = {operator: [] for operator in CORE_OPERATORS}
    scales = np.linspace(0.5, 2.0, 256, dtype=np.float32)
    transforms[name] = [{
        "type": "layernorm_linear_smoothquant",
        "alpha": 0.5,
        "scales": scales.tolist(),
    }]
    profile = build_profile(
        checkpoint_sha256="a" * 64,
        model_metadata_sha256="b" * 64,
        calibration_sha256="c" * 64,
        gpu_identity="RTX PRO 6000 Blackwell|sm120",
        cutlass_commit="d" * 40,
        operator_precision={
            operator: "sm120_block_fp8" if operator == name else "fp16"
            for operator in CORE_OPERATORS
        },
        operator_transforms=transforms,
    )
    destination = tmp_path / "prepared"
    overrides = _prepare_encoding_source(source, profile, [name], destination)
    encoded_weight = np.fromfile(
        destination / "block0_attn_qkv_weight_hxk.fp32", dtype=np.float32
    ).reshape(256, 768)
    np.testing.assert_allclose(encoded_weight, weight / scales[:, None])
    np.testing.assert_array_equal(
        np.frombuffer(overrides[name]["gamma"], dtype=np.float16),
        (gamma * scales).astype(np.float16),
    )
    np.testing.assert_array_equal(
        np.frombuffer(overrides[name]["beta"], dtype=np.float16),
        (beta * scales).astype(np.float16),
    )
