from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sm120_quant_export_weights import selected_operators, sha256_file
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


def test_selected_operators_are_canonical_and_fail_closed() -> None:
    selected = {"blocks.0.attn.in_proj_weight", "blocks.2.ff.0.weight"}
    assert selected_operators(_profile("a" * 64, selected)) == [
        name for name in CORE_OPERATORS if name in selected
    ]
    with pytest.raises(ValueError, match="does not yet support"):
        selected_operators(_profile("a" * 64, {"blocks.0.ff.3.weight"}))
    with pytest.raises(ValueError, match="at least one"):
        selected_operators(_profile("a" * 64, set()))


def test_sha256_file_matches_known_vector(tmp_path: Path) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
