from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.export_stream1_mlp import export_batchnorm_folded
from tools.kaggle_t4_mlp_profiles import supported_model_header, validate_manifest


BASE_MANIFEST = {
    "state_len": 120,
    "num_classes": 120,
    "normalization": "batchnorm_folded",
}


def test_output1_manifest_is_supported() -> None:
    assert (
        validate_manifest(
            BASE_MANIFEST | {"output_dim": 1},
            state_len=120,
            num_classes=120,
            move_count=24,
        )
        == "output1"
    )


def test_move_count_manifest_is_supported() -> None:
    assert (
        validate_manifest(
            BASE_MANIFEST | {"output_dim": 24},
            state_len=120,
            num_classes=120,
            move_count=24,
        )
        == "output_move_count"
    )


def test_arbitrary_output_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="only output_dim=1 or output_dim=move_count",
    ):
        validate_manifest(
            BASE_MANIFEST | {"output_dim": 7},
            state_len=120,
            num_classes=120,
            move_count=24,
        )


@pytest.mark.parametrize(
    ("field", "actual", "expected"),
    [
        ("state_len", 119, 120),
        ("num_classes", 119, 120),
    ],
)
def test_manifest_generator_dimensions_must_match(
    field: str,
    actual: int,
    expected: int,
) -> None:
    manifest = BASE_MANIFEST | {"output_dim": 1, field: actual}
    with pytest.raises(ValueError, match=rf"{field}.*{actual}.*{expected}"):
        validate_manifest(
            manifest,
            state_len=120,
            num_classes=120,
            move_count=24,
        )


def test_header_names_both_checkpoint_formats() -> None:
    text = supported_model_header()
    assert "batchnorm-folded" in text
    assert "resmlp-layernorm" in text
    assert "arbitrary PyTorch" in text
    assert "output_dim=1" in text
    assert "output_dim=move_count" in text


def _bn_state(size: int) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.ones(size),
        "bias": torch.zeros(size),
        "running_mean": torch.zeros(size),
        "running_var": torch.ones(size),
        "num_batches_tracked": torch.tensor(0, dtype=torch.long),
    }


def test_deterministic_batchnorm_fixture_exports(tmp_path: Path) -> None:
    input_dim = 6
    hd1 = 8
    hd2 = 8
    state_dict: dict[str, torch.Tensor] = {
        "input_layer.weight": torch.arange(hd1 * input_dim, dtype=torch.float32).reshape(hd1, input_dim) / 100,
        "input_layer.bias": torch.zeros(hd1),
        "hidden_layer.weight": torch.eye(hd2, hd1),
        "hidden_layer.bias": torch.zeros(hd2),
        "output_layer.weight": torch.ones(1, hd2),
        "output_layer.bias": torch.zeros(1),
        "residual_blocks.0.fc1.weight": torch.eye(hd2),
        "residual_blocks.0.fc1.bias": torch.zeros(hd2),
        "residual_blocks.0.fc2.weight": torch.eye(hd2),
        "residual_blocks.0.fc2.bias": torch.zeros(hd2),
    }
    for prefix, size in (
        ("bn1", hd1),
        ("bn2", hd2),
        ("residual_blocks.0.bn1", hd2),
        ("residual_blocks.0.bn2", hd2),
    ):
        for suffix, tensor in _bn_state(size).items():
            state_dict[f"{prefix}.{suffix}"] = tensor

    checkpoint = tmp_path / "batchnorm_output1.pth"
    out_dir = tmp_path / "export"
    torch.save({"model": state_dict}, checkpoint)
    export_batchnorm_folded(
        checkpoint,
        out_dir,
        dtype="fp16",
        num_classes=3,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_dim"] == 1
    assert (
        validate_manifest(
            manifest,
            state_len=2,
            num_classes=3,
            move_count=24,
        )
        == "output1"
    )
