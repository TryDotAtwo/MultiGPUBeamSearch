from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from tools.cayleypy_public.model import detect_checkpoint_format, export_checkpoint


def write_checkpoint(tmp_path: Path, state_dict: dict[str, torch.Tensor]) -> Path:
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": state_dict}, path)
    return path


def batchnorm_schema() -> dict[str, torch.Tensor]:
    return {
        "input_layer.weight": torch.zeros(8, 12),
        "hidden_layer.weight": torch.zeros(8, 8),
        "output_layer.weight": torch.zeros(1, 8),
        "bn1.running_mean": torch.zeros(8),
        "bn2.running_mean": torch.zeros(8),
        "residual_blocks.0.fc1.weight": torch.zeros(8, 8),
    }


def resmlp_schema() -> dict[str, torch.Tensor]:
    return {
        "embedding.weight": torch.zeros(12, 16),
        "input_stack.0.weight": torch.zeros(8, 32),
        "input_stack.1.weight": torch.zeros(8),
        "input_stack.3.weight": torch.zeros(8, 8),
        "head.weight": torch.zeros(1, 8),
        "res_blocks.0.lin1.weight": torch.zeros(8, 8),
    }


def test_detects_batchnorm_folded(tmp_path: Path) -> None:
    assert detect_checkpoint_format(write_checkpoint(tmp_path, batchnorm_schema())) == "batchnorm-folded"


def test_detects_resmlp_layernorm_after_orig_mod_prefix_normalization(tmp_path: Path) -> None:
    state_dict = {f"_orig_mod.{key}": value for key, value in resmlp_schema().items()}
    assert detect_checkpoint_format(write_checkpoint(tmp_path, state_dict)) == "resmlp-layernorm"


def test_rejects_mixed_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        detect_checkpoint_format(write_checkpoint(tmp_path, batchnorm_schema() | resmlp_schema()))


def test_rejects_unknown_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        detect_checkpoint_format(write_checkpoint(tmp_path, {"weight": torch.zeros(1)}))


def _exportable_batchnorm_state(output_dim: int = 1) -> dict[str, torch.Tensor]:
    state_dict = batchnorm_schema()
    state_dict["output_layer.weight"] = torch.zeros(output_dim, 8)
    for prefix in ("bn1", "bn2", "residual_blocks.0.bn1", "residual_blocks.0.bn2"):
        state_dict.update({
            f"{prefix}.weight": torch.ones(8),
            f"{prefix}.bias": torch.zeros(8),
            f"{prefix}.running_mean": torch.zeros(8),
            f"{prefix}.running_var": torch.ones(8),
        })
    state_dict.update({
        "residual_blocks.0.fc2.weight": torch.zeros(8, 8),
        "output_layer.bias": torch.zeros(output_dim),
    })
    return state_dict

def test_export_checkpoint_sanitizes_manifest_and_hashes_source(tmp_path: Path) -> None:
    state_dict = batchnorm_schema()
    for prefix in ("bn1", "bn2", "residual_blocks.0.bn1", "residual_blocks.0.bn2"):
        state_dict.update({
            f"{prefix}.weight": torch.ones(8),
            f"{prefix}.bias": torch.zeros(8),
            f"{prefix}.running_mean": torch.zeros(8),
            f"{prefix}.running_var": torch.ones(8),
        })
    state_dict.update({
        "residual_blocks.0.fc2.weight": torch.zeros(8, 8),
        "output_layer.bias": torch.zeros(1),
    })
    path = write_checkpoint(tmp_path, state_dict)

    exported = export_checkpoint(path, tmp_path / "export", num_classes=3, state_len=4, move_count=24)

    assert exported.format == "batchnorm-folded"
    assert exported.dtype == "fp16"
    assert exported.checkpoint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert exported.manifest["source_weights"] == path.name
    assert exported.manifest["state_len"] == 4
    assert exported.manifest["num_classes"] == 3


def test_export_checkpoint_rejects_state_len_mismatch(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path, _exportable_batchnorm_state())
    with pytest.raises(ValueError, match="state_len mismatch"):
        export_checkpoint(path, tmp_path / "export", num_classes=3, state_len=5, move_count=24)


def test_export_checkpoint_accepts_move_count_head(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path, _exportable_batchnorm_state(output_dim=24))
    exported = export_checkpoint(path, tmp_path / "export", num_classes=3, state_len=4, move_count=24)
    assert exported.manifest["output_dim"] == 24


def test_export_checkpoint_rejects_other_output_dim(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path, _exportable_batchnorm_state(output_dim=7))
    with pytest.raises(ValueError, match="output_dim"):
        export_checkpoint(path, tmp_path / "export", num_classes=3, state_len=4, move_count=24)
