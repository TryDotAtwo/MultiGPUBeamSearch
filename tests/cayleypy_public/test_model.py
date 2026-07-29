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


def _exportable_resmlp_state(output_dim: int = 1) -> dict[str, torch.Tensor]:
    state_dict = {
        "embedding.weight": torch.zeros(3, 16),
        "input_stack.0.weight": torch.zeros(8, 32),
        "input_stack.0.bias": torch.zeros(8),
        "input_stack.1.weight": torch.ones(8),
        "input_stack.1.bias": torch.zeros(8),
        "input_stack.3.weight": torch.zeros(8, 8),
        "input_stack.3.bias": torch.zeros(8),
        "input_stack.4.weight": torch.ones(8),
        "input_stack.4.bias": torch.zeros(8),
        "res_blocks.0.lin1.weight": torch.zeros(8, 8),
        "res_blocks.0.lin1.bias": torch.zeros(8),
        "res_blocks.0.ln1.weight": torch.ones(8),
        "res_blocks.0.ln1.bias": torch.zeros(8),
        "res_blocks.0.lin2.weight": torch.zeros(8, 8),
        "res_blocks.0.lin2.bias": torch.zeros(8),
        "res_blocks.0.ln2.weight": torch.ones(8),
        "res_blocks.0.ln2.bias": torch.zeros(8),
        "head.weight": torch.zeros(output_dim, 8),
        "head.bias": torch.zeros(output_dim),
    }
    return state_dict


def test_export_checkpoint_exports_resmlp_fp16_with_sanitized_manifest(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path, _exportable_resmlp_state(output_dim=24))

    exported = export_checkpoint(
        path, tmp_path / "export", num_classes=3, state_len=2, move_count=24,
    )

    assert exported.format == "resmlp-layernorm"
    assert exported.dtype == "fp16"
    assert exported.manifest["source_weights"] == path.name
    assert exported.manifest["output_dim"] == 24


def test_export_checkpoint_is_atomic_for_malformed_supported_signature(tmp_path: Path) -> None:
    state_dict = _exportable_batchnorm_state()
    del state_dict["bn1.weight"]
    path = write_checkpoint(tmp_path, state_dict)
    out_dir = tmp_path / "export"

    with pytest.raises(ValueError, match="invalid checkpoint tensors"):
        export_checkpoint(path, out_dir, num_classes=3, state_len=4, move_count=24)

    assert not out_dir.exists()


def test_export_checkpoint_refuses_existing_output_directory(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path, _exportable_batchnorm_state())
    out_dir = tmp_path / "export"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        export_checkpoint(path, out_dir, num_classes=3, state_len=4, move_count=24)


def test_auto_cli_requires_fp16_and_selected_puzzle_contract(monkeypatch, capsys, tmp_path: Path) -> None:
    from tools import export_stream1_mlp

    checkpoint = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_stream1_mlp.py", "--weights", str(checkpoint), "--out", str(tmp_path / "export"),
            "--format", "auto", "--dtype", "bf16",
        ],
    )
    with pytest.raises(SystemExit):
        export_stream1_mlp.main()
    assert "requires --dtype fp16" in capsys.readouterr().err

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_stream1_mlp.py", "--weights", str(checkpoint), "--out", str(tmp_path / "export"),
            "--format", "auto", "--dtype", "fp16",
        ],
    )
    with pytest.raises(SystemExit):
        export_stream1_mlp.main()
    assert "requires --state-len and --move-count" in capsys.readouterr().err


def test_explicit_cli_batchnorm_mode_keeps_bf16(monkeypatch, tmp_path: Path) -> None:
    from tools import export_stream1_mlp

    observed: dict[str, object] = {}

    def record_export(weights: Path, out_dir: Path, dtype: str, num_classes: int) -> None:
        observed.update(dtype=dtype, num_classes=num_classes)

    monkeypatch.setattr(export_stream1_mlp, "export_batchnorm_folded", record_export)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_stream1_mlp.py", "--weights", str(tmp_path / "checkpoint.pt"),
            "--out", str(tmp_path / "export"), "--format", "batchnorm-folded", "--dtype", "bf16",
        ],
    )
    export_stream1_mlp.main()

    assert observed == {"dtype": "bf16", "num_classes": 120}


def test_export_checkpoint_hash_failure_leaves_no_final_destination(monkeypatch, tmp_path: Path) -> None:
    from tools.cayleypy_public import model

    path = write_checkpoint(tmp_path, _exportable_batchnorm_state())
    out_dir = tmp_path / "export"
    monkeypatch.setattr(model, "_sha256", lambda _: (_ for _ in ()).throw(OSError("source unavailable")))

    with pytest.raises(OSError, match="source unavailable"):
        export_checkpoint(path, out_dir, num_classes=3, state_len=4, move_count=24)

    assert not out_dir.exists()


def test_export_checkpoint_creates_nested_output_parent(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path, _exportable_batchnorm_state())
    out_dir = tmp_path / "nested" / "export"

    exported = export_checkpoint(path, out_dir, num_classes=3, state_len=4, move_count=24)

    assert exported.manifest["source_weights"] == path.name
    assert (out_dir / "manifest.json").is_file()

def test_checkpoint_detection_uses_safe_weights_only_load(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_load(path: Path, **kwargs: object) -> dict[str, object]:
        observed.update(path=path, **kwargs)
        return {"model": batchnorm_schema()}

    monkeypatch.setattr(torch, "load", fake_load)
    assert detect_checkpoint_format(tmp_path / "checkpoint.pt") == "batchnorm-folded"
    assert observed["weights_only"] is True


def test_public_exporters_never_disable_weights_only(monkeypatch, tmp_path: Path) -> None:
    from tools import export_stream1_mlp

    observed: list[dict[str, object]] = []

    def fake_load(path: Path, **kwargs: object) -> dict[str, object]:
        observed.append(dict(kwargs))
        state = (
            _exportable_batchnorm_state()
            if Path(path).name.startswith("batchnorm")
            else _exportable_resmlp_state()
        )
        return {"model": state}

    monkeypatch.setattr(torch, "load", fake_load)
    export_stream1_mlp.export_batchnorm_folded(
        tmp_path / "batchnorm.pt", tmp_path / "batchnorm-export", "fp16", 3,
    )
    export_stream1_mlp.export_resmlp_layernorm(
        tmp_path / "resmlp.pt", tmp_path / "resmlp-export", "fp16",
    )

    assert len(observed) == 2
    assert all(call.get("weights_only") is True for call in observed)
