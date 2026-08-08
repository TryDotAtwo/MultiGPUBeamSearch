"""Public, fail-closed Stream1 MLP checkpoint export contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Literal, Mapping

import torch

from tools.export_stream1_mlp import (
    export_batchnorm_folded,
    export_resmlp_layernorm,
    strip_orig_mod,
    unwrap_state_dict,
)

CheckpointFormat = Literal["batchnorm-folded", "resmlp-layernorm"]
BN_REQUIRED = frozenset({
    "input_layer.weight", "hidden_layer.weight", "output_layer.weight",
    "bn1.running_mean", "bn2.running_mean",
})
LN_REQUIRED = frozenset({
    "embedding.weight", "input_stack.0.weight", "input_stack.1.weight",
    "input_stack.3.weight", "head.weight",
})


@dataclass(frozen=True)
class ExportedModel:
    format: CheckpointFormat
    dtype: Literal["fp16"]
    checkpoint_sha256: str
    manifest: Mapping[str, object]


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    return strip_orig_mod(unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=True)))


def _has_residual_key(state_dict: Mapping[str, torch.Tensor], pattern: str) -> bool:
    return any(re.match(pattern, key) for key in state_dict)


def detect_checkpoint_format(path: Path) -> CheckpointFormat:
    """Identify one supported checkpoint schema from its tensor-key signature."""
    state_dict = _state_dict(path)
    batchnorm = BN_REQUIRED.issubset(state_dict) and _has_residual_key(
        state_dict, r"^residual_blocks\.\d+\.fc1\.weight$"
    )
    resmlp = LN_REQUIRED.issubset(state_dict) and _has_residual_key(
        state_dict, r"^res_blocks\.\d+\.lin1\.weight$"
    )
    if batchnorm and resmlp:
        raise ValueError("ambiguous checkpoint schema: matches batchnorm-folded and resmlp-layernorm")
    if batchnorm:
        return "batchnorm-folded"
    if resmlp:
        return "resmlp-layernorm"
    raise ValueError("unsupported checkpoint schema")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_manifest(
    path: Path, *, state_len: int, num_classes: int, move_count: int,
) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("export manifest must be an object")
    if manifest.get("state_len") != state_len:
        raise ValueError(f"export manifest state_len mismatch: {manifest.get('state_len')!r}")
    if manifest.get("num_classes") != num_classes:
        raise ValueError(f"export manifest num_classes mismatch: {manifest.get('num_classes')!r}")
    if manifest.get("normalization") not in {"batchnorm_folded", "layernorm"}:
        raise ValueError("export manifest has unsupported normalization")
    output_dim = manifest.get("output_dim")
    allowed = {1, move_count}
    if not isinstance(output_dim, int) or output_dim not in allowed:
        raise ValueError("export manifest output_dim must be 1 or move_count")
    return manifest


def export_checkpoint(
    path: Path, out_dir: Path, num_classes: int, *, state_len: int, move_count: int,
) -> ExportedModel:
    """Export one supported checkpoint atomically with public metadata only."""
    if out_dir.exists():
        raise ValueError(f"export output directory already exists: {out_dir}")
    checkpoint_sha256 = _sha256(path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    format = detect_checkpoint_format(path)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        try:
            if format == "batchnorm-folded":
                export_batchnorm_folded(path, temporary_dir, dtype="fp16", num_classes=num_classes)
            else:
                export_resmlp_layernorm(path, temporary_dir, dtype="fp16")
        except KeyError as error:
            raise ValueError(f"invalid checkpoint tensors: missing {error.args[0]!r}") from error
        manifest_path = temporary_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_weights"] = path.name
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validated_manifest = _validated_manifest(
            manifest_path, state_len=state_len, num_classes=num_classes, move_count=move_count,
        )
        temporary_dir.replace(out_dir)
        return ExportedModel(
            format=format,
            dtype="fp16",
            checkpoint_sha256=checkpoint_sha256,
            manifest=validated_manifest,
        )
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
