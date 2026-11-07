#!/usr/bin/env python3
"""Export Stream1 piece-transformer weights into runtime weight files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Literal

import torch
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]
ExportDType = Literal["fp16", "bf16"]


def unwrap_state_dict(obj: Any) -> TensorDict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "module"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        return obj
    raise TypeError(f"unsupported checkpoint root type: {type(obj).__name__}")


def strip_prefixes(sd: TensorDict) -> TensorDict:
    prefixes = ("_orig_mod.", "module.")
    out: TensorDict = {}
    for key, value in sd.items():
        clean = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    changed = True
        out[clean] = value
    return out


def tensor_bytes(tensor: torch.Tensor, dtype: ExportDType) -> bytes:
    tensor = tensor.detach().contiguous()
    if dtype == "fp16":
        return tensor.to(torch.float16).cpu().numpy().tobytes()
    if dtype == "bf16":
        return tensor.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes()
    raise ValueError(f"unsupported export dtype: {dtype}")


def write_tensor(path: Path, tensor: torch.Tensor, dtype: ExportDType) -> None:
    path.write_bytes(tensor_bytes(tensor, dtype))


def write_int_tensor(path: Path, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    path.write_bytes(tensor.detach().contiguous().to(dtype).cpu().numpy().tobytes())


def suffix(dtype: ExportDType) -> str:
    return ".bf16" if dtype == "bf16" else ".fp16"


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_metadata(weights: Path) -> Path | None:
    root = weights.resolve()
    for parent in [root.parent, *root.parents]:
        logs = parent / "logs"
        if logs.is_dir():
            matches = sorted(logs.glob("model_*.json"))
            if matches:
                return matches[0]
    return None


def find_generator(weights: Path, group_id: int | None) -> Path | None:
    root = weights.resolve()
    name = f"p{group_id:03d}.json" if group_id is not None else None
    for parent in [root.parent, *root.parents]:
        generators = parent / "generators"
        if generators.is_dir():
            if name is not None and (generators / name).is_file():
                return generators / name
            matches = sorted(generators.glob("p*.json"))
            if matches:
                return matches[0]
    return None


def find_source_root(weights: Path) -> Path | None:
    root = weights.resolve()
    for parent in [root.parent, *root.parents]:
        if (parent / "pilgrim" / "model.py").is_file():
            return parent
    return None


def import_piece_layout(source_root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sys.path.insert(0, str(source_root))
    try:
        from pilgrim.model import make_p900_piece_layout  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise RuntimeError(f"cannot import p900 piece layout from {source_root}: {exc}") from exc
    positions, mask, types = make_p900_piece_layout()
    return positions.cpu(), mask.cpu(), types.cpu()


def require_tensor(sd: TensorDict, key: str, ndim: int | None = None) -> torch.Tensor:
    if key not in sd:
        raise ValueError(f"checkpoint is not a supported piece_transformer: missing tensor {key}")
    tensor = sd[key].detach().float().cpu()
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"tensor {key} expected ndim={ndim}, got shape={tuple(tensor.shape)}")
    return tensor


def contiguous_block_indices(sd: TensorDict) -> list[int]:
    indices = []
    prefix = "blocks."
    suffix_key = ".attn.in_proj_weight"
    for key in sd:
        if key.startswith(prefix) and key.endswith(suffix_key):
            mid = key[len(prefix):-len(suffix_key)]
            if mid.isdigit():
                indices.append(int(mid))
    if not indices:
        raise ValueError("checkpoint has no transformer blocks.*.attn.in_proj_weight tensors")
    indices = sorted(indices)
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError(f"transformer block indices are not contiguous: found={indices}")
    return indices


def parse_generator(generator_path: Path | None) -> tuple[int | None, list[str]]:
    if generator_path is None:
        return None, []
    data = json.loads(generator_path.read_text(encoding="utf-8"))
    names = data.get("names") or data.get("move_names")
    actions = data.get("actions") or data.get("moves")
    if not isinstance(names, list) or not isinstance(actions, list):
        raise ValueError(f"generator spec must contain names/actions: {generator_path}")
    if len(names) != len(actions):
        raise ValueError(f"generator name/action count mismatch: {generator_path}")
    return len(actions), [str(x) for x in names]


def export_piece_transformer(
    weights_path: Path,
    out_dir: Path,
    dtype: ExportDType,
    metadata_path: Path | None,
    generator_path: Path | None,
    source_root: Path | None,
) -> None:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = strip_prefixes(unwrap_state_dict(checkpoint))

    required = [
        "cls_token",
        "local_value_embedding.weight",
        "piece_projection.weight",
        "piece_projection.bias",
        "piece_position_embedding.weight",
        "piece_type_embedding.weight",
        "input_norm.weight",
        "input_norm.bias",
        "output_norm.weight",
        "output_norm.bias",
        "output_layer.weight",
        "output_layer.bias",
    ]
    for key in required:
        require_tensor(sd, key)

    metadata = load_json(metadata_path)
    source_root = source_root or find_source_root(weights_path)
    if source_root is None:
        raise ValueError("cannot locate megaminx-transformer source root; pass --source-root")
    piece_positions, piece_mask, piece_types = import_piece_layout(source_root)

    move_count, move_names = parse_generator(generator_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = suffix(dtype)

    cls_token = require_tensor(sd, "cls_token", 3).reshape(-1)
    d_model = int(cls_token.numel())
    local_embedding = require_tensor(sd, "local_value_embedding.weight", 2)
    piece_projection_weight = require_tensor(sd, "piece_projection.weight", 2)
    piece_projection_bias = require_tensor(sd, "piece_projection.bias", 1)
    position_embedding = require_tensor(sd, "piece_position_embedding.weight", 2)
    type_embedding = require_tensor(sd, "piece_type_embedding.weight", 2)
    output_weight = require_tensor(sd, "output_layer.weight", 2)
    output_bias = require_tensor(sd, "output_layer.bias", 1)

    if piece_projection_weight.shape[0] != d_model:
        raise ValueError("piece_projection output dim does not match cls_token d_model")
    if local_embedding.shape[1] != d_model:
        raise ValueError("local_value_embedding dim does not match d_model")
    max_piece_size = int(piece_positions.shape[1])
    num_pieces = int(piece_positions.shape[0])
    if piece_projection_weight.shape[1] != max_piece_size * d_model:
        raise ValueError("piece_projection input dim does not match piece layout")
    if local_embedding.shape[0] % max_piece_size != 0:
        raise ValueError("local_value_embedding rows must be max_piece_size * num_classes")
    num_classes = int(local_embedding.shape[0] // max_piece_size)
    state_len = int(metadata.get("state_size", 120))
    if int(piece_positions.max().item()) >= state_len:
        raise ValueError("piece layout references state position outside state_len")
    if position_embedding.shape != (num_pieces, d_model):
        raise ValueError("piece_position_embedding shape does not match piece layout")
    if type_embedding.shape[1] != d_model:
        raise ValueError("piece_type_embedding dim does not match d_model")
    if output_weight.shape[1] != d_model:
        raise ValueError("output_layer input dim does not match d_model")
    output_dim = int(output_weight.shape[0])
    if move_count is not None and output_dim != move_count:
        raise ValueError(f"output_dim={output_dim} does not match generator move_count={move_count}")

    slot_rows = []
    for slot in range(max_piece_size):
        start = slot * d_model
        end = (slot + 1) * d_model
        slot_projection = piece_projection_weight[:, start:end]
        token_start = slot * num_classes
        token_end = (slot + 1) * num_classes
        slot_embedding = local_embedding[token_start:token_end]
        slot_rows.append(F.linear(slot_embedding, slot_projection))
    fast_slot_projected = torch.stack(slot_rows, dim=0).contiguous()
    fast_piece_static = position_embedding[torch.arange(num_pieces)]
    fast_piece_static = fast_piece_static + type_embedding[piece_types.long()]
    fast_piece_static = fast_piece_static + piece_projection_bias.view(1, -1)

    write_tensor(out_dir / f"fast_slot_projected{ext}", fast_slot_projected, dtype)
    write_tensor(out_dir / f"fast_piece_static{ext}", fast_piece_static, dtype)
    write_tensor(out_dir / f"cls_token{ext}", cls_token, dtype)
    write_tensor(out_dir / f"input_ln_gamma{ext}", require_tensor(sd, "input_norm.weight", 1), dtype)
    write_tensor(out_dir / f"input_ln_beta{ext}", require_tensor(sd, "input_norm.bias", 1), dtype)
    write_int_tensor(out_dir / "piece_positions.u16", piece_positions, torch.int16)
    write_int_tensor(out_dir / "piece_mask.u8", piece_mask, torch.uint8)
    write_int_tensor(out_dir / "piece_types.u8", piece_types, torch.uint8)

    blocks = contiguous_block_indices(sd)
    ff_dim = None
    for block in blocks:
        prefix = f"blocks.{block}"
        qkv_weight = require_tensor(sd, f"{prefix}.attn.in_proj_weight", 2)
        qkv_bias = require_tensor(sd, f"{prefix}.attn.in_proj_bias", 1)
        out_proj_weight = require_tensor(sd, f"{prefix}.attn.out_proj.weight", 2)
        out_proj_bias = require_tensor(sd, f"{prefix}.attn.out_proj.bias", 1)
        ff0_weight = require_tensor(sd, f"{prefix}.ff.0.weight", 2)
        ff0_bias = require_tensor(sd, f"{prefix}.ff.0.bias", 1)
        ff3_weight = require_tensor(sd, f"{prefix}.ff.3.weight", 2)
        ff3_bias = require_tensor(sd, f"{prefix}.ff.3.bias", 1)
        if qkv_weight.shape != (3 * d_model, d_model) or qkv_bias.shape != (3 * d_model,):
            raise ValueError(f"block {block} qkv shape mismatch")
        if out_proj_weight.shape != (d_model, d_model) or out_proj_bias.shape != (d_model,):
            raise ValueError(f"block {block} attention out projection shape mismatch")
        if ff0_weight.shape[1] != d_model or ff3_weight.shape[0] != d_model or ff3_weight.shape[1] != ff0_weight.shape[0]:
            raise ValueError(f"block {block} FFN shape mismatch")
        ff_dim = int(ff0_weight.shape[0]) if ff_dim is None else ff_dim
        if int(ff0_weight.shape[0]) != ff_dim:
            raise ValueError("all transformer blocks must have the same ff_dim")

        write_tensor(out_dir / f"block{block}_ln1_gamma{ext}", require_tensor(sd, f"{prefix}.norm1.weight", 1), dtype)
        write_tensor(out_dir / f"block{block}_ln1_beta{ext}", require_tensor(sd, f"{prefix}.norm1.bias", 1), dtype)
        write_tensor(out_dir / f"block{block}_qkv_weight_hxk{ext}", qkv_weight.t(), dtype)
        write_tensor(out_dir / f"block{block}_qkv_bias{ext}", qkv_bias, dtype)
        write_tensor(out_dir / f"block{block}_attn_out_weight_hxk{ext}", out_proj_weight.t(), dtype)
        write_tensor(out_dir / f"block{block}_attn_out_bias{ext}", out_proj_bias, dtype)
        write_tensor(out_dir / f"block{block}_ln2_gamma{ext}", require_tensor(sd, f"{prefix}.norm2.weight", 1), dtype)
        write_tensor(out_dir / f"block{block}_ln2_beta{ext}", require_tensor(sd, f"{prefix}.norm2.bias", 1), dtype)
        write_tensor(out_dir / f"block{block}_ff1_weight_hxk{ext}", ff0_weight.t(), dtype)
        write_tensor(out_dir / f"block{block}_ff1_bias{ext}", ff0_bias, dtype)
        write_tensor(out_dir / f"block{block}_ff2_weight_hxk{ext}", ff3_weight.t(), dtype)
        write_tensor(out_dir / f"block{block}_ff2_bias{ext}", ff3_bias, dtype)

    write_tensor(out_dir / f"output_ln_gamma{ext}", require_tensor(sd, "output_norm.weight", 1), dtype)
    write_tensor(out_dir / f"output_ln_beta{ext}", require_tensor(sd, "output_norm.bias", 1), dtype)
    write_tensor(out_dir / f"output_weight_hxk{ext}", output_weight.t(), dtype)
    write_tensor(out_dir / f"output_bias{ext}", output_bias, dtype)

    nhead = int(metadata.get("transformer_heads", 0))
    if nhead <= 0:
        # Infer from the known Kaggle checkpoint if metadata is unavailable.
        nhead = 8
    if d_model % nhead != 0:
        raise ValueError(f"d_model={d_model} is not divisible by nhead={nhead}")
    manifest = {
        "backend": "piece_transformer",
        "model_arch": metadata.get("model_arch", "piece_transformer"),
        "state_len": state_len,
        "num_classes": num_classes,
        "move_count": move_count if move_count is not None else output_dim,
        "output_dim": output_dim,
        "dtype": dtype,
        "piece_layout": metadata.get("piece_layout", "p900"),
        "piece_embed_mode": metadata.get("piece_embed_mode", "full_s120"),
        "num_pieces": num_pieces,
        "max_piece_size": max_piece_size,
        "seq_len": num_pieces + 1,
        "d_model": d_model,
        "nhead": nhead,
        "head_dim": d_model // nhead,
        "num_layers": len(blocks),
        "ff_dim": ff_dim,
        "activation": metadata.get("transformer_activation", "silu"),
        "pooling": metadata.get("transformer_pooling", "cls"),
        "input_embedding": "fast_slot_projected",
        "weight_layout": "hxk files are input-major [input_cols, output_cols]",
        "source_weights": str(weights_path),
        "source_metadata": str(metadata_path) if metadata_path is not None else "",
        "source_generator": str(generator_path) if generator_path is not None else "",
        "source_model_id": metadata.get("model_id"),
        "move_names": move_names,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "stream1_transformer_export_done"
        f" out_dir={out_dir}"
        f" dtype={dtype}"
        f" state_len={state_len}"
        f" num_classes={num_classes}"
        f" move_count={manifest['move_count']}"
        f" d_model={d_model}"
        f" layers={len(blocks)}"
        f" heads={nhead}"
        f" ff_dim={ff_dim}"
        f" num_pieces={num_pieces}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--generators", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    args = parser.parse_args()

    metadata = args.metadata or find_metadata(args.weights)
    metadata_json = load_json(metadata)
    group_id = int(metadata_json["group_id"]) if "group_id" in metadata_json else None
    generators = args.generators or find_generator(args.weights, group_id)
    export_piece_transformer(args.weights, args.out, args.dtype, metadata, generators, args.source_root)


if __name__ == "__main__":
    main()
