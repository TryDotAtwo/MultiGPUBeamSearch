"""Standalone FullBeamNice action24 inference without TorchScript.

This module is an intermediate validation target before C++/CUDA integration:
it loads the existing FullBeamNice weights, folds BatchNorm into Linear layers,
and runs the exact 24-output scorer with explicit tensor operations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

STATE_LEN = 120
STATE_STORAGE_LEN = 128
STATE_VALUE_PAD = 128

from export_fullbeamnice_scorer import (  # noqa: E402
    FullBeamNiceCurrentOrderScorer,
    Pilgrim,
    action_permutation_for_current_order,
    build_model,
    load_info,
)


@dataclass
class StaticFullBeamNiceWeights:
    embed_w_t: torch.Tensor
    embed_bias: torch.Tensor
    hidden_w_t: torch.Tensor
    hidden_bias: torch.Tensor
    res0_fc1_w_t: torch.Tensor
    res0_fc1_bias: torch.Tensor
    res0_fc2_w_t: torch.Tensor
    res0_fc2_bias: torch.Tensor
    res1_fc1_w_t: torch.Tensor
    res1_fc1_bias: torch.Tensor
    res1_fc2_w_t: torch.Tensor
    res1_fc2_bias: torch.Tensor
    out_w_t: torch.Tensor
    out_bias: torch.Tensor
    action_perm: torch.Tensor
    state_size: int
    num_classes: int
    score_scale: float
    score_bias: float


def _fold_bn_linear(
    weight: torch.Tensor,
    bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = bn_weight.float() / torch.sqrt(running_var.float() + eps)
    folded_weight = weight.float() * scale[:, None]
    folded_bias = (bias.float() - running_mean.float()) * scale + bn_bias.float()
    return folded_weight.contiguous(), folded_bias.contiguous()


def load_static_weights(
    fullbeamnice_dir: Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    score_scale: float = 1024.0,
    score_bias: float = 65535.0,
) -> StaticFullBeamNiceWeights:
    root = Path(fullbeamnice_dir)
    metadata_path = root / "logs" / "model_p900-t000-q-sym_1777988767.json"
    weights_path = root / "weights" / "p900-t000-q-sym_1777988767_best.pth"
    target_path = root / "targets" / "p900-t000.pt"
    generator_path = root / "generators" / "p900.json"

    info = load_info(metadata_path)
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    logical_state_size = int(target.numel())
    logical_num_classes = int(torch.unique(target).numel())

    embed_w = sd["input_layer.weight"].float()
    if embed_w.shape[0] == int(info["hd1"]):
        embed_w = embed_w.t().contiguous()
    if logical_state_size == STATE_LEN and logical_num_classes <= STATE_VALUE_PAD:
        expanded_embed_w = torch.zeros(
            (STATE_STORAGE_LEN * STATE_VALUE_PAD, embed_w.shape[1]),
            dtype=embed_w.dtype,
            device=embed_w.device,
        )
        for pos in range(STATE_LEN):
            old_start = pos * logical_num_classes
            new_start = pos * STATE_VALUE_PAD
            expanded_embed_w[new_start:new_start + logical_num_classes].copy_(
                embed_w[old_start:old_start + logical_num_classes]
            )
        embed_w = expanded_embed_w.contiguous()
    embed_bias = sd["input_layer.bias"].float()
    bn1_scale = sd["bn1.weight"].float() / torch.sqrt(sd["bn1.running_var"].float() + 1e-5)
    embed_w = embed_w * bn1_scale.unsqueeze(0)
    embed_bias = (embed_bias - sd["bn1.running_mean"].float()) * bn1_scale + sd["bn1.bias"].float()

    hidden_w, hidden_bias = _fold_bn_linear(
        sd["hidden_layer.weight"],
        sd["hidden_layer.bias"],
        sd["bn2.weight"],
        sd["bn2.bias"],
        sd["bn2.running_mean"],
        sd["bn2.running_var"],
    )

    folded: dict[str, torch.Tensor] = {}
    for block in range(int(info["nrd"])):
        prefix = f"residual_blocks.{block}"
        for fc_name, bn_name in (("fc1", "bn1"), ("fc2", "bn2")):
            w, b = _fold_bn_linear(
                sd[f"{prefix}.{fc_name}.weight"],
                sd[f"{prefix}.{fc_name}.bias"],
                sd[f"{prefix}.{bn_name}.weight"],
                sd[f"{prefix}.{bn_name}.bias"],
                sd[f"{prefix}.{bn_name}.running_mean"],
                sd[f"{prefix}.{bn_name}.running_var"],
            )
            folded[f"res{block}_{fc_name}_w_t"] = w.t().contiguous()
            folded[f"res{block}_{fc_name}_bias"] = b

    action_perm = torch.tensor(action_permutation_for_current_order(generator_path), dtype=torch.long)
    d = torch.device(device)
    weight_dtype = dtype
    return StaticFullBeamNiceWeights(
        embed_w_t=embed_w.contiguous().to(device=d, dtype=weight_dtype),
        embed_bias=embed_bias.contiguous().to(device=d, dtype=weight_dtype),
        hidden_w_t=hidden_w.t().contiguous().to(device=d, dtype=weight_dtype),
        hidden_bias=hidden_bias.contiguous().to(device=d, dtype=weight_dtype),
        res0_fc1_w_t=folded["res0_fc1_w_t"].to(device=d, dtype=weight_dtype),
        res0_fc1_bias=folded["res0_fc1_bias"].to(device=d, dtype=weight_dtype),
        res0_fc2_w_t=folded["res0_fc2_w_t"].to(device=d, dtype=weight_dtype),
        res0_fc2_bias=folded["res0_fc2_bias"].to(device=d, dtype=weight_dtype),
        res1_fc1_w_t=folded["res1_fc1_w_t"].to(device=d, dtype=weight_dtype),
        res1_fc1_bias=folded["res1_fc1_bias"].to(device=d, dtype=weight_dtype),
        res1_fc2_w_t=folded["res1_fc2_w_t"].to(device=d, dtype=weight_dtype),
        res1_fc2_bias=folded["res1_fc2_bias"].to(device=d, dtype=weight_dtype),
        out_w_t=sd["output_layer.weight"].float().t().contiguous().to(device=d, dtype=weight_dtype),
        out_bias=sd["output_layer.bias"].float().contiguous().to(device=d, dtype=weight_dtype),
        action_perm=action_perm.to(d),
        state_size=STATE_STORAGE_LEN,
        num_classes=STATE_VALUE_PAD,
        score_scale=float(score_scale),
        score_bias=float(score_bias),
    )


def static_forward_q(states_u8: torch.Tensor, w: StaticFullBeamNiceWeights) -> torch.Tensor:
    if states_u8.dim() == 2 and states_u8.size(1) == STATE_LEN and int(w.state_size) == STATE_STORAGE_LEN:
        padded = torch.zeros((states_u8.size(0), STATE_STORAGE_LEN), dtype=states_u8.dtype, device=states_u8.device)
        padded[:, :STATE_LEN].copy_(states_u8)
        states_u8 = padded
    states = states_u8.to(device=w.embed_w_t.device, dtype=torch.long)
    offsets = torch.arange(w.state_size, device=states.device, dtype=torch.long) * int(w.num_classes)
    token_ids = states + offsets.unsqueeze(0)
    x = w.embed_w_t.index_select(0, token_ids.reshape(-1)).reshape(states.size(0), w.state_size, -1).sum(dim=1)
    x = torch.relu(x + w.embed_bias)
    x = torch.relu(x @ w.hidden_w_t + w.hidden_bias)

    residual = x
    x = torch.relu(x @ w.res0_fc1_w_t + w.res0_fc1_bias)
    x = torch.relu(x @ w.res0_fc2_w_t + w.res0_fc2_bias + residual)

    residual = x
    x = torch.relu(x @ w.res1_fc1_w_t + w.res1_fc1_bias)
    x = torch.relu(x @ w.res1_fc2_w_t + w.res1_fc2_bias + residual)

    return x @ w.out_w_t + w.out_bias


def static_forward_scores(states_u8: torch.Tensor, w: StaticFullBeamNiceWeights) -> torch.Tensor:
    q = static_forward_q(states_u8, w)
    q_current = q.index_select(1, w.action_perm)
    score = torch.clamp(torch.round(w.score_bias - q_current.float() * w.score_scale), 0.0, 65535.0).to(torch.int32)
    return torch.where(score >= 32768, score - 65536, score).to(torch.int16)


def build_reference_scorer(fullbeamnice_dir: Path, device: torch.device | str = "cpu") -> nn.Module:
    root = Path(fullbeamnice_dir)
    info = load_info(root / "logs" / "model_p900-t000-q-sym_1777988767.json")
    target_path = root / "targets" / "p900-t000.pt"
    weights_path = root / "weights" / "p900-t000-q-sym_1777988767_best.pth"
    generator_path = root / "generators" / "p900.json"
    model = build_model(info, target_path)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return FullBeamNiceCurrentOrderScorer(
        model,
        action_permutation_for_current_order(generator_path),
        score_scale=1024.0,
        score_bias=65535.0,
    ).eval().to(device)


def compare_static_to_reference(fullbeamnice_dir: Path, batch: int, device: str, dtype: str) -> dict[str, float | int | str]:
    d = torch.device(device)
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    w = load_static_weights(fullbeamnice_dir, device=d, dtype=torch_dtype)
    ref = build_reference_scorer(fullbeamnice_dir, device=d)
    states = torch.randint(0, STATE_LEN, (batch, STATE_LEN), dtype=torch.uint8, device=d)
    with torch.no_grad():
        static_scores = static_forward_scores(states, w)
        ref_scores = ref(states)
        q_static = static_forward_q(states, w)
        q_ref_full = ref.base(states)
        q_ref = q_ref_full.index_select(1, w.action_perm)
        q_static_current = q_static.index_select(1, w.action_perm)
    diff = (static_scores.to(torch.int32) - ref_scores.to(torch.int32)).abs()
    q_diff = (q_static_current.float() - q_ref.float()).abs()
    return {
        "batch": int(batch),
        "device": str(d),
        "dtype": dtype,
        "state_size": int(w.state_size),
        "num_classes": int(w.num_classes),
        "max_abs_score_diff": int(diff.max().item()),
        "mean_abs_score_diff": float(diff.float().mean().item()),
        "max_abs_q_diff": float(q_diff.max().item()),
        "mean_abs_q_diff": float(q_diff.mean().item()),
        "q_min": float(q_static.min().item()),
        "q_max": float(q_static.max().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullbeamnice-dir", default=str(PROJECT_DIR / "FullBeamNice"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()
    result = compare_static_to_reference(Path(args.fullbeamnice_dir), args.batch, args.device, args.dtype)
    print("STATIC_FULLBEAMNICE_COMPARE " + json.dumps(result, sort_keys=True))
    max_allowed = 256 if args.dtype == "fp16" else 1
    if result["max_abs_score_diff"] > max_allowed:
        raise SystemExit(f"score mismatch: {result}")


if __name__ == "__main__":
    main()
