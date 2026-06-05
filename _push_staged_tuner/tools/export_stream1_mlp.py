#!/usr/bin/env python3
"""Export QMLP-style Stream1 PyTorch weights into folded FP16 runtime files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Dict, Tuple

import torch


TensorDict = Dict[str, torch.Tensor]


def unwrap_state_dict(obj) -> TensorDict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "module"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"unsupported checkpoint root type: {type(obj).__name__}")


def fold_linear_bn(sd: TensorDict, linear: str, bn: str) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = sd[f"{linear}.weight"].detach().float()
    bias = sd.get(f"{linear}.bias")
    if bias is None:
        bias = torch.zeros(weight.shape[0], dtype=torch.float32)
    else:
        bias = bias.detach().float()
    gamma = sd[f"{bn}.weight"].detach().float()
    beta = sd[f"{bn}.bias"].detach().float()
    mean = sd[f"{bn}.running_mean"].detach().float()
    var = sd[f"{bn}.running_var"].detach().float()
    scale = gamma / torch.sqrt(var + 1e-5)
    return weight * scale[:, None], (bias - mean) * scale + beta


def linear_weight_bias(sd: TensorDict, linear: str) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = sd[f"{linear}.weight"].detach().float()
    bias = sd.get(f"{linear}.bias")
    if bias is None:
        bias = torch.zeros(weight.shape[0], dtype=torch.float32)
    else:
        bias = bias.detach().float()
    return weight, bias


def write_hxk(path: Path, weight_out_in: torch.Tensor) -> None:
    data = weight_out_in.t().contiguous().to(torch.float16).cpu().numpy()
    path.write_bytes(data.tobytes())


def write_bias(path: Path, bias: torch.Tensor) -> None:
    data = bias.contiguous().to(torch.float16).cpu().numpy()
    path.write_bytes(data.tobytes())


def infer_residual_count(sd: TensorDict) -> int:
    blocks = set()
    pattern = re.compile(r"^residual_blocks\.(\d+)\.fc1\.weight$")
    for key in sd:
        match = pattern.match(key)
        if match:
            blocks.add(int(match.group(1)))
    if not blocks:
        raise ValueError("no residual_blocks.*.fc1.weight tensors found")
    expected = set(range(max(blocks) + 1))
    if blocks != expected:
        raise ValueError(f"residual block indices are not contiguous: found={sorted(blocks)}")
    return max(blocks) + 1


def export(weights_path: Path, out_dir: Path) -> None:
    checkpoint = torch.load(weights_path, map_location="cpu")
    sd = unwrap_state_dict(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_weight = sd["input_layer.weight"]
    hidden_weight = sd["hidden_layer.weight"]
    output_weight = sd["output_layer.weight"]
    hd1, input_dim = input_weight.shape
    hd2, hidden_in = hidden_weight.shape
    output_dim, output_in = output_weight.shape
    if hidden_in != hd1:
        raise ValueError(f"hidden_layer input mismatch: hidden_in={hidden_in} hd1={hd1}")
    if output_in != hd2:
        raise ValueError(f"output_layer input mismatch: output_in={output_in} hd2={hd2}")
    if input_dim % 120 != 0:
        raise ValueError(f"input_dim is not divisible by 120 classes: input_dim={input_dim}")
    state_len = input_dim // 120
    num_classes = 120
    residual_count = infer_residual_count(sd)

    weight, bias = fold_linear_bn(sd, "input_layer", "bn1")
    write_hxk(out_dir / "input_weight_hxk.fp16", weight)
    write_bias(out_dir / "input_bias.fp16", bias)

    weight, bias = fold_linear_bn(sd, "hidden_layer", "bn2")
    write_hxk(out_dir / "hidden_weight_hxk.fp16", weight)
    write_bias(out_dir / "hidden_bias.fp16", bias)

    for block in range(residual_count):
        prefix = f"residual_blocks.{block}"
        out_prefix = f"residual{block}"
        weight, bias = fold_linear_bn(sd, f"{prefix}.fc1", f"{prefix}.bn1")
        write_hxk(out_dir / f"{out_prefix}_fc1_weight_hxk.fp16", weight)
        write_bias(out_dir / f"{out_prefix}_fc1_bias.fp16", bias)
        weight, bias = fold_linear_bn(sd, f"{prefix}.fc2", f"{prefix}.bn2")
        write_hxk(out_dir / f"{out_prefix}_fc2_weight_hxk.fp16", weight)
        write_bias(out_dir / f"{out_prefix}_fc2_bias.fp16", bias)

    weight, bias = linear_weight_bias(sd, "output_layer")
    write_hxk(out_dir / "output_weight_hxk.fp16", weight)
    write_bias(out_dir / "output_bias.fp16", bias)

    manifest = {
        "state_len": state_len,
        "num_classes": num_classes,
        "hd1": hd1,
        "hd2": hd2,
        "nrd": residual_count,
        "output_dim": output_dim,
        "dtype": "fp16",
        "layout": "row-major input activations times weight_hxk",
        "batchnorm": "folded into preceding linear weights",
        "embeddingbag": "removed from runtime; input layer exported as position-class table",
        "source_weights": str(weights_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "stream1_export_done"
        f" out_dir={out_dir}"
        f" hd1={hd1}"
        f" hd2={hd2}"
        f" nrd={residual_count}"
        f" output_dim={output_dim}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(args.weights, args.out)


if __name__ == "__main__":
    main()
