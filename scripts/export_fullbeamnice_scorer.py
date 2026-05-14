"""Export the FullBeamNice Q model as a TorchScript scorer for beam_engine."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


CURRENT_ACTION_NAMES = [
    "-B", "-BL", "-BR", "-D", "-DL", "-DR",
    "-F", "-FL", "-FR", "-L", "-R", "-U",
    "B", "BL", "BR", "D", "DL", "DR",
    "F", "FL", "FR", "L", "R", "U",
]


class LegacyCompatibleEmbeddingBagLinear(nn.Module):
    def __init__(self, state_size: int, num_classes: int, out_features: int, bias: bool = True):
        super().__init__()
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.out_features = int(out_features)
        self.in_features = self.state_size * self.num_classes
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.register_buffer(
            "position_offsets",
            torch.arange(self.state_size, dtype=torch.int64) * self.num_classes,
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        token_ids = indices + self.position_offsets.unsqueeze(0)
        flat_ids = token_ids.reshape(-1)
        offsets = torch.arange(0, flat_ids.numel(), self.state_size, dtype=torch.int64, device=indices.device)
        out = F.embedding_bag(flat_ids, self.weight, offsets, mode="sum")
        return out if self.bias is None else out + self.bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "weight"
        if key in state_dict:
            loaded = state_dict[key]
            legacy_shape = (self.out_features, self.in_features)
            native_shape = (self.in_features, self.out_features)
            if loaded.shape == legacy_shape:
                state_dict[key] = loaded.transpose(0, 1)
            elif loaded.shape != native_shape:
                error_msgs.append(f"size mismatch for {key}: got {tuple(loaded.shape)}")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return self.relu(x + residual)


class Pilgrim(nn.Module):
    def __init__(
        self,
        state_size: int,
        hd1: int,
        hd2: int,
        nrd: int,
        output_dim: int,
        dropout_rate: float,
        num_classes: int,
    ):
        super().__init__()
        self.dtype = torch.float32
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.output_dim = int(output_dim)
        self.z_add = 0
        self.input_layer = LegacyCompatibleEmbeddingBagLinear(state_size, num_classes, hd1)
        self.bn1 = nn.BatchNorm1d(hd1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer = nn.Linear(hd1, hd2)
        self.bn2 = nn.BatchNorm1d(hd2)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hd2, dropout_rate) for _ in range(nrd)])
        self.output_layer = nn.Linear(hd2, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(z.long() + self.z_add).to(self.dtype)
        x = self.dropout(self.relu(self.bn1(x)))
        x = self.dropout(self.relu(self.bn2(self.hidden_layer(x))))
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


class FullBeamNiceCurrentOrderScorer(nn.Module):
    def __init__(self, base: nn.Module, action_permutation: Sequence[int], score_scale: float, score_bias: float):
        super().__init__()
        self.base = base
        self.score_scale = float(score_scale)
        self.score_bias = float(score_bias)
        self.register_buffer("action_permutation", torch.tensor(list(action_permutation), dtype=torch.long), persistent=True)

    def forward(self, states_u8: torch.Tensor) -> torch.Tensor:
        q_full_order = self.base(states_u8)
        q_current_order = q_full_order.index_select(1, self.action_permutation)
        score = torch.clamp(torch.round(self.score_bias - q_current_order.float() * self.score_scale), 0.0, 65535.0).to(torch.int32)
        return torch.where(score >= 32768, score - 65536, score).to(torch.int16)


def load_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def action_permutation_for_current_order(generator_path: Path) -> list[int]:
    spec = load_info(generator_path)
    names = list(spec["names"])
    out: list[int] = []
    for name in CURRENT_ACTION_NAMES:
        full_name = name[1:] + "'" if name.startswith("-") else name
        out.append(names.index(full_name))
    return out


def build_model(info: dict, target_path: Path) -> Pilgrim:
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    return Pilgrim(
        state_size=int(target.numel()),
        num_classes=int(torch.unique(target).numel()),
        output_dim=int(info["n_gens"]),
        dropout_rate=float(info.get("dropout", 0.0)),
        hd1=int(info["hd1"]),
        hd2=int(info["hd2"]),
        nrd=int(info["nrd"]),
    )


def export_module(module: nn.Module, example: torch.Tensor, path: Path) -> None:
    traced = torch.jit.trace(module, example, strict=False)
    traced = torch.jit.freeze(traced)
    traced.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullbeamnice-dir", default=str(PROJECT_DIR / "FullBeamNice"))
    ap.add_argument("--out-dir", default=str(PROJECT_DIR / "runtime" / "fullbeamnice_scorers"))
    ap.add_argument("--copies", type=int, default=int(os.environ.get("INFERENCE_PARALLELISM", "1")))
    ap.add_argument("--physical-copies", type=int, default=int(os.environ.get("TORCHSCRIPT_PHYSICAL_COPIES", "1")))
    ap.add_argument("--score-scale", type=float, default=float(os.environ.get("FULLBEAMNICE_SCORE_SCALE", "1024.0")))
    ap.add_argument("--score-bias", type=float, default=float(os.environ.get("FULLBEAMNICE_SCORE_BIAS", "65535.0")))
    args = ap.parse_args()

    root = Path(args.fullbeamnice_dir)
    generator_path = root / "generators" / "p900.json"
    target_path = root / "targets" / "p900-t000.pt"
    metadata_path = root / "logs" / "model_p900-t000-q-sym_1777988767.json"
    weights_path = root / "weights" / "p900-t000-q-sym_1777988767_best.pth"

    info = load_info(metadata_path)
    action_perm = action_permutation_for_current_order(generator_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    example = torch.zeros((16, 120), dtype=torch.uint8, device=device)
    paths: list[str] = []

    physical_copies = max(1, int(args.physical_copies))
    for i in range(physical_copies):
        model = build_model(info, target_path)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        if device.type == "cuda":
            model.half()
            model.dtype = torch.float16
        model.to(device)
        wrapped = FullBeamNiceCurrentOrderScorer(model, action_perm, args.score_scale, args.score_bias).eval()
        wrapped.to(device)
        path = out_dir / f"fullbeamnice_q_to_score_copy{i:02d}.ts"
        export_module(wrapped, example, path)
        paths.append(str(path))

    print("TORCHSCRIPT_SCORER_PATHS=" + os.pathsep.join(paths))
    print("INFERENCE_BACKEND=torchscript_ensemble")
    print(f"INFERENCE_PARALLELISM={args.copies}")
    print(json.dumps({
        "model": info["model_name"],
        "model_id": info["model_id"],
        "inference_lanes": int(args.copies),
        "physical_copies": physical_copies,
        "score_scale": args.score_scale,
        "score_bias": args.score_bias,
        "action_permutation": action_perm,
        "paths": paths,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
