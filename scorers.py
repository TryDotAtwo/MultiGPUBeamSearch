"""
TorchScript-compatible scorer modules for beam_engine.

Contract for C++ TorchScript ensemble backend:
  input:  uint8 tensor [B, 120] on CUDA
  output: int16 tensor [B, 24] with raw uint16 bits interpreted by C++

The int16 output stores uint16 scores by bit pattern:
  0..32767   -> same signed int16 value
  32768..65535 -> signed int16 value q - 65536
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class MLPActionScorer(nn.Module):
    def __init__(self, input_dim: int = 120, hidden_sizes: Sequence[int] = (1024, 256), fanout: int = 24):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.GELU())
            prev = int(h)
        layers.append(nn.Linear(prev, fanout))
        self.net = nn.Sequential(*layers)

    def forward(self, states_u8: torch.Tensor) -> torch.Tensor:
        x = states_u8.to(torch.float32) / 119.0
        return self.net(x)


class QuantizedActionScorer(nn.Module):
    def __init__(self, base: nn.Module, score_mode: str = "sigmoid_u16"):
        super().__init__()
        self.base = base
        self.score_mode = score_mode

    def forward(self, states_u8: torch.Tensor) -> torch.Tensor:
        y = self.base(states_u8)
        if self.score_mode == "sigmoid_u16":
            q = torch.clamp(torch.round(torch.sigmoid(y) * 65535.0), 0.0, 65535.0).to(torch.int32)
        elif self.score_mode == "float_u16":
            q = torch.clamp(torch.round(y), 0.0, 65535.0).to(torch.int32)
        else:
            raise RuntimeError("unsupported score_mode")
        q_signed = torch.where(q >= 32768, q - 65536, q).to(torch.int16)
        return q_signed


def make_default_mlp_scorer(hidden_sizes: Sequence[int] = (1024, 256), fanout: int = 24) -> nn.Module:
    return QuantizedActionScorer(MLPActionScorer(hidden_sizes=hidden_sizes, fanout=fanout))
