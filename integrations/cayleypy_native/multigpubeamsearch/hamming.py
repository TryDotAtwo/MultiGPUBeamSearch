"""Exact, untrained Hamming heuristic in the existing native MLP format.

Only channel zero carries the mismatch count. ReLU and a zero residual
preserve that nonnegative integer; padding channels carry zero throughout.
This is a fixed heuristic, never a replacement for a supplied predictor.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct


def write_hamming_artifact(contract, run_dir: Path) -> Path:
    size = contract.state_len
    classes = max(size, contract.num_classes)
    width = 8
    target = Path(run_dir) / "weights"
    target.mkdir(parents=True, exist_ok=False)

    def blob(name, values):
        with (target / (name + ".fp16")).open("xb") as output:
            for value in values:
                output.write(struct.pack("<e", value))

    # Native weights are input-major [K,H], not PyTorch [H,K].
    blob("input_weight_hxk", (
        -1.0 if channel == 0 and label == contract.center[position] else 0.0
        for position in range(size) for label in range(classes) for channel in range(width)
    ))
    blob("input_bias", (float(size) if channel == 0 else 0.0 for channel in range(width)))
    blob("hidden_weight_hxk", (
        1.0 if incoming == 0 and outgoing == 0 else 0.0
        for incoming in range(width) for outgoing in range(width)
    ))
    blob("hidden_bias", [0.0] * width)
    for fc in (1, 2):
        blob(f"residual0_fc{fc}_weight_hxk", [0.0] * (width * width))
        blob(f"residual0_fc{fc}_bias", [0.0] * width)
    blob("output_weight_hxk", [1.0] + [0.0] * (width - 1))
    blob("output_bias", [0.0])
    manifest = {
        "state_len": size, "num_classes": classes,
        "hidden1": width, "hidden2": width, "residual_count": 1,
        "output_dim": 1, "dtype": "fp16", "normalization": "none",
        "graph_hash": contract.graph_hash, "heuristic": "hamming",
        "trained": False,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target
