#!/usr/bin/env python3
"""Export the numerically gated Cube4 SM120 FlashAttention-4 kernel to C AOT.

The output is a self-contained header/object pair consumed by the optional
``BEAM_ENABLE_SM120_FA4_AOT`` CMake path.  The object embeds the cubin; the
resulting native binary links only the CuTe DSL and TVM-FFI runtime libraries,
not Python or PyTorch in its inference hot path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from cutlass.base_dsl.jit_executor import JitCompiledFunction
from flash_attn.cute.interface import _flash_attn_fwd


HEAD_DIM = 32
HEADS = 8
SEQ_Q = 64
SEQ_KV = 57
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)
FILE_NAME = "fa4_cube4_q64_direct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def torch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=SOFTMAX_SCALE,
        is_causal=False,
    ).transpose(1, 2)


def select_compiled_function() -> tuple[tuple[object, ...], object]:
    cache = _flash_attn_fwd.compile_cache.cache
    candidates = [
        (key, value)
        for key, value in cache.items()
        if len(key) > 36
        and key[1] == HEAD_DIM
        and key[2] == HEAD_DIM
        and key[24:26] == (64, 64)
        and key[30] == 120
        and key[34] is False
        and key[35] is False
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "expected one SM120 FA4 q64k64/mma_pv=false/overlap=false compile-cache entry, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 FA4 export requires CUDA")
    device = torch.cuda.get_device_properties(0)
    if (device.major, device.minor) != (12, 0):
        raise RuntimeError(f"SM120 required, got SM{device.major}{device.minor}")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260901)
    q = torch.randn((4, SEQ_Q, HEADS, HEAD_DIM), device="cuda", dtype=torch.float16, generator=generator)
    k = torch.randn((4, SEQ_KV, HEADS, HEAD_DIM), device="cuda", dtype=torch.float16, generator=generator)
    v = torch.randn((4, SEQ_KV, HEADS, HEAD_DIM), device="cuda", dtype=torch.float16, generator=generator)
    result = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=SOFTMAX_SCALE,
        causal=False,
        tile_mn=(64, 64),
        mma_pv_is_rs=False,
        intra_wg_overlap=False,
    )
    actual = result[0] if isinstance(result, tuple) else result
    reference = torch_reference(q, k, v)
    error = (actual.float() - reference.float()).abs()
    numeric_gate = {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
    }
    if numeric_gate["max_abs_error"] > 0.02:
        raise RuntimeError(f"SM120 FA4 numeric gate failed: {numeric_gate}")

    key, compiled = select_compiled_function()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    JitCompiledFunction.export_to_c(
        compiled,
        str(args.output_dir),
        FILE_NAME,
        FILE_NAME,
    )
    header = args.output_dir / f"{FILE_NAME}.h"
    object_file = args.output_dir / f"{FILE_NAME}.o"
    metadata = {
        "format": "sm120_fa4_c_aot_v1",
        "gpu": device.name,
        "sm": "120",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": [SEQ_Q, SEQ_KV, HEADS, HEAD_DIM],
        "tile_mn": [64, 64],
        "mma_pv_is_rs": False,
        "intra_wg_overlap": False,
        "numeric_gate": numeric_gate,
        "compile_key": [str(value) for value in key],
        "header": {"bytes": header.stat().st_size, "sha256": sha256(header)},
        "object": {"bytes": object_file.stat().st_size, "sha256": sha256(object_file)},
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
