"""CPU-only, reproducible static weight study for the Cube4 SM120 quantization research.

This intentionally does not benchmark CUDA or claim activation/ranking quality.  It measures
weight reconstruction error for candidate formats and emits JSON for the research report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "test_results/cube4-model-bundle/model/model.pth"
OUTPUT = ROOT / "test_results/cube4_sm120_weight_quantization_stats_2026-08-22.json"


def pow2_dequant_scale(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    ratio = amax.to(torch.float64) / qmax
    safe = torch.where(ratio > 0, ratio, torch.ones_like(ratio))
    scale = torch.pow(2.0, torch.ceil(torch.log2(safe)))
    return torch.where(ratio > 0, scale, torch.ones_like(scale)).to(torch.float32)


def quantize_scaled(x: torch.Tensor, dtype: torch.dtype, qmax: float, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x / scale, -qmax, qmax).to(dtype).to(torch.float32) * scale


def quantize_tensor(x: torch.Tensor, dtype: torch.dtype, qmax: float) -> torch.Tensor:
    scale = pow2_dequant_scale(x.abs().amax().reshape(1), qmax)[0]
    return quantize_scaled(x, dtype, qmax, scale)


def quantize_tensor_ideal_amax(x: torch.Tensor, dtype: torch.dtype, qmax: float) -> torch.Tensor:
    amax = x.abs().amax().reshape(1)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))[0]
    return quantize_scaled(x, dtype, qmax, scale)


def quantize_rows(x: torch.Tensor, dtype: torch.dtype, qmax: float) -> torch.Tensor:
    scale = pow2_dequant_scale(x.abs().amax(dim=1, keepdim=True), qmax)
    return quantize_scaled(x, dtype, qmax, scale)


def quantize_rows_ideal_amax(x: torch.Tensor, dtype: torch.dtype, qmax: float) -> torch.Tensor:
    amax = x.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    return quantize_scaled(x, dtype, qmax, scale)


def quantize_mxfp8_block32(x: torch.Tensor, ideal_scale: bool = False) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] % 32 != 0:
        raise ValueError(f"MXFP8 study requires a 2D K-aligned tensor, got {tuple(x.shape)}")
    grouped = x.reshape(x.shape[0], x.shape[1] // 32, 32)
    amax = grouped.abs().amax(dim=2, keepdim=True)
    if ideal_scale:
        scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    else:
        scale = pow2_dequant_scale(amax, 448.0)
    quantized = quantize_scaled(grouped, torch.float8_e4m3fn, 448.0, scale)
    return quantized.reshape_as(x)


def metrics(x: torch.Tensor, q: torch.Tensor) -> dict[str, float]:
    error = (q - x).to(torch.float64)
    signal2 = x.to(torch.float64).square().sum().item()
    error2 = error.square().sum().item()
    abs_error = error.abs().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        x.to(torch.float64).flatten(), q.to(torch.float64).flatten(), dim=0
    ).item()
    return {
        "nmse": error2 / signal2 if signal2 else 0.0,
        "snr_db": 10.0 * math.log10(signal2 / error2) if error2 else math.inf,
        "cosine": cosine,
        "mae": abs_error.mean().item(),
        "p99_abs_error": torch.quantile(abs_error, 0.99).item(),
        "max_abs_error": abs_error.max().item(),
    }


def correct_error_columns(x: torch.Tensor, q: torch.Tensor, columns: int) -> torch.Tensor:
    if columns <= 0:
        return q
    column_error2 = (q - x).to(torch.float64).square().sum(dim=0)
    selected = torch.topk(column_error2, min(columns, x.shape[1])).indices
    corrected = q.clone()
    corrected[:, selected] = x[:, selected]
    return corrected


def correct_error_entries(x: torch.Tensor, q: torch.Tensor, fraction: float) -> torch.Tensor:
    count = max(1, round(x.numel() * fraction))
    indices = torch.topk((q - x).abs().flatten(), count).indices
    corrected = q.flatten().clone()
    source = x.flatten()
    corrected[indices] = source[indices]
    return corrected.reshape_as(x)


def main() -> None:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    matrices = {
        name: value.detach().to(torch.float32).contiguous()
        for name, value in state.items()
        if name.endswith("weight") and value.ndim == 2
    }
    result: dict[str, object] = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "torch_version": torch.__version__,
        "matrix_count": len(matrices),
        "matrix_parameters": sum(value.numel() for value in matrices.values()),
        "matrices": {},
    }
    aggregate: dict[str, dict[str, float]] = {}
    correction_aggregate: dict[str, dict[str, float]] = {}
    sparse_correction_aggregate: dict[str, dict[str, float]] = {}
    low_rank_aggregate: dict[str, dict[str, float]] = {}
    for name, weight in matrices.items():
        rms = weight.to(torch.float64).square().mean().sqrt().item()
        abs_flat = weight.abs().to(torch.float64).flatten()
        schemes = {
            "e4m3_tensor_pow2": quantize_tensor(weight, torch.float8_e4m3fn, 448.0),
            "e4m3_tensor_ideal_amax": quantize_tensor_ideal_amax(weight, torch.float8_e4m3fn, 448.0),
            "e4m3_row_pow2": quantize_rows(weight, torch.float8_e4m3fn, 448.0),
            "e4m3_row_ideal_amax": quantize_rows_ideal_amax(weight, torch.float8_e4m3fn, 448.0),
            "mxfp8_block32_e8m0": quantize_mxfp8_block32(weight),
            "mxfp8_block32_ideal_scale_upper_bound": quantize_mxfp8_block32(weight, ideal_scale=True),
            "e5m2_tensor_pow2": quantize_tensor(weight, torch.float8_e5m2, 57344.0),
        }
        scheme_metrics = {scheme: metrics(weight, quantized) for scheme, quantized in schemes.items()}
        base_mxfp8 = schemes["mxfp8_block32_e8m0"]
        corrections: dict[str, dict[str, float]] = {}
        for columns in sorted(set([1, 2, 4, 8, max(1, weight.shape[1] // 32)])):
            if columns <= weight.shape[1]:
                key = f"top_{columns}_k_columns_fp16"
                corrected = correct_error_columns(weight, base_mxfp8, columns)
                corrections[key] = metrics(weight, corrected)
                signal2 = weight.to(torch.float64).square().sum().item()
                error2 = (corrected - weight).to(torch.float64).square().sum().item()
                row = correction_aggregate.setdefault(
                    key, {"signal2": 0.0, "error2": 0.0, "parameters": 0.0, "fp16_parameters": 0.0}
                )
                row["signal2"] += signal2
                row["error2"] += error2
                row["parameters"] += weight.numel()
                row["fp16_parameters"] += weight.shape[0] * columns
        result["matrices"][name] = {
            "shape": list(weight.shape),
            "parameters": weight.numel(),
            "rms": rms,
            "abs_p99": torch.quantile(abs_flat, 0.99).item(),
            "abs_p999": torch.quantile(abs_flat, 0.999).item(),
            "abs_max": abs_flat.max().item(),
            "max_over_rms": abs_flat.max().item() / rms if rms else 0.0,
            "schemes": scheme_metrics,
            "mxfp8_fp16_column_correction": corrections,
        }
        sparse_corrections: dict[str, dict[str, float]] = {}
        for fraction in (0.001, 0.005, 0.01, 0.02):
            key = f"top_{fraction:.3f}_error_entries_fp16"
            corrected = correct_error_entries(weight, base_mxfp8, fraction)
            sparse_corrections[key] = metrics(weight, corrected)
            signal2 = weight.to(torch.float64).square().sum().item()
            error2 = (corrected - weight).to(torch.float64).square().sum().item()
            row = sparse_correction_aggregate.setdefault(
                key, {"signal2": 0.0, "error2": 0.0, "parameters": 0.0, "fp16_parameters": 0.0}
            )
            row["signal2"] += signal2
            row["error2"] += error2
            row["parameters"] += weight.numel()
            row["fp16_parameters"] += max(1, round(weight.numel() * fraction))
        result["matrices"][name]["mxfp8_fp16_sparse_error_correction"] = sparse_corrections
        if any(marker in name for marker in ("attn.in_proj_weight", "attn.out_proj.weight", ".ff.0.weight", ".ff.3.weight")):
            singular_values = torch.linalg.svdvals((weight - base_mxfp8).to(torch.float32)).to(torch.float64)
            total_error2 = singular_values.square().sum().item()
            low_rank: dict[str, dict[str, float]] = {}
            for rank in (1, 2, 4, 8, 16, 32):
                residual_error2 = singular_values[rank:].square().sum().item()
                correction_parameters = rank * (weight.shape[0] + weight.shape[1])
                key = f"rank_{rank}_fp16_residual"
                low_rank[key] = {
                    "captured_error_fraction": 1.0 - residual_error2 / total_error2,
                    "correction_parameters": correction_parameters,
                    "correction_parameter_fraction": correction_parameters / weight.numel(),
                }
                row = low_rank_aggregate.setdefault(
                    key, {"signal2": 0.0, "error2": 0.0, "parameters": 0.0, "correction_parameters": 0.0}
                )
                row["signal2"] += weight.to(torch.float64).square().sum().item()
                row["error2"] += residual_error2
                row["parameters"] += weight.numel()
                row["correction_parameters"] += correction_parameters
            result["matrices"][name]["mxfp8_best_low_rank_error_upper_bound"] = low_rank
        signal2 = weight.to(torch.float64).square().sum().item()
        for scheme, quantized in schemes.items():
            error2 = (quantized - weight).to(torch.float64).square().sum().item()
            row = aggregate.setdefault(scheme, {"signal2": 0.0, "error2": 0.0, "parameters": 0.0})
            row["signal2"] += signal2
            row["error2"] += error2
            row["parameters"] += weight.numel()
    result["aggregate"] = {
        scheme: {
            "parameters": int(values["parameters"]),
            "nmse": values["error2"] / values["signal2"],
            "snr_db": 10.0 * math.log10(values["signal2"] / values["error2"]),
        }
        for scheme, values in aggregate.items()
    }
    result["mxfp8_fp16_column_correction_aggregate"] = {
        key: {
            "parameters": int(values["parameters"]),
            "fp16_parameters": int(values["fp16_parameters"]),
            "fp16_parameter_fraction": values["fp16_parameters"] / values["parameters"],
            "nmse": values["error2"] / values["signal2"],
            "snr_db": 10.0 * math.log10(values["signal2"] / values["error2"]),
        }
        for key, values in correction_aggregate.items()
    }
    result["mxfp8_fp16_sparse_error_correction_aggregate"] = {
        key: {
            "parameters": int(values["parameters"]),
            "fp16_parameters": int(values["fp16_parameters"]),
            "fp16_parameter_fraction": values["fp16_parameters"] / values["parameters"],
            "nmse": values["error2"] / values["signal2"],
            "snr_db": 10.0 * math.log10(values["signal2"] / values["error2"]),
        }
        for key, values in sparse_correction_aggregate.items()
    }
    result["mxfp8_best_low_rank_error_upper_bound_aggregate"] = {
        key: {
            "parameters": int(values["parameters"]),
            "correction_parameters": int(values["correction_parameters"]),
            "correction_parameter_fraction": values["correction_parameters"] / values["parameters"],
            "nmse": values["error2"] / values["signal2"],
            "snr_db": 10.0 * math.log10(values["signal2"] / values["error2"]),
        }
        for key, values in low_rank_aggregate.items()
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote={OUTPUT}")


if __name__ == "__main__":
    main()
