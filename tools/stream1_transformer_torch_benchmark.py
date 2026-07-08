#!/usr/bin/env python3
"""Plain PyTorch Stream1 piece-transformer inference benchmark.

This intentionally benchmarks the exported Stream1 weights as a PyTorch model.
It is not a runtime fallback path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


SCORE_MAX_Q = 300.0
SCORE_SCALE = 1024.0
FNV64_OFFSET = 1469598103934665603
FNV64_PRIME = 1099511628211
FNV64_MASK = (1 << 64) - 1


def parse_positive_ints(text: str, label: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            value = int(part)
            if value <= 0:
                raise ValueError(f"{label} values must be positive")
            values.append(value)
    if not values:
        raise ValueError(f"at least one {label} value is required")
    return values


def parse_batch_sizes(text: str) -> List[int]:
    return parse_positive_ints(text, "batch size")


def product(shape: Iterable[int]) -> int:
    n = 1
    for value in shape:
        n *= int(value)
    return n


def load_file_tensor(path: Path, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    expected = product(shape)
    tensor = torch.from_file(str(path), shared=False, size=expected, dtype=dtype)
    if tensor.numel() != expected:
        raise ValueError(f"{path}: expected {expected} values, got {tensor.numel()}")
    return tensor.reshape(shape).to(device=device, non_blocking=False)


def score_keys(logits: torch.Tensor) -> torch.Tensor:
    return torch.round(torch.clamp(logits.float(), 0.0, SCORE_MAX_Q) * SCORE_SCALE).to(torch.int64)


def score_key_digest(keys: torch.Tensor) -> int:
    digest = FNV64_OFFSET
    flat = keys.detach().to(device="cpu", dtype=torch.int64).contiguous().view(-1)
    for raw in flat.tolist():
        value = int(raw) & 0xFFFFFFFF
        for shift in (0, 8, 16, 24):
            digest ^= (value >> shift) & 0xFF
            digest = (digest * FNV64_PRIME) & FNV64_MASK
    return digest


class PieceTransformerTorch:
    def __init__(self, weight_dir: Path, device: torch.device) -> None:
        self.weight_dir = weight_dir
        self.device = device
        self.manifest = json.loads((weight_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("backend") != "piece_transformer":
            raise ValueError("manifest backend must be piece_transformer")
        manifest_dtype = str(self.manifest.get("dtype"))
        if manifest_dtype == "fp16":
            self.dtype = torch.float16
            self.dtype_suffix = ".fp16"
        elif manifest_dtype == "bf16":
            self.dtype = torch.bfloat16
            self.dtype_suffix = ".bf16"
        else:
            raise ValueError(f"plain torch benchmark expects exported fp16 or bf16 weights, got {manifest_dtype!r}")

        self.state_len = int(self.manifest["state_len"])
        self.num_classes = int(self.manifest["num_classes"])
        self.output_dim = int(self.manifest["output_dim"])
        self.move_count = int(self.manifest.get("move_count", self.output_dim))
        self.num_pieces = int(self.manifest["num_pieces"])
        self.max_piece_size = int(self.manifest["max_piece_size"])
        self.seq_len = int(self.manifest["seq_len"])
        self.d_model = int(self.manifest["d_model"])
        self.nhead = int(self.manifest["nhead"])
        self.head_dim = int(self.manifest["head_dim"])
        self.num_layers = int(self.manifest["num_layers"])
        self.ff_dim = int(self.manifest["ff_dim"])
        if self.seq_len != self.num_pieces + 1:
            raise ValueError("seq_len must equal num_pieces + cls token")
        if self.d_model != self.nhead * self.head_dim:
            raise ValueError("d_model must equal nhead * head_dim")
        if self.output_dim != self.move_count:
            raise ValueError("output_dim must equal move_count")

        suffix = self.dtype_suffix
        half = self.dtype
        self.fast_slot_projected = load_file_tensor(
            weight_dir / f"fast_slot_projected{suffix}",
            (self.max_piece_size, self.num_classes, self.d_model),
            half,
            device,
        )
        self.fast_piece_static = load_file_tensor(
            weight_dir / f"fast_piece_static{suffix}",
            (self.num_pieces, self.d_model),
            half,
            device,
        )
        self.cls_token = load_file_tensor(weight_dir / f"cls_token{suffix}", (self.d_model,), half, device)
        self.input_ln_gamma = load_file_tensor(weight_dir / f"input_ln_gamma{suffix}", (self.d_model,), half, device)
        self.input_ln_beta = load_file_tensor(weight_dir / f"input_ln_beta{suffix}", (self.d_model,), half, device)
        self.output_ln_gamma = load_file_tensor(weight_dir / f"output_ln_gamma{suffix}", (self.d_model,), half, device)
        self.output_ln_beta = load_file_tensor(weight_dir / f"output_ln_beta{suffix}", (self.d_model,), half, device)
        self.output_weight = load_file_tensor(
            weight_dir / f"output_weight_hxk{suffix}",
            (self.d_model, self.output_dim),
            half,
            device,
        )
        self.output_bias = load_file_tensor(weight_dir / f"output_bias{suffix}", (self.output_dim,), half, device)
        self.piece_positions = load_file_tensor(
            weight_dir / "piece_positions.u16",
            (self.num_pieces, self.max_piece_size),
            torch.int16,
            device,
        ).long()
        self.piece_mask = load_file_tensor(
            weight_dir / "piece_mask.u8",
            (self.num_pieces, self.max_piece_size),
            torch.uint8,
            device,
        ).bool()

        self.blocks: List[Dict[str, torch.Tensor]] = []
        for block in range(self.num_layers):
            prefix = f"block{block}"
            self.blocks.append(
                {
                    "ln1_gamma": load_file_tensor(weight_dir / f"{prefix}_ln1_gamma{suffix}", (self.d_model,), half, device),
                    "ln1_beta": load_file_tensor(weight_dir / f"{prefix}_ln1_beta{suffix}", (self.d_model,), half, device),
                    "qkv_weight": load_file_tensor(
                        weight_dir / f"{prefix}_attn_qkv_weight_hxk{suffix}",
                        (self.d_model, 3 * self.d_model),
                        half,
                        device,
                    ),
                    "qkv_bias": load_file_tensor(
                        weight_dir / f"{prefix}_attn_qkv_bias{suffix}",
                        (3 * self.d_model,),
                        half,
                        device,
                    ),
                    "attn_out_weight": load_file_tensor(
                        weight_dir / f"{prefix}_attn_out_weight_hxk{suffix}",
                        (self.d_model, self.d_model),
                        half,
                        device,
                    ),
                    "attn_out_bias": load_file_tensor(
                        weight_dir / f"{prefix}_attn_out_bias{suffix}",
                        (self.d_model,),
                        half,
                        device,
                    ),
                    "ln2_gamma": load_file_tensor(weight_dir / f"{prefix}_ln2_gamma{suffix}", (self.d_model,), half, device),
                    "ln2_beta": load_file_tensor(weight_dir / f"{prefix}_ln2_beta{suffix}", (self.d_model,), half, device),
                    "ff1_weight": load_file_tensor(
                        weight_dir / f"{prefix}_ff1_weight_hxk{suffix}",
                        (self.d_model, self.ff_dim),
                        half,
                        device,
                    ),
                    "ff1_bias": load_file_tensor(weight_dir / f"{prefix}_ff1_bias{suffix}", (self.ff_dim,), half, device),
                    "ff2_weight": load_file_tensor(
                        weight_dir / f"{prefix}_ff2_weight_hxk{suffix}",
                        (self.ff_dim, self.d_model),
                        half,
                        device,
                    ),
                    "ff2_bias": load_file_tensor(weight_dir / f"{prefix}_ff2_bias{suffix}", (self.d_model,), half, device),
                }
            )

    def layer_norm(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.d_model,), gamma, beta, eps=1.0e-5)

    def build_tokens(self, states: torch.Tensor) -> torch.Tensor:
        states = states[:, : self.state_len].long()
        batch = states.shape[0]
        pieces = self.fast_piece_static.unsqueeze(0).expand(batch, -1, -1).clone()
        for slot in range(self.max_piece_size):
            mask = self.piece_mask[:, slot]
            if not bool(mask.any().item()):
                continue
            positions = self.piece_positions[:, slot]
            state_values = states[:, positions]
            slot_table = self.fast_slot_projected[slot]
            pieces[:, mask, :] += slot_table[state_values[:, mask]]
        cls = self.cls_token.reshape(1, 1, self.d_model).expand(batch, 1, self.d_model)
        return torch.cat((cls, pieces), dim=1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.build_tokens(states)
        x = self.layer_norm(x, self.input_ln_gamma, self.input_ln_beta)
        batch = x.shape[0]
        for block in self.blocks:
            y = self.layer_norm(x, block["ln1_gamma"], block["ln1_beta"])
            qkv = y.matmul(block["qkv_weight"]) + block["qkv_bias"]
            qkv = qkv.reshape(batch, self.seq_len, 3, self.nhead, self.head_dim)
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3).contiguous()
            k = qkv[:, :, 1, :, :].permute(0, 2, 1, 3).contiguous()
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3).contiguous()
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            context = attn.permute(0, 2, 1, 3).reshape(batch, self.seq_len, self.d_model)
            x = x + context.matmul(block["attn_out_weight"]) + block["attn_out_bias"]

            y = self.layer_norm(x, block["ln2_gamma"], block["ln2_beta"])
            y = F.silu(y.matmul(block["ff1_weight"]) + block["ff1_bias"])
            x = x + y.matmul(block["ff2_weight"]) + block["ff2_bias"]

        cls = self.layer_norm(x[:, 0, :], self.output_ln_gamma, self.output_ln_beta)
        return cls.matmul(self.output_weight) + self.output_bias


def validate_reference(model: PieceTransformerTorch, reference_path: Path, tolerance: int) -> Tuple[str, int, int]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    states = torch.tensor(reference["states"], dtype=torch.uint8, device=model.device)
    expected_scores = torch.tensor(reference["scores_fp32"], dtype=torch.float32, device=model.device)
    with torch.inference_mode():
        logits = model.forward(states)
        actual_keys = score_keys(logits)
    expected_keys = score_keys(expected_scores)
    max_error = int((actual_keys - expected_keys).abs().max().item())
    status = "pass" if max_error <= tolerance else "fail"
    return status, int(states.shape[0]), max_error


def make_states(batch: int, state_len: int, num_classes: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(batch * state_len, device=device, dtype=torch.int64)
    values = (values.reshape(batch, state_len) * 17 + 23) % num_classes
    return values.to(torch.uint8)


def benchmark_batch(
    model: PieceTransformerTorch,
    batch: int,
    concurrency: int,
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    states_group = [make_states(batch, model.state_len, model.num_classes, model.device) for _ in range(concurrency)]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(model.device)
    with torch.inference_mode():
        for _ in range(warmup):
            for states in states_group:
                logits = model.forward(states)
        torch.cuda.synchronize(model.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            for states in states_group:
                logits = model.forward(states)
        end.record()
        torch.cuda.synchronize(model.device)
    elapsed_ms = float(start.elapsed_time(end))
    keys = score_keys(logits)
    checksum = int(keys.sum().item())
    score_digest = score_key_digest(keys)
    first_score_keys = ",".join(str(int(x)) for x in keys[0].detach().cpu().tolist()) if batch > 0 else ""
    del logits
    rows_per_launch_group = batch * concurrency
    parents_per_s = (rows_per_launch_group * iters) / (elapsed_ms / 1000.0)
    candidates_per_s = parents_per_s * model.move_count
    peak_gib = torch.cuda.max_memory_allocated(model.device) / (1024.0**3)
    return {
        "batch": batch,
        "concurrency": concurrency,
        "rows_per_launch_group": rows_per_launch_group,
        "iters": iters,
        "elapsed_ms": elapsed_ms,
        "ms_per_iter": elapsed_ms / iters,
        "parents_per_s": parents_per_s,
        "candidates_per_s": candidates_per_s,
        "peak_mem_gib": peak_gib,
        "checksum": checksum,
        "score_key_digest": score_digest,
        "first_score_keys": first_score_keys,
        "status": "ok",
    }


def write_report(path: Path, rows: List[Dict[str, object]], metadata: Dict[str, object]) -> None:
    best = max((row for row in rows if row.get("status") == "ok"), key=lambda r: float(r["candidates_per_s"]), default=None)
    lines = [
        "# Stream1 Piece-Transformer PyTorch Benchmark",
        "",
        f"- torch_version={metadata['torch_version']}",
        f"- cuda_version={metadata['cuda_version']}",
        f"- device={metadata['device_name']}",
        f"- weight_dir={metadata['weight_dir']}",
        f"- reference_status={metadata['reference_status']}",
        f"- reference_rows={metadata['reference_rows']}",
        f"- reference_max_abs_score_key_error={metadata['reference_max_abs_score_key_error']}",
        "",
        "| batch | concurrency | rows/group | iters | ms/iter | parents/s | candidates/s | peak GiB | checksum | score_key_digest | status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row.get("status") == "ok":
            lines.append(
                "| {batch} | {concurrency} | {rows_per_launch_group} | {iters} | {ms_per_iter:.4f} | {parents_per_s:.1f} | {candidates_per_s:.1f} | {peak_mem_gib:.3f} | {checksum} | {score_key_digest} | ok |".format(
                    **row
                )
            )
        else:
            lines.append(f"| {row['batch']} | {row.get('concurrency', '-')} | - | - | - | - | - | - | - | - | {row['status']} |")
    if best is not None:
        lines.extend(
            [
                "",
                f"best_batch={best['batch']}",
                f"best_concurrency={best['concurrency']}",
                f"best_rows_per_launch_group={best['rows_per_launch_group']}",
                f"best_candidates_per_s={float(best['candidates_per_s']):.1f}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", type=Path, default=Path("test_results/stream1_transformer_reference/weights_fp16"))
    parser.add_argument("--reference-json", type=Path, default=Path("test_results/stream1_transformer_reference/reference.json"))
    parser.add_argument("--batch-sizes", default="128,256,512,1024,2048,4096")
    parser.add_argument("--concurrency", default="1")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--reference-tolerance", type=int, default=3072)
    parser.add_argument("--skip-reference", action="store_true", help="Skip reference JSON validation for synthetic benchmark/parity runs.")
    parser.add_argument("--report", type=Path, default=Path("test_results/stream1_transformer_torch_benchmark_2026-06-30.md"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    model = PieceTransformerTorch(args.weight_dir, device)
    if args.skip_reference:
        ref_status, ref_rows, ref_error = "skip", 0, 0
    else:
        ref_status, ref_rows, ref_error = validate_reference(model, args.reference_json, args.reference_tolerance)
        if ref_status != "pass":
            raise RuntimeError(f"reference validation failed: max_abs_score_key_error={ref_error}")

    rows: List[Dict[str, object]] = []
    for batch in parse_batch_sizes(args.batch_sizes):
        for concurrency in parse_positive_ints(args.concurrency, "concurrency"):
            try:
                row = benchmark_batch(model, batch, concurrency, args.warmup, args.iters)
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                rows.append({"batch": batch, "concurrency": concurrency, "status": type(exc).__name__ + ": " + str(exc).splitlines()[0][:160]})
                continue
            rows.append(row)
            print(
                "torch_stream1_transformer"
                f" batch={row['batch']}"
                f" concurrency={row['concurrency']}"
                f" rows_per_launch_group={row['rows_per_launch_group']}"
                f" ms_per_iter={row['ms_per_iter']:.4f}"
                f" parents_per_s={row['parents_per_s']:.1f}"
                f" candidates_per_s={row['candidates_per_s']:.1f}"
                f" peak_mem_gib={row['peak_mem_gib']:.3f}"
                f" checksum={row['checksum']}"
                f" score_key_digest={row['score_key_digest']}"
                f" first_score_keys={row['first_score_keys']}",
                flush=True,
            )

    metadata = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "weight_dir": str(args.weight_dir),
        "reference_status": ref_status,
        "reference_rows": ref_rows,
        "reference_max_abs_score_key_error": ref_error,
    }
    write_report(args.report, rows, metadata)
    with args.report.with_suffix(".tsv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["batch", "concurrency", "rows_per_launch_group", "iters", "elapsed_ms", "ms_per_iter", "parents_per_s", "candidates_per_s", "peak_mem_gib", "checksum", "score_key_digest", "first_score_keys", "status"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"torch_stream1_transformer_report={args.report}")


if __name__ == "__main__":
    main()
