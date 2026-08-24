from __future__ import annotations

import pytest

from tools.sm120_native_benchmark_artifact import parse_native_runner_log


def _log(seconds: float = 79.0, backend: str = "cublaslt") -> str:
    return f"""
cuda_device_name=NVIDIA RTX PRO 6000 Blackwell Server Edition
cuda_device_sm=120
B_MICRO=3584
STREAM1_CONCURRENCY=2
GLOBAL_BEAM_WIDTH_EFFECTIVE=33554432
stream1_backend=piece_transformer
stream1_transformer_micro=896
stream1_transformer_activation=relu
stream1_transformer_fp16_gemm_backend={backend}
stream1_transformer_dims seq_len=57 d_model=256 nhead=8 head_dim=32 layers=4 ff_dim=1024 output_dim=24
[default0]:depth_done=8 depth_sec={seconds} ring_slot_jobs=782 stream3_jobs=391 stream4_jobs=51 next_frontier_size=16777216
"""


def test_parser_emits_selector_native_execution_contract() -> None:
    row = parse_native_runner_log(_log(), name="exact_cublaslt")
    assert row["latency_ms"] == 79000.0
    assert row["effective_beam"] == 2**25
    assert row["native_execution"] == {
        "fp16_gemm_backend": "cublaslt", "target_sm": 120, "workspace_bytes": 0,
    }


@pytest.mark.parametrize("bad", [
    "cuda_device_sm=120", "B_MICRO=3584", "STREAM1_CONCURRENCY=2",
    "stream1_transformer_micro=896", "output_dim=24", "stream3_jobs=391",
    "stream1_transformer_activation=relu",
])
def test_parser_rejects_non_comparable_workload(bad: str) -> None:
    text = _log().replace(bad, bad.split("=")[0] + "=1")
    with pytest.raises(ValueError):
        parse_native_runner_log(text, name="bad")


def test_parser_rejects_fatal_markers() -> None:
    with pytest.raises(ValueError, match="fatal markers"):
        parse_native_runner_log(_log() + "CUDA error: illegal memory access\n", name="bad")
