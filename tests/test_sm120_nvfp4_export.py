from __future__ import annotations

import pytest

import json
import subprocess
from pathlib import Path

from tools.sm120_nvfp4_export import (
    FULL_STREAM1_OPERATORS,
    export_nvfp4_ffn_package,
    export_nvfp4_full_stream1_package,
    selected_fused_ffn_layers,
)
from tools.sm120_quant_tuner import CORE_OPERATORS, build_profile


def _profile(selected: set[str]) -> dict:
    return build_profile(
        checkpoint_sha256="a" * 64,
        model_metadata_sha256="b" * 64,
        calibration_sha256="c" * 64,
        gpu_identity="RTX PRO 6000 Blackwell|sm120",
        cutlass_commit="d" * 40,
        operator_precision={
            name: "sm120_nvfp4" if name in selected else "fp16"
            for name in CORE_OPERATORS
        },
    )


def _ffn_pair(layer: int) -> set[str]:
    return {f"blocks.{layer}.ff.0.weight", f"blocks.{layer}.ff.3.weight"}


def test_selected_fused_ffn_layers_require_complete_nonfinal_pairs() -> None:
    assert selected_fused_ffn_layers(_profile(_ffn_pair(0) | _ffn_pair(2))) == [0, 2]

    with pytest.raises(ValueError, match="complete FF1/FF2 pair"):
        selected_fused_ffn_layers(_profile({"blocks.0.ff.0.weight"}))

    with pytest.raises(ValueError, match="final block"):
        selected_fused_ffn_layers(_profile(_ffn_pair(3)))


def test_selected_fused_ffn_layers_reject_non_ffn_nvfp4_operators() -> None:
    with pytest.raises(ValueError, match="only fused FF1/FF2"):
        selected_fused_ffn_layers(_profile({"blocks.0.attn.in_proj_weight"}))


def test_export_package_is_atomic_and_invokes_native_encoder(tmp_path: Path) -> None:
    source = tmp_path / "fp32"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({
            "backend": "piece_transformer", "dtype": "fp32",
            "output_dim": 24, "d_model": 256, "num_layers": 4,
            "ff_dim": 1024, "activation": "relu", "seq_len": 57,
            "piece_layout": "cube4", "pooling": "cls",
            "checkpoint_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    calls: list[list[str]] = []
    quality_report = tmp_path / "quality.json"
    benchmark_report = tmp_path / "benchmark.json"
    quality_report.write_text('{"status":"accepted"}\n', encoding="utf-8")
    benchmark_report.write_text('{"status":"accepted"}\n', encoding="utf-8")

    def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "runtime_manifest.txt").write_text(
            "schema_version=1\nencoding=sm120_nvfp4_e2m1_k16_ue4m3_sfb\nweights=offline_immutable\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0)

    destination = tmp_path / "package"
    result = export_nvfp4_ffn_package(
        source, destination, _profile(_ffn_pair(0) | _ffn_pair(2)),
        native_exporter=Path("/opt/bin/nvfp4-export"), runner=fake_run,
        quality_report=quality_report,
        benchmark_report=benchmark_report,
        build_fingerprint={
            "cuda": "13.3.73",
            "cutlass_commit": "d" * 40,
            "source_sha256": "e" * 64,
        },
        microbatch=896,
        physical_slots=2,
    )
    assert result == destination
    assert calls == [[
        str(Path("/opt/bin/nvfp4-export")), "--weight-dir", str(source),
        "--output-dir", str(destination) + ".tmp", "--layers", "0,2",
    ]]
    package = json.loads((destination / "package_manifest.json").read_text(encoding="utf-8"))
    assert package["schema_version"] == 4
    assert package["backend"] == "sm120a_nvfp4_persistent"
    assert package["target_sm"] == 120
    assert package["target_arch"] == "sm120a"
    assert package["microbatch"] == 896
    assert package["physical_slots"] == 2
    assert package["profile_scope"] == "ffn_only"
    assert package["promotion_state"] == "component_candidate"
    assert package["quality_report_sha256"] == _sha256_for_test(quality_report)
    assert package["benchmark_report_sha256"] == _sha256_for_test(benchmark_report)
    assert package["layers"] == [0, 2]
    assert package["source_model"]["output_dim"] == 24
    runtime = (destination / "runtime_manifest.txt").read_text(encoding="utf-8")
    assert "schema_version=4\n" in runtime
    assert "backend=sm120a_nvfp4_persistent\n" in runtime
    assert "target_sm=120\n" in runtime
    assert "target_arch=sm120a\n" in runtime
    assert "microbatch=896\n" in runtime
    assert "physical_slots=2\n" in runtime
    assert "profile_scope=ffn_only\n" in runtime
    assert "promotion_state=component_candidate\n" in runtime
    assert f"quality_report_sha256={_sha256_for_test(quality_report)}\n" in runtime
    assert f"benchmark_report_sha256={_sha256_for_test(benchmark_report)}\n" in runtime
    assert "layers=0,2\n" in runtime
    assert f"checkpoint_sha256={'a' * 64}\n" in runtime
    assert not Path(str(destination) + ".tmp").exists()


def _sha256_for_test(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("microbatch", "physical_slots", "message"),
    [
        (0, 2, "microbatch"),
        (896, 1, "exactly two physical slots"),
        (896, 3, "exactly two physical slots"),
    ],
)
def test_export_package_rejects_invalid_persistent_runtime_geometry_before_writing(
    tmp_path: Path,
    microbatch: int,
    physical_slots: int,
    message: str,
) -> None:
    source = tmp_path / "fp32"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({
            "backend": "piece_transformer", "dtype": "fp32",
            "output_dim": 24, "d_model": 256, "num_layers": 4,
            "ff_dim": 1024, "activation": "relu", "seq_len": 57,
            "piece_layout": "cube4", "pooling": "cls",
            "checkpoint_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    quality_report = tmp_path / "quality.json"
    benchmark_report = tmp_path / "benchmark.json"
    quality_report.write_text('{"status":"accepted"}\n', encoding="utf-8")
    benchmark_report.write_text('{"status":"accepted"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        export_nvfp4_ffn_package(
            source,
            tmp_path / "package",
            _profile(_ffn_pair(0)),
            native_exporter=Path("unused"),
            quality_report=quality_report,
            benchmark_report=benchmark_report,
            build_fingerprint={
                "cuda": "13.3.73",
                "cutlass_commit": "d" * 40,
                "source_sha256": "e" * 64,
            },
            microbatch=microbatch,
            physical_slots=physical_slots,
        )


def test_export_package_rejects_non_cube4_contract_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "fp32"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"backend": "piece_transformer", "dtype": "fp16", "output_dim": 24,
                    "checkpoint_sha256": "a" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="FP32 Cube4 ReLU"):
        export_nvfp4_ffn_package(
            source, tmp_path / "package", _profile(_ffn_pair(0)),
            native_exporter=Path("unused"),
        )


def _full_stream1_source(tmp_path: Path) -> Path:
    source = tmp_path / "fp32"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({
            "backend": "piece_transformer", "dtype": "fp32",
            "output_dim": 24, "d_model": 256, "num_layers": 4,
            "ff_dim": 1024, "activation": "relu", "seq_len": 57,
            "piece_layout": "cube4", "pooling": "cls",
            "checkpoint_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    return source


def _full_quality_report(path: Path) -> Path:
    path.write_text(json.dumps({
        "status": "accepted",
        "scope": "full_stream1",
        "replay_valid": True,
        "nvfp4_solved_not_less_than_fp16": True,
        "nvfp4_lengths_not_greater_than_fp16": True,
        "quality_ladder": [
            {"states": 10, "beam": 2**16},
            {"states": 30, "beam": 2**18},
            {"states": 100, "beam": 2**20},
        ],
        "final_depth8": {"beam": 2**25, "completed_depth": 8},
        "full_solve_control": {"replay_valid": True, "length_not_greater_than_fp16": True},
    }) + "\n", encoding="utf-8")
    return path


def _full_benchmark_report(path: Path) -> Path:
    path.write_text(json.dumps({
        "status": "accepted",
        "scope": "full_stream1",
        "batch": 896,
        "median_ms": 0.333252,
        "p95_ms": 0.35,
        "alternating_runs": 7,
        "stage_sum_ms": 0.3,
        "oom": False,
        "nan": False,
        "race": False,
        "barrier_errors": False,
    }) + "\n", encoding="utf-8")
    return path


def test_full_stream1_operator_contract_includes_all_linears_and_padded_head() -> None:
    assert len(FULL_STREAM1_OPERATORS) == 17
    assert FULL_STREAM1_OPERATORS[0][0] == "block0_attn_qkv"
    assert FULL_STREAM1_OPERATORS[-1] == ("output_head", 256, 24, 128)
    assert {item[0] for item in FULL_STREAM1_OPERATORS} >= {
        "block3_attn_out", "block3_ff1", "block3_ff2", "output_head"
    }


def test_export_full_stream1_package_is_fail_closed_and_atomic(tmp_path: Path) -> None:
    source = _full_stream1_source(tmp_path)
    quality = _full_quality_report(tmp_path / "quality.json")
    benchmark = _full_benchmark_report(tmp_path / "benchmark.json")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "runtime_manifest.txt").write_text(
            "schema_version=2\n"
            "encoding=sm120_nvfp4_e2m1_k16_ue4m3_sfb\n"
            "weights=offline_immutable\n"
            "profile_scope=full_stream1\n"
            "operator_count=17\n"
            "operator.output_head.logical_output_cols=24\n"
            "operator.output_head.physical_output_cols=128\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0)

    destination = tmp_path / "full-profile"
    result = export_nvfp4_full_stream1_package(
        source, destination, native_exporter=Path("/opt/bin/nvfp4-export"),
        quality_report=quality, benchmark_report=benchmark,
        build_fingerprint={
            "cuda": "13.3.73", "cutlass_commit": "d" * 40,
            "source_sha256": "e" * 64,
        },
        runner=fake_run,
    )
    assert result == destination
    assert calls == [[
        str(Path("/opt/bin/nvfp4-export")), "--weight-dir", str(source),
        "--output-dir", str(destination) + ".tmp", "--scope", "full_stream1",
    ]]
    package = json.loads((destination / "package_manifest.json").read_text())
    assert package["schema_version"] == 4
    assert package["profile_scope"] == "full_stream1"
    assert package["promotion_state"] == "accepted"
    assert package["operator_mask"] == [item[0] for item in FULL_STREAM1_OPERATORS]
    assert package["operators"][-1] == {
        "name": "output_head", "input_cols": 256,
        "logical_output_cols": 24, "physical_output_cols": 128,
    }
    runtime = (destination / "runtime_manifest.txt").read_text()
    assert "schema_version=4\n" in runtime
    assert "promotion_state=accepted\n" in runtime
    assert "microbatch=896\n" in runtime
    assert "physical_slots=2\n" in runtime


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report.update(median_ms=0.333253), "median"),
        (lambda report: report.update(p95_ms=0.350001), "p95"),
        (lambda report: report.update(alternating_runs=6), "alternating"),
        (lambda report: report.update(stage_sum_ms=0.300001), "stage sum"),
        (lambda report: report.update(race=True), "race"),
    ],
)
def test_full_stream1_export_rejects_unmet_performance_gate(
    tmp_path: Path, mutate, message: str
) -> None:
    source = _full_stream1_source(tmp_path)
    quality = _full_quality_report(tmp_path / "quality.json")
    benchmark = _full_benchmark_report(tmp_path / "benchmark.json")
    payload = json.loads(benchmark.read_text())
    mutate(payload)
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        export_nvfp4_full_stream1_package(
            source, tmp_path / "output", native_exporter=Path("unused"),
            quality_report=quality, benchmark_report=benchmark,
            build_fingerprint={
                "cuda": "13.3.73", "cutlass_commit": "d" * 40,
                "source_sha256": "e" * 64,
            },
        )
