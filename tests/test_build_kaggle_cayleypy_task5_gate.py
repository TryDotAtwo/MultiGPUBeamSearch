from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

import pytest

from tools.build_kaggle_cayleypy_task5_gate import (
    BEAM_WIDTH,
    CUTLASS_GIT_REV,
    DEPTH_LIMIT,
    K1_RADIUS,
    K2_RADIUS,
    MAX_UNIQUE_SOLUTIONS,
    MEASURED_PROFILE,
    PUZZLE_ID,
    SOLVED_RESULT_CAPACITY,
    BASE_GIT_REV,
    KERNEL_SLUG,
    OUT_NOTEBOOK,
    REVIEWED_COMMIT,
    build_notebook,
    decode_embedded_source,
    validate_gate_output,
)


REVIEWED_SOURCE = Path("tools/production_runner.cu")


def test_builder_roundtrips_reviewed_source_into_private_two_t4_notebook(tmp_path: Path) -> None:
    notebook_path, metadata_path = build_notebook(tmp_path / "private_gate")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "id": KERNEL_SLUG,
        "title": "CayleyPy Public Task 5 2xT4 Gate",
        "code_file": OUT_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    assert BASE_GIT_REV == "6f95bd6bdb32b5f6ef7cca32b96967bce6036503"
    assert REVIEWED_COMMIT == "6830401ed2086921d2563c2bc3c11faf6c5a0741"
    assert decode_embedded_source(notebook_path) == REVIEWED_SOURCE.read_bytes()

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("outputs") == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")

    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for forbidden in ("codex/public-cayleypy-notebook", "C:\\Users\\", "ghp_", "github_pat_", "sk-proj-"):
        assert forbidden not in source


def _write_run(root: Path, name: str, combined: str, *, tsv: bytes | None = None) -> dict[str, object]:
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    combined_path = run_dir / "combined.log"
    combined_path.write_text(combined, encoding="utf-8")
    for rank in (0, 1):
        (run_dir / f"rank{rank}.stdout.log").write_text(
            f"rank={rank} normal_completion=1\n", encoding="utf-8"
        )
        (run_dir / f"rank{rank}.stderr.log").write_text("", encoding="utf-8")
    if tsv is not None:
        (run_dir / "solutions.tsv").write_bytes(tsv)
    gpu_rows = ["0, Tesla T4, 1, 15359, 15360", "1, Tesla T4, 1, 15359, 15360"]
    result = {
        "name": name,
        "mode": "first" if name == "first" else "collect",
        "command": [
            "python", "-m", "torch.distributed.run", "--nproc-per-node=2",
            "--redirects=3", "--tee=0:3", "production_runner",
            str(PUZZLE_ID), str(DEPTH_LIMIT), str(BEAM_WIDTH),
        ],
        "return_code": 0,
        "rank_return_codes": {"0": 0, "1": 0},
        "rank_return_code_basis": "torchrun rc=0 requires every local worker to exit zero",
        "timed_out": False,
        "elapsed_sec": 1.25,
        "peak_process_tree_rss_bytes": 123456,
        "rss_sample_count": 1,
        "rss_samples_retained": 1,
        "rss_samples": [{"elapsed_sec": 0.1, "rss_bytes": 123456, "process_count": 3}],
        "combined_log_bytes": len(combined_path.read_bytes()),
        "combined_log_sha256": sha256(combined_path.read_bytes()).hexdigest(),
        "rank_stdout_count": 2,
        "rank_stderr_count": 2,
        "gpu_before": gpu_rows,
        "gpu_after": gpu_rows,
        "fatal_hits": [],
        "normal_completion_both_ranks": True,
    }
    (run_dir / "run_result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
    return result


def _write_fake_gate(
    root: Path,
    *,
    mismatch: bool = False,
    first_solution: str = "BR",
) -> None:
    source_sha = sha256(REVIEWED_SOURCE.read_bytes()).hexdigest()
    binary_sha = "b" * 64
    manifest = {
        "kernel_slug": KERNEL_SLUG,
        "base_git_rev": BASE_GIT_REV,
        "reviewed_commit": REVIEWED_COMMIT,
        "base_production_runner_sha256": "a" * 64,
        "production_runner_sha256": source_sha,
        "production_runner_bytes": REVIEWED_SOURCE.stat().st_size,
        "binary_sha256": binary_sha,
        "binary_bytes": 1234567,
        "cutlass_git_rev": CUTLASS_GIT_REV,
        "build_type": "Release",
        "cuda_architectures": "75",
        "nccl_linked": True,
        "timings_sec": {
            "base_clone": 1.0,
            "cutlass_clone": 1.0,
            "configure": 1.0,
            "compile": 1.0,
        },
    }
    (root / "source_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    gpu_rows = ["0, Tesla T4, 15360, 15359", "1, Tesla T4, 15360, 15359"]
    (root / "environment.json").write_text(
        json.dumps({"gpu_rows": gpu_rows}) + "\n", encoding="utf-8"
    )
    (root / "weights_manifest.json").write_text(
        json.dumps({
            "model": {
                "state_len": 120,
                "num_classes": 120,
                "output_dim": 24,
                "dtype": "fp16",
            },
            "files": {},
        }) + "\n",
        encoding="utf-8",
    )
    (root / "build.log").write_text(
        "cmake -DCMAKE_BUILD_TYPE=Release -DBEAM_CUDA_ARCHITECTURES=75\n"
        "ldd production_runner => libnccl.so\n",
        encoding="utf-8",
    )
    (root / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\n", encoding="utf-8"
    )
    first_line = (
        "[default0]:puzzle_solved=1 puzzle_id=1 seconds=0.5 solution_length=1 "
        f"found_depth=1 touch_depth=0 solution={first_solution}\n"
    )
    first = _write_run(root, "first", first_line)
    header = (
        b"puzzle_id\tdepth_index\tfound_depth\ttotal_depth\tknown_length\t"
        b"delta\towner_rank\tsolution_path\n"
    )
    row = b"1\t0\t1\t1\t1\t0\t0\tBR\n"
    collect_bytes = header + row
    collect_a = _write_run(
        root,
        "collect_a",
        "[default0]:collection_status=depth_reached\n",
        tsv=collect_bytes,
    )
    collect_b = _write_run(
        root,
        "collect_b",
        "[default0]:collection_status=depth_reached\n",
        tsv=collect_bytes + (b"1\t1\t2\t2\t1\t1\t1\tBR.BR\n" if mismatch else b""),
    )
    summary = {
        "status": "ok",
        "kernel_slug": KERNEL_SLUG,
        "base_git_rev": BASE_GIT_REV,
        "reviewed_commit": REVIEWED_COMMIT,
        "source_sha256": source_sha,
        "binary_sha256": binary_sha,
        "cutlass_git_rev": CUTLASS_GIT_REV,
        "hardware": gpu_rows,
        "build": manifest,
        "parameters": {
            "puzzle_id": PUZZLE_ID,
            "beam_width": BEAM_WIDTH,
            "depth_limit": DEPTH_LIMIT,
            "max_unique_solutions": MAX_UNIQUE_SOLUTIONS,
            "solved_result_capacity": SOLVED_RESULT_CAPACITY,
            "k1_radius": K1_RADIUS,
            "k2_radius": K2_RADIUS,
            "profile": MEASURED_PROFILE,
        },
        "first_release_line": first_line.strip(),
        "first_release": {
            "seconds": 0.5,
            "solution_length": 1,
            "found_depth": 1,
            "touch_depth": 0,
            "solution": first_solution,
        },
        "cpu_solution_valid": True,
        "collection_status": {"collect_a": "depth_reached", "collect_b": "depth_reached"},
        "collect_tsv_schema": [
            "puzzle_id", "depth_index", "found_depth", "total_depth",
            "known_length", "delta", "owner_rank", "solution_path",
        ],
        "collect_tsv_rows": 1,
        "collect_tsv_unique_paths": 1,
        "collect_tsv_bytes": len(collect_bytes),
        "collect_tsv_sha256": sha256(collect_bytes).hexdigest(),
        "collect_tsv_byte_identical": True,
        "runs": {"first": first, "collect_a": collect_a, "collect_b": collect_b},
        "no_overflow_oom_timeout_or_collective_hang": True,
        "injected_rank0_failure": "source-test-only-no-safe-runtime-hook",
    }
    (root / "gate_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")


def test_downloaded_gate_validator_requires_raw_deterministic_two_rank_evidence(tmp_path: Path) -> None:
    _write_fake_gate(tmp_path)
    validated = validate_gate_output(tmp_path)
    assert validated["status"] == "ok"
    assert validated["collect_tsv_sha256"] == sha256(
        (tmp_path / "runs/collect_a/solutions.tsv").read_bytes()
    ).hexdigest()


def test_downloaded_gate_validator_rejects_nonidentical_collect_bytes(tmp_path: Path) -> None:
    _write_fake_gate(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="byte-identical"):
        validate_gate_output(tmp_path)


def test_downloaded_gate_validator_replays_paths_instead_of_trusting_summary(
    tmp_path: Path,
) -> None:
    _write_fake_gate(tmp_path, first_solution="U")

    with pytest.raises(ValueError, match="CPU-invalid"):
        validate_gate_output(tmp_path)


def test_downloaded_gate_validator_rejects_build_or_profile_drift(tmp_path: Path) -> None:
    binary_root = tmp_path / "binary"
    binary_root.mkdir()
    _write_fake_gate(binary_root)
    binary_summary_path = binary_root / "gate_summary.json"
    binary_summary = json.loads(binary_summary_path.read_text(encoding="utf-8"))
    binary_summary["binary_sha256"] = "c" * 64
    binary_summary_path.write_text(json.dumps(binary_summary) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact source/build/run contract"):
        validate_gate_output(binary_root)

    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    _write_fake_gate(profile_root)
    profile_summary_path = profile_root / "gate_summary.json"
    profile_summary = json.loads(profile_summary_path.read_text(encoding="utf-8"))
    profile_summary["parameters"]["profile"]["requested_beam"] = 32768
    profile_summary_path.write_text(json.dumps(profile_summary) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact source/build/run contract"):
        validate_gate_output(profile_root)


def test_kernel_metadata_id_matches_kaggle_created_private_slug(tmp_path: Path) -> None:
    _, metadata_path = build_notebook(tmp_path / "slug_gate")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == "trydotatwo/cayleypy-public-task-5-2xt4-gate"
