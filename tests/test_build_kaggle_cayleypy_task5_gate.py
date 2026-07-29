from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

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
    RUN_CELL,
    build_notebook,
    decode_embedded_source,
    validate_gate_output,
)


REVIEWED_SOURCE = Path("tools/production_runner.cu")


class _FakeStdout:
    def __init__(self, *, tail: str = "", read_error: BaseException | None = None) -> None:
        self.tail, self.read_error, self.read_count = tail, read_error, 0

    def readline(self) -> str:
        if self.read_error is not None:
            raise self.read_error
        if self.read_count == 0:
            self.read_count += 1
            return self.tail
        return ""

    def readlines(self) -> list[str]:
        return [self.tail] if self.tail else []


class _FakeProcess:
    def __init__(self, stdout: _FakeStdout, *, stay_running: bool) -> None:
        self.pid, self.stdout, self.returncode = 4242, stdout, None
        self.stay_running, self.poll_count, self.wait_count = stay_running, 0, 0
        self.wait_timeouts: list[float | None] = []
        self.reaped = False

    def poll(self) -> int | None:
        self.poll_count += 1
        if self.returncode is not None:
            return self.returncode
        if self.stay_running:
            return None
        return None if self.poll_count == 1 else 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self.wait_count += 1
        if self.wait_count == 1 and self.stay_running:
            raise __import__("subprocess").TimeoutExpired("fake", timeout)
        self.returncode, self.reaped = 0, True
        return 0


class _FakeSelector:
    def __init__(self, stdout: _FakeStdout, *, readable: bool) -> None:
        self.stdout, self.readable, self.closed = stdout, readable, False

    def register(self, _fileobj: object, _events: object) -> None:
        pass

    def select(self, timeout: float) -> list[tuple[SimpleNamespace, None]]:
        del timeout
        if not self.readable:
            return []
        self.readable = False
        return [(SimpleNamespace(fileobj=self.stdout), None)]

    def close(self) -> None:
        self.closed = True


def _load_generated_run_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess, selector: _FakeSelector,
) -> tuple[dict[str, object], list[tuple[int, int]]]:
    namespace: dict[str, object] = {
        "WORK_DIR": tmp_path / "work", "REPO_DIR": tmp_path / "repo",
        "BUILD_DIR": tmp_path / "build", "PROFILE": MEASURED_PROFILE,
        "GPU_HEADROOM_BYTES": 1, "HISTORY_RAM_BYTES": 1, "HISTORY_DISK_BYTES": 1,
        "K1_RADIUS": K1_RADIUS, "K2_RADIUS": K2_RADIUS, "PUZZLE_ID": PUZZLE_ID,
        "DEPTH_LIMIT": DEPTH_LIMIT, "MAX_UNIQUE_SOLUTIONS": MAX_UNIQUE_SOLUTIONS,
        "SOLVED_RESULT_CAPACITY": SOLVED_RESULT_CAPACITY, "BEAM_WIDTH": BEAM_WIDTH,
        "RUN_TIMEOUT_SEC": 60, "MAX_COMBINED_LOG_BYTES": 32,
        "PROCESS_RSS_MAX_SAMPLES": 4,
    }
    prefix, marker, _ = RUN_CELL.partition('\nfirst = run_solver("first", "first")')
    assert marker
    exec(compile(prefix, "<generated-run-cell>", "exec"), namespace)
    namespace["HISTORY_ROOT"] = tmp_path / "history"
    namespace["free_port"] = lambda: 12345
    namespace["base_env"] = lambda *_args: {}
    namespace["process_tree_rss"] = lambda _pid: (1234, 1)
    namespace["gpu_snapshot"] = lambda: ["0, Tesla T4", "1, Tesla T4"]

    def fake_discover(_torchrun_dir: Path, run_dir: Path) -> dict[str, dict[int, Path]]:
        copied: dict[str, dict[int, Path]] = {"stdout": {}, "stderr": {}}
        for stream in ("stdout", "stderr"):
            for rank in (0, 1):
                path = run_dir / f"rank{rank}.{stream}.log"
                path.write_text("", encoding="utf-8")
                copied[stream][rank] = path
        return copied

    namespace["discover_rank_logs"] = fake_discover
    monkeypatch.setattr(namespace["subprocess"], "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(namespace["selectors"], "DefaultSelector", lambda: selector)
    monkeypatch.setattr(namespace["signal"], "SIGKILL", 9, raising=False)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        namespace["os"], "killpg", lambda pid, sig: signals.append((pid, sig)), raising=False
    )
    return namespace, signals


def test_generated_run_solver_applies_byte_cap_to_eof_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdout = _FakeStdout(tail="x" * 33)
    process = _FakeProcess(stdout, stay_running=False)
    selector = _FakeSelector(stdout, readable=False)
    namespace, _ = _load_generated_run_solver(tmp_path, monkeypatch, process, selector)
    with pytest.raises(RuntimeError, match="bounded capture"):
        namespace["run_solver"]("eof_tail", "first")
    assert process.reaped and selector.closed
    assert all(timeout is not None for timeout in process.wait_timeouts)


def test_generated_run_solver_read_exception_kill_fallback_and_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdout = _FakeStdout(read_error=OSError("capture read failed"))
    process = _FakeProcess(stdout, stay_running=True)
    selector = _FakeSelector(stdout, readable=True)
    namespace, signals = _load_generated_run_solver(tmp_path, monkeypatch, process, selector)
    with pytest.raises(OSError, match="capture read failed"):
        namespace["run_solver"]("capture_error", "first")
    signal_module = namespace["signal"]
    assert signals == [(process.pid, signal_module.SIGTERM), (process.pid, signal_module.SIGKILL)]
    assert process.wait_timeouts == [10, 10]
    assert process.reaped and selector.closed


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


def _write_remote_attestation(
    root: Path,
    *,
    slug: str = KERNEL_SLUG,
    version: int = 3,
    private: bool = True,
    status: str = "COMPLETE",
    last_run_time: str = "2026-07-29 01:05:00.000000",
    push_observed_at: str = "2026-07-29T01:00:00+00:00",
    completion_observed_at: str = "2026-07-29T01:06:00+00:00",
    pulled_notebook: bytes | None = None,
) -> None:
    remote = root / "remote"
    remote.mkdir(exist_ok=True)
    raw_files = {
        "push_receipt.txt": (
            "Warning: Looks like you're using an outdated `kaggle` version "
            "(installed: 2.1.2), please consider upgrading to the latest version (2.2.2)\n"
            f"Kernel version {version} successfully pushed.  Please check progress at "
            f"https://www.kaggle.com/code/{slug}\n"
        ).encode(),
        "status.txt": (
            "Warning: Looks like you're using an outdated `kaggle` version "
            "(installed: 2.1.2), please consider upgrading to the latest version (2.2.2)\n"
            f'{slug} has status "KernelWorkerStatus.{status}"\n'
        ).encode(),
        "list.csv": (
            "Warning: Looks like you're using an outdated `kaggle` version "
            "(installed: 2.1.2), please consider upgrading to the latest version (2.2.2)\n"
            "ref,title,author,lastRunTime,totalVotes\n"
            f"\n{slug},CayleyPy Public Task 5 2xT4 Gate,Ivan Litvak,{last_run_time},0\n\n"
        ).encode(),
        "kernel-metadata.json": (
            json.dumps({
                "id": slug,
                "title": "CayleyPy Public Task 5 2xT4 Gate",
                "code_file": "cayleypy-public-task-5-2xt4-gate.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": private,
                "enable_gpu": True,
                "machine_shape": "NvidiaTeslaT4",
            }) + "\n"
        ).encode(),
        "pulled-notebook.ipynb": pulled_notebook or OUT_NOTEBOOK.read_bytes(),
    }
    for name, payload in raw_files.items():
        (remote / name).write_bytes(payload)
    (remote / "capture_manifest.json").write_text(json.dumps({
        "push_observed_at_utc": push_observed_at,
        "completion_observed_at_utc": completion_observed_at,
        "sha256": {name: sha256(payload).hexdigest() for name, payload in raw_files.items()},
    }) + "\n", encoding="utf-8")


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
    _write_remote_attestation(root)


def test_downloaded_gate_validator_requires_raw_deterministic_two_rank_evidence(tmp_path: Path) -> None:
    _write_fake_gate(tmp_path)
    validated = validate_gate_output(tmp_path)
    assert validated["status"] == "ok"
    assert validated["collect_tsv_sha256"] == sha256(
        (tmp_path / "runs/collect_a/solutions.tsv").read_bytes()
    ).hexdigest()


def test_downloaded_gate_validator_requires_exact_raw_remote_attestation(
    tmp_path: Path,
) -> None:
    cases = (
        {"slug": "trydotatwo/wrong-kernel"},
        {"version": 2},
        {"private": False},
        {"status": "RUNNING"},
        {"pulled_notebook": b'{"cells": []}\n'},
        {
            "last_run_time": "2026-07-29 01:07:00.000000",
            "completion_observed_at": "2026-07-29T01:06:00+00:00",
        },
    )
    for index, overrides in enumerate(cases):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        _write_fake_gate(case_root)
        _write_remote_attestation(case_root, **overrides)
        with pytest.raises(ValueError, match="remote Kaggle attestation"):
            validate_gate_output(case_root)

    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    _write_fake_gate(valid_root)
    validated = validate_gate_output(valid_root)
    assert validated["remote_attestation"] == {
        "slug": KERNEL_SLUG,
        "private": True,
        "pushed_version": 3,
        "status": "COMPLETE",
        "last_run_time_utc": "2026-07-29T01:05:00+00:00",
        "completion_observed_at_utc": "2026-07-29T01:06:00+00:00",
        "pushed_notebook_sha256": sha256(OUT_NOTEBOOK.read_bytes()).hexdigest(),
        "pulled_notebook_sha256": sha256(OUT_NOTEBOOK.read_bytes()).hexdigest(),
        "pulled_notebook_semantic_match": True,
    }


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
