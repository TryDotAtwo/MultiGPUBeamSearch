"""Checkpoint-only public CayleyPy 2xT4 orchestration CLI."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import re
import time
from typing import Iterable, Mapping, Sequence
import uuid

import pandas as pd

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.data import PuzzleContract, load_puzzle_contract
from tools.cayleypy_public.model import ExportedModel, export_checkpoint
from tools.cayleypy_public.paths import SolutionRecord, tokenize_path
from tools.cayleypy_public.profile import RuntimePlan, derive_runtime, serialize_preflight
from tools.cayleypy_public.results import (
    MAX_PUBLISH_REQUEST_BYTES,
    MAX_RESULTS_PER_REQUEST,
    build_result_archives,
    build_result_envelope,
    publish_results,
    publish_result_archive,
)
from tools.cayleypy_public.runner import PublicSearchRunError, RunArtifacts, run_public_search
from tools.kaggle_t4_mlp_profiles import select_profile


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MLP_PROFILE_REGISTRY = _REPO_ROOT / "configs" / "kaggle_t4_mlp_profiles.json"
_TRANSFORMER_PROFILE_REGISTRY = _REPO_ROOT / "configs" / "kaggle_t4_transformer_profiles.json"
_CUTLASS_URL = "https://github.com/NVIDIA/cutlass.git"
_CUTLASS_REV = "afa1772203677c5118fcd82537a9c8fefbcc7008"
_HISTORY_RAM_MAX_BYTES = 29_000_000_000
_HISTORY_RAM_HEADROOM_BYTES = 1_500_000_000
_HISTORY_DISK_BYTES = 50 * 1024**3


def _available_ram_bytes() -> int:
    """Read Linux MemAvailable without importing an optional runtime package."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        try:
            import psutil
        except ImportError as error:
            raise RuntimeError(
                "/proc/meminfo or psutil is required to derive the history RAM budget"
            ) from error
        return int(psutil.virtual_memory().available)
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _derive_history_budgets(available_ram_bytes: int, tmp_free_bytes: int) -> tuple[int, int]:
    """Derive effective budgets before serializing preflight or launching ranks."""
    if available_ram_bytes <= _HISTORY_RAM_HEADROOM_BYTES:
        raise ValueError("available RAM cannot preserve the configured history headroom")
    ram_budget = min(
        _HISTORY_RAM_MAX_BYTES,
        available_ram_bytes - _HISTORY_RAM_HEADROOM_BYTES,
    )
    if tmp_free_bytes < _HISTORY_DISK_BYTES:
        raise ValueError(
            f"/tmp free disk is smaller than the history disk budget "
            f"({tmp_free_bytes} < {_HISTORY_DISK_BYTES} bytes)"
        )
    return ram_budget, _HISTORY_DISK_BYTES

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _public_error(error: BaseException, config: PublicRunConfig | None = None) -> str:
    message = str(error).splitlines()[0] if str(error) else error.__class__.__name__
    redactions: tuple[Path, ...] = ()
    if config is not None:
        redactions = tuple(path for path in (
            config.checkpoint_path, config.puzzle_info_json, config.test_csv,
            config.sample_submission_csv, config.reflect_source_csv,
        ) if path is not None)
        if config.results_ingest_url:
            message = message.replace(config.results_ingest_url, "<results-endpoint>")
    return f"{error.__class__.__name__}: {_sanitize_log(message, redactions)}"[:2048]


def validate_t4_hardware() -> list[str]:
    """Require the exact public runtime target before checkpoint or build work."""
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(names) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in names):
        raise RuntimeError(f"public runner requires exactly two Tesla T4 GPUs; observed {names!r}")
    return names


def _sanitize_log(text: str, redactions: Sequence[Path] = ()) -> str:
    sanitized = text
    for path in (*redactions, Path.home(), _REPO_ROOT):
        value = str(path)
        if value:
            sanitized = sanitized.replace(value, "<redacted-path>")
            sanitized = sanitized.replace(value.replace("\\", "/"), "<redacted-path>")
    sanitized = re.sub(
        r"(?i)(?:ghp_|github_pat_|sk-proj-|cf_api_token)[A-Za-z0-9_.-]+",
        "<redacted-secret>",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(?:CF_API_TOKEN|CLOUDFLARE_API_TOKEN|GITHUB_TOKEN|GH_TOKEN|AUTHORIZATION)"
        r"\s*[:=]\s*(?:Bearer\s+)?[^\s]+",
        "<redacted-secret>",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*",
        "Bearer <redacted-secret>",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@",
        r"\1<redacted-credentials>@",
        sanitized,
    )
    return sanitized


def _run_logged(command: Sequence[object], log_path: Path, *, cwd: Path | None = None, redactions: Sequence[Path] = ()) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = [str(part) for part in command]
    completed = subprocess.run(
        rendered,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(_sanitize_log(completed.stdout or "", redactions), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}; see {log_path.name}")


def _git_stdout(arguments: Sequence[object], *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *map(str, arguments)], cwd=str(cwd), check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def locate_or_build_runner(output_dir: Path, puzzle_info_json: Path | None = None, *, backend: str = "mlp", config: PublicRunConfig | None = None) -> Path:
    """Build the existing runner in Release mode for T4 (SM75), without source edits."""
    logs = output_dir / "logs"
    cutlass = Path(os.environ.get("CAYLEYPY_CUTLASS_DIR", "/tmp/cayleypy_public_cutlass"))
    default_build = f"/tmp/cayleypy_public_build_sm75_{backend}"
    build = Path(os.environ.get("CAYLEYPY_BUILD_DIR", default_build))
    private_paths = tuple(path for path in (cutlass, build, puzzle_info_json) if path is not None)
    if not cutlass.exists():
        _run_logged(
            ["git", "clone", "--filter=blob:none", "--no-checkout", _CUTLASS_URL, cutlass],
            logs / "cutlass-clone.log", redactions=private_paths,
        )
    if not (cutlass / ".git").exists():
        raise RuntimeError("CUTLASS path exists but is not a Git checkout")
    try:
        head = _git_stdout(["rev-parse", "HEAD"], cwd=cutlass)
    except subprocess.CalledProcessError:
        head = ""
    if head != _CUTLASS_REV:
        _run_logged(
            ["git", "fetch", "--depth", "1", "origin", _CUTLASS_REV],
            logs / "cutlass-fetch.log", cwd=cutlass, redactions=private_paths,
        )
        _run_logged(
            ["git", "checkout", "--detach", "FETCH_HEAD"],
            logs / "cutlass-checkout.log", cwd=cutlass, redactions=private_paths,
        )
    if _git_stdout(["rev-parse", "HEAD"], cwd=cutlass) != _CUTLASS_REV:
        raise RuntimeError("CUTLASS checkout does not match the pinned revision")

    if backend not in {"mlp", "piece_transformer"}:
        raise ValueError(f"unsupported Stream1 backend: {backend}")
    target = "production_runner" if backend == "mlp" else "production_runner_libtorch_stream1"
    configure: list[object] = [
        "cmake", "-S", _REPO_ROOT, "-B", build, "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release", "-DBEAM_CUDA_ARCHITECTURES=75",
        f"-DCUTLASS_DIR={cutlass}",
        f"-DBEAM_ENABLE_DEBUG={'ON' if config is None or config.enable_debug else 'OFF'}",
        f"-DBEAM_ENABLE_DEPTH_LOGS={'ON' if config is None or config.enable_depth_logs else 'OFF'}",
        f"-DBEAM_ENABLE_DEBUG_LOGS={'ON' if config is not None and config.enable_debug_logs else 'OFF'}",
        f"-DBEAM_DEBUG_STREAM_TIMING={'ON' if config is not None and config.debug_stream_timing else 'OFF'}",
        f"-DBEAM_DEBUG_INFERENCE_TRACE={'ON' if config is not None and config.debug_inference_trace else 'OFF'}",
        f"-DBEAM_DEBUG_PATH_TRACE={'ON' if config is not None and config.debug_path_trace else 'OFF'}",
        f"-DBEAM_DEBUG_FINAL_VALIDATE={'ON' if config is not None and config.debug_final_validate else 'OFF'}",
        f"-DBEAM_DEBUG_FINAL_EXCHANGE_TRACE={'ON' if config is not None and config.debug_final_exchange_trace else 'OFF'}",
        f"-DBEAM_DEBUG_FINAL_HISTOGRAM_TRACE={'ON' if config is not None and config.debug_final_histogram_trace else 'OFF'}",
        f"-DBEAM_DEBUG_STREAM4_HISTOGRAM_TRACE={'ON' if config is not None and config.debug_stream4_histogram_trace else 'OFF'}",
        f"-DBEAM_DEBUG_DEPTH_FLOW_TRACE={'ON' if config is not None and config.debug_depth_flow_trace else 'OFF'}",
        f"-DBEAM_DEBUG_PIPELINE_STATS={'ON' if config is not None and config.debug_pipeline_stats else 'OFF'}",
    ]
    if backend == "piece_transformer":
        import torch
        nccl_root = Path(torch.__file__).resolve().parents[1] / "nvidia" / "nccl"
        nccl_candidates = sorted((nccl_root / "lib").glob("libnccl.so*"))
        nccl_include = nccl_root / "include"
        if not nccl_candidates or not (nccl_include / "nccl.h").is_file():
            raise RuntimeError(f"PyTorch-compatible NCCL not found under {nccl_root}")
        nccl_library = next((item for item in nccl_candidates if item.name == "libnccl.so.2"), nccl_candidates[0])
        configure.extend([
            "-DBEAM_ENABLE_LIBTORCH_STREAM1=ON",
            f"-DCMAKE_PREFIX_PATH={Path(torch.__file__).resolve().parent / 'share' / 'cmake'}",
            f"-DNCCL_LIBRARY={nccl_library}", f"-DNCCL_INCLUDE_DIR={nccl_include}",
        ])
        os.environ["LD_LIBRARY_PATH"] = f"{nccl_library.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if puzzle_info_json is not None:
        configure.append(f"-DBEAM_PUZZLE_INFO_JSON={puzzle_info_json.resolve()}")
    _run_logged(configure, logs / "cmake-configure.log", redactions=private_paths)
    _run_logged(
        ["cmake", "--build", build, "--target", target, "-j", "2"],
        logs / "cmake-build.log", redactions=private_paths,
    )
    runner = (build / target).resolve()
    if not runner.is_file() or not runner.is_absolute():
        raise RuntimeError("Release SM75 runner was not materialized")
    return runner


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _solution_rows(records: Iterable[SolutionRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        rows.append({
            "collection_index": index,
            "puzzle_id": record.puzzle_id,
            "variant": record.variant,
            "path": record.path,
            "original_oriented_path": record.original_oriented_path,
            "solution_length": len(tokenize_path(record.original_oriented_path)),
            "found_depth": record.found_depth,
            "touch_depth": record.touch_depth,
            "source_solution_sha256": record.source_solution_sha256,
            "reflected_source_path": record.reflected_source_path,
            "valid": record.valid,
            "reached_state": ",".join(map(str, record.reached_state)),
        })
    return rows


def _best_solution_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    best: dict[int, Mapping[str, object]] = {}
    for row in rows:
        puzzle_id = int(row["puzzle_id"])
        candidate = (
            int(row["solution_length"]), str(row["original_oriented_path"]),
            str(row["variant"]), int(row["collection_index"]),
        )
        current = best.get(puzzle_id)
        if current is None or candidate < (
            int(current["solution_length"]), str(current["original_oriented_path"]),
            str(current["variant"]), int(current["collection_index"]),
        ):
            best[puzzle_id] = row
    return [dict(best[puzzle_id]) for puzzle_id in sorted(best)]


def _invocation_rows(artifacts: RunArtifacts, output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    count = max(
        len(artifacts.return_codes), len(artifacts.timing_summaries),
        len(artifacts.combined_logs), len(artifacts.collection_statuses),
    )
    for index in range(count):
        combined = artifacts.combined_logs[index] if index < len(artifacts.combined_logs) else None
        rank_pair = artifacts.rank_logs[index] if index < len(artifacts.rank_logs) else None
        puzzle_id: int | None = None
        variant: str | None = None
        if combined is not None:
            parts = combined.parts
            for position, part in enumerate(parts):
                if part.startswith("puzzle-"):
                    try:
                        puzzle_id = int(part.removeprefix("puzzle-"))
                    except ValueError:
                        pass
                    if position + 1 < len(parts):
                        variant = parts[position + 1]
                    break
        rows.append({
            "invocation_index": index,
            "puzzle_id": puzzle_id,
            "variant": variant,
            "return_code": artifacts.return_codes[index] if index < len(artifacts.return_codes) else None,
            "seconds": artifacts.timing_summaries[index] if index < len(artifacts.timing_summaries) else None,
            "collection_status": (
                artifacts.collection_statuses[index]
                if index < len(artifacts.collection_statuses) else "not_collected"
            ),
            "combined_log": _relative(combined, output_dir) if combined is not None else None,
            "rank0_log": _relative(rank_pair[0], output_dir) if rank_pair is not None else None,
            "rank1_log": _relative(rank_pair[1], output_dir) if rank_pair is not None else None,
        })
    return rows


def _ensure_export_manifest(path: Path, model: ExportedModel) -> None:
    payload = dict(model.manifest)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("export manifest differs from the validated model manifest")
        return
    _write_json(path, payload)


def _materialize_run_artifacts(
    artifacts: RunArtifacts,
    output_dir: Path,
) -> dict[str, object]:
    solutions_dir = output_dir / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    solution_rows = _solution_rows(artifacts.solution_records)
    pd.DataFrame(solution_rows).to_csv(solutions_dir / "all_solutions.csv", index=False)
    pd.DataFrame(_best_solution_rows(solution_rows)).to_csv(solutions_dir / "solutions.csv", index=False)
    invocation_rows = _invocation_rows(artifacts, output_dir)
    pd.DataFrame(invocation_rows).to_csv(output_dir / "beam_run_results.csv", index=False)
    artifacts.submission.to_csv(output_dir / "submission.csv", index=False)
    return {
        "solution_count": len(solution_rows),
        "solved_puzzle_count": len({int(row["puzzle_id"]) for row in solution_rows}),
        "invocation_count": len(invocation_rows),
        "collection_status": artifacts.collection_status,
        "solve_seconds": float(sum(artifacts.timing_summaries)),
    }


def _puzzle_type(contract: PuzzleContract) -> str:
    fingerprint = sha256(_canonical_bytes({
        "central_state": list(contract.central_state),
        "generators": {name: list(value) for name, value in contract.generators.items()},
    })).hexdigest()[:16]
    return f"cayleypy-{contract.state_len}-{contract.move_count}-{fingerprint}"


def _publication_context(
    config: PublicRunConfig,
    contract: PuzzleContract,
    model: ExportedModel,
    profile: Mapping[str, object],
    plan: RuntimePlan,
    hardware_names: Sequence[str],
    wall_seconds: float,
    solve_seconds: float,
) -> dict[str, object]:
    notebook_sha256 = config.kaggle_notebook_sha256
    required = {
        "competition": config.competition,
        "kaggle_owner": config.kaggle_owner,
        "kaggle_slug": config.kaggle_slug,
        "kaggle_version": config.kaggle_version,
        "solver_commit": config.solver_commit,
        "kaggle_notebook_sha256": notebook_sha256,
    }
    missing = sorted(name for name, value in required.items() if value in {None, ""})
    if missing:
        raise ValueError(f"publication provenance is incomplete: {', '.join(missing)}")
    author: dict[str, object] = {"name": config.author_name}
    if config.kaggle_username is not None:
        author["kaggle_username"] = config.kaggle_username
    runtime = {
        "touch_bfs_radius": config.touch_bfs_radius,
        "solution_mode": config.solution_mode,
        "max_depth": config.max_depth,
        "max_collected_solutions": config.max_collected_solutions,
        **dict(plan.runtime),
    }
    profile_payload = {
        "requested_beam": plan.requested_beam,
        "effective_beam": plan.effective_beam,
        "alignment_delta": plan.alignment_delta,
        "selected_profile": f"p{plan.profile_power}-{plan.model_class}",
        "evidence": "measured-kaggle-2xt4",
        "profile_evidence_version": profile.get("profile_registry_schema_version"),
        "profile_power": plan.profile_power,
        "model_class": plan.model_class,
        "world_size": 2,
    }
    return {
        "run_id": f"run-{uuid.uuid4().hex}",
        "author": author,
        "kaggle": {
            "owner": config.kaggle_owner,
            "slug": config.kaggle_slug,
            "version": config.kaggle_version,
            "run_url": f"https://www.kaggle.com/code/{config.kaggle_owner}/{config.kaggle_slug}",
            "notebook_sha256": notebook_sha256,
        },
        "competition": config.competition,
        "puzzle_type": _puzzle_type(contract),
        "proof": {
            "central_state": list(contract.central_state),
            "generators": {name: list(value) for name, value in contract.generators.items()},
        },
        "search_mode": config.reflect_mode,
        "profile": profile_payload,
        "runtime": runtime,
        "model": {
            "filename": config.checkpoint_path.name,
            "sha256": model.checkpoint_sha256,
            "format": model.format,
            "manifest": dict(model.manifest),
        },
        "hardware": {
            "platform": "kaggle", "gpu_names": list(hardware_names),
            "accelerator_count": 2, "world_size": 2,
        },
        "timings": {
            "solve_us": max(0, round(solve_seconds * 1_000_000)),
            "wall_us": max(0, round(wall_seconds * 1_000_000)),
        },
        "solver_commit": config.solver_commit,
    }


def _publication_records(
    solution_mode: str,
    records: Sequence[SolutionRecord],
) -> tuple[SolutionRecord, ...]:
    eligible = tuple(record for record in records if record.valid and record.variant != "source")
    if solution_mode == "collect":
        return eligible
    if solution_mode != "first":
        raise ValueError(f"unsupported solution mode: {solution_mode}")
    best: dict[int, SolutionRecord] = {}
    for record in eligible:
        key = (
            len(tokenize_path(record.original_oriented_path)),
            record.original_oriented_path,
            record.variant,
            record.found_depth,
            record.touch_depth,
        )
        current = best.get(record.puzzle_id)
        if current is None or key < (
            len(tokenize_path(current.original_oriented_path)),
            current.original_oriented_path,
            current.variant,
            current.found_depth,
            current.touch_depth,
        ):
            best[record.puzzle_id] = record
    return tuple(best[puzzle_id] for puzzle_id in sorted(best))

def _publication_envelopes(
    config: PublicRunConfig,
    contract: PuzzleContract,
    model: ExportedModel,
    profile: Mapping[str, object],
    plan: RuntimePlan,
    hardware_names: Sequence[str],
    artifacts: RunArtifacts,
    wall_seconds: float,
) -> list[dict[str, object]]:
    context = _publication_context(
        config, contract, model, profile, plan, hardware_names,
        wall_seconds, float(sum(artifacts.timing_summaries)),
    )
    envelopes: list[dict[str, object]] = []
    collection_status = (
        "first_solution" if config.solution_mode == "first" else artifacts.collection_status
    )
    indices: dict[int, int] = {}
    for record in artifacts.solution_records:
        if not record.valid or record.variant == "source":
            continue
        index = indices.get(record.puzzle_id, 0)
        indices[record.puzzle_id] = index + 1
        result_context = dict(context)
        proof = dict(context["proof"])
        proof["initial_state"] = list(contract.initial_states[record.puzzle_id])
        result_context["proof"] = proof
        solution: dict[str, object] = {
            "puzzle_id": record.puzzle_id,
            "path": record.path,
            "original_oriented_path": record.original_oriented_path,
            "found_depth": record.found_depth,
            "touch_depth": record.touch_depth,
            "variant": record.variant,
            "valid": True,
            "reached_state": list(record.reached_state),
            "collection_status": collection_status,
            "collection_index": index,
        }
        if record.reflected_source_path is not None:
            solution["reflected_source_path"] = record.reflected_source_path
        if record.source_solution_sha256 is not None:
            solution["source_solution_sha256"] = record.source_solution_sha256
        envelopes.append(build_result_envelope(result_context, solution))
    return envelopes


def _chunk_envelopes(envelopes: Sequence[dict[str, object]]) -> list[list[dict[str, object]]]:
    chunks: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for envelope in envelopes:
        candidate = [*current, envelope]
        size = len(_canonical_bytes({"schema_version": 1, "results": candidate}))
        if current and (len(candidate) > MAX_RESULTS_PER_REQUEST or size > MAX_PUBLISH_REQUEST_BYTES):
            chunks.append(current)
            current = [envelope]
        else:
            current = candidate
        if len(_canonical_bytes({"schema_version": 1, "results": current})) > MAX_PUBLISH_REQUEST_BYTES:
            raise ValueError("one result envelope cannot fit the publish request bound")
    if current:
        chunks.append(current)
    return chunks


@contextmanager
def _history_budget_environment(ram_bytes: int, disk_bytes: int):
    names = ("CAYLEYPY_HISTORY_RAM_BYTES", "CAYLEYPY_HISTORY_DISK_BYTES")
    previous = {name: os.environ.get(name) for name in names}
    os.environ[names[0]] = str(ram_bytes)
    os.environ[names[1]] = str(disk_bytes)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_with_history_budgets(ram_bytes: int, disk_bytes: int, *args, **kwargs):
    with _history_budget_environment(ram_bytes, disk_bytes):
        return run_public_search(*args, **kwargs)

@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _publish_best_effort(
    config: PublicRunConfig,
    contract: PuzzleContract,
    model: ExportedModel,
    profile: Mapping[str, object],
    plan: RuntimePlan,
    hardware_names: Sequence[str],
    artifacts: RunArtifacts,
    output_dir: Path,
    wall_seconds: float,
) -> dict[str, object]:
    if not config.publish_results:
        status = {"state": "skipped", "ok": False, "safe_error": None, "reason": "disabled", "result_count": 0}
        _write_json(output_dir / "publish_status.json", status)
        return status
    try:
        envelopes = _publication_envelopes(
            config, contract, model, profile, plan, hardware_names, artifacts, wall_seconds,
        )
        if not envelopes:
            status = {"state": "skipped", "ok": False, "safe_error": None, "reason": "no_valid_solutions", "result_count": 0}
            _write_json(output_dir / "publish_status.json", status)
            return status
        archives = build_result_archives(envelopes)
        statuses = []
        with _working_directory(output_dir):
            for archive_index, archive in enumerate(archives):
                statuses.append(publish_result_archive(
                    config.results_ingest_url,
                    archive,
                    result_count=len(envelopes),
                    archive_index=archive_index,
                    archive_count=len(archives),
                ))
        ok = all(status.ok for status in statuses)
        status_payload = {
            "state": "published" if ok else "failed",
            "ok": ok,
            "retryable": any(status.retryable for status in statuses),
            "safe_error": None if ok else "; ".join(
                sorted({status.safe_error or "publish_failed" for status in statuses if not status.ok})
            )[:2048],
            "result_count": len(envelopes),
            "archive_count": len(archives),
            "client_submission_ids": [item["client_submission_id"] for item in envelopes],
            "archives": [asdict(status) for status in statuses],
        }
    except Exception as error:
        status_payload = {
            "state": "failed", "ok": False, "retryable": True,
            "safe_error": _public_error(error, config), "result_count": 0, "archive_count": 0,
        }
    _write_json(output_dir / "publish_status.json", status_payload)
    return status_payload


def _summary_base(
    config: PublicRunConfig,
    hardware_names: Sequence[str],
    model: ExportedModel,
    plan: RuntimePlan,
) -> dict[str, object]:
    return {
        "hardware": {"gpu_names": list(hardware_names), "world_size": 2},
        "model": {
            "format": model.format, "dtype": model.dtype,
            "checkpoint_sha256": model.checkpoint_sha256,
            "output_dim": model.manifest.get("output_dim"),
        },
        "requested_beam": plan.requested_beam,
        "effective_beam": plan.effective_beam,
        "alignment_delta": plan.alignment_delta,
        "profile_power": plan.profile_power,
        "model_class": plan.model_class,
        "puzzle_id_start": config.puzzle_id_start,
        "puzzle_id_end": config.puzzle_id_end,
        "max_depth": config.max_depth,
        "reflect_mode": config.reflect_mode,
        "solution_mode": config.solution_mode,
        "touch_bfs_radius": config.touch_bfs_radius,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config: PublicRunConfig | None = None
    try:
        raw_config = json.loads(arguments.config_json.read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict):
            raise ValueError("config JSON must contain an object")
        config = PublicRunConfig.from_mapping(raw_config)
        hardware_names = validate_t4_hardware()
        contract = load_puzzle_contract(
            config.puzzle_info_json, config.test_csv, config.sample_submission_csv,
            config.puzzle_id_start, config.puzzle_id_end,
        )
        export_dir = output_dir / "export"
        model = export_checkpoint(
            config.checkpoint_path, export_dir, contract.num_classes,
            state_len=contract.state_len, move_count=contract.move_count,
            metadata_json=config.checkpoint_metadata_json,
            generator_json=config.checkpoint_generator_json,
            source_root=config.checkpoint_source_root,
        )
        _ensure_export_manifest(export_dir / "manifest.json", model)
        if model.backend not in {"mlp", "piece_transformer"}:
            raise ValueError(f"unsupported model backend: {model.backend!r}")
        profile_registry_path = (
            _TRANSFORMER_PROFILE_REGISTRY
            if model.backend == "piece_transformer"
            else _MLP_PROFILE_REGISTRY
        )
        registry = json.loads(profile_registry_path.read_text(encoding="utf-8"))
        if registry.get("backend") != model.backend:
            raise ValueError(
                f"profile backend mismatch: registry={registry.get('backend')!r} model={model.backend!r}"
            )
        output_dim = model.manifest.get("output_dim")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int):
            raise ValueError("export manifest output_dim must be an integer")
        profile = select_profile(registry, config.beam_width, output_dim, contract.move_count)
        plan = derive_runtime(profile, config.beam_width, output_dim, contract.move_count, 2)
        _write_json(output_dir / "selected_profile.json", profile)
        tmp_free_bytes = shutil.disk_usage("/tmp").free
        history_ram_bytes, history_disk_bytes = _derive_history_budgets(
            _available_ram_bytes(), tmp_free_bytes,
        )
        preflight = serialize_preflight(
            plan, profile, contract.move_count,
            history_ram_bytes, history_disk_bytes, tmp_free_bytes,
        )
        preflight["gpu_names"] = hardware_names
        preflight["model_format"] = model.format
        preflight["model_backend"] = model.backend
        preflight["model_dtype"] = model.dtype
        preflight["output_dim"] = output_dim
        _write_json(output_dir / "preflight.json", preflight)
        backend = model.backend
        runner = locate_or_build_runner(
            output_dir, config.puzzle_info_json, backend=backend, config=config,
        )
        log_redactions = tuple(path for path in (
            arguments.config_json, config.checkpoint_path, config.puzzle_info_json,
            config.test_csv, config.sample_submission_csv, config.reflect_source_csv,
            output_dir, export_dir, _REPO_ROOT, Path.home(),
        ) if path is not None)
        log_sanitizer = lambda text: _sanitize_log(text, log_redactions)
        try:
            artifacts = _run_with_history_budgets(
                history_ram_bytes, history_disk_bytes,
                config, contract, model, plan, export_dir, output_dir / "logs",
                runner_path=str(runner), log_sanitizer=log_sanitizer,
            )
        except PublicSearchRunError as error:
            artifact_summary = _materialize_run_artifacts(error.partial_artifacts, output_dir)
            wall_seconds = time.perf_counter() - started
            publish_status = _publish_best_effort(
                config, contract, model, profile, plan, hardware_names,
                error.partial_artifacts, output_dir, wall_seconds,
            )
            summary = {
                "status": "failed", "safe_error": _public_error(error, config),
                **_summary_base(config, hardware_names, model, plan), **artifact_summary,
                "wall_seconds": wall_seconds, "publish_status": publish_status,
            }
            _write_json(output_dir / "run_summary.json", summary)
            return 2

        artifact_summary = _materialize_run_artifacts(artifacts, output_dir)
        wall_seconds = time.perf_counter() - started
        publish_status = _publish_best_effort(
            config, contract, model, profile, plan, hardware_names,
            artifacts, output_dir, wall_seconds,
        )
        summary = {
            "status": "success", "safe_error": None,
            **_summary_base(config, hardware_names, model, plan), **artifact_summary,
            "wall_seconds": wall_seconds, "publish_status": publish_status,
            "artifacts": [
                "selected_profile.json", "preflight.json", "export/manifest.json",
                "beam_run_results.csv", "solutions/all_solutions.csv",
                "solutions/solutions.csv", "submission.csv", "publish_status.json", "logs/",
            ],
        }
        _write_json(output_dir / "run_summary.json", summary)
        return 0
    except Exception as error:
        summary = {
            "status": "failed", "safe_error": _public_error(error, config),
            "wall_seconds": time.perf_counter() - started,
            "publish_status": {"state": "skipped", "reason": "preflight_or_build_failed"},
        }
        _write_json(output_dir / "run_summary.json", summary)
        print(summary["safe_error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
