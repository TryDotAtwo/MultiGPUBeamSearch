"""Local native execution. No publication, cluster jobs or runtime fallback."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from .errors import NativeBackendError, NativeUnavailable
from .options import NativeOutcome


_RANK_KEYS = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
              "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "NODE_RANK", "OMP_NUM_THREADS"}


@dataclass(frozen=True)
class PreparedRuntime:
    runner: Path
    build_metadata: dict
    architectures: tuple[int, ...]


def prepare_runtime(contract, model, options, run_dir, devices) -> PreparedRuntime:
    """Resolve/build before dispatch closes its capability fallback window."""
    if options.touch_bfs_radius and contract.move_count > 32:
        raise NativeUnavailable("native touch-BFS suffix packing supports at most 32 generators")
    # Stream1's scalar head has a separate kernel. The Q head uses unpadded
    # row-major CUTLASS GEMM with eight-element B/C/D access alignment.
    if model.backend == "mlp" and model.manifest["output_dim"] != 1 and model.manifest["output_dim"] % 8:
        raise NativeUnavailable("native MLP Q-head output_dim must be a multiple of 8 for CUTLASS row alignment; scalar output_dim=1 is supported")
    import torch
    try:
        architectures = tuple(sorted({major * 10 + minor for major, minor in
                                      (torch.cuda.get_device_capability(d) for d in devices)}))
    except (RuntimeError, AssertionError) as exc:
        raise NativeUnavailable(f"cannot inspect requested native CUDA architecture: {exc}") from exc
    if model.manifest.get("dtype") == "bf16" and any(sm < 80 for sm in architectures):
        raise NativeUnavailable("native BF16 Stream1 requires SM80+ on every selected GPU; use FP16 on SM75")
    if model.manifest.get("dtype") == "fp16" and any(sm < 75 for sm in architectures):
        raise NativeUnavailable("this native FP16 MLP backend uses SM75+ TensorOp kernels; older GPU executors need a separate adapter")
    from .build import ensure_runner
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    runner, metadata = ensure_runner(contract, model, options, architectures, run_dir)
    return PreparedRuntime(runner, metadata, architectures)


def runtime_devices(graph, options) -> tuple[int, ...]:
    if platform.system() != "Linux":
        raise NativeUnavailable("native beam currently requires Linux/CUDA/NCCL")
    if any(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID")):
        raise NativeUnavailable("native adapter cannot be invoked inside an existing torchrun rank")
    import torch
    if not torch.cuda.is_available():
        raise NativeUnavailable("native beam requires available CUDA devices")
    if options.devices is not None:
        devices = options.devices
    else:
        config = getattr(graph, "device_config", None)
        if config is None:
            # CayleyPy 0.1 exposes graph.device instead of device_config.
            device = getattr(graph, "device", None)
            configured = () if device is None else (device,)
        else:
            configured = getattr(config, "devices", ())
        if not configured or any(torch.device(d).type != "cuda" for d in configured):
            raise NativeUnavailable("graph does not select CUDA devices; set explicit native devices")
        devices = tuple(0 if torch.device(d).index is None else torch.device(d).index for d in configured)
    if not devices or len(set(devices)) != len(devices) or len(devices) > 128 or any(
        type(d) is not int or d < 0 or d >= torch.cuda.device_count() for d in devices
    ):
        raise NativeUnavailable("native devices must be distinct available CUDA indices (at most 128)")
    from .build import prerequisites
    prerequisites(options)
    return tuple(devices)


def cuda_visibility(devices: tuple[int, ...], environ: dict[str, str]) -> str:
    inherited = environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return ",".join(map(str, devices))
    tokens = [token.strip() for token in inherited.split(",")]
    if any(d >= len(tokens) or not tokens[d] or tokens[d] == "-1" for d in devices):
        raise NativeUnavailable("cannot map selected devices through inherited CUDA_VISIBLE_DEVICES")
    return ",".join(tokens[d] for d in devices)


def child_environment(devices: tuple[int, ...], run_dir: Path, environ=None) -> dict[str, str]:
    inherited = dict(os.environ if environ is None else environ)
    env = {k: v for k, v in inherited.items() if not (
        k.startswith(("BEAM_", "TORCHELASTIC_", "TORCHRUN_", "CUDA_", "NCCL_", "PUBLISH_", "RESULTS_INGEST_")) or k in _RANK_KEYS
        or k.startswith(("SHARD_", "STREAM1_", "STREAM3_", "STREAM4_", "GLOBAL_"))
    )}
    env["CUDA_VISIBLE_DEVICES"] = cuda_visibility(devices, inherited)
    if "CUDA_DEVICE_ORDER" in inherited:
        env["CUDA_DEVICE_ORDER"] = inherited["CUDA_DEVICE_ORDER"]
    env["BEAM_NCCL_ID_FILE"] = str(run_dir / "nccl-id.bin")
    return env


def microbatch_environment(contract, model, beam_width: int, world_size: int) -> tuple[dict[str, str], dict]:
    """Fit one inference slot into a native auto candidate without changing beam.

    Native runtime_config.cpp uses 8192 inference rows, four initial sort slots,
    alignment 1024 and capacity scale 1.25. At tiny beams it otherwise rejects
    every candidate when Stream3's one-slot batch exceeds physical shard storage.
    Only lower the row budget; leave shards, ring count and VRAM planning to native.
    """
    default_rows, alignment, initial_sort_slots = 8192, 1024, 4
    divisor = world_size * initial_sort_slots * alignment
    logical_shard = ((beam_width + divisor - 1) // divisor) * alignment
    scaled_capacity = (logical_shard * 5 + 3) // 4
    shard_storage = ((scaled_capacity + alignment - 1) // alignment) * alignment
    rows_per_parent = contract.move_count if model.manifest["output_dim"] == 1 else 1
    default_parents = default_rows // rows_per_parent
    parents = min(default_parents, shard_storage // contract.move_count)
    row_budget = default_rows if parents == default_parents else parents * rows_per_parent
    env = {} if row_budget == default_rows else {"BEAM_B_MICRO": str(row_budget)}
    return env, {"native_default_row_budget": default_rows, "configured_row_budget": row_budget,
                 "derived_parent_batch": parents, "derived_candidates_per_slot": parents * contract.move_count,
                 "reference_logical_shard": logical_shard, "reference_shard_storage": shard_storage,
                 "adjusted_for_small_beam": bool(env)}


def _stop_process_tree(process: subprocess.Popen) -> None:
    if os.name == "posix":
        # Descendants may outlive an exited torchrun parent, so always target the group.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            if sig == signal.SIGTERM:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        process.wait(timeout=5)
    else:
        # Windows is supported only for process tests; no native CUDA launch here.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False, timeout=10, shell=False)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def run_process(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float, log_path: Path) -> float:
    """Run shell-free with bounded lifetime; preserve stdout/stderr and command."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.with_suffix(log_path.suffix + ".command.json").write_text(json.dumps(command) + "\n", encoding="utf-8")
    started = time.monotonic()
    process = None
    try:
        with log_path.open("wb") as log:
            kwargs = {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, shell=False, **kwargs)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                _stop_process_tree(process)
                raise
        if code:
            _stop_process_tree(process)
            raise NativeBackendError(f"native process failed rc={code}; log={log_path}")
        return time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        raise NativeBackendError(f"native process timed out after {timeout}s; log={log_path}") from exc
    except OSError as exc:
        raise NativeBackendError(f"cannot execute native process; log={log_path}: {exc}") from exc


def collect_worker_logs(log_dir: Path, launcher_log: Path, combined_log: Path,
                        world_size: int, *, strict: bool) -> dict:
    """Merge only the exact torchrun attempt_0/rank/stdout|stderr layout.

    No recursive arbitrary-log discovery and no symlink traversal. A failed
    launcher still gets a combined diagnostic log; successful runs require both
    redirected streams from every expected rank and no additional ranks/retries.
    """
    issues, streams = [], []
    root = log_dir.resolve()

    def safe(path, directory=False):
        return (not path.is_symlink() and path.resolve().is_relative_to(root)
                and (path.is_dir() if directory else path.is_file()))

    try:
        sessions = list(log_dir.iterdir())
        if len(sessions) != 1 or not safe(sessions[0], directory=True):
            issues.append("expected exactly one torchrun log session directory")
        else:
            attempts = list(sessions[0].iterdir())
            attempt = sessions[0] / "attempt_0"
            if attempts != [attempt] or not safe(attempt, directory=True):
                issues.append("expected exactly one torchrun attempt_0 directory")
            else:
                ranks = list(attempt.iterdir())
                if {path.name for path in ranks} != {str(rank) for rank in range(world_size)}:
                    issues.append("torchrun rank directory set does not match requested world size")
                for rank in range(world_size):
                    directory = attempt / str(rank)
                    if not safe(directory, directory=True):
                        issues.append(f"missing or unsafe rank directory: {rank}")
                        continue
                    for name in ("stdout.log", "stderr.log"):
                        path = directory / name
                        if safe(path):
                            streams.append((f"rank {rank} {name}", path))
                        else:
                            issues.append(f"missing or unsafe rank stream: {rank}/{name}")
    except OSError as error:
        issues.append(f"cannot enumerate torchrun rank logs: {error}")
    with combined_log.open("w", encoding="utf-8") as destination:
        for label, path in [("launcher", launcher_log), *streams]:
            destination.write(f"[{label}]\n")
            try:
                with path.open(encoding="utf-8", errors="replace") as source:
                    for line in source:
                        destination.write(line)
                destination.write("\n")
            except OSError as error:
                issues.append(f"cannot read {label}: {error}")
        if issues:
            destination.write("[log collection diagnostics]\n" + "\n".join(issues) + "\n")
    if strict and issues:
        raise NativeBackendError(f"incomplete native worker logs: {'; '.join(issues)}; log={combined_log}")
    return {"launcher_log": str(launcher_log), "worker_log_dir": str(log_dir),
            "worker_streams": [str(path) for _, path in streams], "worker_logs_complete": not issues,
            "worker_log_diagnostics": issues}


def parse_terminal(log_path: Path, contract) -> tuple[tuple[int, ...] | None, int | None, dict]:
    records, effective = [], set()
    terminal = re.compile(r"puzzle_solved=([01])\s+puzzle_id=(\d+)\s+seconds=([^\s]+)\s+solution_length=(-?\d+)(.*?)\s+solution=([^\s]*)\s*$")
    width = re.compile(r"(?:GLOBAL_BEAM_WIDTH_EFFECTIVE|global_beam_width_effective)=(\d+)")
    batch_values = {"B_MICRO": set(), "STREAM1_ROWS_PER_JOB": set(), "STREAM3_BATCH_CANDIDATES": set()}
    batches = re.compile(r"\b(B_MICRO|STREAM1_ROWS_PER_JOB|STREAM3_BATCH_CANDIDATES)=(\d+)")
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            effective.update(int(m.group(1)) for m in width.finditer(line))
            for match in batches.finditer(line):
                batch_values[match.group(1)].add(int(match.group(2)))
            if "puzzle_solved=" not in line:
                continue
            match = terminal.search(line.rstrip())
            if match is None:
                raise NativeBackendError(f"malformed native terminal record; log={log_path}")
            found, puzzle, seconds, length, details, path = match.groups()
            try:
                elapsed = float(seconds)
            except ValueError as exc:
                raise NativeBackendError("native terminal record has invalid time") from exc
            if puzzle != "0" or not math.isfinite(elapsed) or elapsed < 0:
                raise NativeBackendError("native terminal record has invalid puzzle/time")
            if found == "0":
                if length != "-1" or path:
                    raise NativeBackendError("native not-found record contains an inconsistent path")
                parsed = None
            else:
                tokens = () if path == "" else tuple(path.split("."))
                if any(re.fullmatch(r"m\d+", token) is None for token in tokens):
                    raise NativeBackendError("native path contains unknown synthetic generator names")
                parsed = tuple(int(token[1:]) for token in tokens)
                if len(parsed) != int(length) or any(move >= contract.move_count for move in parsed):
                    raise NativeBackendError("native solution length/generator IDs are invalid")
                if not contract.replay(parsed):
                    raise NativeBackendError("native solution failed independent original-graph replay")
            records.append((parsed, elapsed))
    if not records:
        raise NativeBackendError(f"native process returned without a terminal result; log={log_path}")
    if any(record[0] != records[0][0] for record in records):
        raise NativeBackendError("native ranks emitted contradictory terminal results")
    if len(effective) > 1 or 0 in effective:
        raise NativeBackendError("native ranks emitted inconsistent effective global beam widths")
    metadata = {"native_seconds": max(r[1] for r in records), "terminal_records": len(records)}
    for label, name in (("B_MICRO", "observed_parent_batch"), ("STREAM1_ROWS_PER_JOB", "observed_inference_rows"),
                        ("STREAM3_BATCH_CANDIDATES", "observed_stream3_batch_candidates")):
        values = batch_values[label]
        # Preserve rank values when they differ; never invent an effective size.
        metadata[name] = next(iter(values)) if len(values) == 1 else None
        if len(values) > 1:
            metadata[name + "_by_rank_values"] = sorted(values)
    return records[0][0], next(iter(effective), None), metadata


def run_native(contract, model, options, beam_width, max_steps, run_dir, devices, *, runtime=None) -> NativeOutcome:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if type(beam_width) is not int or beam_width <= 0 or type(max_steps) is not int or max_steps < 0:
        raise NativeUnavailable("native beam_width must be positive and max_steps nonnegative integers")
    if contract.replay(()):
        return NativeOutcome((), 0.0, None, run_dir, {"already_solved": True, "replay_valid": True})
    if max_steps == 0:
        return NativeOutcome(None, 0.0, None, run_dir, {"budget_exhausted": True})
    if runtime is None:
        runtime = prepare_runtime(contract, model, options, run_dir, devices)
    else:
        # Fail closed if the verified executable was replaced after preparation.
        from .build import file_sha256, shape_contract
        if (runtime.build_metadata.get("shape") != shape_contract(contract)
                or runtime.build_metadata.get("backend") != model.backend):
            raise NativeBackendError("prepared native runner does not match graph/model")
        if not runtime.runner.is_file() or file_sha256(runtime.runner) != runtime.build_metadata.get("binary_sha256"):
            raise NativeBackendError("prepared native runner changed before launch")
    runner, build_metadata = runtime.runner, runtime.build_metadata
    puzzle = run_dir / "puzzle_info.json"
    puzzle.write_text(json.dumps(contract.to_puzzle_info()) + "\n", encoding="utf-8")
    inputs = run_dir / "test.csv"
    # The native CSV reader requires a bare numeric ID and a quoted state cell.
    inputs.write_text('initial_state_id,initial_state\n0,"' + ",".join(map(str, contract.start)) + '"\n', encoding="utf-8")
    env = child_environment(tuple(devices), run_dir)
    env.update({"BEAM_GENERATOR_PATH": str(puzzle), "BEAM_PUZZLE_INFO_JSON": str(puzzle),
                "BEAM_TEST_CSV": str(inputs), "BEAM_WEIGHT_DIR": str(model.weights_dir),
                "BEAM_RUNTIME_CONFIG_MODE": "auto", "BEAM_SOLVED_NEIGHBORHOOD_RADIUS": str(options.touch_bfs_radius),
                "BEAM_STREAM2_SUFFIX_RADIUS": "0", "BEAM_SOLVE_BUCKET_MODE": "0", "BEAM_HISTORY_MODE": "disk",
                "BEAM_HISTORY_DIR": str(run_dir / "history"), "BEAM_HISTORY_DISK_PATH": str(run_dir / "history")})
    microbatch_env, microbatch_metadata = microbatch_environment(contract, model, beam_width, len(devices))
    env.update(microbatch_env)
    (run_dir / "runtime-config.json").write_text(json.dumps({"mode": "auto", "microbatch": microbatch_metadata}, indent=2) + "\n", encoding="utf-8")
    nccl = build_metadata.get("nccl")
    if nccl is not None:
        if not isinstance(nccl, dict) or not isinstance(nccl.get("library"), str):
            raise NativeBackendError("native build metadata contains invalid NCCL provenance")
        from .build import file_sha256
        library = Path(nccl["library"])
        try:
            if file_sha256(library) != nccl.get("library_sha256"):
                raise NativeBackendError("native NCCL library changed since build preparation")
        except OSError as exc:
            raise NativeBackendError("native NCCL library is unavailable before launch") from exc
        # LD_LIBRARY_PATH takes precedence over the build RPATH. Keep the
        # dependency chosen during preparation first in the private child.
        nccl_dir = str(library.parent)
        inherited_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = nccl_dir + (os.pathsep + inherited_ld if inherited_ld else "")
    if model.backend == "piece_transformer":
        env["BEAM_STREAM1_EXECUTOR"] = "libtorch_eager"
    # Legacy production_runner writes its solution/no-solution logs here.
    legacy_results = run_dir / "test_results"
    legacy_results.mkdir(exist_ok=True)
    if legacy_results.is_symlink() or not legacy_results.resolve().is_relative_to(run_dir):
        raise NativeBackendError("native result directory must remain inside the private run directory")
    args = [str(runner), "0", str(max_steps), str(beam_width)]
    log = run_dir / "native.log"
    process_log = log
    log_metadata = {}
    if len(devices) > 1:
        worker_logs = run_dir / "worker-logs"
        try:
            worker_logs.mkdir()
        except FileExistsError as error:
            raise NativeBackendError("native worker log directory must be fresh") from error
        process_log = run_dir / "launcher.log"
        args = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                f"--nproc-per-node={len(devices)}", f"--log-dir={worker_logs}", "--redirects=3",
                "--max-restarts=0", "--no-python", *args]
    else:
        args += ["1", "0"]
    failure = None
    try:
        elapsed = run_process(args, cwd=run_dir, env=env, timeout=options.timeout_seconds, log_path=process_log)
    except BaseException as error:
        failure = error
        raise
    finally:
        if len(devices) > 1:
            try:
                log_metadata = collect_worker_logs(worker_logs, process_log, log, len(devices), strict=failure is None)
            except Exception as log_error:
                if failure is None:
                    raise
                if hasattr(failure, "add_note"):
                    failure.add_note(f"native worker log collection also failed: {log_error}")
    path, effective, terminal_metadata = parse_terminal(log, contract)
    if effective is not None and effective < beam_width:
        raise NativeBackendError("observed effective global beam is smaller than requested")
    metadata = {"build": build_metadata, "model_artifact_hash": model.artifact_hash,
                "graph_hash": contract.graph_hash, "devices": list(devices), "requested_beam_width": beam_width,
                "microbatch": microbatch_metadata,
                "replay_valid": path is not None, "log_path": str(log), **log_metadata, **terminal_metadata}
    (run_dir / "native-outcome.json").write_text(json.dumps({"path": path, "elapsed_seconds": elapsed,
        "effective_beam_width": effective, "metadata": metadata}, indent=2) + "\n", encoding="utf-8")
    return NativeOutcome(path, elapsed, effective, run_dir, metadata)
