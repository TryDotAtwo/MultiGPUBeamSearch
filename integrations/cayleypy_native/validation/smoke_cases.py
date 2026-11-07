"""Real Linux/two-CUDA-device acceptance through the public CayleyPy API.

This synthetic, untrained model tests integration and replay, not search quality.
Nothing is downloaded, submitted, built, or patched at module import time.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time
import traceback

import torch
from torch import nn


SEED = 1729
STATE_LEN = 8


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(16, 16), nn.Linear(16, 16)
        self.bn1, self.bn2 = nn.BatchNorm1d(16), nn.BatchNorm1d(16)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0)

    def forward(self, x):
        return self.relu(x + self.bn2(self.fc2(self.dropout(self.relu(self.bn1(self.fc1(x)))))))


class Pilgrim(nn.Module):
    """Small instance of the native exporter's existing BatchNorm/ReLU schema."""

    def __init__(self):
        super().__init__()
        self.state_size, self.num_classes, self.z_add = STATE_LEN, STATE_LEN, 0
        self.output_dim, self.n_outputs = 1, 1
        self.input_layer = nn.Linear(STATE_LEN * STATE_LEN, 32)
        self.hidden_layer, self.output_layer = nn.Linear(32, 16), nn.Linear(16, 1)
        self.bn1, self.bn2 = nn.BatchNorm1d(32), nn.BatchNorm1d(16)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0)
        self.residual_blocks = nn.ModuleList([ResidualBlock()])

    def forward(self, states):
        x = nn.functional.one_hot(states.long(), self.num_classes).float().flatten(1)
        x = self.relu(self.bn1(self.input_layer(x)))
        x = self.relu(self.bn2(self.hidden_layer(x)))
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x).squeeze(-1)


def make_model():
    # Construct on CPU without changing caller CPU or CUDA RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        return Pilgrim().eval()


def replay(start, path, generators):
    """Independent literal tuple-gather replay, with no adapter/private hashes."""
    state = tuple(start)
    for move in path:
        if type(move) is not int or not 0 <= move < len(generators):
            raise ValueError(f"invalid generator id: {move!r}")
        state = tuple(state[index] for index in generators[move])
    return state


def _shortest_path(start, center, generators, limit):
    pending = deque([(start, ())])
    seen = {start}
    while pending:
        state, path = pending.popleft()
        if state == center:
            return path
        if len(path) < limit:
            for move in range(len(generators)):
                child = replay(state, (move,), generators)
                if child not in seen:
                    pending.append((child, path + (move,)))
                    seen.add(child)
    raise ValueError("no reference path within bounded exact BFS")


@dataclass(frozen=True)
class SmokeCase:
    name: str
    start: tuple[int, ...]
    max_steps: int
    exact_distance: int
    reference_solution: tuple[int, ...]
    expected_found: bool
    native_workers_required: bool
    beam_width: int = 7


def make_cases(definition):
    center = tuple(definition.central_state)
    generators = tuple(tuple(g) for g in definition.generators)
    if len(center) != STATE_LEN or len(generators) != 3:
        raise ValueError("smoke oracle expects LRX8")
    layer, seen, sizes = {center}, {center}, [1]
    for _ in range(3):
        layer = {replay(state, (move,), generators) for state in layer for move in range(3)}
        unseen = layer - seen
        seen.update(layer)
        sizes.append(len(layer))
    multi = min(unseen)
    one = replay(center, (2,), generators)
    specifications = [("already_goal", center, 0), ("one_step", one, 1),
                      ("multi_step", multi, 3), ("zero_budget", multi, 0),
                      ("not_found", multi, 1), ("multi_step_warm", multi, 3)]
    cases = []
    for name, start, budget in specifications:
        solution = _shortest_path(start, center, generators, 3)
        cases.append(SmokeCase(name, start, budget, len(solution), solution,
                               len(solution) <= budget, start != center and budget > 0))
    if sizes != [1, 3, 7, 15]:
        raise ValueError(f"unexpected LRX8 layer sizes: {sizes}")
    return cases, {"method": "exact CPU tuple BFS and tuple-gather replay", "exact_layer_sizes": sizes,
                   "layer_semantics": "unique states at exact walk length; no visited-state exclusion",
                   "scoring_note": "beam7 retains all depth2 states; torch scores that layer before distance3 goal"}


def rank_evidence(log_text, devices):
    # In a standalone single-node torchrun, production_runner labels global RANK
    # as LOCAL_RANK and prints actual cudaSetDevice(LOCAL_RANK) separately.
    tokens = {}
    for key in ("WORLD_SIZE", "LOCAL_RANK", "CUDA_DEVICE_LOCAL_RANK"):
        tokens[key] = sorted({int(value) for value in re.findall(rf"(?m)^{key}=(\d+)\s*$", log_text)})
    expected = list(range(len(devices)))
    passed = (tokens["WORLD_SIZE"] == [len(devices)] and tokens["LOCAL_RANK"] == expected
              and tokens["CUDA_DEVICE_LOCAL_RANK"] == expected)
    return {"passed": passed, "expected_worker_rank_ids": expected,
            "observed_world_sizes": tokens["WORLD_SIZE"], "observed_rank_ids": tokens["LOCAL_RANK"],
            "observed_cuda_local_device_ids": tokens["CUDA_DEVICE_LOCAL_RANK"],
            "selected_parent_cuda_indices": list(devices),
            "evidence_scope": "worker startup after cudaSetDevice and NCCL initialization; not utilization/scaling"}


def _exception_record(error):
    return {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}


def _run_directories(cache_dir):
    parent = Path(cache_dir) / "runs"
    return set(parent.iterdir()) if parent.is_dir() else set()


def _artifacts(directories):
    return [{"run_dir": str(directory), "files": sorted(str(path) for path in directory.rglob("*")
            if path.is_file() and path.suffix in (".log", ".json", ".csv"))}
            for directory in sorted(directories)]


def exported_weights_evidence(weights_dir):
    """Hash numerical blobs/schema, excluding per-export checkpoint paths."""
    weights_dir = Path(weights_dir)
    manifest = json.loads((weights_dir / "manifest.json").read_text(encoding="utf-8"))
    shape = {key: manifest.get(key) for key in ("state_len", "num_classes", "hd1", "hd2", "nrd",
                                               "output_dim", "dtype", "normalization")}
    blobs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(weights_dir.glob("*." + manifest["dtype"]))}
    if not blobs:
        raise ValueError(f"no exported numerical weight blobs: {weights_dir}")
    digest = hashlib.sha256(json.dumps({"shape": shape, "blobs": blobs}, sort_keys=True).encode()).hexdigest()
    return {"numerical_model_sha256": digest, "shape": shape, "blob_sha256": blobs,
            "weights_dir": str(weights_dir)}


def run_case(graph, predictor, case, *, backend, devices, cache_dir, synchronize):
    """One public search call, including replay/metadata checks and exception capture."""
    record = {"case": asdict(case), "backend": backend, "status": "running"}
    before = _run_directories(cache_dir)
    started = time.perf_counter()
    try:
        synchronize()
        result = graph.beam_search(backend=backend, start_state=list(case.start), predictor=predictor,
                                  beam_width=case.beam_width, max_steps=case.max_steps,
                                  beam_mode="simple", return_path=True)
        synchronize()
        record["wall_seconds"] = time.perf_counter() - started
        path = result.path
        replay_valid = None if path is None else replay(case.start, path, graph.definition.generators) == tuple(graph.definition.central_state)
        checks = {"found_matches_exact_budget": bool(result.path_found) == case.expected_found,
                  "result_graph_matches": result.graph == graph.definition,
                  "path_contract": (path is not None and replay_valid and len(path) == result.path_length
                                    and len(path) == case.exact_distance) if case.expected_found
                                   else (path is None and not result.path_found and result.path_length == 0)}
        record["result"] = {"path_found": bool(result.path_found), "path_length": int(result.path_length),
                            "path": path, "replay_valid": replay_valid,
                            "result_type": f"{type(result).__module__}.{type(result).__name__}",
                            "debug_scores": {str(k): float(v) for k, v in result.debug_scores.items()}}
        if backend == "native":
            metadata = getattr(result, "native_metadata", {})
            record["native_metadata"] = metadata
            record["exported_weights"] = exported_weights_evidence(Path(metadata["run_dir"]) / "weights")
            checks["strict_native_result"] = metadata.get("backend") == "native"
            checks["explicit_devices_preserved"] = metadata.get("devices") == list(devices)
            checks["graph_and_model_hash_present"] = all(isinstance(metadata.get(key), str) and len(metadata[key]) == 64
                                                         for key in ("graph_hash", "model_hash"))
            if case.native_workers_required:
                log = Path(metadata["log_path"])
                evidence = rank_evidence(log.read_text(encoding="utf-8", errors="replace"), devices)
                evidence["log_path"] = str(log)
                record["rank_evidence"] = evidence
                checks["both_workers_observed"] = evidence["passed"]
                checks["effective_beam_at_least_requested"] = int(metadata["effective_beam_width"]) >= case.beam_width
                checks["build_shape"] = metadata.get("build", {}).get("shape") == {
                    "state_len": 8, "storage_len": 16, "alignment": 16, "move_count": 3}
                checks["native_process_and_search_timings"] = all(isinstance(metadata.get(key), (int, float))
                    and metadata[key] >= 0 for key in ("elapsed_seconds", "native_seconds"))
                record["configure_ran_this_call"] = (Path(metadata["run_dir"]) / "cmake-configure.log").is_file()
            else:
                checks["documented_host_shortcut"] = metadata.get("execution") == (
                    "already_at_goal" if case.exact_distance == 0 else "zero_depth_budget")
                record["rank_evidence"] = {"required": False, "reason": "host shortcut; no worker execution claimed"}
        elif case.name.startswith("multi_step"):
            checks["upstream_model_scoring_exercised"] = bool(result.debug_scores)
        record["checks"] = checks
        record["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as error:
        record["status"] = "failed"
        record["exception"] = _exception_record(error)
    finally:
        record.setdefault("wall_seconds", time.perf_counter() - started)
        record["artifacts"] = _artifacts(_run_directories(cache_dir) - before)
    return record


def _save(report, output_dir):
    temporary = output_dir / "acceptance_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "acceptance_report.json")


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_acceptance(*, source_dir, cutlass_dir, output_dir, devices=(0, 1),
                   timeout_seconds=180, build_timeout_seconds=1800, build_jobs=2, cache_dir=None):
    """Return/save JSON evidence; no automatic fallback and no downloads.

    Call from a dedicated Python process. The opt-in patch is restored on exit.
    output_dir holds the report/checkpoint; cache_dir may be on /tmp for builds.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir).resolve() if cache_dir is not None else output_dir / "cache"
    report = {"schema_version": 1, "status": "running", "passed": False,
              "started_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
              "configuration": {"source_dir": str(Path(source_dir).resolve()),
                  "cutlass_dir": str(Path(cutlass_dir).resolve()), "cache_dir": str(cache_dir),
                  "devices": list(devices), "seed": SEED, "timeout_seconds": timeout_seconds,
                  "build_timeout_seconds": build_timeout_seconds, "build_jobs": build_jobs},
              "limitations": ["Synthetic untrained scalar network; not learned-model quality evidence.",
                  "Shallow LRX8 with all pre-goal states retained; not large-puzzle correctness coverage.",
                  "Single observations and warm repeat; not a throughput/scaling benchmark.",
                  "Torch baseline runs on one selected GPU; native workers use both selected GPUs.",
                  "Zero-depth/goal shortcuts do not launch native workers.",
                  "FP32 source model and FP16 exported native model are not bitwise-score-equivalent."]}
    _save(report, output_dir)
    installed = False
    try:
        import cayleypy
        from cayleypy import CayleyGraph, PermutationGroups, Predictor
        from cayleypy_native import NativeOptions, enable_native, disable_native

        if tuple(devices) != (0, 1):
            raise ValueError("this acceptance workflow explicitly requires devices=(0, 1)")
        if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("real acceptance requires Linux and at least two visible CUDA GPUs; no CPU simulation")
        report["environment"] = {"platform": platform.platform(), "python": sys.version,
            "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "cayleypy_version": _version("cayleypy"), "cayleypy_module": str(Path(cayleypy.__file__).resolve()),
            "adapter_version": _version("cayleypy-native"), "gpus": [
                {"index": index, "name": torch.cuda.get_device_name(index),
                 "capability": list(torch.cuda.get_device_capability(index)),
                 "total_memory": torch.cuda.get_device_properties(index).total_memory} for index in devices]}
        torch.cuda.set_device(devices[0])
        # Both current CayleyPy and PyPI 0.1 accept device='cuda'; the old
        # release harmlessly ignores num_gpus and uses the selected device.
        graph = CayleyGraph(PermutationGroups.lrx(STATE_LEN), device="cuda", num_gpus=1, random_seed=SEED, verbose=0)
        cases, oracle = make_cases(graph.definition)
        report["graph"] = {"name": graph.definition.name, "center": list(graph.definition.central_state),
            "generators": [list(row) for row in graph.definition.generators],
            "generator_names": list(graph.definition.generator_names), "oracle": oracle}
        model = make_model()
        checkpoint = output_dir / "synthetic_model_state.pt"
        torch.save(model.state_dict(), checkpoint)
        digest = hashlib.sha256()
        for key, tensor in sorted(model.state_dict().items()):
            digest.update(key.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0" + tensor.numpy().tobytes())
        report["model"] = {"class": "Pilgrim", "state_len": STATE_LEN, "num_classes": STATE_LEN,
                           "hd1": 32, "hd2": 16, "nrd": 1, "output_dim": 1,
                           "training": False, "trained": False, "state_content_sha256": digest.hexdigest(),
                           "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        predictor = Predictor(graph, model)
        options = NativeOptions(source_dir=source_dir, cutlass_dir=cutlass_dir, cache_dir=cache_dir,
            devices=tuple(devices), timeout_seconds=timeout_seconds, build_timeout_seconds=build_timeout_seconds,
            build_jobs=build_jobs, touch_bfs_radius=0)
        enable_native(options, default_backend="native")
        installed = True
        _save(report, output_dir)
        schedule = [(case, backend) for case in cases for backend in ("torch", "native")]
        for index, (case, backend) in enumerate(schedule):
            record = run_case(graph, predictor, case, backend=backend, devices=devices, cache_dir=cache_dir,
                              synchronize=lambda: [torch.cuda.synchronize(device) for device in devices])
            report["cases"].append(record)
            _save(report, output_dir)
            print(f"{backend}/{case.name}: {record['status']} wall={record['wall_seconds']:.6f}s", flush=True)
            if backend == "native" and "exception" in record:
                # Build/worker errors are not expected case outcomes. Do not spend
                # quota repeating the same failed compile, timeout or crash.
                report["cases"].extend({"case": asdict(pending), "backend": mode, "status": "not_run",
                    "reason": f"stopped after native exception in {case.name}"} for pending, mode in schedule[index + 1:])
                break
        native = [case for case in report["cases"] if case["backend"] == "native"]
        hashes = [(case.get("native_metadata", {}).get("graph_hash"),
                   case.get("exported_weights", {}).get("numerical_model_sha256")) for case in native]
        report["cross_call_checks"] = {"all_calls_same_graph_and_exported_model": len(set(hashes)) == 1 and all(all(pair) for pair in hashes),
            "warm_repeat_reused_build": native[-1].get("configure_ran_this_call") is False}
        report["passed"] = all(case["status"] == "passed" for case in report["cases"]) and all(report["cross_call_checks"].values())
        report["status"] = "passed" if report["passed"] else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["exception"] = _exception_record(error)
    finally:
        if installed:
            try:
                disable_native()
            except Exception as error:
                report.update(status="failed", passed=False, cleanup_exception=_exception_record(error))
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _save(report, output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "cutlass-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--build-timeout-seconds", type=float, default=1800)
    parser.add_argument("--build-jobs", type=int, default=2)
    args = parser.parse_args()
    args.devices = tuple(int(value) for value in args.devices.split(","))
    report = run_acceptance(**vars(args))
    print(json.dumps({"passed": report["passed"], "report": str(args.output_dir / "acceptance_report.json")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
