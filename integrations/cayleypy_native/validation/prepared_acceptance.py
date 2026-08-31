"""Real GPU checks of the explicit prepare-once public API; no import-time work."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback


def run_prepared_acceptance(source_dir, cutlass_dir, output_dir, cache_dir, devices=(0, 1)):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "cases": [], "scope": "prepare-once synthetic correctness; not speedup"}
    installed = False
    try:
        from cayleypy import CayleyGraph, PermutationGroups
        from cayleypy_native import NativeOptions, prepare_native, enable_native, disable_native
        if __package__:
            from .smoke_cases import make_cases, make_model, rank_evidence, replay
        else:
            from smoke_cases import make_cases, make_model, rank_evidence, replay
        graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu", random_seed=1729)
        options = NativeOptions(source_dir=source_dir, cutlass_dir=cutlass_dir, cache_dir=cache_dir,
                                devices=tuple(devices), timeout_seconds=120, build_timeout_seconds=900)
        original_model = make_model()
        began = time.perf_counter()
        prepared = prepare_native(graph, original_model, native_options=options)
        report["preparation_wall_seconds"] = time.perf_counter() - began
        report["preparation_dir"] = str(prepared.preparation_dir)
        report["runner_sha256"] = prepared.runner_sha256
        report["model_hash"] = prepared.model.expected_artifact_hash
        if prepared.model.expected_artifact_hash is None or prepared.options.runner_path is None:
            raise ValueError("prepare-once did not pin a model artifact and runner")
        enable_native(prepared.options, default_backend="native")
        installed = True
        cases = make_cases(graph.definition)[0]
        for case in (cases[1], cases[2]):
            began = time.perf_counter()
            result = graph.beam_search(start_state=case.start, predictor=prepared.model,
                beam_width=case.beam_width, max_steps=case.max_steps, return_path=True)
            row = {"name": case.name, "wall_seconds": time.perf_counter() - began,
                   "metadata": result.native_metadata, "path": result.path}
            report["cases"].append(row)
            if (result.backend != "native" or not result.path_found or
                    result.path_length != case.exact_distance or
                    replay(case.start, result.path, graph.definition.generators) != tuple(graph.definition.central_state)):
                raise ValueError("prepared public API path did not pass independent replay")
            metadata = result.native_metadata
            if metadata["model_hash"] != prepared.model.expected_artifact_hash:
                raise ValueError("prepared model artifact identity changed")
            directory = Path(metadata["run_dir"])
            if (directory / "model_export.log").exists() or (directory / "cmake-configure.log").exists():
                raise ValueError("prepared query unexpectedly repeated export or configure")
            row["rank_evidence"] = rank_evidence(Path(metadata["log_path"]).read_text(), devices)
            if not row["rank_evidence"]["passed"]:
                raise ValueError("prepared query lacks observed two-GPU worker evidence")
            row["status"] = "passed"
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", exception={"type": type(error).__name__, "message": str(error),
                                                "traceback": traceback.format_exc()})
    finally:
        if installed:
            disable_native()
        (output / "prepared_acceptance_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report
