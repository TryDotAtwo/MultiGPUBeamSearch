#!/usr/bin/env python3
"""Molab-native checkpoint-only CayleyPy orchestration CLI."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.data import load_puzzle_contract
from tools.cayleypy_public.model import export_checkpoint
from tools.cayleypy_public.profile import (
    derive_portable_runtime,
    serialize_portable_preflight,
)
from tools.cayleypy_public.runner import (
    PublicSearchRunError,
    maximum_history_depth,
    run_public_search,
)
from tools.kaggle_t4_mlp_profiles import align_beam, select_profile
from tools.molab_assets import prepare_molab_public_config
from tools.molab_toolchain import prepare_molab_build_environment
import tools.run_cayleypy_public as public_runtime


ROOT = Path(__file__).resolve().parents[1]
MLP_PROFILES = ROOT / "configs" / "kaggle_t4_mlp_profiles.json"
TRANSFORMER_PROFILES = ROOT / "configs" / "kaggle_t4_transformer_profiles.json"
DEFAULT_MOLAB_CACHE_ROOT = Path("/tmp/cayleypy_molab")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def detect_molab_hardware() -> tuple[list[str], str]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("Molab GPU is not attached; select a GPU compute spec first")
    names: list[str] = []
    arches: set[str] = set()
    for row in rows:
        name, separator, capability = row.rpartition(",")
        if not separator:
            raise RuntimeError(f"cannot parse nvidia-smi GPU row: {row!r}")
        arch = capability.strip().replace(".", "")
        if not arch.isdigit():
            raise RuntimeError(f"cannot parse GPU compute capability: {capability!r}")
        names.append(name.strip())
        arches.add(arch)
    if len(arches) != 1:
        raise RuntimeError(f"heterogeneous Molab GPU architectures are unsupported: {sorted(arches)}")
    if not 1 <= len(names) <= 8:
        raise RuntimeError(f"Molab runner supports 1..8 local GPUs; observed={len(names)}")
    return names, arches.pop()


def _history_budgets() -> tuple[int, int, int]:
    available_ram = public_runtime._available_ram_bytes()
    tmp_free = shutil.disk_usage("/tmp").free
    ram_headroom = 2 * 1024**3
    disk_headroom = 2 * 1024**3
    ram = min(64 * 1024**3, available_ram - ram_headroom)
    disk = min(50 * 1024**3, tmp_free - disk_headroom)
    if ram <= 0 or disk <= 0:
        raise RuntimeError(
            f"Molab runtime lacks history headroom: ram={available_ram} tmp_free={tmp_free}"
        )
    return ram, disk, tmp_free


def _toolchain_workspace() -> Path:
    configured = os.environ.get("CAYLEYPY_MOLAB_CACHE_ROOT")
    workspace = Path(configured) if configured else DEFAULT_MOLAB_CACHE_ROOT
    return workspace.expanduser().resolve()


def _portable_profile(registry: dict, beam_width: int, output_dim: int,
                      move_count: int, world_size: int) -> dict:
    profile = select_profile(registry, beam_width, output_dim, move_count)
    hardware = f"molab_{world_size}xgpu"
    profile["hardware"] = hardware
    profile["effective_beam"] = align_beam(
        beam_width, world_size, profile["runtime"]["shard_count"],
    )
    profile["alignment_delta"] = profile["effective_beam"] - beam_width
    profile["evidence_origin"] = "conservative-kaggle-2xt4-seed-validated-on-molab"
    profile["source_hardware"] = "kaggle_2xt4"
    return profile


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config: PublicRunConfig | None = None
    try:
        raw = json.loads(args.notebook_config.read_text(encoding="utf-8"))
        workspace = args.notebook_config.resolve().parent
        public_mapping = prepare_molab_public_config(raw, workspace)
        config = PublicRunConfig.from_mapping(public_mapping)
        gpu_names, cuda_arch = detect_molab_hardware()
        world_size = len(gpu_names)
        hardware = f"molab_{world_size}xgpu"
        import torch
        torch_cuda = torch.version.cuda
        if not isinstance(torch_cuda, str) or not torch_cuda:
            raise RuntimeError("Molab PyTorch build does not report a CUDA toolkit version")
        toolchain_environment = prepare_molab_build_environment(
            _toolchain_workspace(), torch_cuda, sys.executable,
        )
        os.environ.update(toolchain_environment)
        print("molab_preflight=" + json.dumps({
            "gpu_names": gpu_names, "cuda_arch": cuda_arch,
            "world_size": world_size, "requested_beam": config.beam_width,
            "torch_cuda": torch_cuda,
        }, sort_keys=True), flush=True)

        contract = load_puzzle_contract(
            config.puzzle_info_json, config.test_csv, config.sample_submission_csv,
            config.puzzle_id_start, config.puzzle_id_end,
        )
        export_dir = output / "export"
        model = export_checkpoint(
            config.checkpoint_path, export_dir, contract.num_classes,
            state_len=contract.state_len, move_count=contract.move_count,
            metadata_json=config.checkpoint_metadata_json,
            generator_json=config.checkpoint_generator_json,
            source_root=config.checkpoint_source_root,
        )
        public_runtime._ensure_export_manifest(export_dir / "manifest.json", model)
        registry_path = TRANSFORMER_PROFILES if model.backend == "piece_transformer" else MLP_PROFILES
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        output_dim = model.manifest.get("output_dim")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int):
            raise ValueError("export manifest output_dim must be an integer")
        profile = _portable_profile(
            registry, config.beam_width, output_dim, contract.move_count, world_size,
        )
        plan = derive_portable_runtime(
            profile, config.beam_width, output_dim, contract.move_count,
            world_size=world_size, hardware=hardware,
        )
        public_runtime._write_json(output / "selected_profile.json", profile)

        history_ram, history_disk, tmp_free = _history_budgets()
        requested_depth = config.max_depth
        budget_depth = maximum_history_depth(
            plan, contract.move_count, config.touch_bfs_radius,
            history_ram, history_disk,
        )
        if config.max_depth > budget_depth:
            config = replace(
                config, max_depth=budget_depth,
                collect_until_depth=min(config.collect_until_depth, budget_depth),
            )
        preflight = serialize_portable_preflight(
            plan, profile, contract.move_count, history_ram, history_disk, tmp_free,
        )
        preflight.update({
            "gpu_names": gpu_names, "cuda_arch": cuda_arch,
            "requested_max_depth": requested_depth,
            "budget_max_depth": budget_depth,
            "effective_max_depth": config.max_depth,
            "model_format": model.format, "model_backend": model.backend,
            "model_dtype": model.dtype, "output_dim": output_dim,
        })
        public_runtime._write_json(output / "preflight.json", preflight)
        runner = public_runtime.locate_or_build_runner(
            output, config.puzzle_info_json, backend=model.backend, config=config,
            cuda_arch=cuda_arch, build_tag="molab",
        )
        try:
            artifacts = public_runtime._run_with_history_budgets(
                history_ram, history_disk,
                config, contract, model, plan, export_dir, output / "logs",
                runner_path=str(runner), log_sanitizer=public_runtime._sanitize_log,
            )
        except PublicSearchRunError as error:
            artifact_summary = public_runtime._materialize_run_artifacts(error.partial_artifacts, output)
            wall = time.perf_counter() - started
            publish = public_runtime._publish_best_effort(
                config, contract, model, profile, plan, gpu_names,
                error.partial_artifacts, output, wall,
            )
            public_runtime._write_json(output / "run_summary.json", {
                "status": "failed", "safe_error": public_runtime._public_error(error, config),
                **public_runtime._summary_base(config, gpu_names, model, plan),
                **artifact_summary, "wall_seconds": wall, "publish_status": publish,
            })
            return 2

        artifact_summary = public_runtime._materialize_run_artifacts(artifacts, output)
        wall = time.perf_counter() - started
        publish = public_runtime._publish_best_effort(
            config, contract, model, profile, plan, gpu_names, artifacts, output, wall,
        )
        public_runtime._write_json(output / "run_summary.json", {
            "status": "success", "safe_error": None,
            **public_runtime._summary_base(config, gpu_names, model, plan),
            **artifact_summary, "wall_seconds": wall, "publish_status": publish,
            "artifacts": [
                "selected_profile.json", "preflight.json", "export/manifest.json",
                "beam_run_results.csv", "solutions/all_solutions.csv",
                "solutions/solutions.csv", "submission.csv", "publish_status.json", "logs/",
            ],
        })
        print("publish_status=" + json.dumps(publish, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        public_runtime._write_json(output / "run_summary.json", {
            "status": "failed", "safe_error": public_runtime._public_error(error, config),
            "wall_seconds": time.perf_counter() - started,
            "publish_status": {"state": "skipped", "reason": "preflight_or_build_failed"},
        })
        print(public_runtime._public_error(error, config), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
