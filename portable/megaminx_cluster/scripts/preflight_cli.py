"""Compute-node preflight CLI executed before torchrun."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

from portable.megaminx_cluster.profile import HardwareKey, derive_capacities, select_profile
from portable.megaminx_cluster.profile_cache import load_registry
from portable.megaminx_cluster.scripts.preflight import (
    inspect_gpus,
    validate_allocation,
    verify_payload_hashes,
    write_record,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--beam", type=int)
    parser.add_argument("--hardware-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.hardware_only and (args.beam is None or args.beam <= 0):
            raise ValueError("--beam must be positive unless --hardware-only is used")
        root = args.archive_root.resolve()
        run_dir = args.run_dir.resolve()
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8-sig"))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible:
            raise ValueError("CUDA_VISIBLE_DEVICES is empty")
        query = subprocess.run(
            [
                "nvidia-smi", f"--id={visible}",
                "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if query.returncode != 0:
            raise ValueError(f"nvidia-smi failed: {query.stderr.strip()}")
        gpus = inspect_gpus(query.stdout)
        if len(gpus) != args.gpu_count:
            raise ValueError(f"expected {args.gpu_count} GPUs; observed {len(gpus)}")
        verify_payload_hashes(root, manifest["payloads"])

        if args.hardware_only:
            allocation_plan = {
                "world_size": args.gpu_count,
                "required_vram_mib": int(manifest["minimum_vram_mib"]),
                "history_disk_bytes": 0,
            }
            record = validate_allocation(
                manifest, allocation_plan, gpus, shutil.disk_usage(run_dir).free
            )
            write_record(run_dir / "preflight.json", record)
            return 0

        cache = Path(os.environ.get("MEGAMINX_PROFILE_CACHE", root / "profile-cache" / "registry.json"))
        registry = load_registry(root / "profiles" / "registry.json", cache)
        first = gpus[0]
        selected = select_profile(
            registry,
            HardwareKey(first.family, first.vram_mib, first.sm, args.gpu_count),
            args.beam,
            str(manifest["backend"]),
            str(manifest["model_class"]),
        )
        runtime = derive_capacities(
            selected,
            args.beam,
            int(manifest["move_count"]),
            int(manifest["output_dim"]),
        )
        allocation_plan = {
            "world_size": args.gpu_count,
            "required_vram_mib": int(manifest["minimum_vram_mib"]),
            "history_disk_bytes": int(manifest["history_disk_bytes"]),
        }
        record = validate_allocation(
            manifest, allocation_plan, gpus, shutil.disk_usage(run_dir).free
        )
        selected_record = {
            **asdict(runtime),
            "hardware": asdict(selected.hardware),
            "backend": selected.backend,
            "model_class": selected.model_class,
            "status": selected.status,
            "runtime": dict(runtime.runtime),
        }
        write_record(run_dir / "selected_profile.json", selected_record)
        env_values = {
            "BEAM_RUNTIME_CONFIG_MODE": "manual",
            "BEAM_B_MICRO": runtime.runtime["b_micro"],
            "BEAM_STREAM1_CONCURRENCY": runtime.runtime["stream1_concurrency"],
            "BEAM_STREAM3_RING_SLOTS": runtime.runtime["stream3_ring_slots"],
            "BEAM_SHARD_COUNT": runtime.runtime["shard_count"],
            "BEAM_SHARD_CAPACITY_CANDIDATES": runtime.shard_capacity_candidates,
            "BEAM_STREAM4_BATCH_CANDIDATES": runtime.runtime["stream4_batch_candidates"],
            "BEAM_STREAM4_TRIGGER_CANDIDATES": runtime.runtime["stream4_trigger_candidates"],
            "BEAM_STREAM4_ACTIVE_SORT_SLOTS": runtime.runtime["stream4_active_sort_slots"],
            "BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES": runtime.runtime["final_materialize_chunk_candidates"],
            "BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM": 1_000_000 * args.gpu_count,
        }
        (run_dir / "selected_profile.env").write_text(
            "".join(f"export {key}={value}\n" for key, value in env_values.items()),
            encoding="ascii",
        )
        write_record(run_dir / "preflight.json", record)
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"preflight_failed={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
