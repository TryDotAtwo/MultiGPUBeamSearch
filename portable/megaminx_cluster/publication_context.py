"""Build the whitelisted SLURM v2 publication context from run artifacts."""

from __future__ import annotations

from typing import Mapping


def build_publication_context(manifest: Mapping[str, object], preflight: Mapping[str, object], selected: Mapping[str, object], proof: Mapping[str, object], *, run_id: str, job_id: str, cluster_name: str, author_name: str, search_mode: str, depth: int, solve_us: int, wall_us: int) -> dict[str, object]:
    hardware_key = selected["hardware"]
    if not isinstance(hardware_key, Mapping):
        raise ValueError("selected profile hardware is invalid")
    gpus = preflight["gpus"]
    if not isinstance(gpus, list) or not gpus or not all(isinstance(gpu, Mapping) for gpu in gpus):
        raise ValueError("preflight GPU record is invalid")
    first = gpus[0]
    runtime_source = selected["runtime"]
    if not isinstance(runtime_source, Mapping):
        raise ValueError("selected runtime is invalid")
    profile = {
        "requested_beam": selected["requested_beam"], "effective_beam": selected["effective_beam"], "alignment_delta": selected["alignment_delta"],
        "profile_power": selected["profile_power"], "profile_anchor_beam": 1 << int(selected["profile_power"]), "profile_status": selected["status"],
        "profile_evidence_id": selected["evidence_id"], "gpu_family": hardware_key["gpu_family"], "vram_mib": hardware_key["vram_mib"],
        "native_sm": hardware_key["sm"], "world_size": hardware_key["world_size"], "backend": selected["backend"], "model_class": selected["model_class"],
    }
    runtime_keys = ("b_micro", "stream1_concurrency", "stream3_ring_slots", "shard_count", "shard_capacity_scale_ppm", "stream4_batch_candidates", "stream4_trigger_candidates", "stream4_active_sort_slots")
    runtime = {"touch_bfs_radius": manifest["touch_bfs_radius"], "solution_mode": manifest["solution_mode"], "max_depth": depth, "max_collected_solutions": manifest["max_collected_solutions"], **{key: runtime_source[key] for key in runtime_keys}}
    model = {"filename": manifest["model_filename"], "sha256": manifest["model_sha256"], "format": manifest["model_format"], "manifest": manifest["model_manifest"]}
    hardware = {"platform": "slurm", "gpu_names": [gpu["name"] for gpu in gpus], "accelerator_count": len(gpus), "world_size": len(gpus), "native_sm": first["sm"], "vram_mib_per_gpu": first["vram_mib"], "driver_version": str(first.get("driver_version", first["driver_major"]))}
    provenance = {"platform": "slurm", "cluster_name": cluster_name, "slurm_job_id": job_id, "slurm_array_task_id": None, "run_id": run_id, "release_tag": manifest["release_tag"], "release_asset": manifest["release_asset"], "release_manifest_sha256": manifest["release_manifest_sha256"], "solver_commit": manifest["solver_commit"]}
    return {"run_id": run_id, "author": {"name": author_name, "verification": "claimed"}, "competition": manifest["competition"], "puzzle_type": manifest["puzzle_type"], "search_mode": search_mode, "proof": dict(proof), "profile": profile, "runtime": runtime, "model": model, "hardware": hardware, "timings": {"solve_us": solve_us, "wall_us": wall_us}, "provenance": provenance}
