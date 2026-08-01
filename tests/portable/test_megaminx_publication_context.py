from portable.megaminx_cluster.publication_context import build_publication_context


def test_context_is_built_only_from_manifest_preflight_profile_and_proof():
    manifest = {
        "competition": "santa-2023", "puzzle_type": "megaminx", "release_tag": "v1",
        "release_asset": "megaminx-sm90-linux-x86_64.tar.zst", "release_manifest_sha256": "a" * 64,
        "solver_commit": "b" * 40, "model_filename": "manifest.json", "model_sha256": "c" * 64,
        "model_format": "fp16-files", "model_manifest": {"state_len": 2, "num_classes": 2, "output_dim": 12, "dtype": "fp16"},
        "touch_bfs_radius": 0, "solution_mode": "first", "max_collected_solutions": 16,
    }
    preflight = {"gpus": [{"name": "NVIDIA H100 80GB HBM3", "family": "H100", "vram_mib": 81559, "sm": 90, "driver_major": 570}] * 4}
    selected = {"requested_beam": 1000, "effective_beam": 1024, "alignment_delta": 24, "profile_power": 10, "evidence_id": "h100-p10", "status": "measured", "hardware": {"gpu_family": "H100", "vram_mib": 81559, "sm": 90, "world_size": 4}, "backend": "mlp", "model_class": "output_move_count", "runtime": {"b_micro": 2048, "stream1_concurrency": 4, "stream3_ring_slots": 4, "shard_count": 4, "shard_capacity_scale_ppm": 1050000, "stream4_batch_candidates": 98304, "stream4_trigger_candidates": 98304, "stream4_active_sort_slots": 4}}
    context = build_publication_context(manifest, preflight, selected, {"initial_state": [1, 0], "central_state": [0, 1], "generators": {"swap": [1, 0]}}, run_id="slurm-9", job_id="9", cluster_name="basis", author_name="Ivan", search_mode="off", depth=120, solve_us=100, wall_us=200)
    assert context["profile"]["native_sm"] == 90
    assert context["hardware"]["world_size"] == 4
    assert context["provenance"]["slurm_job_id"] == "9"
    assert context["author"] == {"name": "Ivan", "verification": "claimed"}
    assert "private_path" not in context
