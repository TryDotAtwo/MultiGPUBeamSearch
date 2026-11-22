use beam_runtime::cluster::{cuda_ordinal, scheduler_rank, ClusterError};
use std::collections::BTreeMap;

fn env(values: &[(&str, &str)]) -> BTreeMap<String, String> {
    values
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn slurm_step_and_torchrun_rank_are_preserved() {
    let rank = scheduler_rank(&env(&[
        ("SLURM_PROCID", "12"),
        ("SLURM_LOCALID", "4"),
        ("SLURM_NODEID", "1"),
        ("SLURM_STEP_NUM_TASKS", "14"),
        ("SLURM_NTASKS", "32"),
        ("SLURM_STEP_TASKS_PER_NODE", "8,6"),
        ("SLURM_TASKS_PER_NODE", "8(x4)"),
    ]))
    .unwrap();
    assert_eq!(
        (
            rank.global_rank,
            rank.node_rank,
            rank.local_rank,
            rank.world_size,
            rank.local_world_size
        ),
        (12, 1, 4, 14, 6)
    );
    let rank = scheduler_rank(&env(&[
        ("TORCHELASTIC_RUN_ID", "test"),
        ("SLURM_PROCID", "0"),
        ("WORLD_SIZE", "16"),
        ("RANK", "15"),
        ("LOCAL_WORLD_SIZE", "8"),
        ("LOCAL_RANK", "7"),
        ("GROUP_RANK", "1"),
    ]))
    .unwrap();
    assert_eq!(
        (
            rank.global_rank,
            rank.node_rank,
            rank.local_rank,
            rank.world_size,
            rank.local_world_size
        ),
        (15, 1, 7, 16, 8)
    );
}

#[test]
fn malformed_assignments_are_rejected() {
    for counts in ["8(x0)", "8(x129)", "08,6", "8(x2),", "8(x2)x", "8(x2)(x2)"] {
        let result = scheduler_rank(&env(&[
            ("SLURM_PROCID", "0"),
            ("SLURM_LOCALID", "0"),
            ("SLURM_NODEID", "0"),
            ("SLURM_STEP_NUM_TASKS", "16"),
            ("SLURM_STEP_TASKS_PER_NODE", counts),
        ]));
        assert!(result.is_err(), "accepted {counts}");
    }
    for (world, rank, local_size, local, node) in [
        ("2", "0", "0", "0", "0"),
        ("2", "2", "1", "0", "0"),
        ("129", "0", "1", "0", "0"),
        ("2", "0", "1", "1", "0"),
        ("2", "0", "1", "0", "2"),
        ("2", "0", "1", "1.0", "0"),
    ] {
        let result = scheduler_rank(&env(&[
            ("TORCHELASTIC_RUN_ID", "test"),
            ("WORLD_SIZE", world),
            ("RANK", rank),
            ("LOCAL_WORLD_SIZE", local_size),
            ("LOCAL_RANK", local),
            ("GROUP_RANK", node),
        ]));
        assert!(result.is_err());
    }
}

#[test]
fn slurm_global_rank_must_match_node_and_local_assignment() {
    for (rank, local, node) in [("11", "4", "1"), ("13", "4", "1"), ("4", "4", "1")] {
        let result = scheduler_rank(&env(&[
            ("SLURM_PROCID", rank),
            ("SLURM_LOCALID", local),
            ("SLURM_NODEID", node),
            ("SLURM_STEP_NUM_TASKS", "14"),
            ("SLURM_STEP_TASKS_PER_NODE", "8,6"),
        ]));
        assert!(
            result.is_err(),
            "accepted rank={rank}, node={node}, local={local}"
        );
    }
}

#[test]
fn typed_cluster_plan_preserves_rank_order_and_capacity() {
    use beam_runtime::cluster::{plan_cluster, PlanRequest, PlannedRank};
    let plan = plan_cluster(&PlanRequest {
        gpus_per_node: vec![2, 4, 1],
        beam_width: 1_000_003,
        shard_count: 16,
        batch_alignment: 1024,
    })
    .unwrap();
    assert_eq!(
        plan.ranks,
        vec![
            PlannedRank {
                global_rank: 0,
                node_rank: 0,
                local_rank: 0,
            },
            PlannedRank {
                global_rank: 1,
                node_rank: 0,
                local_rank: 1,
            },
            PlannedRank {
                global_rank: 2,
                node_rank: 1,
                local_rank: 0,
            },
            PlannedRank {
                global_rank: 3,
                node_rank: 1,
                local_rank: 1,
            },
            PlannedRank {
                global_rank: 4,
                node_rank: 1,
                local_rank: 2,
            },
            PlannedRank {
                global_rank: 5,
                node_rank: 1,
                local_rank: 3,
            },
            PlannedRank {
                global_rank: 6,
                node_rank: 2,
                local_rank: 0,
            },
        ]
    );
    assert_eq!(plan.requested_beam, 1_000_003);
    assert_eq!(plan.effective_beam, 1_000_003);
    assert_eq!(plan.alignment, 114_688);
    assert_eq!(plan.local_frontier_capacity, 142_858);
    assert_eq!(plan.shard_count, 16);
    assert!(!plan.hardware_verified);
}

#[test]
fn typed_cluster_plan_rejects_world_and_frontier_overflow() {
    use beam_runtime::cluster::{plan_cluster, PlanRequest};
    assert!(plan_cluster(&PlanRequest {
        gpus_per_node: vec![129],
        beam_width: 1,
        shard_count: 16,
        batch_alignment: 1024,
    })
    .is_err());
    assert!(plan_cluster(&PlanRequest {
        gpus_per_node: vec![1],
        beam_width: u64::MAX,
        shard_count: 16,
        batch_alignment: 1024,
    })
    .is_err());
    assert!(plan_cluster(&PlanRequest {
        gpus_per_node: vec![128],
        beam_width: 1,
        shard_count: u32::MAX,
        batch_alignment: u32::MAX,
    })
    .is_err());
}

#[test]
fn scheduled_cuda_ordinal_preserves_visible_namespace_mapping() {
    assert_eq!(cuda_ordinal(7, 1).unwrap(), 0);
    assert_eq!(cuda_ordinal(0, 4).unwrap(), 0);
    assert_eq!(cuda_ordinal(3, 4).unwrap(), 3);
    assert_eq!(cuda_ordinal(-1, 4).unwrap(), -1);
}

#[test]
fn scheduled_cuda_ordinal_rejects_missing_or_insufficient_visibility() {
    assert!(matches!(
        cuda_ordinal(0, 0),
        Err(ClusterError::Invalid(message)) if message == "no visible CUDA devices"
    ));
    assert!(matches!(
        cuda_ordinal(4, 4),
        Err(ClusterError::Invalid(message)) if message == "LOCAL_RANK exceeds the visible CUDA devices"
    ));
}

#[test]
fn cuda_visibility_preserves_inherited_token_order_and_whitespace_rules() {
    use beam_runtime::cluster::cuda_visibility;

    assert_eq!(cuda_visibility(&[2, 0], None).unwrap(), "2,0");
    assert_eq!(
        cuda_visibility(&[2, 0], Some(" GPU-a, GPU-b ,GPU-c ")).unwrap(),
        "GPU-c,GPU-a"
    );
    assert_eq!(
        cuda_visibility(&[1, 1], Some("GPU-a,GPU-b")).unwrap(),
        "GPU-b,GPU-b"
    );
    assert_eq!(cuda_visibility(&[], Some("")).unwrap(), "");
}

#[test]
fn cuda_visibility_rejects_missing_indices_empty_tokens_and_disabled_devices() {
    use beam_runtime::cluster::cuda_visibility;

    for selected in [
        cuda_visibility(&[2], Some("GPU-a,GPU-b")),
        cuda_visibility(&[1], Some("GPU-a, ")),
        cuda_visibility(&[1], Some("GPU-a,-1")),
    ] {
        assert_eq!(
            selected.unwrap_err().to_string(),
            "cannot map selected devices through inherited CUDA_VISIBLE_DEVICES"
        );
    }
}


#[test]
fn child_environment_filters_rank_controls_and_maps_cuda_visibility() {
    use beam_runtime::cluster::child_environment;

    let inherited = env(&[
        ("PATH", "safe"),
        ("CUDA_VISIBLE_DEVICES", " GPU-a, GPU-b ,GPU-c "),
        ("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        ("RANK", "3"),
        ("ROLE_RANK", "2"),
        ("MASTER_ADDR", "coordinator"),
        ("NCCL_DEBUG", "TRACE"),
        ("BEAM_WEIGHT_DIR", "wrong"),
        ("STREAM4_BATCH_ALIGNMENT", "999"),
        ("PUBLISH_RESULTS", "1"),
        ("RESULTS_INGEST_TOKEN", "filtered"),
        ("CUSTOM_VALUE", "kept"),
    ]);
    let result = child_environment(&inherited, &[2, 0], "/rank/nccl-id.bin").unwrap();
    assert_eq!(result.get("PATH").map(String::as_str), Some("safe"));
    assert_eq!(result.get("CUSTOM_VALUE").map(String::as_str), Some("kept"));
    assert_eq!(result.get("CUDA_VISIBLE_DEVICES").map(String::as_str), Some("GPU-c,GPU-a"));
    assert_eq!(result.get("CUDA_DEVICE_ORDER").map(String::as_str), Some("PCI_BUS_ID"));
    assert_eq!(result.get("BEAM_NCCL_ID_FILE").map(String::as_str), Some("/rank/nccl-id.bin"));
    for removed in [
        "RANK", "ROLE_RANK", "MASTER_ADDR", "NCCL_DEBUG", "BEAM_WEIGHT_DIR",
        "STREAM4_BATCH_ALIGNMENT", "PUBLISH_RESULTS", "RESULTS_INGEST_TOKEN",
    ] {
        assert!(!result.contains_key(removed), "kept reserved key {removed}");
    }
}

#[test]
fn child_environment_rejects_invalid_cuda_mapping() {
    use beam_runtime::cluster::child_environment;

    let inherited = env(&[("CUDA_VISIBLE_DEVICES", "GPU-a,-1")]);
    assert_eq!(
        child_environment(&inherited, &[1], "/run/nccl-id.bin")
            .unwrap_err()
            .to_string(),
        "cannot map selected devices through inherited CUDA_VISIBLE_DEVICES"
    );
}
