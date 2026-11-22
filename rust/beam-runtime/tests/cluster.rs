use beam_runtime::cluster::scheduler_rank;
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
        gpus_per_node: vec![1],
        beam_width: 1,
        shard_count: u32::MAX,
        batch_alignment: u32::MAX,
    })
    .is_err());
}
