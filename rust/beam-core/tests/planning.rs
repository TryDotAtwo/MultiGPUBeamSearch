use beam_core::planning::{rank_frontier_counts, validate_rank_plans, RankPlan};

fn plans(width: u64) -> Vec<RankPlan> {
    (0..3)
        .map(|rank| RankPlan {
            global_beam_width: width,
            beam_alignment: 32,
            shards: 4,
            outer_batch: 128,
            world: 3,
            batch_alignment: 32,
            rank: Some(rank),
            required_bytes: 1024,
            budget_bytes: 2048,
        })
        .collect()
}

#[test]
fn exact_width_quotas_do_not_overflow_or_include_spare_slots() {
    assert_eq!(rank_frontier_counts(&plans(5), 5).unwrap(), vec![2, 2, 1]);
    let large = plans(u64::MAX);
    let counts = rank_frontier_counts(&large, u64::MAX).unwrap();
    assert_eq!(
        counts.iter().map(|n| *n as u128).sum::<u128>(),
        u64::MAX as u128
    );
    assert!(validate_rank_plans(&plans(6), 5).is_err());
}

#[test]
fn admission_requires_every_rank_and_identical_protocol() {
    let mut rows = plans(5);
    rows[1].rank = Some(0);
    assert!(validate_rank_plans(&rows, 5).is_err());
    rows = plans(5);
    rows[1].outer_batch += 1;
    assert!(validate_rank_plans(&rows, 5).is_err());
    rows = plans(5);
    rows[1].required_bytes = 2049;
    let error = validate_rank_plans(&rows, 5).unwrap_err().to_string();
    assert!(error.contains("required=2049 available=2048"), "{error}");
    rows = plans(5);
    rows[0].rank = None;
    assert!(validate_rank_plans(&rows, 5).is_err());
    assert!(validate_rank_plans(&[], 5).is_err());
}
