use beam_core::components::{
    service_envelope, validate_components, ComponentPlan, ComponentReceipt,
};
use serde_json::json;

fn fixture() -> (Vec<ComponentPlan>, Vec<ComponentReceipt>) {
    let plans = (0..2)
        .map(|rank| {
            serde_json::from_value(json!({
                "GLOBAL_BEAM_WIDTH_EFFECTIVE":65,"BEAM_WIDTH_ALIGNMENT":2048,"SHARD_COUNT":2,
                "B_MICRO":16,"WORLD_SIZE":2,"STREAM4_BATCH_ALIGNMENT":1024,"LOCAL_RANK":rank,
                "estimated_required_device_bytes":1000,"gpu_budget_bytes":2000,
                "frontier_state_capacity":33,"STREAM3_RING_SLOTS":2,"STREAM4_BATCH_CANDIDATES":1024,
                "SHARD_CAPACITY_CANDIDATES":2048,"STREAM4_ACTIVE_SORT_SLOTS":2
            }))
            .unwrap()
        })
        .collect();
    let rows = (0..2)
        .map(|rank| {
            serde_json::from_value(json!({
                "component_calibration":true,"correctness_passed":true,"full_step_verified":false,
                "rank":rank,"outer_candidates":768,"shard_capacity":2048,"sort_jobs_concurrent":2,
                "stream3_seconds":vec![0.01;5],"stream4_group_seconds":vec![0.02;5],
                "union_seconds":vec![0.03;5],"transport":[{"items":768,"seconds":vec![0.04;5]}]
            }))
            .unwrap()
        })
        .collect();
    (plans, rows)
}
#[test]
fn exact_service_contracts_and_reference_proxy() {
    let (plans, rows) = fixture();
    validate_components(&rows, &plans, 65, 24).unwrap();
    let estimate = service_envelope(&rows, &plans[0], 24, 0.001, None).unwrap();
    assert_eq!(
        estimate.scope,
        "isolated exact-capacity service envelope; not full-step timing"
    );
    // Per-rank gross=792. S3=2*.01, sort=ceil(2*ceil(396/1024)/2)*.02,
    // transport=2*.04, final union=2*.03, inference=33*2*.001.
    assert!((estimate.service_envelope_seconds - 0.14).abs() < 1e-12);
    assert!((estimate.downstream_service_work_seconds - 0.18).abs() < 1e-12);
    assert!((estimate.service_work_seconds - 0.246).abs() < 1e-12);
    assert_eq!(
        serde_json::to_value(service_envelope(&rows, &plans[0], 24, 0.001, Some(0)).unwrap())
            .unwrap(),
        serde_json::to_value(&estimate).unwrap()
    );
    for mutate in 0..7 {
        let mut bad = rows.clone();
        match mutate {
            0 => bad[1].rank = 0,
            1 => bad[1].outer_candidates += 1,
            2 => bad[1].shard_capacity -= 1,
            3 => bad[1].correctness_passed = false,
            4 => bad[1].stream3_seconds[0] = f64::NAN,
            5 => bad[1].transport[0].seconds.truncate(4),
            _ => bad[1].sort_jobs_concurrent = 0,
        }
        assert!(
            validate_components(&bad, &plans, 65, 24).is_err(),
            "{mutate}"
        );
    }
    assert!(validate_components(&rows, &plans, 64, 24).is_err());
    assert!(service_envelope(&rows, &plans[0], 24, -1., None).is_err());
    assert!(service_envelope(&rows, &plans[0], 24, 0.001, Some(5)).is_err());
    // Native probes cap concurrent jobs at the two physical buffers per shard.
    let mut bounded_plans = plans.clone();
    let mut bounded_rows = rows.clone();
    for plan in &mut bounded_plans {
        plan.sort_slots = 10;
    }
    for row in &mut bounded_rows {
        row.sort_jobs_concurrent = 4;
    }
    validate_components(&bounded_rows, &bounded_plans, 65, 24).unwrap();
}
