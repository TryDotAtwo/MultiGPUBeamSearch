use beam_runtime::{
    maximum_beam::{discover, Config},
    Error,
};
use serde_json::json;
use std::time::{Duration, Instant};

#[test]
fn every_shard_profile_has_its_own_frozen_search_and_only_capacity_rejection_is_a_bound() {
    let config = Config {
        upper_bound: 10000,
        alignment: 256,
        world: 2,
        outer: 16,
        minimum_flush: 2048,
    };
    let mut calls = 0;
    let result = discover(
        &config,
        true,
        Instant::now() + Duration::from_secs(10),
        |beam, env| {
            calls += 1;
            let shards: u64 = env["BEAM_SHARD_COUNT"].parse().unwrap();
            let flush: u64 = env["BEAM_STREAM4_BATCH_CANDIDATES"].parse().unwrap();
            assert_eq!(beam % (2 * shards * 256), 0);
            assert_eq!(env.len(), 5);
            if beam > 8192 || (flush > 2048 && beam > 4096) {
                return Err(Error::CapacityRejected("fixture memory bound".into()));
            }
            Ok((0..2)
                .map(|rank| {
                    json!({"GLOBAL_BEAM_WIDTH_EFFECTIVE":beam,"BEAM_WIDTH_ALIGNMENT":256,
            "SHARD_COUNT":shards,"B_MICRO":16,"WORLD_SIZE":2,"STREAM4_BATCH_ALIGNMENT":256,
            "LOCAL_RANK":rank,"estimated_required_device_bytes":beam,"gpu_budget_bytes":10000,
            "SHARD_CAPACITY_CANDIDATES":4096})
                })
                .collect())
        },
    )
    .unwrap();
    assert_eq!(result.capacity.effective_beam, 4096);
    assert_eq!(result.memory_ceiling.effective_beam, 8192);
    assert_eq!(result.capacity.probes, calls);
    assert!(result.capacity.search_complete);
    assert!(!result.capacity.full_step_verified);
    assert_eq!(
        result.capacity.profile["BEAM_STREAM4_BATCH_CANDIDATES"],
        "4096"
    );
    let mut attempts = 0;
    assert!(discover(
        &config,
        false,
        Instant::now() + Duration::from_secs(10),
        |_, _| {
            attempts += 1;
            Err(Error::Message("rank failure".into()))
        }
    )
    .is_err());
    assert_eq!(attempts, 1);
}
