use beam_core::profile_policy::{Environment, Hardware};
use beam_runtime::{
    component_tuning::{tune_components, ComponentEvaluator, ComponentProbe, TuneConfig},
    telemetry::{parse, Receipt},
    Error, Result,
};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

fn baseline() -> Environment {
    [
        ("BEAM_ENSEMBLE_INFERENCE_MICRO", "16"),
        ("BEAM_B_MICRO", "16"),
        ("BEAM_SHARD_COUNT", "2"),
        ("BEAM_STREAM3_RING_SLOTS", "2"),
        ("BEAM_STREAM4_ACTIVE_SORT_SLOTS", "1"),
        ("BEAM_STREAM4_BATCH_CANDIDATES", "128"),
    ]
    .map(|(k, v)| (k.into(), v.into()))
    .into()
}
fn config() -> TuneConfig {
    TuneConfig {
        beam: 4097,
        moves: 24,
        micro: 16,
        row_factor: 1,
        inference_seconds: 0.001,
        hardware: Hardware {
            sm_count: 28,
            l2_bytes: 3 << 20,
        },
        gpu_uuids: vec!["a".into(), "b".into()],
        device_indices: vec![0, 1],
    }
}
#[derive(Default)]
struct Engine {
    events: Vec<String>,
    closed: bool,
    throttle: bool,
    fail_rank: bool,
    unstable_first: bool,
    malformed: bool,
    baseline_calls: usize,
    scalar: bool,
}
impl ComponentEvaluator for Engine {
    fn admit(&mut self, env: &Environment) -> Result<Vec<Value>> {
        self.events.push(format!("admit:{}", env["BEAM_B_MICRO"]));
        if env["BEAM_SHARD_COUNT"] == "1" {
            return Err(Error::CapacityRejected("precise required/available".into()));
        }
        let n = |key: &str| env[key].parse::<u64>().unwrap();
        Ok((0..2).map(|rank|json!({"GLOBAL_BEAM_WIDTH_EFFECTIVE":4097,"BEAM_WIDTH_ALIGNMENT":32,
            "WORLD_SIZE":2,"LOCAL_RANK":rank,"B_MICRO":n("BEAM_B_MICRO")/if self.scalar {24}else{1},"SHARD_COUNT":n("BEAM_SHARD_COUNT"),
            "STREAM4_BATCH_ALIGNMENT":32,"estimated_required_device_bytes":512,"gpu_budget_bytes":1024,
            "frontier_state_capacity":2049,"STREAM3_RING_SLOTS":n("BEAM_STREAM3_RING_SLOTS"),
            "STREAM4_BATCH_CANDIDATES":n("BEAM_STREAM4_BATCH_CANDIDATES"),"SHARD_CAPACITY_CANDIDATES":4096,
            "STREAM4_ACTIVE_SORT_SLOTS":n("BEAM_STREAM4_ACTIVE_SORT_SLOTS")})).collect())
    }
    fn measure(&mut self, env: &Environment, plans: &[Value]) -> Result<ComponentProbe> {
        self.events.push(format!("measure:{}", env["BEAM_B_MICRO"]));
        if self.fail_rank {
            return Err(Error::Message("rank one died".into()));
        }
        let is_baseline = *env == baseline();
        if is_baseline {
            self.baseline_calls += 1;
        }
        let seconds = if is_baseline { 0.01 } else { 0.001 };
        let times = if is_baseline && self.unstable_first && self.baseline_calls == 1 {
            vec![0.01, 0.02, 0.03, 0.04, 0.05]
        } else {
            vec![seconds; 5]
        };
        let rows=(0..2).map(|rank|{let p=&plans[rank];
            let q=p["B_MICRO"].as_u64().unwrap()*p["STREAM3_RING_SLOTS"].as_u64().unwrap()*24;
            json!({"component_calibration":true,"correctness_passed":true,"full_step_verified":false,
                "rank":if self.malformed {0}else{rank},"outer_candidates":q,"shard_capacity":4096,
                "sort_jobs_concurrent":p["STREAM4_ACTIVE_SORT_SLOTS"],"stream3_seconds":times,
                "stream4_group_seconds":times,"union_seconds":times,
                "scope":"raw metadata retained","transport":[{"items":q,"seconds":times}]})
        }).collect();
        let sample=parse("0,GPU-a,60,Not Active,Not Active,Not Active\n1,GPU-b,60,Not Active,Not Active,Not Active").unwrap();
        let mut telemetry = Receipt {
            samples: vec![sample.clone(), sample],
            errors: vec![],
            interval_seconds: 5.,
        };
        telemetry.samples[0][1].throttled = self.throttle;
        Ok(ComponentProbe { rows, telemetry })
    }
    fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.events.push("close".into());
        Ok(())
    }
}
#[test]
fn bounded_component_tuning_keeps_exact_admission_raw_receipts_and_no_full_step_claim() {
    let mut engine = Engine::default();
    let result = tune_components(
        &mut engine,
        &config(),
        &baseline(),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert!(engine.closed);
    assert!(!result.pipeline_verified);
    assert_eq!(result.workload_parents, 4097);
    assert_ne!(result.selection, "baseline");
    assert!(result.tested.len() <= 5);
    assert!(result.tested.iter().any(|r| r
        .rejected
        .as_deref()
        .is_some_and(|s| s.contains("required/available"))));
    for row in result.tested.iter().filter(|r| r.rejected.is_none()) {
        assert_eq!(row.environment["BEAM_ENSEMBLE_INFERENCE_MICRO"], "16");
        assert_eq!(row.rank_measurements[0]["scope"], "raw metadata retained");
        assert_eq!(row.plans[0]["GLOBAL_BEAM_WIDTH_EFFECTIVE"], 4097);
    }
    for pair in engine.events[..engine.events.len() - 1].windows(2) {
        if pair[1].starts_with("measure:") {
            assert!(pair[0].starts_with("admit:"));
        }
    }
}
#[test]
fn cancellation_or_rank_failure_does_not_become_capacity_and_bad_cohorts_never_select() {
    for mut engine in [
        Engine {
            throttle: true,
            ..Engine::default()
        },
        Engine {
            fail_rank: true,
            ..Engine::default()
        },
        Engine {
            malformed: true,
            ..Engine::default()
        },
    ] {
        let result = tune_components(
            &mut engine,
            &config(),
            &baseline(),
            Instant::now() + Duration::from_secs(10),
        );
        assert!(result.is_err());
        assert!(engine.closed);
        assert_eq!(
            engine
                .events
                .iter()
                .filter(|s| s.starts_with("measure:"))
                .count(),
            1
        );
    }
    let mut engine = Engine {
        unstable_first: true,
        ..Engine::default()
    };
    let result = tune_components(
        &mut engine,
        &config(),
        &baseline(),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert_eq!(engine.baseline_calls, 2);
    assert!(result.initial_baseline.is_some());
    assert_eq!(result.baseline_retry, "recovered_after_one_retry");
    assert!(engine.closed);
}

#[test]
fn scalar_scorer_keeps_row_batch_distinct_from_parent_batch() {
    let mut engine = Engine {
        scalar: true,
        ..Engine::default()
    };
    let mut config = config();
    config.row_factor = 24;
    let mut env = baseline();
    env.insert("BEAM_B_MICRO".into(), "384".into());
    let result = tune_components(
        &mut engine,
        &config,
        &env,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    let base = &result.tested[0];
    assert_eq!(base.environment["BEAM_B_MICRO"], "384");
    assert_eq!(base.plans[0]["B_MICRO"], 16);
    assert_eq!(base.rank_measurements[0]["outer_candidates"], 16 * 2 * 24);
    assert!(engine.closed);
}

#[test]
fn component_cache_recomputes_evidence_and_readmits_exact_geometry() {
    use beam_runtime::pipeline_cache;
    let dir = std::env::temp_dir().join(format!("beam-component-cache-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.json");
    let mut engine = Engine::default();
    let config = config();
    let selected = tune_components(
        &mut engine,
        &config,
        &baseline(),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    let identity = json!({"fixture":"exact-admission","beam":4097});
    pipeline_cache::write_components(&path, &identity, &config, &selected).unwrap();
    let mut admitted = 0;
    let hit = pipeline_cache::read_components(&path, &identity, &config, |env| {
        admitted += 1;
        Engine::default().admit(env)
    })
    .unwrap()
    .unwrap();
    assert_eq!(admitted, 1);
    assert!(hit.selection.cache_hit);
    assert_eq!(hit.current_rank_plans.len(), 2);
    assert!(
        pipeline_cache::read_components(&path, &identity, &config, |env| {
            let mut p = Engine::default().admit(env)?;
            for row in &mut p {
                row["SHARD_CAPACITY_CANDIDATES"] = json!(8192);
            }
            Ok(p)
        })
        .unwrap()
        .is_none()
    );
    assert!(
        pipeline_cache::read_components(&path, &identity, &config, |_| Err(Error::Message(
            "rank died".into()
        )))
        .is_err()
    );
    let valid: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    for mutation in 0..5 {
        let mut invalid = valid.clone();
        let d = &mut invalid["selection"];
        match mutation {
            0 => d["environment"]["BEAM_SOLUTION_MODE"] = json!("collect"),
            1 => d["tested"][0]["rank_measurements"][1]["rank"] = json!(0),
            2 => d["selection"] = json!("baseline"),
            3 => d["tested"][0]["telemetry"]["samples"][0][0]["throttled"] = json!(true),
            _ => d["estimate"]["service_work_seconds"] = json!(0.000001),
        }
        std::fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(
            pipeline_cache::read_components(&path, &identity, &config, |_| panic!(
                "bad cache must not launch a measurement/admission"
            ))
            .unwrap()
            .is_none(),
            "{mutation}"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}
