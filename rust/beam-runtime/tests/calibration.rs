use beam_core::calibration::Measurement;
use beam_runtime::calibration::{find_capacity, tune_profiles, Profile, ProfileEvaluator};
use beam_runtime::{Error, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

fn plans(beam: u64) -> Vec<Value> {
    (0..2)
        .map(|rank| {
            json!({"GLOBAL_BEAM_WIDTH_EFFECTIVE":beam,"BEAM_WIDTH_ALIGNMENT":32,
        "WORLD_SIZE":2,"LOCAL_RANK":rank,"B_MICRO":64,"SHARD_COUNT":1,"STREAM4_BATCH_ALIGNMENT":32,
        "estimated_required_device_bytes":512,"gpu_budget_bytes":1024})
        })
        .collect()
}
#[test]
fn capacity_only_retreats_on_explicit_rejection_and_keeps_exact_lower_bound() {
    let deadline = Instant::now() + Duration::from_secs(10);
    let profiles = vec![BTreeMap::new()];
    let result = find_capacity(&profiles, 100, 4, deadline, |beam, _| {
        if beam > 63 {
            Err(Error::CapacityRejected("capacity".into()))
        } else {
            Ok(plans(beam))
        }
    })
    .unwrap();
    assert_eq!(result.requested_beam, 60);
    assert_eq!(result.upper_rejected_beam, Some(64));
    assert!(result.search_complete);
    assert!(!result.full_step_verified);
    assert!(
        find_capacity(&profiles, 100, 4, deadline, |_, _| Err(Error::Message(
            "rank died".into()
        )))
        .unwrap_err()
        .to_string()
        .contains("rank died")
    );
    assert!(find_capacity(&profiles, 100, 4, deadline, |beam, _| Ok(plans(beam + 4))).is_err());
}
struct Engine {
    events: Vec<String>,
}
impl ProfileEvaluator for Engine {
    fn admit(&mut self, environment: &BTreeMap<String, String>) -> Result<Vec<Value>> {
        self.events
            .push(format!("admit:{}", environment["BEAM_B_MICRO"]));
        Ok(plans(100))
    }
    fn finish_planning(&mut self) -> Result<()> {
        self.events.push("release".into());
        Ok(())
    }
    fn measure(
        &mut self,
        environment: &BTreeMap<String, String>,
        _: &[Value],
    ) -> Result<Vec<Measurement>> {
        assert!(self.events.iter().any(|e| e == "release"));
        self.events
            .push(format!("measure:{}", environment["BEAM_B_MICRO"]));
        let seconds = if environment["BEAM_B_MICRO"] == "128" {
            8.
        } else {
            10.
        };
        Ok(vec![
            Measurement {
                profile: String::new(),
                parents: 100,
                seconds_by_rank: vec![seconds, seconds],
                correctness_passed: true,
                memory_admitted: true,
                throttled: false
            };
            5
        ])
    }
}
#[test]
fn rust_owns_admit_release_measure_select_sequence_and_freezes_microbatch() {
    let environment = |outer: &str| {
        BTreeMap::from([
            ("BEAM_B_MICRO".into(), outer.into()),
            ("BEAM_ENSEMBLE_INFERENCE_MICRO".into(), "32".into()),
        ])
    };
    let profiles = vec![
        Profile {
            name: "baseline".into(),
            environment: environment("64"),
        },
        Profile {
            name: "large".into(),
            environment: environment("128"),
        },
    ];
    let mut engine = Engine { events: vec![] };
    let result = tune_profiles(
        &mut engine,
        &profiles,
        100,
        32,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert_eq!(result.estimate.profile, "large");
    assert_eq!(
        engine.events,
        vec![
            "admit:64",
            "admit:128",
            "release",
            "measure:64",
            "measure:128"
        ]
    );
    let mut bad = profiles;
    bad[1]
        .environment
        .insert("BEAM_ENSEMBLE_INFERENCE_MICRO".into(), "64".into());
    assert!(tune_profiles(
        &mut engine,
        &bad,
        100,
        32,
        Instant::now() + Duration::from_secs(10)
    )
    .is_err());
}
