use beam_core::calibration::{
    estimate, paired_promotion, select, Measurement, PairedMeasurement, Workload,
};

fn sample(profile: &str, seconds: f64) -> Measurement {
    Measurement {
        profile: profile.into(),
        parents: 100,
        seconds_by_rank: vec![seconds * 0.8, seconds],
        correctness_passed: true,
        memory_admitted: true,
        throttled: false,
    }
}
#[test]
fn complete_slow_rank_cohorts_select_stable_gains_only() {
    let mut rows = vec![sample("base", 10.); 5];
    rows.extend(vec![sample("fast", 8.); 5]);
    rows.extend(vec![sample("few", 1.); 4]);
    let (winner, rejected) = select(&rows, "base", 0.05).unwrap();
    assert_eq!(winner.profile, "fast");
    assert_eq!(winner.median, 0.08);
    assert!(rejected.contains_key("few"));
    rows[0].throttled = true;
    assert!(select(&rows, "base", 0.05).is_err());
    rows = vec![sample("base", 10.); 5];
    rows[1].seconds_by_rank[0] = f64::NAN;
    assert!(estimate("base", &rows).is_err());
    rows = vec![sample("base", 10.); 5];
    rows[1].parents = 101;
    assert!(estimate("base", &rows).is_err());
}

fn pair(seconds: f64, bytes: u64) -> PairedMeasurement {
    let workload = Workload {
        model: "sha-model".into(),
        input: "sha-input".into(),
        beam: 100,
        precision: "fp16-fp32".into(),
        ranks: 2,
    };
    PairedMeasurement {
        baseline_workload: workload.clone(),
        candidate_workload: workload,
        baseline: sample("base", 10.),
        candidate: sample("candidate", seconds),
        baseline_peak_bytes: 2000,
        candidate_peak_bytes: bytes,
    }
}
#[test]
fn optimization_promotion_needs_five_matching_pairs_and_material_full_step_gain() {
    assert!(!paired_promotion(&vec![pair(8., 2000); 4]).unwrap().accepted);
    assert!(
        paired_promotion(&vec![pair(9.4, 2000); 5])
            .unwrap()
            .accepted
    );
    assert!(
        !paired_promotion(&vec![pair(9.8, 2000); 5])
            .unwrap()
            .accepted
    );
    assert!(
        paired_promotion(&vec![pair(10.2, 1600); 5])
            .unwrap()
            .accepted
    );
    assert!(
        !paired_promotion(&vec![pair(10.4, 1600); 5])
            .unwrap()
            .accepted
    );
    let mut rows = vec![pair(8., 1600); 5];
    rows[0].candidate_workload.precision = "fp8".into();
    assert!(paired_promotion(&rows).is_err());
    rows = vec![pair(8., 1600); 5];
    rows[0].candidate.throttled = true;
    assert!(paired_promotion(&rows).is_err());
    rows = vec![pair(8., 1600); 5];
    rows[0].candidate.seconds_by_rank[0] = 30.;
    assert!(!paired_promotion(&rows).unwrap().accepted);
}
