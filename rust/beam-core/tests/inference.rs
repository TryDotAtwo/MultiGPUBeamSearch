use beam_core::{
    calibration::Measurement,
    inference::{coarse_batches, refinement_batches, select_inference},
};
use std::collections::BTreeSet;

#[test]
fn bounded_inference_candidates_preserve_non_power_of_two_cap() {
    let coarse = coarse_batches(1000).unwrap();
    assert_eq!(&coarse[..2], &[256, 1000]);
    assert_eq!(coarse.len(), coarse.iter().collect::<BTreeSet<_>>().len());
    assert_eq!(
        coarse.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([32, 64, 128, 256, 512, 1000])
    );
    assert_eq!(coarse_batches(17).unwrap(), vec![17]);
    assert!(coarse_batches(0).is_err());
    assert!(coarse_batches(65537).is_err());
    let visited = coarse.into_iter().collect();
    let refined = refinement_batches(512, 1000, &visited).unwrap();
    assert_eq!(
        refined.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([320, 384, 448, 576, 640, 768])
    );
    assert!(refinement_batches(1001, 1000, &visited).is_err());
}

#[test]
fn noisy_baseline_does_not_disqualify_complete_stable_inference_cohort() {
    let row = |profile: &str, seconds: f64| Measurement {
        profile: profile.into(),
        parents: 1024,
        seconds_by_rank: vec![seconds / 2., seconds],
        correctness_passed: true,
        memory_admitted: true,
        throttled: false,
    };
    let mut samples = vec![
        row("256", 1.),
        row("256", 4.),
        row("256", 2.),
        row("256", 7.),
        row("256", 3.),
    ];
    samples.extend(vec![row("512", 0.5); 7]);
    let (winner, rejected) = select_inference(&samples, "256").unwrap();
    assert_eq!(winner.profile, "512");
    assert!(rejected.contains_key("256"));
    for sample in &mut samples {
        sample.throttled = true;
    }
    assert!(select_inference(&samples, "256").is_err());
}
