use beam_runtime::telemetry::{parse, Receipt};
#[test]
fn telemetry_requires_selected_cohorts_and_rejects_missing_or_throttled_rows() {
    let sample=parse("0, GPU-abc, 63, Not Active, Not Active, Not Active\n1, GPU-def, 62, Not Active, Not Active, Not Active").unwrap();
    let mut receipt = Receipt {
        samples: vec![sample.clone(), sample],
        errors: vec![],
        interval_seconds: 5.,
    };
    assert!(receipt.verified(&["abc".into(), "def".into()], &[0, 1]));
    assert_eq!(serde_json::to_value(&receipt).unwrap()["throttled"], false);
    receipt.samples[1][1].throttled = true;
    assert_eq!(serde_json::to_value(&receipt).unwrap()["throttled"], true);
    assert!(!receipt.verified(&["abc".into(), "def".into()], &[0, 1]));
    assert!(!receipt.verified(&["unknown".into()], &[0]));
    assert!(parse("0, GPU-a, 60, N/A, Not Active, Not Active").is_err());
    assert!(parse("").is_err());
}
