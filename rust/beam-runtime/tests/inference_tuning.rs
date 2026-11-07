use beam_runtime::{
    inference::{Batch, RankFailure},
    inference_tuning::{tune_inference, InferenceEvaluator, ProbeResult, TuneConfig},
    telemetry::{parse, Receipt},
    Result,
};
use serde_json::json;
use std::time::Duration;
struct Engine {
    calls: Vec<u64>,
    closed: bool,
    oom: bool,
    throttled: bool,
}
impl InferenceEvaluator for Engine {
    fn capacity(&self) -> u64 {
        512
    }
    fn starts(&self) -> u64 {
        2
    }
    fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }
    fn measure(&mut self, batch: u64, parents: u64, _timeout: Duration) -> Result<ProbeResult> {
        self.calls.push(batch);
        let sample=parse("0,GPU-a,60,Not Active,Not Active,Not Active\n1,GPU-b,60,Not Active,Not Active,Not Active").unwrap();
        let mut telemetry = Receipt {
            samples: vec![sample.clone(), sample],
            errors: vec![],
            interval_seconds: 5.,
        };
        telemetry.samples[1][0].throttled = self.throttled;
        let failures = if self.oom && batch >= 256 {
            vec![RankFailure {
                rank: 1,
                error: "CUDA capacity rejected".into(),
                capacity: true,
            }]
        } else {
            vec![]
        };
        let seconds = if batch == 384 {
            0.5
        } else if batch == 512 {
            0.75
        } else {
            1.
        };
        let rows = if failures.is_empty() {
            (0..2).map(|rank|json!({"batch":batch,"parents":parents,"device":rank,"seconds":vec![seconds;7],"numeric_error":0,"correctness_passed":true,"torch_reserved_peak_bytes":batch*1000})).collect()
        } else {
            vec![]
        };
        Ok(ProbeResult {
            batch: Batch { rows, failures },
            telemetry,
        })
    }
}
fn engine() -> Engine {
    Engine {
        calls: vec![],
        closed: false,
        oom: false,
        throttled: false,
    }
}
fn config() -> TuneConfig {
    TuneConfig {
        cap: 512,
        parents: 8192,
        world: 2,
        capacity_bootstrap: false,
        shared_batch: None,
        device_indices: vec![0, 1],
        gpu_uuids: vec!["a".into(), "b".into()],
    }
}
#[test]
fn inference_control_owns_coarse_refinement_reserve_and_complete_rank_statistics() {
    let mut engine = engine();
    let result = tune_inference(&mut engine, &config(), Duration::from_secs(60)).unwrap();
    assert!(engine.closed);
    assert_eq!(result.parent_batch, 384);
    assert_eq!(result.reserve_bytes, 384000 + (512 << 20));
    assert_eq!(result.estimate.repeats, 7);
    assert_eq!(result.estimate.median, 0.5 / 16384.);
    assert_eq!(result.probe_process_starts, 2);
    assert!(!result.pipeline_verified);
    assert_eq!(&engine.calls[..2], &[256, 512]);
    assert!(result.refinement_candidates.contains(&384));
    assert_eq!(
        engine.calls.len(),
        engine
            .calls
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
}
#[test]
fn explicit_capacity_rejection_uses_smaller_rows_and_throttling_rejects_entire_selection() {
    let mut smaller = engine();
    smaller.oom = true;
    let result = tune_inference(&mut smaller, &config(), Duration::from_secs(60)).unwrap();
    assert!(result.capacity_limited_search);
    assert!(result.parent_batch < 256);
    assert!(smaller.closed);
    assert!(smaller.calls.iter().skip(1).all(|batch| *batch < 256));
    let mut bad = engine();
    bad.throttled = true;
    assert!(tune_inference(&mut bad, &config(), Duration::from_secs(60)).is_err());
    assert!(bad.closed);
}
#[test]
fn shared_proposal_and_capacity_seed_require_fresh_verified_cohorts() {
    let mut shared = engine();
    let mut options = config();
    options.shared_batch = Some(128);
    let result = tune_inference(&mut shared, &options, Duration::from_secs(60)).unwrap();
    assert!(result.shared_profile_validated);
    assert_eq!(shared.calls, vec![128]);
    let mut seed = engine();
    options.shared_batch = None;
    options.cap = 32;
    options.capacity_bootstrap = true;
    let result = tune_inference(&mut seed, &options, Duration::from_secs(60)).unwrap();
    assert_eq!(result.parent_batch, 32);
    assert_eq!(seed.calls, vec![32]);
    assert!(result.refinement_candidates.is_empty());
}
