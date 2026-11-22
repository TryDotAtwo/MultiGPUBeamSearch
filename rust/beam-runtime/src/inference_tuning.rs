//! Inference calibration control, process lifetime, telemetry and selection.
use crate::{
    inference::{validate_receipt, Batch, InferencePool},
    telemetry::{Receipt, Telemetry},
    Error, Result,
};
use beam_core::{
    calibration::{estimate, Estimate, Measurement},
    inference::{coarse_batches, refinement_batches, select_inference},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneConfig {
    pub cap: u64,
    pub parents: u64,
    pub world: u32,
    pub capacity_bootstrap: bool,
    pub shared_batch: Option<u64>,
    pub device_indices: Vec<u32>,
    pub gpu_uuids: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ProbeResult {
    pub batch: Batch,
    pub telemetry: Receipt,
}
pub trait InferenceEvaluator {
    fn measure(&mut self, batch: u64, parents: u64, timeout: Duration) -> Result<ProbeResult>;
    fn capacity(&self) -> u64;
    fn starts(&self) -> u64;
    fn close(&mut self) -> Result<()>;
}
struct PoolEvaluator<'a> {
    pool: &'a mut InferencePool,
}
impl InferenceEvaluator for PoolEvaluator<'_> {
    fn measure(&mut self, batch: u64, parents: u64, timeout: Duration) -> Result<ProbeResult> {
        let telemetry = Telemetry::start()?;
        let rows = self.pool.measure(batch, parents, timeout);
        let telemetry = telemetry.finish()?;
        let result = ProbeResult {
            batch: rows?,
            telemetry,
        };
        crate::atomic_json(
            &self
                .pool
                .log_directory()
                .join(format!("batch-{batch}-receipt.json")),
            &result,
        )?;
        Ok(result)
    }
    fn capacity(&self) -> u64 {
        self.pool.capacity()
    }
    fn starts(&self) -> u64 {
        self.pool.starts()
    }
    fn close(&mut self) -> Result<()> {
        self.pool.close()
    }
}
impl InferencePool {
    pub fn tune(&mut self, config: &TuneConfig, timeout: Duration) -> Result<Selection> {
        if config.parents != self.parents() || config.world != self.world() {
            return Err(Error::Message(
                "tuning workload does not match its inference pool".into(),
            ));
        }
        let selection = tune_inference(&mut PoolEvaluator { pool: self }, config, timeout)?;
        crate::atomic_json(
            &self.log_directory().join("inference-selection.json"),
            &selection,
        )?;
        Ok(selection)
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Selection {
    pub search_policy: String,
    pub phase: String,
    pub parent_batch: u64,
    pub reserve_bytes: u64,
    pub estimate: Estimate,
    pub records: Vec<Value>,
    pub measurements: Vec<Measurement>,
    pub rejected: BTreeMap<String, String>,
    pub stat_rejected: BTreeMap<String, String>,
    pub gpu_telemetry: BTreeMap<String, Receipt>,
    pub coarse_candidates: Vec<u64>,
    pub refinement_candidates: Vec<u64>,
    pub calibration_wall_seconds: f64,
    pub probe_process_starts: u64,
    pub capacity_limited_search: bool,
    pub shared_profile_validated: bool,
    pub measured_candidates: Vec<u64>,
    pub pipeline_verified: bool,
    pub cache_hit: bool,
}
fn probe_allowance(records: &[Value]) -> f64 {
    records
        .iter()
        .filter_map(|r| r["seconds"].as_array())
        .flatten()
        .filter_map(Value::as_f64)
        .fold(12., |limit, seconds| limit.max(10. * seconds + 3.))
}
pub fn tune_inference(
    evaluator: &mut impl InferenceEvaluator,
    config: &TuneConfig,
    timeout: Duration,
) -> Result<Selection> {
    let started = Instant::now();
    let result = tune(evaluator, config, timeout);
    let cleanup = evaluator.close();
    let mut selection = result?;
    cleanup?;
    selection.probe_process_starts = evaluator.starts();
    selection.calibration_wall_seconds = started.elapsed().as_secs_f64();
    Ok(selection)
}
fn tune(
    evaluator: &mut impl InferenceEvaluator,
    config: &TuneConfig,
    timeout: Duration,
) -> Result<Selection> {
    if config.cap == 0
        || config.cap > evaluator.capacity()
        || config.cap > 65536
        || config.parents < config.cap
        || config.parents > u32::MAX as u64
        || config.world == 0
        || config.world > 65535
        || config.device_indices.len() != config.world as usize
        || timeout.is_zero()
        || config.device_indices.iter().collect::<BTreeSet<_>>().len() != config.world as usize
    {
        return Err(Error::Message(
            "invalid inference tuning bounds/workload/devices".into(),
        ));
    }
    let started = Instant::now();
    let budget = if config.capacity_bootstrap {
        timeout.min(Duration::from_secs(30))
    } else {
        timeout
    };
    let deadline = started
        .checked_add(budget)
        .ok_or_else(|| Error::Message("inference deadline overflow".into()))?;
    let coarse_deadline = started + budget.mul_f64(0.7);
    let mut candidates = if config.capacity_bootstrap {
        std::iter::once(config.cap)
            .chain([16, 8, 4, 2, 1].into_iter().filter(|v| *v < config.cap))
            .collect()
    } else {
        coarse_batches(config.cap)?
    };
    let baseline = candidates[0].to_string();
    let shared = config
        .shared_batch
        .filter(|b| !config.capacity_bootstrap && *b > 0 && *b <= config.cap);
    if let Some(batch) = shared {
        candidates.retain(|v| *v != batch);
        candidates.insert(0, batch);
    }
    let mut coarse = candidates.clone();
    let mut refined = Vec::new();
    let mut refinement_queued = config.capacity_bootstrap;
    let mut index = 0;
    let mut samples = Vec::new();
    let mut records = Vec::new();
    let mut rejected = BTreeMap::new();
    let mut reserves = BTreeMap::new();
    let mut telemetry_records = BTreeMap::new();
    let mut capacity_limited = false;
    let mut shared_used = false;
    while index < candidates.len() {
        let batch = candidates[index];
        let current = index;
        index += 1;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero()
            || (current > 0 && remaining.as_secs_f64() < probe_allowance(&records))
        {
            break;
        }
        if !samples.is_empty()
            && current < coarse.len()
            && current > 0
            && Instant::now() >= coarse_deadline
        {
            if !refinement_queued {
                if let Ok((winner, _)) = select_inference(&samples, &baseline) {
                    refined = refinement_batches(
                        winner.profile.parse().map_err(|_| {
                            Error::Message("invalid selected inference batch".into())
                        })?,
                        config.cap,
                        &candidates.iter().copied().collect(),
                    )?;
                    candidates.extend(&refined);
                    refinement_queued = true;
                }
            }
            // No new coarse starts after the boundary; an in-flight healthy
            // probe always receives the full remaining phase deadline.
            if refinement_queued {
                continue;
            }
        }
        if batch > evaluator.capacity() {
            rejected.insert(
                batch.to_string(),
                "batch exceeds reduced inference capacity".into(),
            );
            capacity_limited = true;
            continue;
        }
        let probe = evaluator.measure(batch, config.parents, remaining)?;
        let name = batch.to_string();
        for row in &probe.batch.rows {
            let rank = row["device"]
                .as_u64()
                .filter(|r| *r < config.world as u64)
                .ok_or_else(|| Error::Message("invalid inference receipt rank".into()))?;
            validate_receipt(row, rank as u32, batch, config.parents)?;
        }
        let ranks: BTreeSet<_> = probe
            .batch
            .rows
            .iter()
            .filter_map(|r| r["device"].as_u64())
            .collect();
        if ranks.len() != probe.batch.rows.len() {
            return Err(Error::Message("duplicate inference receipt rank".into()));
        }
        let failures = &probe.batch.failures;
        if !failures.is_empty() {
            rejected.insert(
                name.clone(),
                failures
                    .iter()
                    .map(|f| format!("rank {}: {}", f.rank, f.error))
                    .collect::<Vec<_>>()
                    .join("; "),
            );
        }
        capacity_limited |= failures.iter().any(|f| f.capacity);
        let verified = probe
            .telemetry
            .verified(&config.gpu_uuids, &config.device_indices);
        telemetry_records.insert(name.clone(), probe.telemetry);
        let complete = probe.batch.rows.len() == config.world as usize && failures.is_empty();
        if complete {
            if !verified {
                rejected.insert(
                    name.clone(),
                    "GPU telemetry missing, incomplete or throttled".into(),
                );
            }
            let repeats = probe
                .batch
                .rows
                .iter()
                .map(|r| r["seconds"].as_array().unwrap().len())
                .min()
                .unwrap();
            for repeat in 0..repeats {
                samples.push(Measurement {
                    profile: name.clone(),
                    parents: config.parents * config.world as u64,
                    seconds_by_rank: probe
                        .batch
                        .rows
                        .iter()
                        .map(|r| r["seconds"][repeat].as_f64().unwrap())
                        .collect(),
                    correctness_passed: true,
                    memory_admitted: true,
                    throttled: !verified,
                });
            }
            let peak = probe
                .batch
                .rows
                .iter()
                .map(|r| r["torch_reserved_peak_bytes"].as_u64().unwrap())
                .max()
                .unwrap();
            reserves.insert(
                name.clone(),
                peak.checked_add(512 << 20)
                    .ok_or_else(|| Error::Message("inference reserve size overflow".into()))?,
            );
        } else if name == baseline && !(current == 0 && shared.is_some()) {
            if failures.is_empty() || !failures.iter().all(|f| f.capacity) {
                return Err(Error::Message(
                    "baseline inference calibration failed; inspect rank logs".into(),
                ));
            }
            let mut smaller: Vec<_> = candidates[index..]
                .iter()
                .copied()
                .filter(|b| *b < batch)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .rev()
                .collect();
            candidates.truncate(index);
            candidates.append(&mut smaller);
            coarse = candidates.clone();
        }
        records.extend(probe.batch.rows);
        if complete && estimate(&name, &samples).is_ok() {
            if current == 0 && shared == Some(batch) {
                shared_used = true;
                break;
            }
            if config.capacity_bootstrap {
                break;
            }
        }
        if index == coarse.len() && !refinement_queued && Instant::now() < deadline {
            let (winner, _) = select_inference(&samples, &baseline)?;
            refined = refinement_batches(
                winner
                    .profile
                    .parse()
                    .map_err(|_| Error::Message("invalid selected inference batch".into()))?,
                config.cap,
                &candidates.iter().copied().collect(),
            )?;
            candidates.extend(&refined);
            refinement_queued = true;
        }
    }
    let (estimate, stat_rejected) = select_inference(&samples, &baseline)?;
    let parent_batch = estimate
        .profile
        .parse()
        .map_err(|_| Error::Message("invalid selected inference batch".into()))?;
    let reserve_bytes = reserves[&estimate.profile];
    let measured_candidates = samples
        .iter()
        .filter_map(|r| r.profile.parse().ok())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(Selection {
        search_policy: "rust-inference-v1-splitmix64".into(),
        phase: "inference_verified".into(),
        parent_batch,
        reserve_bytes,
        estimate,
        records,
        measurements: samples,
        rejected,
        stat_rejected,
        gpu_telemetry: telemetry_records,
        coarse_candidates: coarse,
        refinement_candidates: refined,
        calibration_wall_seconds: started.elapsed().as_secs_f64(),
        probe_process_starts: 0,
        capacity_limited_search: capacity_limited,
        shared_profile_validated: shared_used,
        measured_candidates,
        pipeline_verified: false,
        cache_hit: false,
    })
}
