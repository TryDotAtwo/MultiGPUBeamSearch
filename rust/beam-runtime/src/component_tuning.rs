//! Exact-capacity downstream probes. Their sum is a proxy, never a depth time.
use crate::{telemetry::Receipt, Error, Result};
use beam_core::{
    calibration::{estimate, select, Measurement},
    components::{
        service_envelope, validate_components, ComponentPlan, ComponentReceipt, ServiceEnvelope,
    },
    profile_policy::{self, Environment, Hardware, Proposals, SearchGeometry},
    ContractError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneConfig {
    pub beam: u64,
    pub moves: u32,
    pub micro: u64,
    pub row_factor: u64,
    pub inference_seconds: f64,
    pub hardware: Hardware,
    pub gpu_uuids: Vec<String>,
    pub device_indices: Vec<u32>,
}
pub struct ComponentProbe {
    pub rows: Vec<Value>,
    pub telemetry: Receipt,
}
pub trait ComponentEvaluator {
    fn admit(&mut self, environment: &Environment) -> Result<Vec<Value>>;
    fn measure(&mut self, environment: &Environment, plans: &[Value]) -> Result<ComponentProbe>;
    fn close(&mut self) -> Result<()>;
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trial {
    pub name: String,
    pub environment: Environment,
    pub plans: Vec<Value>,
    pub rank_measurements: Vec<Value>,
    pub telemetry: Option<Receipt>,
    pub estimate: Option<ServiceEnvelope>,
    pub rejected: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Selection {
    pub phase: String,
    pub environment: Environment,
    pub requested_beam_effective: u64,
    pub workload_parents: u64,
    pub measurement_scope: String,
    pub pipeline_verified: bool,
    pub cache_hit: bool,
    pub tested: Vec<Trial>,
    pub selection: String,
    pub estimate: ServiceEnvelope,
    pub selection_rejections: BTreeMap<String, String>,
    pub selection_objective: String,
    pub search_policy: String,
    pub search_complete: bool,
    pub proposals: Proposals,
    pub baseline_retry: String,
    pub initial_baseline: Option<Trial>,
}
fn contract(reason: &'static str) -> Error {
    ContractError::Invalid(reason).into()
}
pub fn checked_plans(config: &TuneConfig, raw: &[Value]) -> Result<Vec<ComponentPlan>> {
    let plans: Vec<ComponentPlan> = raw
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<std::result::Result<_, _>>()?;
    beam_core::planning::validate_rank_plans(
        &plans.iter().map(|p| p.base.clone()).collect::<Vec<_>>(),
        config.beam,
    )?;
    let first = &plans[0];
    let common = |p: &ComponentPlan| {
        (
            p.frontier_state_capacity,
            p.staging,
            p.flush,
            p.capacity,
            p.sort_slots,
        )
    };
    // Native plans report parents; BEAM_B_MICRO is input rows for a scalar MLP.
    let quantum = config.micro;
    if config.device_indices.len() != plans.len()
        || quantum == 0
        || plans.iter().enumerate().any(|(rank, p)| {
            p.base.rank != Some(rank as u32)
                || common(p) != common(first)
                || p.base.outer_batch % quantum != 0
        })
        || first.frontier_state_capacity < config.beam.div_ceil(plans.len() as u64)
        || [first.staging, first.flush, first.capacity, first.sort_slots].contains(&0)
    {
        return Err(contract(
            "component admission rank cohort/geometry mismatch",
        ));
    }
    Ok(plans)
}
pub fn normalized_environment(config: &TuneConfig, plans: &[Value]) -> Result<Environment> {
    let typed = checked_plans(config, plans)?;
    let p = &typed[0];
    let environment = [
        ("BEAM_ENSEMBLE_INFERENCE_MICRO", config.micro),
        (
            "BEAM_B_MICRO",
            p.base
                .outer_batch
                .checked_mul(config.row_factor)
                .ok_or(ContractError::Overflow("profile outer input rows"))?,
        ),
        ("BEAM_SHARD_COUNT", p.base.shards),
        ("BEAM_STREAM3_RING_SLOTS", p.staging),
        ("BEAM_STREAM4_ACTIVE_SORT_SLOTS", p.sort_slots),
        ("BEAM_STREAM4_BATCH_CANDIDATES", p.flush),
    ]
    .map(|(k, v)| (k.into(), v.to_string()))
    .into();
    profile_policy::validate_environment(&environment, config.micro)?;
    Ok(environment)
}
/// Recompute every selection input from the original native rows on cache reads.
pub fn validate_trial(config: &TuneConfig, trial: &Trial) -> Result<Vec<Measurement>> {
    if trial.rejected.is_some()
        || trial.name.is_empty()
        || trial.environment != normalized_environment(config, &trial.plans)?
    {
        return Err(contract("component profile/environment receipt mismatch"));
    }
    let plans = checked_plans(config, &trial.plans)?;
    let rows: Vec<ComponentReceipt> = trial
        .rank_measurements
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<std::result::Result<_, _>>()?;
    validate_components(&rows, &plans, config.beam, config.moves)?;
    if !trial
        .telemetry
        .as_ref()
        .is_some_and(|r| r.verified(&config.gpu_uuids, &config.device_indices))
    {
        return Err(contract("component GPU telemetry missing or throttled"));
    }
    (0..5)
        .map(|i| {
            let service = service_envelope(
                &rows,
                &plans[0],
                config.moves,
                config.inference_seconds,
                Some(i),
            )?;
            Ok(Measurement {
                profile: trial.name.clone(),
                parents: config.beam,
                seconds_by_rank: vec![service.downstream_service_work_seconds],
                correctness_passed: true,
                memory_admitted: true,
                throttled: false,
            })
        })
        .collect()
}
pub fn measure_profile(
    evaluator: &mut impl ComponentEvaluator,
    config: &TuneConfig,
    name: &str,
    environment: &Environment,
) -> Result<Trial> {
    profile_policy::validate_environment(environment, config.micro)?;
    let plans = evaluator.admit(environment)?;
    let normalized = normalized_environment(config, &plans)?;
    let probe = evaluator.measure(&normalized, &plans)?;
    let mut trial = Trial {
        name: name.into(),
        environment: normalized,
        plans,
        rank_measurements: probe.rows,
        telemetry: Some(probe.telemetry),
        estimate: None,
        rejected: None,
    };
    validate_trial(config, &trial)?;
    let typed = checked_plans(config, &trial.plans)?;
    let rows = trial
        .rank_measurements
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    trial.estimate = Some(service_envelope(
        &rows,
        &typed[0],
        config.moves,
        config.inference_seconds,
        None,
    )?);
    Ok(trial)
}
fn run(
    evaluator: &mut impl ComponentEvaluator,
    config: &TuneConfig,
    baseline: &Environment,
    deadline: Instant,
) -> Result<Selection> {
    if config.beam == 0
        || !(1..=255).contains(&config.moves)
        || config.device_indices.is_empty()
        || config.device_indices.len() > 65535
        || config.device_indices.iter().collect::<BTreeSet<_>>().len()
            != config.device_indices.len()
        || (config.row_factor != 1 && config.row_factor != config.moves as u64)
        || !config.inference_seconds.is_finite()
        || config.inference_seconds <= 0.
        || Instant::now() >= deadline
    {
        return Err(contract("component tuning configuration/deadline"));
    }
    let mut base = measure_profile(evaluator, config, "baseline", baseline)?;
    let mut samples = validate_trial(config, &base)?;
    let mut initial_baseline = None;
    let mut baseline_retry = "not_needed";
    if let Err(error) = estimate("baseline", &samples) {
        if !matches!(error, ContractError::Invalid("unstable timing samples"))
            || Instant::now() >= deadline
        {
            return Err(error.into());
        }
        let retry = measure_profile(evaluator, config, "baseline", &base.environment)?;
        let retry_samples = validate_trial(config, &retry)?;
        estimate("baseline", &retry_samples)?;
        initial_baseline = Some(base);
        base = retry;
        samples = retry_samples;
        baseline_retry = "recovered_after_one_retry";
    }
    let plan = checked_plans(config, &base.plans)?.remove(0);
    let shape = SearchGeometry {
        beam: config.beam,
        world: plan.base.world,
        moves: config.moves,
        alignment: plan.base.batch_alignment,
        row_factor: config.row_factor,
    };
    let proposals = profile_policy::proposals(
        &shape,
        &base.environment,
        &config.hardware,
        base.estimate.as_ref().expect("validated service estimate"),
        false,
        &BTreeSet::new(),
    )?;
    let mut tested = vec![base];
    let mut search_complete = true;
    for proposal in &proposals.candidates {
        if Instant::now() >= deadline {
            search_complete = false;
            break;
        }
        let name = format!("coarse-{}", proposal.name);
        match measure_profile(evaluator, config, &name, &proposal.environment) {
            Ok(trial) => {
                samples.extend(validate_trial(config, &trial)?);
                tested.push(trial);
            }
            Err(error @ (Error::CapacityRejected(_) | Error::Contract(_))) => tested.push(Trial {
                name,
                environment: proposal.environment.clone(),
                plans: vec![],
                rank_measurements: vec![],
                telemetry: None,
                estimate: None,
                rejected: Some(error.to_string()),
            }),
            Err(error) => return Err(error),
        }
    }
    let (winner, mut rejections) = select(&samples, "baseline", 0.05)?;
    for trial in &tested {
        if let Some(reason) = &trial.rejected {
            rejections.insert(trial.name.clone(), reason.clone());
        }
    }
    let best = tested
        .iter()
        .find(|r| r.name == winner.profile)
        .expect("selection from measured profiles");
    Ok(Selection {
        phase: "component_calibrated".into(),
        environment: best.environment.clone(),
        requested_beam_effective: config.beam,
        workload_parents: config.beam,
        measurement_scope: "exact_admitted_component_capacities".into(),
        pipeline_verified: false,
        cache_hit: false,
        selection: best.name.clone(),
        estimate: best.estimate.clone().expect("validated service estimate"),
        tested,
        selection_rejections: rejections,
        selection_objective:
            "minimize summed downstream service work; proxy only, not full-step wall time".into(),
        search_policy: profile_policy::POLICY.into(),
        search_complete,
        proposals,
        baseline_retry: baseline_retry.into(),
        initial_baseline,
    })
}
pub fn tune_components(
    evaluator: &mut impl ComponentEvaluator,
    config: &TuneConfig,
    baseline: &Environment,
    deadline: Instant,
) -> Result<Selection> {
    let result = run(evaluator, config, baseline, deadline);
    let cleanup = evaluator.close();
    let value = result?;
    cleanup?;
    Ok(value)
}
