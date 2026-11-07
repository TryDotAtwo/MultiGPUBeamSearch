//! Component evidence is reusable only after fresh all-rank native admission.
use crate::{
    artifacts,
    component_tuning::{checked_plans, validate_trial, Selection, TuneConfig},
    Error, Result,
};
use beam_core::{
    calibration::select,
    components::{service_envelope, ComponentReceipt},
    profile_policy::{self, Environment},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, fs::File, io::Read, path::Path};
const MAX_CACHE_BYTES: u64 = 64 << 20;
#[derive(Serialize, Deserialize)]
struct Record {
    schema: u32,
    identity: Value,
    selection: Selection,
}
pub struct Hit {
    pub selection: Selection,
    pub current_rank_plans: Vec<Value>,
}
fn invalid() -> Error {
    Error::Message("invalid or incomplete component cache".into())
}
fn service_matches(
    left: &beam_core::components::ServiceEnvelope,
    right: &beam_core::components::ServiceEnvelope,
) -> bool {
    let values = |s: &beam_core::components::ServiceEnvelope| {
        [
            s.service_envelope_seconds,
            s.service_work_seconds,
            s.downstream_service_work_seconds,
            s.inference_seconds,
            s.stream3_seconds,
            s.stream4_seconds,
            s.transport_seconds,
            s.union_seconds,
        ]
    };
    left.scope == right.scope
        && values(left).iter().zip(values(right)).all(|(a, b)| {
            a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= a.abs().max(b.abs()).max(1e-300) * 1e-12
        })
}
pub fn validate_components(config: &TuneConfig, data: &Selection) -> Result<()> {
    if data.phase != "component_calibrated"
        || data.pipeline_verified
        || !data.search_complete
        || data.search_policy != profile_policy::POLICY
        || data.workload_parents != config.beam
        || data.requested_beam_effective != config.beam
        || data.tested.first().is_none_or(|r| r.name != "baseline")
    {
        return Err(invalid());
    }
    profile_policy::validate_environment(&data.environment, config.micro)?;
    let mut names = BTreeSet::new();
    let mut samples = Vec::new();
    for trial in &data.tested {
        if !names.insert(&trial.name) {
            return Err(invalid());
        }
        if trial.rejected.is_some() {
            continue;
        }
        samples.extend(validate_trial(config, trial)?);
        let plans = checked_plans(config, &trial.plans)?;
        let rows: Vec<ComponentReceipt> = trial
            .rank_measurements
            .iter()
            .cloned()
            .map(serde_json::from_value)
            .collect::<std::result::Result<_, _>>()?;
        let actual = service_envelope(
            &rows,
            &plans[0],
            config.moves,
            config.inference_seconds,
            None,
        )?;
        if !trial
            .estimate
            .as_ref()
            .is_some_and(|claimed| service_matches(claimed, &actual))
        {
            return Err(invalid());
        }
    }
    let (winner, _) = select(&samples, "baseline", 0.05)?;
    let chosen = data
        .tested
        .iter()
        .find(|r| r.name == winner.profile)
        .ok_or_else(invalid)?;
    if winner.profile != data.selection
        || chosen.environment != data.environment
        || !chosen
            .estimate
            .as_ref()
            .is_some_and(|s| service_matches(s, &data.estimate))
    {
        return Err(invalid());
    }
    Ok(())
}
pub fn same_geometry(left: &[Value], right: &[Value]) -> bool {
    let keys = [
        "GLOBAL_BEAM_WIDTH_EFFECTIVE",
        "frontier_state_capacity",
        "B_MICRO",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "SHARD_COUNT",
        "SHARD_CAPACITY_CANDIDATES",
        "STREAM3_RING_SLOTS",
        "RING_COUNT",
        "STREAM4_ACTIVE_SORT_SLOTS",
        "STREAM4_BATCH_CANDIDATES",
        "STREAM4_BATCH_ALIGNMENT",
        "BEAM_WIDTH_ALIGNMENT",
    ];
    !left.is_empty()
        && left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| keys.iter().all(|k| a[*k] == b[*k]))
}
pub fn write_components(
    path: &Path,
    identity: &Value,
    config: &TuneConfig,
    data: &Selection,
) -> Result<()> {
    validate_components(config, data)?;
    artifacts::atomic_json(
        path,
        &Record {
            schema: 1,
            identity: identity.clone(),
            selection: data.clone(),
        },
    )
}
pub fn read_components(
    path: &Path,
    identity: &Value,
    config: &TuneConfig,
    mut admit: impl FnMut(&Environment) -> Result<Vec<Value>>,
) -> Result<Option<Hit>> {
    let read = (|| -> Result<Record> {
        let file = File::open(path)?;
        if file.metadata()?.len() > MAX_CACHE_BYTES {
            return Err(invalid());
        }
        let mut bytes = Vec::new();
        file.take(MAX_CACHE_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_CACHE_BYTES {
            return Err(invalid());
        }
        let record: Record = serde_json::from_slice(&bytes)?;
        if record.schema != 1 || record.identity != *identity {
            return Err(invalid());
        }
        validate_components(config, &record.selection)?;
        Ok(record)
    })();
    let Ok(mut record) = read else {
        return Ok(None);
    };
    let current = match admit(&record.selection.environment) {
        Ok(plans) => plans,
        Err(Error::CapacityRejected(_) | Error::Contract(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    if checked_plans(config, &current).is_err() {
        return Ok(None);
    }
    let old = &record
        .selection
        .tested
        .iter()
        .find(|r| r.name == record.selection.selection)
        .ok_or_else(invalid)?
        .plans;
    if !same_geometry(old, &current) {
        return Ok(None);
    }
    record.selection.cache_hit = true;
    Ok(Some(Hit {
        selection: record.selection,
        current_rank_plans: current,
    }))
}
