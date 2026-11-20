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


const PUBLIC_PROFILE_ENV_KEYS: &[&str] = &[
    "BEAM_B_MICRO",
    "BEAM_STREAM3_RING_SLOTS",
    "BEAM_SHARD_COUNT",
    "BEAM_STREAM4_BATCH_CANDIDATES",
    "BEAM_STREAM4_ACTIVE_SORT_SLOTS",
    "BEAM_ENSEMBLE_INFERENCE_MICRO",
];

fn public_json_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().is_some_and(|number| number != 0.0),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
    }
}

fn public_nonnegative_integer(value: Option<&Value>) -> Option<u64> {
    value?.as_u64()
}

fn public_positive_seconds(value: &Value) -> bool {
    let number = match value {
        // Python's math.isfinite accepts bool because bool subclasses int.
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Number(value) => value.as_f64(),
        _ => None,
    };
    number.is_some_and(|number| number.is_finite() && number > 0.0)
}

fn public_rank_plans_valid(plans: &[Value]) -> bool {
    if plans.is_empty() {
        return false;
    }
    let Some(first) = plans[0].as_object() else {
        return false;
    };
    let common = [
        "GLOBAL_BEAM_WIDTH_EFFECTIVE",
        "BEAM_WIDTH_ALIGNMENT",
        "SHARD_COUNT",
        "B_MICRO",
        "WORLD_SIZE",
        "STREAM4_BATCH_ALIGNMENT",
    ];
    for key in common {
        let Some(expected) = first.get(key) else {
            return false;
        };
        if plans.iter().any(|plan| {
            plan.as_object()
                .and_then(|row| row.get(key))
                .is_none_or(|value| value != expected)
        }) {
            return false;
        }
    }

    let Some(world) = public_nonnegative_integer(first.get("WORLD_SIZE")) else {
        return false;
    };
    if world == 0 || world != plans.len() as u64 {
        return false;
    }

    if plans.iter().any(|plan| {
        plan.as_object()
            .is_none_or(|row| row.contains_key("LOCAL_RANK"))
    }) {
        let mut ranks = vec![false; plans.len()];
        for plan in plans {
            let Some(rank) = plan
                .as_object()
                .and_then(|row| public_nonnegative_integer(row.get("LOCAL_RANK")))
            else {
                return false;
            };
            let Ok(rank) = usize::try_from(rank) else {
                return false;
            };
            if rank >= ranks.len() || ranks[rank] {
                return false;
            }
            ranks[rank] = true;
        }
        if ranks.iter().any(|seen| !seen) {
            return false;
        }
    }

    plans.iter().all(|plan| {
        let Some(row) = plan.as_object() else {
            return false;
        };
        let Some(required) = public_nonnegative_integer(row.get("estimated_required_device_bytes"))
        else {
            return false;
        };
        let Some(budget) = public_nonnegative_integer(row.get("gpu_budget_bytes")) else {
            return false;
        };
        required <= budget
    })
}

fn valid_public_component_receipt(data: &Value, identity: &Value) -> bool {
    let (Some(data), Some(identity)) = (data.as_object(), identity.as_object()) else {
        return false;
    };
    if data.get("cache_identity") != Some(&Value::Object(identity.clone()))
        || data.get("phase").and_then(Value::as_str) != Some("component_calibrated")
        || data.get("pipeline_verified") != Some(&Value::Bool(false))
    {
        return false;
    }

    let Some(micro) = public_nonnegative_integer(identity.get("parent_batch")) else {
        return false;
    };
    let Some(signature) = identity.get("signature").and_then(Value::as_object) else {
        return false;
    };
    let Some(world) = public_nonnegative_integer(signature.get("world_size")) else {
        return false;
    };
    let Some(requested_width) = public_nonnegative_integer(signature.get("requested_beam_width"))
    else {
        return false;
    };
    if world == 0 {
        return false;
    }

    let Some(environment_value) = data.get("environment") else {
        return false;
    };
    let Some(environment) = environment_value.as_object() else {
        return false;
    };
    if environment.keys().any(|key| !PUBLIC_PROFILE_ENV_KEYS.contains(&key.as_str()))
        || environment.get("BEAM_ENSEMBLE_INFERENCE_MICRO")
            != Some(&Value::String(micro.to_string()))
    {
        return false;
    }
    if environment.values().any(|value| {
        let Some(value) = value.as_str() else {
            return true;
        };
        if value.is_empty() || !value.is_ascii() || !value.bytes().all(|byte| byte.is_ascii_digit())
        {
            return true;
        }
        match value.parse::<u128>() {
            Ok(value) => value == 0 || value > (1_u128 << 40),
            Err(_) => true,
        }
    }) {
        return false;
    }

    let Some(selection) = data.get("selection").and_then(Value::as_str) else {
        return false;
    };
    let Some(tested) = data.get("tested").and_then(Value::as_array) else {
        return false;
    };
    let chosen: Vec<_> = tested
        .iter()
        .filter(|row| row.get("name").and_then(Value::as_str) == Some(selection))
        .collect();
    if chosen.len() != 1 || chosen[0].get("environment") != Some(environment_value) {
        return false;
    }
    let Some(plans) = chosen[0].get("plans").and_then(Value::as_array) else {
        return false;
    };
    if !public_rank_plans_valid(plans) || plans.len() as u64 != world {
        return false;
    }
    let Some(plan_width) = plans[0]
        .get("GLOBAL_BEAM_WIDTH_EFFECTIVE")
        .and_then(Value::as_u64)
    else {
        return false;
    };
    if data.get("requested_beam_width").and_then(Value::as_u64) != Some(requested_width)
        || data.get("requested_beam_effective").and_then(Value::as_u64) != Some(plan_width)
        || plan_width != requested_width
    {
        return false;
    }

    let Some(rows) = chosen[0].get("rank_measurements").and_then(Value::as_array) else {
        return false;
    };
    if rows.len() as u64 != world {
        return false;
    }
    let mut ranks = vec![false; rows.len()];
    for row in rows {
        let Some(rank) = row.get("rank").and_then(Value::as_u64) else {
            return false;
        };
        let Ok(rank) = usize::try_from(rank) else {
            return false;
        };
        if rank >= ranks.len() || ranks[rank] {
            return false;
        }
        ranks[rank] = true;
        if row.get("correctness_passed") != Some(&Value::Bool(true))
            || row.get("full_step_verified") != Some(&Value::Bool(false))
            || row.get("shard_capacity")
                != plans[rank].get("SHARD_CAPACITY_CANDIDATES")
        {
            return false;
        }
        for key in ["stream3_seconds", "stream4_group_seconds", "union_seconds"] {
            let Some(samples) = row.get(key).and_then(Value::as_array) else {
                return false;
            };
            if samples.len() < 5 || samples.iter().any(|sample| !public_positive_seconds(sample)) {
                return false;
            }
        }
        let Some(transfers) = row.get("transport").and_then(Value::as_array) else {
            return false;
        };
        for transfer in transfers {
            let Some(items) = transfer.get("items").and_then(Value::as_u64) else {
                return false;
            };
            let Some(samples) = transfer.get("seconds").and_then(Value::as_array) else {
                return false;
            };
            if items == 0
                || samples.len() < 5
                || samples.iter().any(|sample| !public_positive_seconds(sample))
            {
                return false;
            }
        }
    }
    ranks.iter().all(|seen| *seen)
}

/// Returns None for full-frontier receipts or JSON the Python compatibility
/// validator must handle. Component receipts are validated here and return Some.
pub fn validate_pipeline_profile_receipt_json(data: &str, identity: &str) -> Option<bool> {
    let data: Value = serde_json::from_str(data).ok()?;
    let identity: Value = serde_json::from_str(identity).ok()?;
    if public_json_truthy(identity.get("full_frontier")) {
        return None;
    }
    Some(valid_public_component_receipt(&data, &identity))
}
