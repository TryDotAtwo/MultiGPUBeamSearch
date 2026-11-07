//! Preserve the numeric frontier; only verified measured batches may replace it.
use crate::{profile_cache, Error, Result};
use beam_core::{
    calibration::Estimate,
    planning::{validate_rank_plans, RankPlan},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub parent_batch: u64,
    pub reserve_bytes: u64,
    pub estimate: Estimate,
}
pub fn measured_candidates(data: &Value, max_batch: u64) -> Result<Vec<Candidate>> {
    let rows = data["records"]
        .as_array()
        .ok_or_else(|| Error::Message("missing inference records".into()))?;
    let signature = &data["signature"];
    let cap = signature["max_batch"]
        .as_u64()
        .unwrap_or(max_batch)
        .min(max_batch);
    let batches: BTreeSet<_> = rows
        .iter()
        .filter_map(|r| r["batch"].as_u64())
        .filter(|n| *n > 0 && *n <= cap)
        .collect();
    let mut candidates = Vec::new();
    for batch in batches {
        let peak = rows
            .iter()
            .filter(|r| r["batch"].as_u64() == Some(batch))
            .filter_map(|r| r["torch_reserved_peak_bytes"].as_u64())
            .max();
        let Some(reserve) = peak.and_then(|n| n.checked_add(512 << 20)) else {
            continue;
        };
        let mut candidate = data.clone();
        candidate["parent_batch"] = json!(batch);
        candidate["reserve_bytes"] = json!(reserve);
        // Recompute quality, rank membership, timing and telemetry from raw rows.
        if let Ok(verified) = profile_cache::normalize(&candidate, signature, cap) {
            candidates.push(Candidate {
                parent_batch: batch,
                reserve_bytes: reserve,
                estimate: serde_json::from_value(verified["estimate"].clone())?,
            });
        }
    }
    candidates.sort_by(|a, b| {
        (Some(a.parent_batch) != data["parent_batch"].as_u64())
            .cmp(&(Some(b.parent_batch) != data["parent_batch"].as_u64()))
            .then_with(|| a.estimate.median.total_cmp(&b.estimate.median))
            .then_with(|| a.reserve_bytes.cmp(&b.reserve_bytes))
            .then_with(|| a.parent_batch.cmp(&b.parent_batch))
    });
    Ok(candidates)
}
pub fn with_candidate(data: &Value, candidate: &Candidate) -> Value {
    let mut result = data.clone();
    result["parent_batch"] = json!(candidate.parent_batch);
    result["reserve_bytes"] = json!(candidate.reserve_bytes);
    result["estimate"] = json!(candidate.estimate);
    result
}
pub fn capacity_seed(data: &Value, max_batch: u64) -> Result<Value> {
    let candidate = measured_candidates(data, max_batch)?
        .into_iter()
        .min_by(|a, b| a.estimate.median.total_cmp(&b.estimate.median))
        .ok_or_else(|| {
            Error::CapacityRejected("no verified inference batch for maximum frontier".into())
        })?;
    let mut result = with_candidate(data, &candidate);
    result["capacity_bootstrap"] = json!({"objective":"fastest verified inference first; frontier uses remaining memory",
        "unconstrained_batch":data["parent_batch"]});
    Ok(result)
}
pub fn select_admitted(
    data: &Value,
    max_batch: u64,
    beam: u64,
    mut admit: impl FnMut(&Candidate) -> Result<Vec<Value>>,
) -> Result<(Value, Vec<Value>)> {
    if data["signature"]["requested_beam_width"].as_u64() != Some(beam) || beam == 0 {
        return Err(Error::Message(
            "inference admission requires exact requested frontier".into(),
        ));
    }
    let mut attempts = vec![];
    for candidate in measured_candidates(data, max_batch)? {
        let plans = match admit(&candidate) {
            Ok(plans) => plans,
            Err(Error::CapacityRejected(reason)) => {
                attempts.push(json!({"candidate":candidate,"admitted":false,"reason":reason}));
                continue;
            }
            Err(error) => return Err(error),
        };
        let typed: Vec<RankPlan> = serde_json::from_value(json!(plans))?;
        validate_rank_plans(&typed, beam)?;
        if data["signature"]["world_size"].as_u64() != Some(typed.len() as u64)
            || typed
                .iter()
                .enumerate()
                .any(|(rank, plan)| plan.rank != Some(rank as u32))
        {
            return Err(Error::Message(
                "inference admission does not contain the requested ordered rank cohort".into(),
            ));
        }
        attempts.push(json!({"candidate":candidate,"admitted":true}));
        let mut result = with_candidate(data, &candidate);
        result["inference_admission"] = json!({"unconstrained_batch":data["parent_batch"],
            "unconstrained_estimate":data["estimate"],"attempts":attempts,"requested_beam":beam,"plans":plans});
        return Ok((result, plans));
    }
    Err(Error::CapacityRejected(format!(
        "no measured inference batch admits requested frontier {beam}"
    )))
}
