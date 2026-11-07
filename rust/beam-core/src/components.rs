//! Typed exact-capacity component receipts and a scheduling proxy, not wall time.
use crate::{
    planning::{validate_rank_plans, RankPlan},
    ContractError,
};
use serde::{Deserialize, Serialize};
type Result<T> = std::result::Result<T, ContractError>;
fn invalid() -> ContractError {
    ContractError::Invalid("component calibration contract")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentPlan {
    #[serde(flatten)]
    pub base: RankPlan,
    pub frontier_state_capacity: u64,
    #[serde(rename = "STREAM3_RING_SLOTS")]
    pub staging: u64,
    #[serde(rename = "STREAM4_BATCH_CANDIDATES")]
    pub flush: u64,
    #[serde(rename = "SHARD_CAPACITY_CANDIDATES")]
    pub capacity: u64,
    #[serde(rename = "STREAM4_ACTIVE_SORT_SLOTS")]
    pub sort_slots: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransportReceipt {
    pub items: u64,
    pub seconds: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentReceipt {
    pub component_calibration: bool,
    pub correctness_passed: bool,
    pub full_step_verified: bool,
    pub rank: u32,
    pub outer_candidates: u64,
    pub shard_capacity: u64,
    pub sort_jobs_concurrent: u64,
    pub stream3_seconds: Vec<f64>,
    pub stream4_group_seconds: Vec<f64>,
    pub union_seconds: Vec<f64>,
    pub transport: Vec<TransportReceipt>,
}
fn times(values: &[f64]) -> Result<()> {
    if values.len() < 5 || values.iter().any(|v| !v.is_finite() || *v <= 0.) {
        Err(invalid())
    } else {
        Ok(())
    }
}
fn validate_row(
    row: &ComponentReceipt,
    plan: &ComponentPlan,
    moves: u32,
    rank: usize,
) -> Result<()> {
    let outer = plan
        .base
        .outer_batch
        .checked_mul(plan.staging)
        .and_then(|n| n.checked_mul(moves as u64))
        .ok_or(ContractError::Overflow("component outer candidates"))?;
    if !(1..=255).contains(&moves)
        || plan.staging == 0
        || plan.flush == 0
        || plan.capacity == 0
        || plan.sort_slots == 0
        || plan.base.shards == 0
        || plan.frontier_state_capacity == 0
        || !row.component_calibration
        || !row.correctness_passed
        || row.full_step_verified
        || row.rank != rank as u32
        || row.outer_candidates != outer
        || row.shard_capacity != plan.capacity
        || row.sort_jobs_concurrent == 0
        || row.sort_jobs_concurrent
            != plan.sort_slots.min(
                plan.base
                    .shards
                    .checked_mul(2)
                    .ok_or(ContractError::Overflow("physical component shard buffers"))?,
            )
    {
        return Err(invalid());
    }
    for values in [
        &row.stream3_seconds,
        &row.stream4_group_seconds,
        &row.union_seconds,
    ] {
        times(values)?;
    }
    let mut previous = 0;
    for volume in &row.transport {
        if volume.items <= previous || volume.items > outer {
            return Err(invalid());
        }
        previous = volume.items;
        times(&volume.seconds)?;
    }
    if (plan.base.world > 1 && previous != outer)
        || (plan.base.world == 1 && !row.transport.is_empty())
    {
        return Err(invalid());
    }
    Ok(())
}
pub fn validate_components(
    rows: &[ComponentReceipt],
    plans: &[ComponentPlan],
    beam: u64,
    moves: u32,
) -> Result<()> {
    validate_rank_plans(
        &plans.iter().map(|p| p.base.clone()).collect::<Vec<_>>(),
        beam,
    )?;
    if rows.len() != plans.len() {
        return Err(invalid());
    }
    for (rank, (row, plan)) in rows.iter().zip(plans).enumerate() {
        validate_row(row, plan, moves, rank)?;
    }
    Ok(())
}
fn sample(values: &[f64], repeat: Option<usize>) -> Result<f64> {
    if let Some(index) = repeat {
        return values.get(index).copied().ok_or_else(invalid);
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    Ok(if ordered.len() % 2 == 0 {
        ordered[middle - 1] * 0.5 + ordered[middle] * 0.5
    } else {
        ordered[middle]
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceEnvelope {
    pub service_envelope_seconds: f64,
    pub service_work_seconds: f64,
    pub downstream_service_work_seconds: f64,
    pub inference_seconds: f64,
    pub stream3_seconds: f64,
    pub stream4_seconds: f64,
    pub transport_seconds: f64,
    pub union_seconds: f64,
    pub scope: String,
}
pub fn service_envelope(
    rows: &[ComponentReceipt],
    plan: &ComponentPlan,
    moves: u32,
    inference_seconds: f64,
    repeat: Option<usize>,
) -> Result<ServiceEnvelope> {
    if !inference_seconds.is_finite()
        || inference_seconds <= 0.
        || rows.len() != plan.base.world as usize
        || rows.is_empty()
    {
        return Err(invalid());
    }
    for (rank, row) in rows.iter().enumerate() {
        validate_row(row, plan, moves, rank)?;
    }
    let gross = plan.frontier_state_capacity as u128 * moves as u128;
    let slow = |get: fn(&ComponentReceipt) -> &[f64]| -> Result<f64> {
        rows.iter()
            .map(|r| sample(get(r), repeat))
            .try_fold(0.0f64, |a, b| Ok(a.max(b?)))
    };
    let inference =
        plan.frontier_state_capacity as f64 * plan.base.world as f64 * inference_seconds;
    let s3 =
        gross.div_ceil(rows[0].outer_candidates as u128) as f64 * slow(|r| &r.stream3_seconds)?;
    let jobs = plan.base.shards as u128
        * gross
            .div_ceil(plan.base.shards as u128)
            .div_ceil(plan.flush.min(plan.capacity) as u128);
    let s4 = jobs.div_ceil(rows[0].sort_jobs_concurrent as u128) as f64
        * slow(|r| &r.stream4_group_seconds)?;
    let mut transport = 0.0f64;
    for row in rows {
        if let Some(last) = row.transport.last() {
            transport = transport
                .max(gross.div_ceil(last.items as u128) as f64 * sample(&last.seconds, repeat)?);
        }
    }
    let union = plan.base.shards as f64 * slow(|r| &r.union_seconds)?;
    let downstream = s3 + s4 + transport + union;
    let work = inference + downstream;
    let envelope = inference.max(s3).max(s4).max(transport) + union;
    if !work.is_finite() || !envelope.is_finite() {
        return Err(ContractError::Overflow("service timing proxy"));
    }
    Ok(ServiceEnvelope {
        service_envelope_seconds: envelope,
        service_work_seconds: work,
        downstream_service_work_seconds: downstream,
        inference_seconds: inference,
        stream3_seconds: s3,
        stream4_seconds: s4,
        transport_seconds: transport,
        union_seconds: union,
        scope: "isolated exact-capacity service envelope; not full-step timing".into(),
    })
}
