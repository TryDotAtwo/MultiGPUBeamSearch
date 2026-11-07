//! Exact logical frontier and all-rank memory admission contracts.
use crate::{require_capacity, ContractError};
use serde::{Deserialize, Serialize};

pub const TUNABLE_KEYS: [&str; 5] = [
    "BEAM_B_MICRO",
    "BEAM_STREAM3_RING_SLOTS",
    "BEAM_SHARD_COUNT",
    "BEAM_STREAM4_BATCH_CANDIDATES",
    "BEAM_STREAM4_ACTIVE_SORT_SLOTS",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RankPlan {
    #[serde(rename = "GLOBAL_BEAM_WIDTH_EFFECTIVE")]
    pub global_beam_width: u64,
    #[serde(rename = "BEAM_WIDTH_ALIGNMENT")]
    pub beam_alignment: u64,
    #[serde(rename = "SHARD_COUNT")]
    pub shards: u64,
    #[serde(rename = "B_MICRO")]
    pub outer_batch: u64,
    #[serde(rename = "WORLD_SIZE")]
    pub world: u32,
    #[serde(rename = "STREAM4_BATCH_ALIGNMENT")]
    pub batch_alignment: u64,
    // Historical native LOCAL_RANK means communicator rank, not device ordinal.
    #[serde(rename = "LOCAL_RANK")]
    pub rank: Option<u32>,
    #[serde(rename = "estimated_required_device_bytes")]
    pub required_bytes: u64,
    #[serde(rename = "gpu_budget_bytes")]
    pub budget_bytes: u64,
}

pub fn validate_rank_plans(plans: &[RankPlan], requested: u64) -> Result<(), ContractError> {
    let first = plans
        .first()
        .ok_or(ContractError::Invalid("no rank plans"))?;
    if requested == 0 || first.global_beam_width != requested {
        return Err(ContractError::Invalid("exact requested frontier"));
    }
    if first.world == 0 || first.world as usize != plans.len() {
        return Err(ContractError::Invalid("missing rank admission"));
    }
    let common = |p: &RankPlan| {
        (
            p.global_beam_width,
            p.beam_alignment,
            p.shards,
            p.outer_batch,
            p.world,
            p.batch_alignment,
        )
    };
    if first.beam_alignment == 0
        || first.shards == 0
        || first.outer_batch == 0
        || first.batch_alignment == 0
    {
        return Err(ContractError::Invalid("zero plan geometry"));
    }
    let ranks_present = plans.iter().any(|p| p.rank.is_some());
    let mut seen = std::collections::BTreeSet::new();
    for row in plans {
        if common(row) != common(first) {
            return Err(ContractError::Invalid("rank plans disagree on protocol"));
        }
        if ranks_present && !row.rank.is_some_and(|r| r < first.world && seen.insert(r)) {
            return Err(ContractError::Invalid(
                "duplicate or missing communicator rank admission",
            ));
        }
        require_capacity("rank GPU memory", row.required_bytes, row.budget_bytes)?;
    }
    Ok(())
}

pub fn rank_frontier_counts(plans: &[RankPlan], requested: u64) -> Result<Vec<u64>, ContractError> {
    validate_rank_plans(plans, requested)?;
    let world = plans.len() as u128;
    Ok((0..world)
        .map(|rank| {
            (((rank + 1) * requested as u128).div_ceil(world)
                - (rank * requested as u128).div_ceil(world)) as u64
        })
        .collect())
}
